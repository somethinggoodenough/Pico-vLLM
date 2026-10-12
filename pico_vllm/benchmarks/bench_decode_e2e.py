"""
End-to-end engine benchmark: Triton vs FlashInfer attention inside the full model.

For each (batch, prompt length) the Engine prefills B random prompts, then decodes
--gen-len tokens per request (EOS ignored) with CUDA-Graph decode. Prefix caching
is off, so requests share no KV. The model is loaded once and only the attention
backend is swapped, so both backends run identical weights.

Reported per config:
  prefill_ms        wall time of the step that prefills all B requests (one at a time,
                    as the engine does)
  itl_p50/p90/p99   decode step latency (ms); every decode step emits B tokens
  decode_tok_s      B * decode_steps / decode time
  ideal_itl_ms      (weight bytes + KV bytes read per step) / measured copy bandwidth
  bw_util_pct       ideal_itl_ms / itl_p50 -- how close a step gets to the memory roofline
  kv_share_pct      estimated KV share of weight+KV traffic; not an attention time fraction
  wrapper_plan_host_us  host duration inside FlashInfer wrapper.plan(), excluding metadata build

Weights: ./weights or --weights PATH. --random-weights skips the download; kernel
speed does not depend on weight values (tokens are meaningless, which is fine here).

Usage (from pico_vllm/):
  python benchmarks/bench_decode_e2e.py --batch-sizes 1 4 8 --prompt-lens 512 2048 8192 \
      --gen-len 128 --out results/e2e.csv
  # timeline for nsys (NVTX ranges per step):
  nsys profile -t cuda,nvtx,osrt -o e2e python benchmarks/bench_decode_e2e.py \
      --batch-sizes 8 --prompt-lens 8192 --repeats 1 --nvtx
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import gc
import random
import statistics
import time
from pathlib import Path

import torch

from bench_common import environment_info, measure_copy_bandwidth_gbps, print_table, write_results
from blockmanager import BlockManager
from cache import PagedKVCache
from engine import Engine
from model import ModelConfig, Qwen25_15B
from ops import get_ops_backend

DTYPE = torch.bfloat16


class StubTokenizer:
    """Prompts are submitted as token ids; generated text is never needed."""
    eos_token_id = -1

    def encode(self, text):
        raise RuntimeError("benchmark submits token ids directly")

    def decode(self, ids):
        return ""


def cdiv(a, b):
    return (a + b - 1) // b


def find_weights(path):
    here = Path(__file__).resolve().parent.parent
    candidates = [Path(path)] if path else [here / "weights", here.parent / "weights"]
    for c in candidates:
        if (c / "model.safetensors").exists() or any(c.glob("*.safetensors")):
            return str(c)
    return None


def build_model(args):
    cfg = ModelConfig(use_cuda=True, ops_backend="triton")
    torch.manual_seed(0)
    model = Qwen25_15B(cfg)
    if not args.random_weights:
        weights = find_weights(args.weights)
        if weights is None:
            raise SystemExit("no weights found; pass --weights PATH or use --random-weights")
        from weights import load_weights
        model = load_weights(model, weights)
        print(f"loaded weights from {weights}")
    else:
        print("using random weights (timing only)")
    return model.to(DTYPE).to("cuda").eval()


def set_attention_backend(model, ops):
    """Point every module of the model at `ops` (all backends share the same norms/params)."""
    for module in model.modules():
        if "ops" in module.__dict__:
            module.ops = ops
    model.cfg.ops_backend = ops.name


def run_workload(engine, batch, prompt_len, gen_len, vocab, rng, nvtx=False):
    if gen_len < 2:
        raise ValueError("gen_len must be >= 2: prefill emits the first token")
    prompts = [[rng.randrange(vocab) for _ in range(prompt_len)] for _ in range(batch)]
    rids = {engine.submit_token_ids(p, max_new_tokens=gen_len, temperature=0.0, top_p=1.0) for p in prompts}
    done, prefill_s, decode_s = set(), 0.0, []
    while not rids <= done:
        is_prefill = len(engine.scheduler.waiting) > 0
        if nvtx:
            torch.cuda.nvtx.range_push("prefill_step" if is_prefill else "decode_step")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        completed = engine.step()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if nvtx:
            torch.cuda.nvtx.range_pop()
        if is_prefill:
            prefill_s += dt
        else:
            decode_s.append(dt)
        done.update(rid for rid, _ in completed)
    # Completion is reported one step before Scheduler moves requests to
    # FINISHED. Drain outside the timed region so the next prefill does not
    # inherit this workload's KV release cost or block occupancy.
    engine.step()
    torch.cuda.synchronize()
    engine.scheduler.clear_finished()
    return prefill_s, decode_s


def percentile(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * (len(xs) - 1)))]


def bench_config(model, ops, args, batch, prompt_len, copy_bw, weight_bytes, rng):
    cfg = model.cfg
    gen = args.gen_len
    max_seq = prompt_len + gen + cfg.BLOCK_SIZE
    num_blocks = batch * cdiv(max_seq, cfg.BLOCK_SIZE) + 8  # + padding page + slack
    kv_cache_bytes = (2 * num_blocks * cfg.num_hidden_layers * cfg.local_num_key_value_heads
                      * cfg.BLOCK_SIZE * cfg.head_dim * 2)
    free, _ = torch.cuda.mem_get_info()
    if kv_cache_bytes > free - args.reserve_gb * 2**30:
        print(f"  skip {ops.name} B={batch} P={prompt_len}: KV cache needs {kv_cache_bytes / 2**30:.2f} GiB, "
              f"{free / 2**30:.2f} GiB free")
        return None

    bm = BlockManager(
        num_gpu_blocks=num_blocks, num_cpu_blocks=0, block_size=cfg.BLOCK_SIZE,
        num_layers=cfg.num_hidden_layers, num_kv_heads=cfg.local_num_key_value_heads,
        head_dim=cfg.head_dim, dtype=DTYPE, device="cuda",
    )
    cache_kwargs = dict(
        block_manager=bm, num_layers=cfg.num_hidden_layers, max_seq_len=max_seq,
        num_kv_heads=cfg.local_num_key_value_heads, head_dim=cfg.head_dim,
        device=torch.device("cuda"), dtype=DTYPE,
    )
    engine = Engine(
        model=model, tokenizer=StubTokenizer(), block_manager=bm, cache_cls=PagedKVCache,
        cache_kwargs=cache_kwargs, device="cuda", use_cuda_graph=not args.no_cuda_graph,
        max_batch_size=batch, max_num_seqs=batch, enable_prefix_cache=False, ignore_eos=True,
        detokenize_outputs=False,
    )
    run_workload(engine, batch, prompt_len, min(gen, 8), cfg.vocab_size, rng)  # warmup / JIT

    stats = getattr(ops, "stats", None)
    if stats is not None:
        stats.reset()
    prefills, itls = [], []
    for _ in range(args.repeats):
        prefill_s, decode_s = run_workload(engine, batch, prompt_len, gen, cfg.vocab_size, rng, nvtx=args.nvtx)
        prefills.append(prefill_s)
        itls.extend(decode_s)

    kv_per_step = (batch * (prompt_len + gen / 2) * cfg.num_hidden_layers * 2
                   * cfg.local_num_key_value_heads * cfg.head_dim * 2)
    ideal_itl_s = (weight_bytes + kv_per_step) / (copy_bw * 1e9)
    itl_p50 = statistics.median(itls)
    row = dict(
        backend=ops.name, batch=batch, prompt_len=prompt_len, gen_len=gen,
        prefill_ms=1e3 * statistics.median(prefills),
        itl_p50_ms=1e3 * itl_p50, itl_p90_ms=1e3 * percentile(itls, 0.90), itl_p99_ms=1e3 * percentile(itls, 0.99),
        decode_tok_s=batch * len(itls) / sum(itls),
        ideal_itl_ms=1e3 * ideal_itl_s, bw_util_pct=100 * ideal_itl_s / itl_p50,
        kv_share_pct=100 * kv_per_step / (kv_per_step + weight_bytes),
        cuda_graph=engine.use_cuda_graph,
    )
    if stats is not None:
        calls = stats.decode_plans + stats.prefill_plans
        row["wrapper_plan_host_us"] = 1e6 * stats.wrapper_plan_host_seconds / max(calls, 1)
        row["fallback_plans"] = stats.fallback_plans

    # cache_kwargs also owns bm; release it before empty_cache so the next
    # configuration's mem_get_info does not include the previous KV pool.
    del engine, bm, cache_kwargs
    gc.collect()
    torch.cuda.empty_cache()
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backends", nargs="+", default=["triton", "flashinfer"], choices=["triton", "flashinfer"])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8])
    p.add_argument("--prompt-lens", type=int, nargs="+", default=[512, 2048, 8192])
    p.add_argument("--gen-len", type=int, default=128)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--weights", default=None)
    p.add_argument("--random-weights", action="store_true")
    p.add_argument("--no-cuda-graph", action="store_true")
    p.add_argument("--reserve-gb", type=float, default=0.75, help="memory kept free for activations/graphs")
    p.add_argument("--nvtx", action="store_true", help="NVTX range per engine step (for nsys)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.gen_len < 2 or args.repeats < 1:
        p.error("--gen-len must be >= 2 and --repeats must be >= 1")

    env = environment_info()
    copy_bw = measure_copy_bandwidth_gbps()
    env["copy_bandwidth_gbps"] = round(copy_bw, 1)
    print(f"{env.get('gpu')} | torch {env['torch']} | triton {env['triton']} | flashinfer {env['flashinfer']}")
    print(f"measured copy bandwidth: {copy_bw:.0f} GB/s")

    model = build_model(args)
    weight_bytes = sum(t.numel() * t.element_size() for t in model.parameters())
    env["weight_gb"] = round(weight_bytes / 1e9, 3)
    print(f"weights: {weight_bytes / 1e9:.2f} GB -> roofline for B=1, short context: "
          f"{copy_bw * 1e9 / weight_bytes:.0f} tok/s\n")

    rows = []
    for backend in args.backends:
        ops = get_ops_backend("cuda", backend)
        set_attention_backend(model, ops)
        for batch in args.batch_sizes:
            for prompt_len in args.prompt_lens:
                rng = random.Random(args.seed * 1000003 + batch * 1009 + prompt_len)
                print(f"[{backend}] B={batch} prompt={prompt_len} gen={args.gen_len}")
                row = bench_config(model, ops, args, batch, prompt_len, copy_bw, weight_bytes, rng)
                if row is not None:
                    rows.append(row)
        if hasattr(ops, "describe"):
            env[f"{backend}_backend_info"] = ops.describe()

    by_key = {(r["batch"], r["prompt_len"]): r for r in rows if r["backend"] == "triton"}
    for r in rows:
        base = by_key.get((r["batch"], r["prompt_len"]))
        if base:
            r["itl_speedup_vs_triton"] = base["itl_p50_ms"] / r["itl_p50_ms"]
            r["prefill_speedup_vs_triton"] = base["prefill_ms"] / r["prefill_ms"]

    print()
    print_table(rows, ["batch", "prompt_len", "backend", "prefill_ms", "itl_p50_ms", "itl_p99_ms",
                       "decode_tok_s", "bw_util_pct", "kv_share_pct", "itl_speedup_vs_triton",
                       "prefill_speedup_vs_triton", "wrapper_plan_host_us"])
    write_results(rows, args.out, env, vars(args))


if __name__ == "__main__":
    main()
