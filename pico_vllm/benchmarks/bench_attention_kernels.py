"""
Kernel-level attention benchmark: Triton vs FlashInfer on paged KV (one layer).

Only the attention kernel is timed, with Qwen2.5-1.5B shapes by default
(12 Q heads, 2 KV heads, head_dim 128, page 16, bf16). KV pages are randomly
permuted, every call starts from a cold L2, and each output is checked against
the Triton kernel and an fp32 reference before its time is reported.

Usage (from pico_vllm/):
  # decode: batch x context sweep
  python benchmarks/bench_attention_kernels.py decode \
      --batch-sizes 1 8 32 --ctx-lens 512 2048 8192 32768 --out results/decode_uniform.csv
  # decode: ragged lengths (uniform in [ctx/8, ctx]) or one long + many short requests
  python benchmarks/bench_attention_kernels.py decode --dist skew --batch-sizes 8 32 --ctx-lens 8192 32768
  # isolate GQA packing: tensor-core vs CUDA-core FlashInfer decode
  python benchmarks/bench_attention_kernels.py decode --backends triton flashinfer flashinfer-cc
  # prefill (one request, Q = new tokens, KV = prefix + new)
  python benchmarks/bench_attention_kernels.py prefill --q-lens 128 512 2048 8192 --prefix-lens 0 2048

Backends: triton | flashinfer (tensor cores) | flashinfer-cc (CUDA cores) | flashinfer-cudnn (decode only)

Timing: decode is captured in a CUDA graph and replayed (no Python launch overhead,
like the engine's decode path). Prefill runs eagerly for both backends; the known
CPU query length is passed as max_new_len, just as in the latest Engine, avoiding
a redundant .item() synchronization. Triton kernels follow PICO_DECODE_ATTN and
PICO_PREFILL_ATTN (defaults: flash / v2).
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import gc
import random
import time
import warnings

import torch

from bench_common import (environment_info, make_flush_buffer, measure_copy_bandwidth_gbps,
                          print_table, time_cuda, write_results)
from ops.triton.backend import TritonOps

_triton_ops = TritonOps()
paged_decode_attention = _triton_ops.paged_decode_attention
paged_prefill_attention = _triton_ops.paged_prefill_attention

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}
ENGINE_BLOCK_TABLE_WIDTH = 8192  # ModelConfig.MAX_BLOCKS_PER_SEQ: Triton specializes on it


def cdiv(a, b):
    return (a + b - 1) // b


# ---------------------------------------------------------------------------
# Problem construction
# ---------------------------------------------------------------------------
def make_lens(batch, ctx, dist, rng):
    if dist == "uniform":
        return [ctx] * batch
    if dist == "ragged":
        return [rng.randint(max(1, ctx // 8), ctx) for _ in range(batch)]
    if dist == "skew":
        return [ctx] + [max(1, ctx // 16)] * (batch - 1)
    raise ValueError(dist)


class PagedProblem:
    """One layer's paged KV cache for a batch, laid out the way BlockManager does it."""

    def __init__(self, kv_lens, a, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.kv_lens = kv_lens
        self.page_size = a.page_size
        pages = [cdiv(n, a.page_size) for n in kv_lens]
        self.num_pages = sum(pages) + 1  # page 0 is reserved as the padding page, like the engine
        shape = (self.num_pages, a.kv_heads, a.page_size, a.head_dim)
        self.k_cache = torch.randn(shape, dtype=a.dtype, device="cuda")
        self.v_cache = torch.randn(shape, dtype=a.dtype, device="cuda")
        perm = (torch.randperm(self.num_pages - 1, generator=g) + 1).to(torch.int32)
        self.tables, off = [], 0
        for p in pages:
            self.tables.append(perm[off:off + p].cuda())
            off += p
        width = max(ENGINE_BLOCK_TABLE_WIDTH, max(pages))
        bt = torch.full((len(kv_lens), width), -1, dtype=torch.int32)
        off = 0
        for i, p in enumerate(pages):
            bt[i, :p] = perm[off:off + p]
            off += p
        self.block_table = bt.cuda()
        self.context_lens = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")

    def gather(self, cache, i):
        n = self.kv_lens[i]
        pages = cache[self.tables[i].long()]                     # (p, KVH, page, D)
        return pages.permute(0, 2, 1, 3).reshape(-1, cache.shape[1], cache.shape[3])[:n]

    @staticmethod
    def bytes_needed(kv_lens, a):
        pages = sum(cdiv(n, a.page_size) for n in kv_lens) + 1
        return 2 * pages * a.kv_heads * a.page_size * a.head_dim * torch.finfo(a.dtype).bits // 8


def ref_attention(q, k, v, causal):
    """q (M, H, D), k/v (N, KVH, D) -> fp32 (M, H, D); causal is bottom-right aligned."""
    group = q.shape[1] // k.shape[1]
    k = k.float().repeat_interleave(group, dim=1)
    v = v.float().repeat_interleave(group, dim=1)
    s = torch.einsum("mhd,nhd->hmn", q.float(), k) / q.shape[-1] ** 0.5
    if causal:
        M, N = q.shape[0], k.shape[0]
        mask = (torch.arange(M, device=q.device)[:, None] + (N - M)) >= torch.arange(N, device=q.device)[None, :]
        s = s.masked_fill(~mask, float("-inf"))
    return torch.einsum("hmn,nhd->mhd", s.softmax(-1), v)


def make_flashinfer_ops(backend, a):
    from ops.flashinfer import FlashInferOps
    kwargs = {
        "flashinfer": dict(use_tensor_cores=True),
        "flashinfer-cc": dict(use_tensor_cores=False),
        "flashinfer-cudnn": dict(use_tensor_cores=True, decode_backend="cudnn"),
    }[backend]
    ops = FlashInferOps(**kwargs)
    ops.setup_attention(num_qo_heads=a.heads, num_kv_heads=a.kv_heads, head_dim=a.head_dim,
                        page_size=a.page_size, dtype=a.dtype, device="cuda", dummy_page=0)
    return ops


def median_plan_host_us(plan, reps=20):
    """Host call duration including metadata construction, not GPU completion latency."""
    plan()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        plan()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2] * 1e6


def max_diff(a, b):
    return (a.float() - b.float()).abs().max().item()


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------
def bench_decode_config(a, batch, ctx, flush, copy_bw, rng):
    lens = make_lens(batch, ctx, a.dist, rng)
    if PagedProblem.bytes_needed(lens, a) > a.max_kv_bytes:
        print(f"  skip B={batch} ctx={ctx}: KV needs {PagedProblem.bytes_needed(lens, a) / 2**30:.2f} GiB")
        return []
    prob = PagedProblem(lens, a, seed=batch * 100003 + ctx)
    q = torch.randn(batch, a.heads, 1, a.head_dim, dtype=a.dtype, device="cuda")
    width = prob.block_table.shape[1]
    elsize = torch.finfo(a.dtype).bits // 8
    io_bytes = sum(lens) * a.kv_heads * a.head_dim * 2 * elsize + 2 * batch * a.heads * a.head_dim * elsize

    # fp32 reference for the first and last request (cheap, catches indexing bugs)
    refs = {}
    for i in {0, batch - 1}:
        refs[i] = ref_attention(q[i, :, 0][None], prob.gather(prob.k_cache, i),
                                prob.gather(prob.v_cache, i), causal=False)[0]

    out_triton = paged_decode_attention(q, prob.k_cache, prob.v_cache, prob.block_table,
                                        prob.context_lens, MAX_BLOCKS_PER_SEQ=width)
    rows = []
    for backend in a.backends:
        row = dict(mode="decode", dist=a.dist, batch=batch, ctx=ctx, total_kv_tokens=sum(lens), backend=backend)
        try:
            if backend == "triton":
                fn = lambda: paged_decode_attention(q, prob.k_cache, prob.v_cache, prob.block_table,
                                                    prob.context_lens, MAX_BLOCKS_PER_SEQ=width)
                out = out_triton
            else:
                ops = make_flashinfer_ops(backend, a)
                graph = not a.no_graph
                if graph:
                    ops.init_cuda_graph(max_batch_size=batch, max_num_pages=prob.num_pages + batch)
                plan = lambda: ops.plan_decode(prob.tables, lens, use_cuda_graph=graph,
                                               block_table=prob.block_table, context_lens=prob.context_lens)
                row["plan_host_us"] = median_plan_host_us(plan)
                plan()
                fn = lambda: ops.paged_decode_attention(q, prob.k_cache, prob.v_cache, prob.block_table,
                                                        prob.context_lens, MAX_BLOCKS_PER_SEQ=width)
                out = fn()
                info = ops.describe()
                row["resolved"] = info.get("decode_graph_resolved_backend" if graph else "decode_eager_resolved_backend")
                assert ops.stats.fallback_plans == 0
            row["max_diff_vs_triton"] = max_diff(out, out_triton)
            row["max_diff_vs_ref"] = max(max_diff(out[i, :, 0], r) for i, r in refs.items())
            if row["max_diff_vs_ref"] > a.tol:
                print(f"  WARNING {backend} B={batch} ctx={ctx}: max diff vs fp32 reference {row['max_diff_vs_ref']:.3g}")
            t = time_cuda(fn, iters=a.iters, flush=flush, use_graph=not a.no_graph)
            row.update(t)
            row["gbps"] = io_bytes / (t["median_us"] * 1e-6) / 1e9
            row["pct_copy_bw"] = 100 * row["gbps"] / copy_bw
        except Exception as exc:  # keep sweeping; record why this backend failed
            row["error"] = f"{type(exc).__name__}: {exc}"[:200]
            print(f"  {backend} B={batch} ctx={ctx} failed: {row['error']}")
        rows.append(row)

    base = next((r.get("median_us") for r in rows if r["backend"] == "triton"), None)
    for r in rows:
        if base and r.get("median_us"):
            r["speedup_vs_triton"] = base / r["median_us"]
    return rows


# ---------------------------------------------------------------------------
# Prefill
# ---------------------------------------------------------------------------
def bench_prefill_config(a, q_len, prefix, flush):
    kv_len = prefix + q_len
    if PagedProblem.bytes_needed([kv_len], a) > a.max_kv_bytes:
        return []
    prob = PagedProblem([kv_len], a, seed=q_len * 7919 + prefix)
    q = torch.randn(q_len, a.heads, a.head_dim, dtype=a.dtype, device="cuda")
    new_lens = torch.tensor([q_len], dtype=torch.int32, device="cuda")
    q_start = torch.tensor([0], dtype=torch.int32, device="cuda")
    width = prob.block_table.shape[1]
    flops = 4 * a.heads * a.head_dim * (q_len * prefix + q_len * (q_len + 1) // 2)

    ref = None
    if q_len * kv_len * a.heads * 4 <= 512 * 2**20:  # fp32 score matrix fits comfortably
        ref = ref_attention(q, prob.gather(prob.k_cache, 0), prob.gather(prob.v_cache, 0), causal=True)

    tri = lambda: paged_prefill_attention(q, prob.k_cache, prob.v_cache, prob.block_table, prob.context_lens,
                                          new_lens, q_start, MAX_BLOCKS_PER_SEQ=width,
                                          max_new_len=q_len)
    out_triton = tri()
    rows = []
    for backend in a.backends:
        if backend == "flashinfer-cudnn":
            continue
        row = dict(mode="prefill", q_len=q_len, prefix_len=prefix, kv_len=kv_len, backend=backend)
        try:
            if backend == "triton":
                fn, out = tri, out_triton
            else:
                ops = make_flashinfer_ops(backend, a)
                plan = lambda: ops.plan_prefill(prob.tables, [kv_len], [q_len],
                                                block_table=prob.block_table, context_lens=prob.context_lens,
                                                new_token_lens=new_lens, q_start_loc=q_start)
                row["plan_host_us"] = median_plan_host_us(plan)
                plan()
                fn = lambda: ops.paged_prefill_attention(q, prob.k_cache, prob.v_cache, prob.block_table,
                                                         prob.context_lens, new_lens, q_start,
                                                         MAX_BLOCKS_PER_SEQ=width, max_new_len=q_len)
                out = fn()
                assert ops.stats.fallback_plans == 0
            row["max_diff_vs_triton"] = max_diff(out, out_triton)
            if ref is not None:
                row["max_diff_vs_ref"] = max_diff(out, ref)
                if row["max_diff_vs_ref"] > a.tol:
                    print(f"  WARNING {backend} q={q_len} prefix={prefix}: diff vs reference {row['max_diff_vs_ref']:.3g}")
            t = time_cuda(fn, iters=a.iters, flush=flush, use_graph=False)
            row.update(t)
            row["tflops"] = flops / (t["median_us"] * 1e-6) / 1e12
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"[:200]
            print(f"  {backend} q={q_len} prefix={prefix} failed: {row['error']}")
        rows.append(row)

    base = next((r.get("median_us") for r in rows if r["backend"] == "triton"), None)
    for r in rows:
        if base and r.get("median_us"):
            r["speedup_vs_triton"] = base / r["median_us"]
    return rows


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["decode", "prefill"])
    p.add_argument("--backends", nargs="+", default=["triton", "flashinfer"],
                   choices=["triton", "flashinfer", "flashinfer-cc", "flashinfer-cudnn"])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    p.add_argument("--ctx-lens", type=int, nargs="+", default=[512, 2048, 8192, 16384, 32768])
    p.add_argument("--dist", choices=["uniform", "ragged", "skew"], default="uniform")
    p.add_argument("--q-lens", type=int, nargs="+", default=[128, 512, 2048, 8192])
    p.add_argument("--prefix-lens", type=int, nargs="+", default=[0, 2048])
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--kv-heads", type=int, default=2)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--page-size", type=int, default=16)
    p.add_argument("--dtype", choices=list(DTYPES), default="bf16")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--no-graph", action="store_true", help="time decode eagerly instead of CUDA-graph replay")
    p.add_argument("--no-flush", action="store_true", help="keep L2 warm between calls (not realistic)")
    p.add_argument("--max-kv-gb", type=float, default=None, help="skip configs whose KV exceeds this (default: 60%% of free memory)")
    p.add_argument("--tol", type=float, default=3e-2, help="max |diff| vs fp32 reference before warning")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="CSV path; a .json with environment and arguments is written next to it")
    a = p.parse_args()
    a.dtype = DTYPES[a.dtype]
    free, _ = torch.cuda.mem_get_info()
    a.max_kv_bytes = a.max_kv_gb * 2**30 if a.max_kv_gb else 0.6 * free
    warnings.filterwarnings("ignore", message=".*without an explicit plan.*")

    env = environment_info()
    copy_bw = measure_copy_bandwidth_gbps()
    env["copy_bandwidth_gbps"] = round(copy_bw, 1)
    print(f"{env.get('gpu')} | torch {env['torch']} | triton {env['triton']} | flashinfer {env['flashinfer']}")
    print(f"measured copy bandwidth: {copy_bw:.0f} GB/s; L2 {env.get('l2_cache_mb')} MB; SMs {env.get('sm_count')}\n")

    flush = None if a.no_flush else make_flush_buffer()
    rng = random.Random(a.seed)
    rows = []
    if a.mode == "decode":
        for batch in a.batch_sizes:
            for ctx in a.ctx_lens:
                rows += bench_decode_config(a, batch, ctx, flush, copy_bw, rng)
                gc.collect()
                torch.cuda.empty_cache()  # drop this config's KV before allocating the next
        cols = ["batch", "ctx", "backend", "median_us", "p90_us", "gbps", "pct_copy_bw",
                "speedup_vs_triton", "plan_host_us", "max_diff_vs_ref", "resolved"]
    else:
        for q_len in a.q_lens:
            for prefix in a.prefix_lens:
                rows += bench_prefill_config(a, q_len, prefix, flush)
                gc.collect()
                torch.cuda.empty_cache()  # drop this config's KV before allocating the next
        cols = ["q_len", "prefix_len", "backend", "median_us", "p90_us", "tflops",
                "speedup_vs_triton", "plan_host_us", "max_diff_vs_ref"]

    print()
    print_table(rows, cols)
    write_results(rows, a.out, env, {k: v for k, v in vars(a).items() if k != "max_kv_bytes"})


if __name__ == "__main__":
    main()
