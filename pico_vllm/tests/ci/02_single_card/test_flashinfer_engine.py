"""End-to-end: the same tiny model through Engine with the Triton and FlashInfer backends.

Covers the three places the engine plans attention: CUDA-Graph decode (with ghost
padding), eager decode, and prefill with a prefix-cache hit. The model is random,
and greedy tokens are compared on these fixed smoke-test inputs. This does not
guarantee identical argmax for arbitrary prompts near a numerical tie.
"""

import pytest
import torch

flashinfer = pytest.importorskip("flashinfer")

from blockmanager import BlockManager  # noqa: E402
from cache import PagedKVCache  # noqa: E402
from engine import Engine  # noqa: E402
from model import ModelConfig, Qwen25_15B  # noqa: E402

pytestmark = [pytest.mark.cuda, pytest.mark.single_card]

DTYPE = torch.bfloat16


class StubTokenizer:
    eos_token_id = -1

    def encode(self, text):
        return [ord(c) % 256 for c in text]

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


def tiny_cfg(backend):
    # head_dim = 512 / 4 = 128, GQA group 2
    return ModelConfig(
        vocab_size=256, hidden_size=512, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, intermediate_size=512,
        max_position_embeddings=2048, use_cuda=True, ops_backend=backend,
    )


@pytest.fixture(scope="module")
def models():
    torch.manual_seed(0)
    reference = Qwen25_15B(tiny_cfg("triton"))
    with torch.no_grad():
        reference.norm.weight.mul_(8.0)  # identical logit scale for both test models
    state = reference.state_dict()
    out = {}
    for backend in ("triton", "flashinfer"):
        m = Qwen25_15B(tiny_cfg(backend))
        m.load_state_dict(state)
        out[backend] = m.to(device="cuda", dtype=DTYPE).eval()
    return out


def make_engine(model, *, use_cuda_graph, prefix_cache=False):
    cfg = model.cfg
    bm = BlockManager(
        num_gpu_blocks=256, num_cpu_blocks=0, block_size=cfg.BLOCK_SIZE,
        num_layers=cfg.num_hidden_layers, num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim, dtype=DTYPE, device="cuda",
    )
    return Engine(
        model=model, tokenizer=StubTokenizer(), block_manager=bm,
        cache_cls=PagedKVCache, device="cuda",
        use_cuda_graph=use_cuda_graph, max_batch_size=4, max_num_seqs=4,
        enable_prefix_cache=prefix_cache, ignore_eos=True,
    )


def generate(engine, prompts, max_new_tokens):
    ids = [engine.submit_ids(p, max_new_tokens=max_new_tokens, temperature=0.0, top_p=1.0) for p in prompts]
    out = {}
    while len(out) < len(ids):
        for rid, text in engine.step():
            out[rid] = [int(t) for t in text.split()][-max_new_tokens:]
    return [out[i] for i in ids]


PROMPTS = [
    list(range(1, 6)),                        # 5 tokens: shorter than a page
    [(7 * i) % 256 for i in range(37)],        # crosses page boundaries
    [(13 * i + 5) % 256 for i in range(100)],
]


@pytest.mark.parametrize("use_cuda_graph", [True, False], ids=["graph", "eager"])
def test_greedy_tokens_match_triton(models, use_cuda_graph):
    results = {}
    for backend, model in models.items():
        if backend == "flashinfer":
            model.ops.stats.reset()
        engine = make_engine(model, use_cuda_graph=use_cuda_graph)
        results[backend] = generate(engine, PROMPTS, max_new_tokens=24)
        if backend == "flashinfer":
            stats = engine.attn_ops.stats
            assert stats.fallback_plans == 0, "engine must always plan explicitly"
            assert stats.prefill_plans == len(PROMPTS)
            assert stats.decode_plans >= 23
    assert results["flashinfer"] == results["triton"]


def test_prefix_cache_hit_path_matches_triton(models):
    shared = [(3 * i + 1) % 256 for i in range(64)]   # 4 full pages, cacheable
    first = shared + [9] * 20
    second = shared + [(5 * i) % 256 for i in range(30)]
    results = {}
    for backend, model in models.items():
        engine = make_engine(model, use_cuda_graph=True, prefix_cache=True)
        generate(engine, [first], max_new_tokens=4)           # populates the radix tree
        rid = engine.submit_ids(second, max_new_tokens=16, temperature=0.0, top_p=1.0)
        request = engine.scheduler.waiting[-1]
        assert request.request_id == rid and request.matched_len == 64, "expected a 64-token prefix hit"
        out = {}
        while rid not in out:
            for r, text in engine.step():
                out[r] = [int(t) for t in text.split()][-16:]
        results[backend] = out[rid]
    assert results["flashinfer"] == results["triton"]
