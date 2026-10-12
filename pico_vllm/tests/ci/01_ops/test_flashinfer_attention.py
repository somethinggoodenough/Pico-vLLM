"""FlashInfer backend vs Triton kernels vs an fp32 reference, on paged KV.

Physical pages are randomly permuted so any page-indexing bug shows up.
Shapes follow Qwen2.5-1.5B: 12 query heads, 2 KV heads (GQA group 6), head_dim 128.
"""

import warnings

import pytest
import torch

flashinfer = pytest.importorskip("flashinfer")

from ops.flashinfer import FlashInferOps  # noqa: E402
from ops.triton.backend import TritonOps  # noqa: E402

# Follow the same kernel selection as the engine (flash decode / v2 prefill
# by default), rather than silently testing only the legacy kernels.
_triton_ops = TritonOps()
paged_decode_attention = _triton_ops.paged_decode_attention
paged_prefill_attention = _triton_ops.paged_prefill_attention

pytestmark = [pytest.mark.cuda, pytest.mark.ops]

H, KVH, D, PAGE = 12, 2, 128, 16
DTYPE = torch.bfloat16
TOL = dict(atol=2e-2, rtol=2e-2)


def cdiv(a, b):
    return (a + b - 1) // b


def make_paged_kv(lens, extra_pages=5, seed=0, width_pad=3):
    """Random K/V spread over shuffled physical pages, plus Pico-style metadata."""
    g = torch.Generator().manual_seed(seed)
    pages_needed = [cdiv(n, PAGE) for n in lens]
    total = sum(pages_needed) + extra_pages
    perm = torch.randperm(total, generator=g).tolist()
    k_cache = torch.randn(total, KVH, PAGE, D, generator=g).to("cuda", DTYPE)
    v_cache = torch.randn(total, KVH, PAGE, D, generator=g).to("cuda", DTYPE)

    tables, off = [], 0
    for p in pages_needed:
        tables.append(perm[off:off + p])
        off += p
    block_table = torch.full((len(lens), max(pages_needed) + width_pad), -1, dtype=torch.int32, device="cuda")
    for i, t in enumerate(tables):
        block_table[i, :len(t)] = torch.tensor(t, dtype=torch.int32)
    context_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    return k_cache, v_cache, tables, block_table, context_lens


def gather_kv(cache, table, n):
    """(n, KVH, D) contiguous view of one request's KV."""
    pages = cache[torch.tensor(table, device=cache.device)]          # (p, KVH, PAGE, D)
    return pages.permute(0, 2, 1, 3).reshape(-1, KVH, D)[:n]


def ref_attention(q, k, v, causal):
    """q (M, H, D), k/v (N, KVH, D) -> (M, H, D) in fp32; causal is bottom-right."""
    group = q.shape[1] // k.shape[1]
    k = k.float().repeat_interleave(group, dim=1)
    v = v.float().repeat_interleave(group, dim=1)
    s = torch.einsum("mhd,nhd->hmn", q.float(), k) / D ** 0.5
    if causal:
        M, N = q.shape[0], k.shape[0]
        q_pos = torch.arange(M, device=q.device)[:, None] + (N - M)
        k_pos = torch.arange(N, device=q.device)[None, :]
        s = s.masked_fill(~(q_pos >= k_pos), float("-inf"))
    return torch.einsum("hmn,nhd->mhd", s.softmax(-1), v)


def make_ops(**kwargs):
    ops = FlashInferOps(**kwargs)
    ops.setup_attention(num_qo_heads=H, num_kv_heads=KVH, head_dim=D, page_size=PAGE,
                        dtype=DTYPE, device="cuda", dummy_page=0)
    return ops


def check_decode(out, q, k_cache, v_cache, tables, lens):
    for i, n in enumerate(lens):
        ref = ref_attention(q[i, :, 0][None], gather_kv(k_cache, tables[i], n),
                            gather_kv(v_cache, tables[i], n), causal=False)[0]
        torch.testing.assert_close(out[i, :, 0].float(), ref, **TOL)


DECODE_LENS = [1, 16, 17, 300, 1000]


@pytest.mark.parametrize(
    "use_tensor_cores",
    [
        True,
        pytest.param(False, marks=pytest.mark.xfail(
            reason="CUDA-core decode may not support GQA group size 6 in every FlashInfer build",
            strict=False)),
    ],
)
def test_decode_matches_reference_and_triton(use_tensor_cores):
    k, v, tables, bt, cl = make_paged_kv(DECODE_LENS)
    q = torch.randn(len(DECODE_LENS), H, 1, D, device="cuda", dtype=DTYPE)
    ops = make_ops(use_tensor_cores=use_tensor_cores)

    ops.plan_decode(tables, DECODE_LENS, block_table=bt, context_lens=cl)
    out_fi = ops.paged_decode_attention(q, k, v, bt, cl, MAX_BLOCKS_PER_SEQ=bt.shape[1])
    out_tr = paged_decode_attention(q, k, v, bt, cl, MAX_BLOCKS_PER_SEQ=bt.shape[1])

    assert out_fi.shape == q.shape
    check_decode(out_fi, q, k, v, tables, DECODE_LENS)
    torch.testing.assert_close(out_fi.float(), out_tr.float(), **TOL)
    assert ops.stats.fallback_plans == 0


def test_decode_page_tables_as_gpu_tensors():
    """The engine passes GPU block-table views, not Python lists."""
    k, v, tables, bt, cl = make_paged_kv(DECODE_LENS, seed=1)
    q = torch.randn(len(DECODE_LENS), H, 1, D, device="cuda", dtype=DTYPE)
    ops = make_ops()
    ops.plan_decode(list(bt.unbind(0)), DECODE_LENS, block_table=bt, context_lens=cl)
    out = ops.paged_decode_attention(q, k, v, bt, cl, MAX_BLOCKS_PER_SEQ=bt.shape[1])
    check_decode(out, q, k, v, tables, DECODE_LENS)


def test_decode_without_plan_falls_back_once_per_forward():
    k, v, tables, bt, cl = make_paged_kv(DECODE_LENS, seed=2)
    q = torch.randn(len(DECODE_LENS), H, 1, D, device="cuda", dtype=DTYPE)
    ops = make_ops()

    with pytest.warns(UserWarning, match="without an explicit plan"):
        out_fallback = ops.paged_decode_attention(q, k, v, bt, cl, MAX_BLOCKS_PER_SEQ=bt.shape[1])
    # Same tensors again (= the next layer of the same forward): plan is reused.
    ops.paged_decode_attention(q, k, v, bt, cl, MAX_BLOCKS_PER_SEQ=bt.shape[1])
    assert ops.stats.fallback_plans == 1

    ops.plan_decode(tables, DECODE_LENS, block_table=bt, context_lens=cl)
    out_planned = ops.paged_decode_attention(q, k, v, bt, cl, MAX_BLOCKS_PER_SEQ=bt.shape[1])
    torch.testing.assert_close(out_fallback, out_planned)
    check_decode(out_fallback, q, k, v, tables, DECODE_LENS)


def test_stale_plan_is_never_reused():
    k, v, tables, bt, cl = make_paged_kv([40, 70], seed=3)
    q = torch.randn(2, H, 1, D, device="cuda", dtype=DTYPE)
    ops = make_ops()
    ops.plan_decode(tables, [40, 70], block_table=bt, context_lens=cl)

    # Next step: same buffers refilled in place with longer lengths.
    cl.copy_(torch.tensor([41, 71], dtype=torch.int32))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = ops.paged_decode_attention(q, k, v, bt, cl, MAX_BLOCKS_PER_SEQ=bt.shape[1])
    assert ops.stats.fallback_plans == 1
    check_decode(out, q, k, v, tables, [41, 71])


def test_decode_cuda_graph_with_ghost_padding_and_replan():
    max_bs = 8
    lens = [5, 40, 129]
    k, v, tables, bt_small, _ = make_paged_kv([n + 1 for n in lens], seed=4)  # room for one more token
    width = bt_small.shape[1]
    ops = make_ops()

    static_q = torch.zeros(max_bs, H, 1, D, device="cuda", dtype=DTYPE)
    static_bt = torch.full((max_bs, width), -1, dtype=torch.int32, device="cuda")
    static_cl = torch.zeros(max_bs, dtype=torch.int32, device="cuda")
    static_bt[:len(lens)] = bt_small
    static_cl[:len(lens)] = torch.tensor(lens, dtype=torch.int32)

    ops.init_cuda_graph(max_batch_size=max_bs, max_num_pages=max_bs * width)
    ops.plan_decode(tables, lens, use_cuda_graph=True, block_table=static_bt, context_lens=static_cl)

    q = torch.randn(len(lens), H, 1, D, device="cuda", dtype=DTYPE)
    static_q[:len(lens)] = q

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):  # warmup outside capture
        ops.paged_decode_attention(static_q, k, v, static_bt, static_cl, MAX_BLOCKS_PER_SEQ=width)
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = ops.paged_decode_attention(static_q, k, v, static_bt, static_cl, MAX_BLOCKS_PER_SEQ=width)

    graph.replay()
    torch.cuda.synchronize()
    check_decode(static_out, q, k, v, tables, lens)
    assert torch.isfinite(static_out).all(), "ghost rows must stay finite"

    # Next decode step: one more token per request, re-plan outside the graph, replay.
    lens2 = [n + 1 for n in lens]
    static_cl[:len(lens)] = torch.tensor(lens2, dtype=torch.int32)
    ops.plan_decode(tables, lens2, use_cuda_graph=True, block_table=static_bt, context_lens=static_cl)
    q2 = torch.randn(len(lens), H, 1, D, device="cuda", dtype=DTYPE)
    static_q[:len(lens)] = q2
    graph.replay()
    torch.cuda.synchronize()
    check_decode(static_out, q2, k, v, tables, lens2)


@pytest.mark.parametrize("kv_len, q_len", [(100, 36), (50, 50), (16, 1), (257, 257)])
def test_prefill_with_prefix_matches_reference_and_triton(kv_len, q_len):
    """kv_len > q_len is the prefix-cache case: Q covers only the newly computed tail."""
    k, v, tables, bt, cl = make_paged_kv([kv_len], seed=kv_len)
    q = torch.randn(q_len, H, D, device="cuda", dtype=DTYPE)
    new_lens = torch.tensor([q_len], dtype=torch.int32, device="cuda")
    q_start = torch.tensor([0], dtype=torch.int32, device="cuda")
    ops = make_ops()

    ops.plan_prefill(tables, [kv_len], [q_len], block_table=bt, context_lens=cl,
                     new_token_lens=new_lens, q_start_loc=q_start)
    out_fi = ops.paged_prefill_attention(q, k, v, bt, cl, new_lens, q_start,
                                       MAX_BLOCKS_PER_SEQ=bt.shape[1], max_new_len=q_len)
    out_tr = paged_prefill_attention(q, k, v, bt, cl, new_lens, q_start, MAX_BLOCKS_PER_SEQ=bt.shape[1])
    ref = ref_attention(q, gather_kv(k, tables[0], kv_len), gather_kv(v, tables[0], kv_len), causal=True)

    torch.testing.assert_close(out_fi.float(), ref, **TOL)
    torch.testing.assert_close(out_tr.float(), ref, **TOL)
    assert ops.stats.fallback_plans == 0


def test_ragged_batched_prefill():
    """Pico prefills one request at a time today; the backend already handles ragged batches."""
    kv_lens, q_lens = [100, 33], [36, 33]
    k, v, tables, bt, cl = make_paged_kv(kv_lens, seed=7)
    q = torch.randn(sum(q_lens), H, D, device="cuda", dtype=DTYPE)
    new_lens = torch.tensor(q_lens, dtype=torch.int32, device="cuda")
    q_start = torch.tensor([0, q_lens[0]], dtype=torch.int32, device="cuda")
    ops = make_ops()

    ops.plan_prefill(tables, kv_lens, q_lens, block_table=bt, context_lens=cl,
                     new_token_lens=new_lens, q_start_loc=q_start)
    out = ops.paged_prefill_attention(q, k, v, bt, cl, new_lens, q_start, MAX_BLOCKS_PER_SEQ=bt.shape[1])
    start = 0
    for i, (kv_len, q_len) in enumerate(zip(kv_lens, q_lens)):
        ref = ref_attention(q[start:start + q_len], gather_kv(k, tables[i], kv_len),
                            gather_kv(v, tables[i], kv_len), causal=True)
        torch.testing.assert_close(out[start:start + q_len].float(), ref, **TOL)
        start += q_len


def test_prefill_changed_partition_matches_fp32():
    k, v, tables, bt, cl = make_paged_kv([64, 64], seed=13)
    q = torch.randn(64, H, D, device="cuda", dtype=DTYPE)
    nl = torch.tensor([32, 32], dtype=torch.int32, device="cuda")
    starts = torch.tensor([0, 32], dtype=torch.int32, device="cuda")
    ops = make_ops()
    ops.plan_prefill(tables, [64, 64], [32, 32], block_table=bt, context_lens=cl,
                     new_token_lens=nl, q_start_loc=starts)
    nl.copy_(torch.tensor([16, 48], dtype=torch.int32))
    starts.copy_(torch.tensor([0, 16], dtype=torch.int32))
    with pytest.warns(UserWarning, match="without an explicit plan"):
        out = ops.paged_prefill_attention(q, k, v, bt, cl, nl, starts, bt.shape[1])
    for i, (start, count) in enumerate([(0, 16), (16, 48)]):
        ref = ref_attention(q[start:start + count], gather_kv(k, tables[i], 64),
                            gather_kv(v, tables[i], 64), causal=True)
        torch.testing.assert_close(out[start:start + count].float(), ref, **TOL)


def test_ghost_rope_q_is_zero_and_existing_kv_is_untouched(monkeypatch):
    from ops.triton.fused_rope_kvcache_store import fused_decode_rope_and_cache

    q = torch.randn(2, 1, H, D, device="cuda", dtype=DTYPE)
    k = torch.randn(2, 1, KVH, D, device="cuda", dtype=DTYPE)
    v = torch.randn_like(k)
    kc = torch.full((2, KVH, PAGE, D), 5.0, device="cuda", dtype=DTYPE)
    vc = torch.full_like(kc, 7.0)
    cos, sin = torch.ones(2, 1, D, device="cuda", dtype=DTYPE), torch.zeros(2, 1, D, device="cuda", dtype=DTYPE)
    slots = torch.tensor([PAGE, -1], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([1, 0], dtype=torch.int32, device="cuda")
    # Poison the wrapper's output allocation to make the old uninitialized-Q
    # behavior deterministic instead of depending on allocator reuse.
    monkeypatch.setattr(torch, "empty_like", lambda x: torch.full_like(x, float("nan")))
    rotated = fused_decode_rope_and_cache(q, k, v, cos, sin, kc, vc, slots, lengths)
    assert torch.count_nonzero(rotated[1]).item() == 0
    torch.testing.assert_close(rotated[0], q[0])
    assert (kc[0] == 5).all() and (vc[0] == 7).all()


def test_fused_prefill_does_not_interpret_request_lengths_as_token_lengths():
    from ops.triton.backend import TritonOps

    # Prefill has many tokens but only one request-length entry. Zero here
    # deliberately proves the decode-only ghost check is disabled.
    n = 31
    q = torch.randn(1, n, H, D, device="cuda", dtype=DTYPE)
    k = torch.randn(1, n, KVH, D, device="cuda", dtype=DTYPE)
    v = torch.randn_like(k)
    kc = torch.zeros(2, KVH, PAGE, D, device="cuda", dtype=DTYPE)
    vc = torch.zeros_like(kc)
    cos = torch.ones(1, n, D, device="cuda", dtype=DTYPE)
    slots = torch.arange(n, device="cuda", dtype=torch.int32)
    cl = torch.zeros(1, device="cuda", dtype=torch.int32)
    rotated = TritonOps().fused_rope_and_cache(
        q, k, v, cos, torch.zeros_like(cos), kc, vc, slots, cl, check_ctx_len=False)
    torch.testing.assert_close(rotated, q)
    torch.testing.assert_close(gather_kv(kc, [0, 1], n), k[0])
    torch.testing.assert_close(gather_kv(vc, [0, 1], n), v[0])
