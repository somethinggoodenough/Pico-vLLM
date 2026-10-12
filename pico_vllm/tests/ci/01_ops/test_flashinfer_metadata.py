"""CPU tests for the FlashInfer backend's host-side logic.

These need neither a GPU nor the flashinfer package: the CSR builder is pure
torch, and `flashinfer` is only imported when FlashInferOps is constructed.
"""

import pytest
import torch

from ops.flashinfer.backend import AttentionShape, FlashInferOps, PlanStats, _make_key, _PlanState
from ops.flashinfer.metadata import build_paged_kv_metadata, build_qo_indptr

pytestmark = [pytest.mark.cpu, pytest.mark.ops]

PAGE = 16


def test_csr_lengths_at_page_boundaries():
    lens = [1, 16, 17, 33]
    tables = [[7], [3], [5, 9], [2, 4, 6]]
    meta = build_paged_kv_metadata(tables, lens, PAGE)

    assert meta.indptr.tolist() == [0, 1, 2, 4, 7]
    assert meta.indices.tolist() == [7, 3, 5, 9, 2, 4, 6]
    assert meta.last_page_len.tolist() == [1, 16, 1, 1]
    assert meta.kv_lens == lens
    assert meta.num_real == 4 and meta.batch_size == 4
    assert meta.indptr.dtype == torch.int32 and meta.indices.dtype == torch.int32
    assert meta.last_page_len.dtype == torch.int32


def test_tensor_tables_are_sliced_to_needed_pages():
    # Rows of a padded dense block table: entries past the needed pages are -1.
    block_table = torch.tensor([[4, 8, -1, -1], [1, -1, -1, -1]], dtype=torch.int32)
    meta = build_paged_kv_metadata(list(block_table.unbind(0)), [20, 3], PAGE)

    assert meta.indptr.tolist() == [0, 2, 3]
    assert meta.indices.tolist() == [4, 8, 1]
    assert meta.last_page_len.tolist() == [4, 3]


def test_extra_allocated_pages_are_ignored():
    # PagedKVCache may already own the page for the next token.
    meta = build_paged_kv_metadata([[1, 2, 3]], [32], PAGE)
    assert meta.indices.tolist() == [1, 2]
    assert meta.last_page_len.tolist() == [16]


def test_zero_length_slot_becomes_one_token_on_dummy_page():
    meta = build_paged_kv_metadata([[5], [], [6]], [3, 0, 16], PAGE, dummy_page=42)
    assert meta.indptr.tolist() == [0, 1, 2, 3]
    assert meta.indices.tolist() == [5, 42, 6]
    assert meta.last_page_len.tolist() == [3, 1, 16]
    assert meta.kv_lens == [3, 1, 16]


def test_padding_to_fixed_batch_uses_ghost_slots():
    meta = build_paged_kv_metadata([[1, 2]], [17], PAGE, pad_to=4, dummy_page=0)
    assert meta.batch_size == 4 and meta.num_real == 1
    assert meta.indptr.tolist() == [0, 2, 3, 4, 5]
    assert meta.indices.tolist() == [1, 2, 0, 0, 0]
    assert meta.last_page_len.tolist() == [1, 1, 1, 1]


def test_all_ghost_batch_for_graph_capture():
    meta = build_paged_kv_metadata([], [], PAGE, pad_to=3, dummy_page=9)
    assert meta.indptr.tolist() == [0, 1, 2, 3]
    assert meta.indices.tolist() == [9, 9, 9]
    assert meta.num_real == 0


def test_empty_batch_without_padding():
    meta = build_paged_kv_metadata([], [], PAGE)
    assert meta.indptr.tolist() == [0]
    assert meta.indices.numel() == 0


@pytest.mark.parametrize(
    "tables, lens, kwargs",
    [
        ([[1]], [17], {}),                    # needs 2 pages, has 1
        ([[1], [2]], [1], {}),                # length mismatch
        ([[1], [2]], [1, 1], {"pad_to": 1}),  # batch larger than fixed size
        ([[1]], [-1], {}),                    # negative length
    ],
)
def test_invalid_inputs_raise(tables, lens, kwargs):
    with pytest.raises(ValueError):
        build_paged_kv_metadata(tables, lens, PAGE, **kwargs)


def test_qo_indptr():
    assert build_qo_indptr([36, 1, 50]).tolist() == [0, 36, 37, 87]
    with pytest.raises(ValueError):
        build_qo_indptr([3, 0])


def test_plan_key_tracks_identity_and_inplace_writes():
    block_table = torch.zeros(2, 4, dtype=torch.int32)
    context_lens = torch.zeros(2, dtype=torch.int32)
    state = _PlanState(wrapper=None, num_rows=2, graph=False, key=_make_key(block_table, context_lens))

    assert state.matches(block_table, context_lens)
    # A different tensor with identical contents is a different forward.
    assert not state.matches(block_table.clone(), context_lens)
    # In-place refill of a static buffer (how the CUDA-Graph path updates it)
    # invalidates the plan, including writes through a view.
    block_table[0, :2].copy_(torch.tensor([1, 2], dtype=torch.int32))
    assert not state.matches(block_table, context_lens)


def test_plan_without_key_matches_anything():
    state = _PlanState(wrapper=None, num_rows=1, graph=False, key=_make_key(None, None))
    assert state.matches(torch.zeros(1), torch.zeros(1))


class RecordingPrefill:
    def __init__(self):
        self.partitions = []

    def plan(self, qo_indptr, *args, **kwargs):
        self.partitions.append(qo_indptr.tolist())


def cpu_prefill_ops():
    # Exercise the real plan/cache implementation; only replace the GPU wrapper.
    ops = object.__new__(FlashInferOps)
    ops.shape = AttentionShape(2, 1, 8, PAGE, torch.float32, torch.device("cpu"))
    ops._prefill = RecordingPrefill()
    ops._prefill_state = None
    ops._warned_fallback = True
    ops.stats = PlanStats()
    return ops


def test_prefill_replans_when_query_partition_changes():
    ops = cpu_prefill_ops()
    bt = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32)
    cl = torch.tensor([64, 64], dtype=torch.int32)
    nl = torch.tensor([32, 32], dtype=torch.int32)
    starts = torch.tensor([0, 32], dtype=torch.int32)
    q, k = torch.zeros(64, 2, 8), torch.zeros(8, 1, PAGE, 8)
    ops.plan_prefill(list(bt.unbind()), [64, 64], [32, 32],
                     block_table=bt, context_lens=cl, new_token_lens=nl, q_start_loc=starts)
    ops._resolve_prefill(q, k, bt, cl, nl, starts)
    assert ops.stats.fallback_plans == 0
    nl.copy_(torch.tensor([16, 48]))
    starts.copy_(torch.tensor([0, 16]))
    ops._resolve_prefill(q, k, bt, cl, nl, starts)
    assert ops._prefill.partitions == [[0, 32, 64], [0, 16, 64]]
    ops._resolve_prefill(q, k, bt, cl, nl, starts)
    assert ops.stats.fallback_plans == 1  # next layer reuses the new plan
    starts[1] = 17
    with pytest.raises(NotImplementedError, match="packed in request order"):
        ops._resolve_prefill(q, k, bt, cl, nl, starts)


def test_unbound_prefill_plan_is_validated_against_actual_query_metadata():
    ops = cpu_prefill_ops()
    bt = torch.tensor([[0], [1]], dtype=torch.int32)
    cl = torch.tensor([16, 16], dtype=torch.int32)
    ops.plan_prefill([[0], [1]], [16, 16], [8, 8])
    nl, starts = torch.tensor([4, 12]), torch.tensor([0, 4])
    ops._resolve_prefill(torch.zeros(16, 2, 8), torch.zeros(2, 1, PAGE, 8), bt, cl, nl, starts)
    assert ops._prefill.partitions[-1] == [0, 4, 16]
    assert ops.stats.fallback_plans == 1


def test_stale_decode_plan_is_rejected_during_capture(monkeypatch):
    bt, cl = torch.zeros(2, 4, dtype=torch.int32), torch.ones(2, dtype=torch.int32)
    ops = object.__new__(FlashInferOps)
    ops._decode_state = _PlanState(None, 2, True, _make_key(bt, cl))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    cl.add_(1)
    with pytest.raises(RuntimeError, match="stale decode plan"):
        ops._resolve_decode(torch.zeros(2, 2, 1, 8), None, bt, cl)
