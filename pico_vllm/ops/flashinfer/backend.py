"""FlashInfer attention backend.

Only the two paged-attention ops are replaced. RMSNorm, SwiGLU, fused RoPE and
the KV-cache store kernels are inherited from `TritonOps`, so comparing
`ops_backend="triton"` with `ops_backend="flashinfer"` changes exactly one
variable: the attention kernels.

FlashInfer splits attention into `plan()` (CPU-side scheduling, once per forward
step, shared by all layers) and `run()` (once per layer). This backend holds the
current plan. The Engine plans explicitly before each forward from host-side
lengths and GPU page tables (no device->host sync). Callers that do not plan
(older scripts, tests that call the model directly) still get correct results:
the backend notices the missing plan and derives one from the `block_table` /
`context_lens` tensors it receives, at the cost of one sync per step.

A plan is tied to the forward's metadata tensors (identity and in-place version
counter). Prefill also binds `new_token_lens` and `q_start_loc`; callers omitting
these bindings pay a one-time fallback to validate the actual query partition.

KV layout: Pico-vLLM stores each layer's cache as
(num_blocks, num_kv_heads, block_size, head_dim), which is FlashInfer's "HND"
layout with page_size = block_size, so the cache is passed zero-copy as a
(k_cache, v_cache) tuple.

Environment knobs (read once, at construction):
    PICO_FLASHINFER_TENSOR_CORES   1/0  tensor-core decode (default 1; Qwen2.5's
                                        GQA group of 6 packs onto tensor cores)
    PICO_FLASHINFER_DECODE_BACKEND      FlashInfer decode backend (default "auto")
    PICO_FLASHINFER_WORKSPACE_MB        float workspace size (default 128)
"""

from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass
from typing import Sequence

import torch

from ..triton.backend import TritonOps
from .metadata import build_paged_kv_metadata, build_qo_indptr

_DEFAULT_WORKSPACE_MB = 128


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "no", "off", "")


@dataclass(frozen=True)
class AttentionShape:
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    dtype: torch.dtype
    device: torch.device
    dummy_page: int = 0


@dataclass
class PlanStats:
    decode_plans: int = 0
    prefill_plans: int = 0
    fallback_plans: int = 0
    # Host duration inside FlashInfer wrapper.plan(), excluding metadata build.
    # Asynchronous device copies may still be pending when this timer stops.
    wrapper_plan_host_seconds: float = 0.0

    def reset(self) -> None:
        self.decode_plans = self.prefill_plans = self.fallback_plans = 0
        self.wrapper_plan_host_seconds = 0.0


def _make_key(*tensors: torch.Tensor | None) -> tuple | None:
    """Tie a plan to specific tensor objects and their current in-place versions.

    Holding strong references guarantees a new tensor can never alias an old
    key by reusing its Python id; `_version` catches in-place refills of the
    static CUDA-Graph buffers (fill_/copy_ on a view bump the shared counter).
    """
    if any(t is None for t in tensors):
        return None
    return tuple((t, t._version) for t in tensors)


@dataclass
class _PlanState:
    wrapper: object
    num_rows: int        # decode: batch size; prefill: total query tokens
    graph: bool
    key: tuple | None    # None: explicitly planned without tensors, valid until replanned

    def matches(self, *tensors: torch.Tensor | None) -> bool:
        if self.key is None:
            return True
        if len(tensors) != len(self.key):
            return False
        return all(t is kt and t._version == ver for t, (kt, ver) in zip(tensors, self.key))


class FlashInferOps(TritonOps):
    """Triton ops with FlashInfer paged attention."""

    name = "flashinfer"
    device_type = "cuda"
    supports_cuda_graph = True
    needs_attention_plan = True

    def __init__(
        self,
        *,
        use_tensor_cores: bool | None = None,
        decode_backend: str | None = None,
        workspace_mb: int | None = None,
    ) -> None:
        try:
            import flashinfer
        except ImportError as exc:
            raise ImportError(
                "ops_backend='flashinfer' requires the `flashinfer-python` package "
                "built for your torch/CUDA versions (see https://docs.flashinfer.ai)."
            ) from exc
        self._fi = flashinfer
        self.use_tensor_cores = (
            _env_bool("PICO_FLASHINFER_TENSOR_CORES", True)
            if use_tensor_cores is None else use_tensor_cores
        )
        self.decode_backend = (
            os.environ.get("PICO_FLASHINFER_DECODE_BACKEND", "auto")
            if decode_backend is None else decode_backend
        )
        mb = (
            int(os.environ.get("PICO_FLASHINFER_WORKSPACE_MB", _DEFAULT_WORKSPACE_MB))
            if workspace_mb is None else workspace_mb
        )
        self.workspace_bytes = mb * 1024 * 1024

        self.shape: AttentionShape | None = None
        self._workspace: torch.Tensor | None = None
        self._decode_eager = None
        self._decode_graph = None
        self._graph_buffers: tuple[torch.Tensor, ...] | None = None
        self._graph_batch_size = 0
        self._prefill = None
        self._decode_state: _PlanState | None = None
        self._prefill_state: _PlanState | None = None
        self._warned_fallback = False
        self.stats = PlanStats()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup_attention(
        self,
        *,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device | str,
        dummy_page: int = 0,
    ) -> None:
        if num_qo_heads % num_kv_heads != 0:
            raise ValueError(f"num_qo_heads={num_qo_heads} is not a multiple of num_kv_heads={num_kv_heads}")
        shape = AttentionShape(
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            dtype=dtype,
            device=torch.device(device),
            dummy_page=dummy_page,
        )
        if self.shape is not None and self.shape.device != shape.device:
            self._workspace = None
            self._decode_eager = self._decode_graph = self._prefill = None
        self.shape = shape
        self._decode_state = self._prefill_state = None

    def init_cuda_graph(self, *, max_batch_size: int, max_num_pages: int) -> None:
        shape = self._require_shape()
        dev = shape.device
        self._graph_buffers = (
            torch.zeros(max_batch_size + 1, dtype=torch.int32, device=dev),
            torch.zeros(max_num_pages, dtype=torch.int32, device=dev),
            torch.zeros(max_batch_size, dtype=torch.int32, device=dev),
        )
        indptr_buf, indices_buf, last_page_len_buf = self._graph_buffers
        self._decode_graph = self._fi.BatchDecodeWithPagedKVCacheWrapper(
            self._workspace_buffer(),
            "HND",
            use_cuda_graph=True,
            use_tensor_cores=self.use_tensor_cores,
            paged_kv_indptr_buffer=indptr_buf,
            paged_kv_indices_buffer=indices_buf,
            paged_kv_last_page_len_buffer=last_page_len_buf,
            **self._decode_backend_kwargs(),
        )
        self._graph_batch_size = max_batch_size
        if self._decode_state is not None and self._decode_state.graph:
            self._decode_state = None

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------
    def plan_decode(
        self,
        page_tables: Sequence,
        seq_lens: Sequence[int],
        *,
        use_cuda_graph: bool = False,
        block_table: torch.Tensor | None = None,
        context_lens: torch.Tensor | None = None,
    ) -> None:
        shape = self._require_shape()
        if use_cuda_graph:
            if self._decode_graph is None:
                raise RuntimeError("call init_cuda_graph() before planning CUDA-Graph decode")
            wrapper, pad_to = self._decode_graph, self._graph_batch_size
        else:
            wrapper, pad_to = self._get_decode_eager(), None
        meta = build_paged_kv_metadata(
            page_tables, seq_lens, shape.page_size,
            pad_to=pad_to, dummy_page=shape.dummy_page, device=shape.device,
        )
        self._plan_decode_wrapper(wrapper, meta)
        self._decode_state = _PlanState(
            wrapper=wrapper,
            num_rows=meta.batch_size,
            graph=use_cuda_graph,
            key=_make_key(block_table, context_lens),
        )

    def plan_prefill(
        self,
        page_tables: Sequence,
        kv_lens: Sequence[int],
        q_lens: Sequence[int],
        *,
        block_table: torch.Tensor | None = None,
        context_lens: torch.Tensor | None = None,
        new_token_lens: torch.Tensor | None = None,
        q_start_loc: torch.Tensor | None = None,
    ) -> None:
        shape = self._require_shape()
        if not (len(page_tables) == len(kv_lens) == len(q_lens)):
            raise ValueError("page_tables, kv_lens and q_lens must have the same length")
        for kv_len, q_len in zip(kv_lens, q_lens):
            if not 0 < int(q_len) <= int(kv_len):
                raise ValueError(f"need 0 < q_len <= kv_len, got q_len={q_len}, kv_len={kv_len}")
        meta = build_paged_kv_metadata(
            page_tables, kv_lens, shape.page_size,
            dummy_page=shape.dummy_page, device=shape.device,
        )
        qo_indptr = build_qo_indptr(q_lens)
        wrapper = self._get_prefill()
        t0 = time.perf_counter()
        # causal=True aligns Q to the *end* of its KV (bottom-right), which is
        # exactly the prefix-cache case: Q = new tokens, KV = prefix + new.
        wrapper.plan(
            qo_indptr,
            meta.indptr,
            meta.indices,
            meta.last_page_len,
            shape.num_qo_heads,
            shape.num_kv_heads,
            shape.head_dim,
            shape.page_size,
            causal=True,
            q_data_type=shape.dtype,
            kv_data_type=shape.dtype,
        )
        self.stats.wrapper_plan_host_seconds += time.perf_counter() - t0
        self.stats.prefill_plans += 1
        self._prefill_state = _PlanState(
            wrapper=wrapper,
            num_rows=int(qo_indptr[-1]),
            graph=False,
            key=_make_key(block_table, context_lens, new_token_lens, q_start_loc),
        )

    # ------------------------------------------------------------------
    # Attention ops (signatures match TritonOps so the model is unchanged)
    # ------------------------------------------------------------------
    def paged_decode_attention(
        self,
        q: torch.Tensor,                 # (B, n_heads, 1, head_dim)
        k_cache: torch.Tensor,           # (num_blocks, n_kv_heads, block_size, head_dim)
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        context_lens: torch.Tensor,
        MAX_BLOCKS_PER_SEQ: int,
        BLOCK_SIZE: int = 16,
    ) -> torch.Tensor:
        B, H, _, D = q.shape
        state = self._resolve_decode(q, k_cache, block_table, context_lens)
        out = state.wrapper.run(q.reshape(B, H, D), (k_cache, v_cache))
        return out.view(B, H, 1, D)

    def paged_prefill_attention(
        self,
        q: torch.Tensor,                 # (total_new_tokens, n_heads, head_dim)
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        context_lens: torch.Tensor,
        new_token_lens: torch.Tensor,
        q_start_loc: torch.Tensor,
        MAX_BLOCKS_PER_SEQ: int,
        BLOCK_SIZE: int = 16,
        BLOCK_M: int = 16,
        max_new_len: int | None = None,
    ) -> torch.Tensor:
        # Upstream supplies this CPU-side grid hint. FlashInfer already knows
        # query lengths from plan_prefill(), so no grid calculation is needed.
        state = self._resolve_prefill(q, k_cache, block_table, context_lens, new_token_lens, q_start_loc)
        return state.wrapper.run(q.contiguous(), (k_cache, v_cache))

    # ------------------------------------------------------------------
    # Introspection (used by the benchmarks to record what actually ran)
    # ------------------------------------------------------------------
    def describe(self) -> dict:
        info = {
            "flashinfer_version": getattr(self._fi, "__version__", "unknown"),
            "use_tensor_cores": self.use_tensor_cores,
            "decode_backend_requested": self.decode_backend,
        }
        for label, wrapper in (("decode_eager", self._decode_eager), ("decode_graph", self._decode_graph)):
            if wrapper is not None and hasattr(wrapper, "resolved_backend"):
                try:
                    info[f"{label}_resolved_backend"] = wrapper.resolved_backend
                except Exception:  # property may need a prior plan()
                    pass
        return info

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _require_shape(self) -> AttentionShape:
        if self.shape is None:
            raise RuntimeError("FlashInferOps.setup_attention() has not been called")
        return self.shape

    def _ensure_shape(self, num_qo_heads: int, k_cache: torch.Tensor, dtype: torch.dtype) -> AttentionShape:
        _, num_kv_heads, page_size, head_dim = k_cache.shape
        if self.shape is None:
            self.setup_attention(
                num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
                page_size=page_size, dtype=dtype, device=k_cache.device,
            )
        s = self.shape
        assert s is not None
        if (s.num_qo_heads, s.num_kv_heads, s.head_dim, s.page_size) != (num_qo_heads, num_kv_heads, head_dim, page_size):
            raise ValueError(
                f"attention inputs (heads={num_qo_heads}/{num_kv_heads}, head_dim={head_dim}, "
                f"page_size={page_size}) do not match setup_attention() ({s})"
            )
        return s

    def _workspace_buffer(self) -> torch.Tensor:
        if self._workspace is None:
            shape = self._require_shape()
            # FlashInfer requires the float workspace to be zero on first use. One
            # buffer is shared by all wrappers: they never run concurrently.
            self._workspace = torch.zeros(self.workspace_bytes, dtype=torch.uint8, device=shape.device)
        return self._workspace

    def _decode_backend_kwargs(self) -> dict:
        # Older FlashInfer releases have no `backend` argument on the decode
        # wrapper; only pass it when a non-default backend is requested.
        return {} if self.decode_backend == "auto" else {"backend": self.decode_backend}

    def _get_decode_eager(self):
        if self._decode_eager is None:
            self._decode_eager = self._fi.BatchDecodeWithPagedKVCacheWrapper(
                self._workspace_buffer(),
                "HND",
                use_tensor_cores=self.use_tensor_cores,
                **self._decode_backend_kwargs(),
            )
        return self._decode_eager

    def _get_prefill(self):
        if self._prefill is None:
            self._prefill = self._fi.BatchPrefillWithPagedKVCacheWrapper(self._workspace_buffer(), "HND")
        return self._prefill

    def _plan_decode_wrapper(self, wrapper, meta) -> None:
        s = self._require_shape()
        t0 = time.perf_counter()
        wrapper.plan(
            meta.indptr,
            meta.indices,
            meta.last_page_len,
            s.num_qo_heads,
            s.num_kv_heads,
            s.head_dim,
            s.page_size,
            # plan() defaults q_data_type to float16; the model runs in bf16.
            q_data_type=s.dtype,
            kv_data_type=s.dtype,
        )
        self.stats.wrapper_plan_host_seconds += time.perf_counter() - t0
        self.stats.decode_plans += 1

    def _warn_fallback(self, what: str) -> None:
        if not self._warned_fallback:
            warnings.warn(
                f"FlashInferOps: {what} attention ran without an explicit plan; deriving it from "
                "block_table/context_lens costs one GPU->CPU sync per step. The Engine plans "
                "explicitly; direct model callers can call plan_decode()/plan_prefill().",
                stacklevel=4,
            )
            self._warned_fallback = True

    def _resolve_decode(self, q, k_cache, block_table, context_lens) -> _PlanState:
        batch = q.shape[0]
        state = self._decode_state
        if torch.cuda.is_current_stream_capturing():
            # Planning needs host data, which is illegal inside capture.
            if state is None or not state.graph:
                raise RuntimeError(
                    "FlashInfer decode is being captured into a CUDA graph without a CUDA-Graph "
                    "plan: call init_cuda_graph() and plan_decode(..., use_cuda_graph=True) first."
                )
            if not state.matches(block_table, context_lens):
                raise RuntimeError("stale decode plan before CUDA graph capture; re-plan outside capture")
        elif state is None or not state.matches(block_table, context_lens):
            state = self._fallback_plan_decode(q, k_cache, block_table, context_lens)
        if state.num_rows != batch:
            raise RuntimeError(f"decode plan is for batch {state.num_rows}, but q has batch {batch}")
        return state

    def _fallback_plan_decode(self, q, k_cache, block_table, context_lens) -> _PlanState:
        shape = self._ensure_shape(q.shape[1], k_cache, q.dtype)
        self._warn_fallback("decode")
        lens = context_lens.tolist()  # the one sync
        meta = build_paged_kv_metadata(
            list(block_table.unbind(0)), lens, shape.page_size,
            dummy_page=shape.dummy_page, device=shape.device,
        )
        wrapper = self._get_decode_eager()
        self._plan_decode_wrapper(wrapper, meta)
        self.stats.fallback_plans += 1
        self._decode_state = _PlanState(wrapper, meta.batch_size, False, _make_key(block_table, context_lens))
        return self._decode_state

    def _resolve_prefill(self, q, k_cache, block_table, context_lens, new_token_lens, q_start_loc) -> _PlanState:
        state = self._prefill_state
        # An unbound prefill plan cannot establish how Q is partitioned. Rebuild
        # it once from the actual metadata, then reuse it across model layers.
        if state is None or state.key is None or not state.matches(
            block_table, context_lens, new_token_lens, q_start_loc
        ):
            self._ensure_shape(q.shape[1], k_cache, q.dtype)
            self._warn_fallback("prefill")
            kv_lens = context_lens.tolist()
            q_lens = new_token_lens.tolist()
            starts = q_start_loc.tolist()
            expected = [sum(q_lens[:i]) for i in range(len(q_lens))]
            if starts != expected:
                raise NotImplementedError(
                    f"FlashInfer prefill expects Q packed in request order; q_start_loc={starts}, "
                    f"expected {expected}"
                )
            self.plan_prefill(
                list(block_table.unbind(0)), kv_lens, q_lens,
                block_table=block_table, context_lens=context_lens,
                new_token_lens=new_token_lens, q_start_loc=q_start_loc,
            )
            self.stats.fallback_plans += 1
            state = self._prefill_state
            assert state is not None
        if state.num_rows != q.shape[0]:
            raise RuntimeError(f"prefill plan is for {state.num_rows} query tokens, but q has {q.shape[0]}")
        return state
