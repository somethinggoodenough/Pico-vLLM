from abc import ABC, abstractmethod
from typing import Sequence

import torch
import torch.nn as nn


class OpsBackend(ABC):
    """Backend-neutral operator surface used by the model.

    The interface is intentionally close to the current Triton wrappers so the
    model can be migrated one call site at a time.
    """

    name: str
    device_type: str
    supports_cuda_graph: bool = False

    # ------------------------------------------------------------------
    # Optional attention-planning hooks.
    #
    # Some attention libraries (e.g. FlashInfer) split attention into a
    # CPU-side `plan` (scheduling + metadata upload, done once per forward
    # step and shared by every layer) and a GPU-side `run` (once per layer).
    # Such backends set `needs_attention_plan = True` and override the hooks
    # below; the Engine calls them before each forward. Backends whose
    # kernels consume `block_table` / `context_lens` directly (Triton, Torch)
    # keep these no-op defaults and behave exactly as before.
    # ------------------------------------------------------------------
    needs_attention_plan: bool = False

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
        """Bind the (per-rank) attention shape once, before any plan.

        `dummy_page` is a physical KV block that padded/ghost batch slots may
        point at; it is only ever read, and its output is discarded.
        """

    def init_cuda_graph(self, *, max_batch_size: int, max_num_pages: int) -> None:
        """Pre-allocate fixed-address metadata buffers for CUDA Graph decode."""

    def plan_decode(
        self,
        page_tables: Sequence,
        seq_lens: Sequence[int],
        *,
        use_cuda_graph: bool = False,
        block_table: torch.Tensor | None = None,
        context_lens: torch.Tensor | None = None,
    ) -> None:
        """Plan one decode step.

        page_tables[i]: physical block ids of request i (list or 1-D int tensor).
        seq_lens[i]:    KV length of request i *including* the token written
                        in this step.
        block_table / context_lens: the exact tensors that will be passed to
                        the model in this forward; used to tie the plan to it.
        """

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
        """Plan prefill; pass all four metadata tensors to reuse the plan without a sync."""

    @abstractmethod
    def create_rms_norm(self, hidden_size: int, eps: float = 1e-6) -> nn.Module:
        ...

    @abstractmethod
    def swiglu(self, gate_up: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def store_kvcache(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_size: int = 16,
    ) -> None:
        ...

    @abstractmethod
    def decode_rope_and_cache(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache_k: torch.Tensor,
        kv_cache_v: torch.Tensor,
        slot_mapping: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> torch.Tensor:
        ...

    @abstractmethod
    def paged_decode_attention(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        context_lens: torch.Tensor,
        MAX_BLOCKS_PER_SEQ: int,
        BLOCK_SIZE: int = 16,
    ) -> torch.Tensor:
        ...

    @abstractmethod
    def paged_prefill_attention(
        self,
        q: torch.Tensor,
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
        ...
