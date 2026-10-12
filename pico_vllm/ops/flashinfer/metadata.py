"""Paged-KV metadata in the CSR form FlashInfer expects.

Pico-vLLM describes a batch with a padded dense table, `block_table (B, MAX_BLOCKS)`,
plus `context_lens (B,)`. FlashInfer instead wants, per batch:

    kv_indptr        [B + 1]  request i owns kv_indices[indptr[i] : indptr[i+1]]
    kv_indices       [nnz]    physical page ids of all requests, concatenated
    kv_last_page_len [B]      valid tokens in each request's last page, in [1, page_size]

Only `kv_indices` can be large (sum of pages over the batch), so it is built by
concatenating per-request page tensors on whatever device they live on: the
engine passes GPU views of `PagedKVCache.gpu_block_table`, so building it costs
one `torch.cat` and no host sync. `indptr` / `last_page_len` are tiny and are
built on the host, which is where FlashInfer's `plan()` reads them anyway.

This module is pure torch and runs on CPU, so it is unit-tested without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


@dataclass
class PagedKVMetadata:
    indptr: torch.Tensor         # int32, host, [B + 1]
    indices: torch.Tensor        # int32, on `device`, [indptr[-1]]
    last_page_len: torch.Tensor  # int32, host, [B]
    kv_lens: list[int]           # effective KV length per slot (ghost slots -> 1)
    num_real: int                # slots [0, num_real) are real requests, the rest are padding

    @property
    def batch_size(self) -> int:
        return len(self.kv_lens)

    @property
    def num_pages(self) -> int:
        return int(self.indptr[-1])


def _as_page_tensor(pages, num_pages: int, device: torch.device) -> torch.Tensor:
    """First `num_pages` entries of one request's page list, as int32 on `device`."""
    if isinstance(pages, torch.Tensor):
        if pages.dim() != 1:
            raise ValueError(f"page table must be 1-D, got shape {tuple(pages.shape)}")
        if pages.numel() < num_pages:
            raise ValueError(f"request needs {num_pages} pages but its table has {pages.numel()}")
        out = pages[:num_pages]
        if out.dtype != torch.int32:
            out = out.to(torch.int32)
        if out.device != device:
            out = out.to(device, non_blocking=True)
        return out
    pages = list(pages)
    if len(pages) < num_pages:
        raise ValueError(f"request needs {num_pages} pages but its table has {len(pages)}")
    return torch.tensor(pages[:num_pages], dtype=torch.int32, device=device)


def build_paged_kv_metadata(
    page_tables: Sequence,
    seq_lens: Sequence[int],
    page_size: int,
    *,
    pad_to: int | None = None,
    dummy_page: int = 0,
    device: torch.device | str = "cpu",
) -> PagedKVMetadata:
    """Convert per-request page tables + lengths to FlashInfer CSR metadata.

    Slots with seq_len == 0 (CUDA-Graph ghost requests) and padding slots up to
    `pad_to` point at `dummy_page` with a length of 1 token: FlashInfer derives
    kv_len = (num_pages - 1) * page_size + last_page_len, so a slot must own at
    least one page. Their outputs are garbage-but-finite and must be discarded.
    """
    if len(page_tables) != len(seq_lens):
        raise ValueError(f"{len(page_tables)} page tables but {len(seq_lens)} lengths")
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    num_real = len(seq_lens)
    if pad_to is not None and num_real > pad_to:
        raise ValueError(f"batch of {num_real} does not fit the fixed batch size {pad_to}")
    device = torch.device(device)

    indptr = [0]
    last_page_len: list[int] = []
    kv_lens: list[int] = []
    chunks: list[torch.Tensor] = []
    dummy: torch.Tensor | None = None
    num_ghost_pages = 0

    def add_ghost() -> None:
        nonlocal dummy, num_ghost_pages
        if dummy is None:
            dummy = torch.tensor([dummy_page], dtype=torch.int32, device=device)
        chunks.append(dummy)
        num_ghost_pages += 1
        indptr.append(indptr[-1] + 1)
        last_page_len.append(1)
        kv_lens.append(1)

    for pages, n in zip(page_tables, seq_lens):
        n = int(n)
        if n < 0:
            raise ValueError(f"negative sequence length {n}")
        if n == 0:
            add_ghost()
            continue
        num_pages = cdiv(n, page_size)
        chunks.append(_as_page_tensor(pages, num_pages, device))
        indptr.append(indptr[-1] + num_pages)
        last_page_len.append(n - (num_pages - 1) * page_size)
        kv_lens.append(n)

    for _ in range(0 if pad_to is None else pad_to - num_real):
        add_ghost()

    if chunks:
        indices = chunks[0] if len(chunks) == 1 else torch.cat(chunks)
    else:
        indices = torch.empty(0, dtype=torch.int32, device=device)

    return PagedKVMetadata(
        indptr=torch.tensor(indptr, dtype=torch.int32),
        indices=indices,
        last_page_len=torch.tensor(last_page_len, dtype=torch.int32),
        kv_lens=kv_lens,
        num_real=num_real,
    )


def build_qo_indptr(q_lens: Sequence[int]) -> torch.Tensor:
    """Host int32 [B + 1] prefix sum of query lengths (ragged Q packing)."""
    indptr = [0]
    for n in q_lens:
        n = int(n)
        if n <= 0:
            raise ValueError(f"query length must be positive, got {n}")
        indptr.append(indptr[-1] + n)
    return torch.tensor(indptr, dtype=torch.int32)
