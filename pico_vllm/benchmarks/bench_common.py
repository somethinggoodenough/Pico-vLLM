"""Shared helpers for the attention benchmarks: environment capture, GPU timing, output."""

from __future__ import annotations

import csv
import datetime as _dt
import json
import os
import platform
import statistics
from pathlib import Path
from typing import Callable

import torch


def environment_info() -> dict:
    from ops.triton.backend import TritonOps

    info = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "triton_attention_selection": {
            "decode_requested": TritonOps._decode_attn,
            "prefill_requested": TritonOps._prefill_attn,
            "decode_resolved": "legacy" if TritonOps._decode_attn == "legacy" else "flash",
            "prefill_resolved": (TritonOps._prefill_attn
                                 if TritonOps._prefill_attn in ("legacy", "v1") else "v2"),
            "environment": {key: os.environ.get(key) for key in
                            ("PICO_ATTN", "PICO_DECODE_ATTN", "PICO_PREFILL_ATTN")},
        },
    }
    try:
        import triton
        info["triton"] = triton.__version__
    except ImportError:
        info["triton"] = None
    try:
        import flashinfer
        info["flashinfer"] = getattr(flashinfer, "__version__", "unknown")
    except ImportError:
        info["flashinfer"] = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            gpu=props.name,
            sm_count=props.multi_processor_count,
            compute_capability=f"{props.major}.{props.minor}",
            total_memory_gb=round(props.total_memory / 2**30, 2),
            l2_cache_mb=round(getattr(props, "L2_cache_size", 0) / 2**20, 1),
        )
    return info


def make_flush_buffer() -> torch.Tensor:
    """A buffer larger than L2. Zeroing it before each timed call evicts the KV
    cache from L2: in the real model ~90 MB of weights stream through L2 between
    two layers' attention, so a cold L2 is the realistic condition."""
    l2 = getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 0) or 64 * 2**20
    return torch.empty(max(2 * l2, 64 * 2**20), dtype=torch.uint8, device="cuda")


def measure_copy_bandwidth_gbps(num_bytes: int = 512 * 2**20, iters: int = 10) -> float:
    """Achievable DRAM bandwidth (device-to-device copy: read + write)."""
    src = torch.empty(num_bytes, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    for _ in range(3):
        dst.copy_(src)
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        dst.copy_(src)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / 1e3)
    del src, dst
    return 2 * num_bytes / statistics.median(times) / 1e9


def time_cuda(
    fn: Callable[[], object],
    *,
    iters: int = 50,
    warmup: int = 5,
    flush: torch.Tensor | None = None,
    use_graph: bool = True,
) -> dict:
    """GPU time of `fn` in microseconds (median / p10 / p90).

    use_graph=True captures one call into a CUDA graph and times replays, which
    removes Python/launcher overhead; that matches the engine's decode path.
    The L2 flush runs before the start event, so it is not timed.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    call = fn
    if use_graph:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            fn()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        call = graph.replay

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        if flush is not None:
            flush.zero_()
        starts[i].record()
        call()
        ends[i].record()
    torch.cuda.synchronize()
    us = sorted(s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends))

    def pct(p: float) -> float:
        return us[min(len(us) - 1, int(p * (len(us) - 1)))]

    return {"median_us": statistics.median(us), "p10_us": pct(0.10), "p90_us": pct(0.90)}


def write_results(rows: list[dict], out_path: str | None, env: dict, args: dict) -> None:
    if not out_path or not rows:
        return
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    meta = {"env": env, "args": args}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"\nwrote {path} (+ {path.with_suffix('.json').name} with environment and arguments)")


def print_table(rows: list[dict], columns: list[str]) -> None:
    if not rows:
        return
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.3g}" if abs(v) < 1000 else f"{v:.0f}"
        return "" if v is None else str(v)
    cells = [[fmt(r.get(c)) for c in columns] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(columns)]
    print("  ".join(c.rjust(w) for c, w in zip(columns, widths)))
    for row in cells:
        print("  ".join(v.rjust(w) for v, w in zip(row, widths)))
