# 14 让 benchmark 跟随最新 Triton 基线

## 问题

原 microbenchmark 直接导入 `ops.triton.attention`（legacy），新版 Engine 默认
使用 Flash Decode 和 Prefill v2。两者都叫 triton 会掩盖基线差异，误导加速比比较。
旧 prefill 计时还包含 `.item()`，但新版 Engine 已传入 CPU 已知长度。

## 改动

- microbenchmark 和 FlashInfer 数值回归通过 `TritonOps` 选择算子，与 Engine 一致。
- prefill 传入已知 `max_new_len`，避免引入 Engine 已消除的同步；仍使用 eager CUDA event 计时。
- 两种 benchmark 的 JSON 记录 `env.triton_attention_selection`，包含原始环境变量、
  requested/resolved kernel。默认 decode=flash、prefill=v2。
- 保留 `PICO_DECODE_ATTN=legacy` / `PICO_PREFILL_ATTN=legacy` 选择旧基线的能力。
- FLASHINFER.md 明确旧 CSV 与新基线不能不加区分地比较；不改写用户原结果。

## 文件与验证

`bench_attention_kernels.py`、`bench_common.py`、`test_flashinfer_attention.py`、`FLASHINFER.md`。
最新版默认基线的 decode/prefill/E2E smoke 及 JSON 检查结果见 17。
