# 12 保留新版 Engine API 并适配 FlashInfer

## 问题

上游新增 `submit_token_ids`、`detokenize_outputs`、`ignore_eos` 和 CPU-side
`max_new_len` 提示；原补丁使用 `submit_ids`，FlashInfer prefill 不接受新参数。
直接恢复旧文件会丢失上游 API，并导致 prefill unexpected keyword 错误。

## 改动

- 保留上游 `submit_token_ids` 为正式实现，`submit_ids` 作为转发别名保留旧调用兼容性。
- 保留 detokenize/EOS 行为，在构造函数末尾增加 `max_num_seqs`，不改变上游已有位置参数。
- 保留 Engine 传入 `max_new_len=new_len` 的优化；FlashInfer 接受该参数，长度仍来自 plan。
- E2E benchmark 使用正式 token-ID API，明确关闭 detokenize。
- 保留每 forward 一次的 FlashInfer plan 和 graph 固定元数据机制。

## 文件与验证

`pico_vllm/engine.py`、`ops/flashinfer/backend.py`、`benchmarks/bench_decode_e2e.py`。
CPU lifecycle 测试新增 3 种提交 API × 2 种 detokenize 设置；GPU 测试覆盖显式
`max_new_len`、graph/eager、prefix-cache。实际结果见 17。
