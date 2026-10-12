# 02 修复 prefill plan 缓存键不完整

## 问题

旧 key 只追踪 `block_table/context_lens`。当 Q 分段由 `[32, 32]` 变为 `[16, 48]`，
KV 不变且 Q 总数仍是 64，旧 `qo_indptr` 会被复用，造成请求间错误的 query 分段。
仅改变 `q_start_loc` 也可能跳过合法性校验。当前 Engine 显式每步 plan，但直接调用
backend 或 ragged batch 使用者可触发此问题。

## 改动

`plan_prefill` 新增可选 `new_token_lens/q_start_loc` 绑定参数。
prefill key 追踪四个 metadata tensor 的对象和原地修改版本；Engine、microbenchmark、
GPU 测试都传完整绑定。旧调用未传完整 metadata 时，从真实输入重建一次 plan，
校验 packed Q 顺序，并把完整 key 缓存给后续模型层。

CUDA Graph capture 的 decode 路径也检查现有 plan key；若 capture 前 metadata 已改变，
明确报错要求在 capture 外重新 plan。

## 涉及文件

`pico_vllm/ops/base.py`、`ops/flashinfer/backend.py`、`engine.py`、
`benchmarks/bench_attention_kernels.py` 和 FlashInfer metadata/attention 测试。

## 验证

CPU 回归用真实 adapter 和记录型 wrapper 验证分段改变会 re-plan、下一层可复用、
非法 start 会报错、未绑定 plan 会重新验证。GPU 回归将变化后的结果与 fp32 reference 比较。
执行结果见 08。
