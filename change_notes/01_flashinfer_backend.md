# 01 引入 FlashInfer 后端与 benchmark

## 问题 / 需求

项目原有 Triton attention，缺少通过同一个 Engine 切换 FlashInfer 的实现，以及区分
kernel 性能和完整推理性能的对照 benchmark。

## 改动

应用用户附件中的两个提交内容，增加可选 `ops_backend="flashinfer"`。
新增 CSR metadata 构造、plan/run adapter、Engine planning hook 和 backend selector。
保留 Triton 的 RMSNorm、SwiGLU、RoPE、KV store；HND KV cache 直接传给 FlashInfer。
Engine 增加 `submit_ids`、`max_num_seqs` 和 benchmark 使用的 `ignore_eos`。

新增 `bench_attention_kernels.py`（kernel）和 `bench_decode_e2e.py`（Engine），
支持 CSV/JSON 环境信息输出和 E2E step 级 NVTX。另增加 CPU/GPU/Engine 测试。

## 涉及文件

- `pico_vllm/ops/flashinfer/`、`ops/base.py`、`ops/selector.py`
- `pico_vllm/engine.py`
- `pico_vllm/benchmarks/bench_attention_kernels.py`、`bench_common.py`、`bench_decode_e2e.py`
- `pico_vllm/tests/ci/01_ops/test_flashinfer_*.py`、`tests/ci/02_single_card/test_flashinfer_engine.py`
- `scripts/local_ci.py`、`FLASHINFER.md`

## 验证与边界

补丁对原始 HEAD 可应用；本目录其他说明记录应用后的修复。实际执行结果见 08。
microbenchmark 使用随机 Q/K/V；E2E 可使用真实权重或随机权重，不依赖训练数据。
它们评估执行速度和实现数值正确性，不提供语言任务质量分数，也未完成 TP/PD 多卡验证。
