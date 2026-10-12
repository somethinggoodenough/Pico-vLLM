# 17 最新版本同步后的验证

## 验证对象与环境

上游 `1691ebb522316e285dcc98bd29a9fe6d54f9ca24` 加本地 FlashInfer 适配。
使用现有 WSL / Python 3.12.3 / RTX 4060 8 GiB，torch 2.6.0+cu124、Triton 3.2.0、
FlashInfer 0.2.5+cu124torch2.6。未升级依赖。

## 已完成测试

- 首次回归：61 passed、1 xfailed、2 failed；两处失败为上游 tiny-model 旧 logits 断言，见 18。
- 修复后单次回归：**63 passed、1 xfailed**，40.10 秒。
- 完整默认 `scripts/local_ci.py`：10 个步骤全部通过，包含 compileall，
  pytest 各步骤合计 **65 passed、1 xfailed**（比上述套件多 2 个环境检查）。
- 迁移后的 prefill 测试脚本直接运行：12 组通过，退出码 0。
- XFAIL 沿用原补丁：FlashInfer CUDA-core decode 的 GQA group size 6 支持限制；
  默认 tensor-core 路径、最新 Triton 基线、graph/eager、prefix-cache 测试通过。
- local CI 日志：`results/sync-1691ebb-validation/local-ci/summary.log` 和相邻逐项日志。

## Benchmark smoke

- Decode：两后端，B=1/3、context 上限 33/129、ragged 分布、iters=3；8 行有效结果。
- Prefill：两后端，Q=17/32、prefix=0/19、iters=3；8 行有效结果。
- 完整 Qwen E2E：随机权重、两后端、B=1/2、prompt=17、gen_len=4、repeats=2，
  CUDA Graph + NVTX；4 行有效结果，FlashInfer `fallback_plans=0`。
- 两个 microbenchmark 均无 error 行，GPU 时间为正，检查的 max diff vs fp32 均小于 0.03。
- CSV/JSON 位于 `results/sync-1691ebb-validation/`，核验 JSON 的默认 Triton 选择为 flash/v2。
- E2E 同样核验 flash/v2 基线、graph 模式及正数计时。只启用 NVTX，没有运行 Nsight 采集。
- 最终 `git diff --check` 通过，Git 无未解决冲突、无暂存更改；保留 stash，不提交或推送。

## 手动运行（已经在 WSL 中）

```bash
cd /mnt/c/Users/admin/Pico-vLLM
source .venv-wsl/bin/activate

# 全部默认 local CI；自动生成带时间戳的日志目录。
python scripts/local_ci.py

# 单次回归，避免逐文件启动 Python 的额外耗时。
python -m pytest -q -rx --tb=short \
  pico_vllm/tests/ci/01_ops \
  pico_vllm/tests/ci/02_single_card/test_flashinfer_engine.py \
  pico_vllm/tests/ci/02_single_card/test_attention_benchmark_lifecycle.py \
  pico_vllm/tests/ci/02_single_card/test_tiny_model_smoke.py \
  pico_vllm/tests/ci/02_single_card/test_torch_cpu_tiny_model.py

python pico_vllm/benchmarks/bench_attention_kernels.py decode \
  --backends triton flashinfer --batch-sizes 1 3 --ctx-lens 33 129 \
  --dist ragged --iters 3 --out results/manual-sync/decode.csv

python pico_vllm/benchmarks/bench_attention_kernels.py prefill \
  --backends triton flashinfer --q-lens 17 32 --prefix-lens 0 19 \
  --iters 3 --out results/manual-sync/prefill.csv

python pico_vllm/benchmarks/bench_decode_e2e.py \
  --backends triton flashinfer --batch-sizes 1 2 --prompt-lens 17 \
  --gen-len 4 --repeats 2 --random-weights --nvtx \
  --out results/manual-sync/e2e.csv
```

这些短 sweep 用来验证运行和集成，不足以给出稳定性能排名。
实际验证产物使用独立的 `results/sync-1691ebb-validation/` 目录；上面的手动命令另用 `manual-sync/`。
没有预训练权重质量评估、TP/PD 多卡、H200、新依赖配置或真实 Nsight 采集的验证结果。
