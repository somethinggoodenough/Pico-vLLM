# 08 回归测试与验证记录

> 历史记录：以下结果属于同步前的 `ddafca2` 加本地补丁，命令保留原路径。
> 最新版本的测试路径已迁移至 `tests/ci/`，请看 [17_sync_validation.md](17_sync_validation.md)。

## 改动

新增 query 分段变化、未绑定 prefill plan、ghost Q 初始化、dummy page 按需分配、
连续 benchmark 生命周期等回归测试，并把 CPU lifecycle 测试接入 local CI。
CPU 测试可独立验证 host metadata 和 scheduler 生命周期；GPU 测试验证实际 kernel 结果。

## 涉及文件

`pico_vllm/tests/01_ops/test_flashinfer_metadata.py`、`test_flashinfer_attention.py`、
`pico_vllm/tests/02_single_card/test_attention_benchmark_lifecycle.py`、
`test_flashinfer_engine.py`、`scripts/local_ci.py`。

## 实际执行结果

- 最终扩展回归：**56 passed，1 xfailed，0 failed，0 error**（41.87 秒）。
  包含本次新增测试、原有 CPU ops、12 组 Triton prefill，以及 CPU/GPU tiny model。
  XFAIL 为原补丁已标记的 `use_tensor_cores=False` / GQA group size 6 用例；
  默认 tensor-core FlashInfer、graph/eager Engine、prefix-cache 路径均通过。
- metadata CPU 回归在最终套件中共 17 项通过。
- 已安装 FlashInfer 0.2.5+cu124torch2.6；PyTorch 2.6.0+cu124，Triton 3.2.0。
- 扩展回归曾发现两个原有测试问题，现已修复并重跑通过，详见 09、10。
- GPU：NVIDIA GeForce RTX 4060，8 GiB，24 SM，L2 24 MiB。
- decode microbenchmark：B=1，context=32/64，各 3 次；两后端均成功，CSV/JSON 已生成。
- prefill microbenchmark：Q=16/32，prefix=0/16，各 3 次；两后端均成功，CSV/JSON 已生成。
- 完整 Qwen E2E：随机权重，B=1，prompt=16，gen_len=4，repeats=2，CUDA Graph + NVTX；
  两后端均成功，CSV/JSON 已生成。启用 NVTX 不等于已经运行 Nsight 采集。
- Python compileall 和 git diff --check 通过。

## 重现命令

以下在仓库根目录的 WSL shell 中运行；也可从 PowerShell 用
`wsl --cd /mnt/c/Users/admin/Pico-vLLM --exec /mnt/c/Users/admin/Pico-vLLM/.venv-wsl/bin/python`
代替命令开头的 `python`。

```bash
python -m pytest -q -rx pico_vllm/tests/01_ops pico_vllm/tests/02_single_card/test_flashinfer_engine.py pico_vllm/tests/02_single_card/test_attention_benchmark_lifecycle.py pico_vllm/tests/02_single_card/test_tiny_model_smoke.py pico_vllm/tests/02_single_card/test_torch_cpu_tiny_model.py
python pico_vllm/benchmarks/bench_attention_kernels.py decode --batch-sizes 1 --ctx-lens 32 64 --iters 3 --out /tmp/pico-flashinfer-review-decode.csv
python pico_vllm/benchmarks/bench_attention_kernels.py prefill --q-lens 16 32 --prefix-lens 0 16 --iters 3 --out /tmp/pico-flashinfer-review-prefill.csv
python pico_vllm/benchmarks/bench_decode_e2e.py --batch-sizes 1 --prompt-lens 16 --gen-len 4 --repeats 2 --random-weights --nvtx --out /tmp/pico-flashinfer-review-e2e.csv
```

运行产物位于 WSL 的 `/tmp/pico-flashinfer-review-{decode,prefill,e2e}.{csv,json}`。
这些短时间 smoke run 用于验证可执行性和输出格式，不作为稳定性能或加速比结论。

本记录只报告实际运行结果。没有权重质量评估结果，也没有 TP/PD 多卡或 Nsight 采集结果。
