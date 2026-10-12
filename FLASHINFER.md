# FlashInfer attention backend

通过 `ModelConfig(ops_backend="flashinfer")` 选择 FlashInfer paged decode/prefill。
RMSNorm、SwiGLU、RoPE、KV store 仍用 Triton；KV cache 直接以 HND tuple 传入。
逐项修改原因见 [change_notes/README.md](change_notes/README.md)。

## 安装与环境

当前基于上游 `1691ebb`。上游 `requirements.txt` 是 H200 的 torch 2.11/cu128 +
Triton 3.6 配置；下面是现有 RTX 4060 / WSL 的替代配置，不是叠加安装。
两个 requirements 文件不能在同一个环境混装；保留 `.venv-wsl`，不要为了同步源码
直接运行 `pip install -r requirements.txt`。本轮不升级现有依赖。

在项目根目录、WSL 的已激活虚拟环境中执行：

```bash
python -m pip install -r requirements-flashinfer.txt
```

可从 PowerShell 调用现有 WSL 环境：

```powershell
wsl --cd /mnt/c/Users/admin/Pico-vLLM --exec /mnt/c/Users/admin/Pico-vLLM/.venv-wsl/bin/python -m pip install -r requirements-flashinfer.txt
```

固定配置：Linux x86_64 / WSL、PyTorch 2.6.0+cu124、Triton 3.2.0、
FlashInfer 0.2.5+cu124torch2.6。wheel 来自
[FlashInfer 官方 CUDA 12.4 / torch 2.6 索引](https://flashinfer.ai/whl/cu124/torch2.6/flashinfer-python/)。
这是适配现有项目环境的版本配置，其他 CUDA/PyTorch 组合需要另选匹配 wheel。
安装成功不等于 GPU 测试通过；实际验证记录见 change_notes。
首次遇到新 kernel 形状仍可能需要编译或加载缓存。

## plan 和 CUDA Graph

Engine 每次 forward 前 plan 一次，所有层共享。decode 的 key 包含
`block_table/context_lens`；prefill 还包含 `new_token_lens/q_start_loc`。
key 追踪 tensor 对象和原地修改版本；没有完整 prefill key 的调用会从实际
输入重建一次 plan，因此可能触发 GPU→CPU 同步。正常 Engine 路径传入完整 key。

CUDA Graph 使用固定地址 metadata buffers，plan 在 capture/replay 外执行。
仅 graph decode Engine 预留一个 dummy page；ghost 的 KV 长度在 FlashInfer
metadata 中为 1，fused RoPE 对 ghost 写零 Q、不写 KV，最终输出丢弃。
dummy block 在 Engine 使用期间保留；目前没有新增可复用 BlockManager 的 Engine close API。

可选环境变量：

- `PICO_FLASHINFER_TENSOR_CORES=0/1`：decode 是否使用 tensor cores，默认 1。
- `PICO_FLASHINFER_DECODE_BACKEND`：默认 auto；非默认值是否可用取决于安装版本。
- `PICO_FLASHINFER_WORKSPACE_MB`：workspace 大小，默认 128 MiB。

固定的 0.2.5 配置不承诺支持较新版本的 cuDNN decode wrapper API。
`flashinfer-cudnn` 是实验选项，异常写入 microbenchmark CSV 的 error 列。

## 测试

以下命令在仓库根目录的 WSL shell 执行：

```bash
python -m pytest -q pico_vllm/tests/ci/01_ops/test_flashinfer_metadata.py pico_vllm/tests/ci/01_ops/test_flashinfer_attention.py pico_vllm/tests/ci/02_single_card/test_flashinfer_engine.py pico_vllm/tests/ci/02_single_card/test_attention_benchmark_lifecycle.py
python scripts/local_ci.py --layer 01_ops
python scripts/local_ci.py --layer 02_single_card
```

数值测试比较 Triton、FlashInfer 和 fp32 reference，覆盖页边界、ragged prefill、
prefix、stale plan、ghost 和 graph re-plan。tiny model 的 token 比较只验证固定
smoke 输入；不能据此宣称任意 prompt 都逐 token 一致或模型质量已评估。

## Benchmark 与 Nsight

以下命令从仓库根目录运行；不依赖 training data。

`triton` 现在跟随最新 Engine：默认 Flash Decode + Prefill v2，**不是旧版 legacy kernel**。
JSON 的 `env.triton_attention_selection` 记录实际选择；旧 CSV 不能不加区分地与新结果比较。
Prefill microbenchmark 传入已知的 `max_new_len`，不额外加入每层 `.item()` 同步。
如需旧基线，在启动 Python 前指定环境变量，并用新的输出文件名：

```bash
PICO_DECODE_ATTN=legacy PICO_PREFILL_ATTN=legacy python pico_vllm/benchmarks/bench_attention_kernels.py decode --batch-sizes 1 --ctx-lens 512 --out results/decode_legacy.csv
```

```bash
python pico_vllm/benchmarks/bench_attention_kernels.py decode --batch-sizes 1 4 8 --ctx-lens 512 2048 8192 --out results/decode.csv
python pico_vllm/benchmarks/bench_attention_kernels.py prefill --q-lens 128 512 2048 --prefix-lens 0 2048 --out results/prefill.csv
python pico_vllm/benchmarks/bench_decode_e2e.py --batch-sizes 1 4 8 --prompt-lens 512 2048 --gen-len 128 --out results/e2e.csv
```

E2E 使用 `--weights PATH` 指定 safetensors；没有权重时可加
`--random-weights` 测固定形状的执行性能，生成内容没有语言质量意义。
`--gen-len` 至少为 2：第一个 token 在 prefill 生成，剩余 token 才产生 decode ITL。
每轮完成后在计时区间外 drain/释放请求；`prefill_ms` 仍是包含调度、分配和采样的
Engine prefill step wall time，并非纯 attention kernel 时间。

指标：

- `median_us`：microbenchmark GPU event 区间，decode 默认 CUDA Graph replay。
- `plan_host_us`：microbenchmark 的整次 backend plan host 调用时间，含 metadata 构造。
- `wrapper_plan_host_us`：E2E 统计的 FlashInfer wrapper.plan host 调用时间，不含 metadata 构造。
- 上述两个 host 指标都不保证异步 GPU copy 已完成，不能直接当完整 plan latency。
- `itl_p50_ms/p90/p99`、`decode_tok_s`：完整 Engine 的同步 wall time 和 decode 吞吐量。
- `kv_share_pct`：估算的 KV 流量占比，不是 attention 时间占比或加速上限。
- `bw_util_pct`：按估算流量和实测 D2D copy 带宽计算的参考值，不是 Nsight 实测 DRAM 利用率。

Nsight 示例（WSL，工具需已安装）：

```bash
nsys profile -t cuda,nvtx,osrt -o e2e python pico_vllm/benchmarks/bench_decode_e2e.py --batch-sizes 1 --prompt-lens 512 --gen-len 16 --repeats 1 --random-weights --nvtx
ncu --set basic --launch-count 6 python pico_vllm/benchmarks/bench_attention_kernels.py decode --backends flashinfer --batch-sizes 1 --ctx-lens 2048 --iters 3
```

NVTX 标注到 prefill/decode step。完整采集也会包含初始化/预热，
分析稳态性能时选择 NVTX 标记的测量区间；profiler 开销下的速度不替代普通 benchmark。
`benchmark_tp.py` 自行 capture graph，切换 FlashInfer 时还需要显式初始化并 plan；
本改动尚不构成 TP/PD 多卡路径的完整验证。
