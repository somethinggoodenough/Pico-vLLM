# 05 固定 WSL FlashInfer 依赖配置

> 同步后注意：最新上游的 H200 requirements 与此配置互斥，不要叠加安装。
> 当前依赖说明见 [16_dependency_profiles.md](16_dependency_profiles.md)，最新验证见 17。

## 问题

原补丁只注释提示安装 `flashinfer-python`，没有固定版本；不同版本的 wrapper API、
支持的 CUDA/PyTorch 组合和自动选择的 kernel 不同，难以复现结果。

## 改动

读取现有 `.venv-wsl` 后确认 PyTorch 为 2.6.0+cu124。
新增 `requirements-flashinfer.txt` 固定 torch 2.6.0+cu124、Triton 3.2.0 和
FlashInfer 0.2.5+cu124torch2.6，使用官方 CUDA 12.4 / torch 2.6 wheel 索引。
这是可选配置，基础 requirements 不强制安装 FlashInfer。
`FLASHINFER.md` 增加仓库根目录的 WSL 和 PowerShell 调用命令。

## 涉及文件

`requirements-flashinfer.txt`、`requirements.txt`、`FLASHINFER.md`。

## 验证与来源

已在项目 `.venv-wsl` 安装该 FlashInfer wheel，原有 torch/triton 版本满足约束，无需升级。
[官方 wheel 索引](https://flashinfer.ai/whl/cu124/torch2.6/flashinfer-python/) 提供该组合。
实际 GPU 运行结果见 08；本配置不承诺新版 cuDNN decode API 可用。
