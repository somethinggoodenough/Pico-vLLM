# 16 分开上游 H200 与现有 WSL 依赖配置

## 问题

最新 `requirements.txt` 固定 torch 2.11.0+cu128 / Triton 3.6.0；现有 WSL
环境为 torch 2.6.0+cu124 / Triton 3.2.0 / FlashInfer 0.2.5+cu124torch2.6。
这两个配置不是可叠加的 requirements；混装会产生冲突或替换可运行环境。

## 改动

- 保留上游 H200 版本固定值，不私自升级现有 `.venv-wsl`。
- `requirements-flashinfer.txt` 明确为替代配置，并补全 transformers、safetensors、
  pytest、numpy、matplotlib，便于单独安装；不是在上游 requirements 之上叠加。
- 两份 requirements 和 FLASHINFER.md 明确应使用不同虚拟环境。

## 验证与限制

本轮只在现有 WSL 环境测试源码兼容性，没有安装或升级包，也没有验证
H200 / torch 2.11 配置或从零重建虚拟环境。具体测试结果见 17。
