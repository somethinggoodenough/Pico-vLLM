# 15 迁移测试并合并上游已有修复

## 问题

上游将分层测试移至 `pico_vllm/tests/ci/`，删除旧多卡 smoke 入口。
原补丁的 local CI 和说明仍引用旧路径；上游还已修复 helper 命名、pytest assert、
GPU tiny-model head_dim，不能覆盖回旧实现。

## 改动

- 将新增的 4 个 FlashInfer/benchmark 测试文件迁入相应 `tests/ci/` 子目录。
- 更新 local CI 和当前文档路径，不恢复上游删除的多卡入口。
- 沿用上游 `run_case`、assert、hidden_size=64；09/10 说明标记这些已由上游包含。
- 保留 prefill 测试脚本直接运行时的成功计数及非零失败退出码修复。
- 修正该脚本独立运行的 sys.path：迁入 ci 后多了一层，仍须定位到 pico_vllm。
- 08 保留为明确标注的历史验证记录，当前验证单独记录在 17。

## 文件与验证

`pico_vllm/tests/ci/`、`scripts/local_ci.py`、`FLASHINFER.md`、`change_notes/`。
pytest、新路径 local CI 与脚本直接运行检查见 17。
