# FlashInfer 改动说明索引

本目录按问题或改动拆分记录。每份说明包含问题、修改内容、涉及文件和验证方式。
01 记录用户提供补丁的引入；02–07 记录补丁修复；08 汇总测试结果；09–10 记录扩展回归发现的原有测试问题。

当前已同步上游 `1691ebb`；11–18 记录同步适配。08 是旧版本的历史结果，当前结果见 17。

| 文件 | 对应问题或改动 |
| --- | --- |
| [01_flashinfer_backend.md](01_flashinfer_backend.md) | 引入 FlashInfer 可选后端与两层 benchmark |
| [02_prefill_plan_key.md](02_prefill_plan_key.md) | 防止相同总 Q 长度下误用旧 ragged plan |
| [03_graph_ghost_rows.md](03_graph_ghost_rows.md) | 初始化 ghost Q、按需预留 dummy block，并更正原评审 |
| [04_benchmark_cleanup.md](04_benchmark_cleanup.md) | 在轮次之间回收请求，隔离上一轮清理时间 |
| [05_dependency_versions.md](05_dependency_versions.md) | 固定适配现有 WSL 的依赖组合 |
| [06_benchmark_metrics.md](06_benchmark_metrics.md) | 区分 host plan 时间与 GPU latency，澄清带宽估算 |
| [07_test_state_isolation.md](07_test_state_isolation.md) | 消除复用后端统计计数导致的测试误报 |
| [08_validation.md](08_validation.md) | 新增回归测试、接入 local CI、记录实际验证结果 |
| [09_prefill_test_collection.md](09_prefill_test_collection.md) | 修复原有 prefill 测试误收集和布尔返回值不触发失败 |
| [10_tiny_gpu_test_shape.md](10_tiny_gpu_test_shape.md) | 使用 Triton 支持的 tiny model head dimension |
| [11_sync_latest_fork.md](11_sync_latest_fork.md) | 备份原改动并同步最新 fork |
| [12_engine_api_compatibility.md](12_engine_api_compatibility.md) | 保留上游 API、适配 max_new_len |
| [13_fused_prefill_ghost_guard.md](13_fused_prefill_ghost_guard.md) | 区分 fused prefill 和 decode ghost 检查 |
| [14_benchmark_kernel_baseline.md](14_benchmark_kernel_baseline.md) | benchmark 使用新版 Triton 基线并记录选择 |
| [15_tests_ci_migration.md](15_tests_ci_migration.md) | 迁移 tests/ci、合并上游已有修复 |
| [16_dependency_profiles.md](16_dependency_profiles.md) | 隔离 H200 与 WSL 依赖配置 |
| [17_sync_validation.md](17_sync_validation.md) | 最新版本实际验证与运行命令 |
| [18_tiny_logits_contract.md](18_tiny_logits_contract.md) | 修正最后 token logits 测试并检查完整 KV |

运行说明与 PowerShell/WSL 命令见 [FLASHINFER.md](../FLASHINFER.md)。
