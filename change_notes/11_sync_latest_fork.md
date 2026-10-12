# 11 同步 fork 并保留旧改动

## 问题

GitHub Sync fork 已更新远程，但本地停留在 `ddafca2`，且有未提交的 FlashInfer
代码、说明文件、虚拟环境和用户生成的结果。直接覆盖会丢失本地修改。

## 改动

- 将原代码修改及新增说明保存为 stash，未包括 `.venv-wsl/` 和 `results/`。
- 备份 SHA：`49c35f534a514ce430a47b5e07175dd26304fc6b`，名称
  `backup-before-sync-flashinfer-2026-10-12`；使用 apply 恢复，未 pop/drop。
- 本地 master 从 `ddafca2` 快进到 `1691ebb522316e285dcc98bd29a9fe6d54f9ca24`，共 14 个提交。
- Windows Git 的 SSH 认证不可用，使用同一 fork 的 HTTPS URL 获取；未修改 origin URL。
- 刷新 `origin/master` 跟踪引用；逐项解决冲突后保留为未暂存修改，不创建新提交、不 push。
- 上游迁移/删除的文件遵循上游版本，仍可从旧提交或上述备份恢复；未清理虚拟环境和结果。

## 验证

`git log -1`、`git status`、`git ls-files -u` 和最终回归见 17。
不要在当前改动上直接再次 apply 该备份，以免重复叠加；备份用于需要时单独比对或恢复。
