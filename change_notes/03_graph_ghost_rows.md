# 03 修复 ghost Q 未初始化并减少 dummy block 占用

## 原评审更正

上一轮认为 graph warmup 会因 slot=0 覆盖 physical page 0。进一步检查
`fused_rope_kvcache_store.py` 后发现，`context_lens == 0` 会提前返回，不写 KV。
因此该覆盖场景在当前实现中不成立；这里明确撤回那项 bug 判断。

## 实际问题

提前返回也没有写 `torch.empty_like` 分配的 Q。FlashInfer 使用 dummy page 计算
ghost 行，会读取未初始化 Q，其中可能有 NaN/Inf。另一个问题是 eager-only 或
prefill-only Engine 也预留 dummy block，导致可用 KV 容量无谓减少。

## 改动

Triton fused RoPE kernel 在 ghost 分支写零 Q 后返回，仍不写 KV。
graph 初始 slot mapping 使用与 replay ghost 一致的 -1 哨兵值。
仅启用 graph 且角色为 `d/pd` 时分配 dummy block；使用实际分配的 page id。
如果分配到 CPU block 无法用于 graph，则先归还该 block 再报错。

## 涉及文件

`pico_vllm/engine.py`、`ops/triton/fused_rope_kvcache_store.py`、
`tests/ci/01_ops/test_flashinfer_attention.py`、`tests/ci/02_single_card/test_attention_benchmark_lifecycle.py`。

## 验证与边界

GPU 回归把 Q 输出分配预先填成 NaN，检查 ghost Q 变零且现有 KV page 保持原值。
CPU 回归覆盖 graph/eager、p/d/pd 和 physical page 0 已被占用的分配情况。
dummy block 在 graph Engine 生命周期内保留，本次没有增加 Engine close API。
执行结果见 08。
