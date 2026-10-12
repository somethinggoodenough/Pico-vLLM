# 10 修复原有 tiny GPU 测试的形状配置

> 同步更新：`1691ebb` 已包含 hidden_size=64；现在直接保留上游设置，
> 不再将它算作本地差异。新发现的输出形状断言问题见 [18_tiny_logits_contract.md](18_tiny_logits_contract.md)。

## 问题

原 GPU smoke test 设置 hidden_size=32、4 个 attention heads，得到 head_dim=8。
在当前 Triton 3.2.0 / RTX 4060 上，prefill 的 tl.dot 编译要求 M/N/K 至少 16，
测试在进入 decode 前就编译失败。这是原有测试配置与 kernel 支持形状不匹配。

## 改动

只将该 GPU 测试的 hidden_size 改为 64，保持 4 heads，得到 head_dim=16。
保持原有 prefill/decode 执行流程、输出形状和有限值断言。
CPU tiny test 不使用 Triton，保留其原始 head_dim=8。

## 涉及文件

`pico_vllm/tests/ci/02_single_card/test_tiny_model_smoke.py`。

## 验证与边界

重新运行该测试及相关算子/Engine 回归，结果见 08。
这不增加生产 Triton kernel 对 head_dim=8 的支持；测试改为使用当前已支持的形状。
