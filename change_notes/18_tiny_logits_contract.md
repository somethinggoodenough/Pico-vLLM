# 18 tiny-model 测试适配最后 token logits 输出

## 问题

上游模型在所有 transformer 层后切到 `x[:, -1:, :]`，只计算最后 token 的
norm/lm_head，以减少 prefill 无用计算。CPU/GPU tiny 测试仍断言
`(1, seq_len, vocab_size)`，因此本轮首次回归有 2 个失败。

## 改动

- 两个测试的 prefill 输出断言改为新契约 `(1, 1, vocab_size)`，不修改模型行为。
- KV cache 先填 NaN；prefill 后检查整个 prompt 的 KV 都已写入且有限，下一 slot
  尚未写入；decode 后检查新增 slot 也已写入。
- 保留 logits/decode shape 和有限值检查，避免只改输出断言却漏检完整 KV 构建。

## 文件与验证

`tests/ci/02_single_card/test_tiny_model_smoke.py` 和 `test_torch_cpu_tiny_model.py`。
初次回归为 61 passed、1 xfailed、2 failed；修复后的结果见 17。
