# 13 兼容统一 RoPE/KV kernel 的 prefill 与 decode

## 问题

上游将 prefill 也接入 fused RoPE/KV kernel，并增加 `CHECK_CTX_LEN`。
prefill 中 token 数与 request 数不同；直接套回旧 ghost 判断会按 token 索引
读取 request-length 数组，可能越界或错误跳过 token。

## 改动

保留上游 `CHECK_CTX_LEN` 和 `check_ctx_len=False` 的 prefill 路径。
只在 decode 检查开启且 context_len=0 时写零 Q 并返回，仍不写 ghost KV。
FlashInfer 继承新版 Triton fused prefill，未恢复旧的分离 RoPE/store 路径。

## 文件与验证

`pico_vllm/ops/triton/fused_rope_kvcache_store.py`。
`tests/ci/01_ops/test_flashinfer_attention.py` 新增 31-token prefill、单个 context-length
entry 的回归，验证全部 Q/K/V；保留 poison-Q ghost 测试与 graph Engine 测试。
结果见 17。
