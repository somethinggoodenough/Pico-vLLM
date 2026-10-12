# 07 修复共享 fixture 的统计计数误报

## 问题

`test_flashinfer_engine.py` 的模型 fixture 是 module scope；graph 和 eager 参数化测试
复用同一个 FlashInferOps。PlanStats 原来持续累加，但测试断言 prefill_plans 等于本轮
prompt 数，因此第二个参数组合可能失败，与推理是否正确无关。

## 改动

每轮创建 Engine 前重置 FlashInfer stats，让计数断言只覆盖本轮。
另外更正测试注释：放大最终 norm 并不能保证任意输入 argmax 不翻转；这里的 greedy
token equality 仅是固定输入的 smoke test。

## 涉及文件

`pico_vllm/tests/ci/02_single_card/test_flashinfer_engine.py`。

## 验证

在同一 pytest 会话中运行 graph、eager、prefix-cache 测试，执行结果见 08。
