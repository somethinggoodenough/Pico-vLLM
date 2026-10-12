# 09 修复原有 prefill 测试的 pytest 接口

> 同步更新：`1691ebb` 已包含 helper 改名为 `run_case` 及 pytest assert 修复，
> 现在沿用上游版本。仅保留直接运行脚本时的成功统计和失败退出码修复；路径已更新。

## 问题

扩展回归发现原有 `test_case(name, B, ...)` 是辅助函数，却因为名字以 test_ 开头
被 pytest 收集，触发 `fixture 'name' not found`。12 个测试入口返回 bool，
pytest 不把 False 返回值当失败，导致数值检查可能误报通过，并产生 12 条警告。

## 改动

辅助函数改名为 `run_attention_case`，测试入口使用 assert 检查其结果并自然返回 None。
保留直接执行脚本的入口；它按是否抛出异常记录通过状态，全部结束后用退出码报告结果。

## 涉及文件

`pico_vllm/tests/ci/01_ops/test_triton_prefill_attention.py`。

## 验证

在原有 12 组 prefill 用例上重跑 pytest，既检查收集成功，也让真实数值误差触发断言。
执行结果见 08。本项仅修测试接口，没有调整 attention kernel 或误差阈值。
