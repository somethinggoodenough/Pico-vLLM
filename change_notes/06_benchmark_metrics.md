# 06 澄清 plan 计时和带宽估算

## 问题

两个 benchmark 都叫 `plan_us`，但 microbenchmark 包围整个 backend plan，
E2E 只累计 wrapper.plan；前者包含 metadata 构造，后者不包含。
两者使用 host 时钟，可能在异步设备 copy 完成前返回，不能作为完整 GPU plan latency。
原说明还把 KV 字节占比当作 attention 影响上限，字节比例不能直接等同时间比例。

## 改动

microbenchmark 字段改为 `plan_host_us`；E2E 改为 `wrapper_plan_host_us`，
内部累计值改为 `wrapper_plan_host_seconds`。文档说明两个指标不同且不等待异步 copy。
保持计时不额外同步，避免为测量 host 开销而改变 Engine 稳态执行行为。

明确 `kv_share_pct` 是估算流量占比；`bw_util_pct` 是模型流量和 D2D copy 带宽推算值，
不是 Nsight 硬件计数器测得的 DRAM 利用率，也不是实际 attention 时间占比。

## 涉及文件与兼容性

`pico_vllm/ops/flashinfer/backend.py`、两个 benchmark 脚本、`FLASHINFER.md`。
读取补丁早期 CSV 的下游脚本需要适配字段改名。kernel 时间不包含 plan；E2E wall time 包含 plan。

## 验证

代码搜索确认旧统计字段不再被执行代码引用；测试与 benchmark smoke 结果见 08。
