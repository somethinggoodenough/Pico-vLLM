# 04 隔离 benchmark 轮次的请求清理

## 问题

Engine 返回 completion 的那一步仅设置完成标记；下一步 schedule 才移动请求并释放 KV。
原 `run_workload` 收齐 completion 就返回，下一轮的 prefill 时间因此包含上一轮清理。
新一轮分配发生在旧请求释放之前，KV 容量紧张时还可能错误地触发 OOM。

## 改动

完成测量后额外执行一次不计时 `engine.step()`，同步设备并清空已关闭的 finished 列表。
warmup 和正式测量都走相同清理逻辑。
释放配置级 KV pool 时也删除持有 BlockManager 的 `cache_kwargs`，再调用 empty_cache，
避免下一配置通过 mem_get_info 看到上一配置本应已释放的显存。
限制 `gen_len >= 2`、`repeats >= 1`，避免只有 prefill token 时计算空 ITL 分布。

## 涉及文件

`pico_vllm/benchmarks/bench_decode_e2e.py`、
`pico_vllm/tests/ci/02_single_card/test_attention_benchmark_lifecycle.py`。

## 验证与口径

CPU 回归使用真实 Engine/Scheduler/BlockManager，两个 17-token prompt 和仅 4 个 KV blocks，
连续运行三轮；每轮须释放全部 blocks，得到 3 个 decode step，并清空运行及 finished 队列。
只替换测试中的 CUDA synchronize，以便在 CPU 上验证生命周期。
`prefill_ms` 仍包含本轮调度、分配、计算和采样，不能称为纯 attention kernel 时间。
执行结果见 08。
