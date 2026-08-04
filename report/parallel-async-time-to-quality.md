# AgentDescent 并行与异步 Time-to-Quality 实验报告

日期：2026-08-04
运行目录：`/home/qiukaixi/agentdescent`
结论性质：受控调度基准，用于验证并行扩展和异步消除 barrier 的效果

## 1. 摘要

本实验不比较模型答案优劣，而是在相同任务、算法、质量阈值和最大 rollout 预算下，直接
测量 AgentDescent 的串行、同步并行和异步执行路径需要多久达到目标质量。

| 核心比较 | 基线 TTQ 中位数 | 优化后 TTQ 中位数 | 时间减少 | 加速比 | 重复结果 |
|---|---:|---:|---:|---:|---:|
| 串行 -> 8 路同步并行，质量 `1.00` | 0.6744 s | **0.1115 s** | **83.46%** | **6.05x** | 5 次 |
| 8 路同步并行 -> 8 worker 异步，质量 `0.75` | 0.4300 s | **0.0763 s** | **82.26%** | **5.64x** | **5/5 异步更快** |

结论：并行让相同的 8 个 rollout 更快完成；存在慢任务时，异步不等待整轮 barrier，在 6
个快任务产生可迁移改进后就达到 `0.75` 的目标质量。后者相对同步并行的逐次配对加速为
`5.23x` 到 `5.87x`。

这里的 TTQ 是 `EvolutionResult.time_to_quality(目标值)`：历史中首次达到质量阈值的时间，
不是函数最终返回时间。异步达到阈值后仍需清理不可取消的在途慢 rollout，所以异步函数
返回墙钟时间中位数为 0.4211 s，同步为 0.4401 s。若应用只在函数返回后读取结果，这个
短实验中的尾部清理会遮住大部分 TTQ 优势；长任务应通过 `on_round` 消费进度，并让后端
支持取消，才能把内部提前产出的收益完整转成端到端响应收益。

## 2. 产物

代码：

- [`experiment/parallel_async_time_to_quality.py`](../experiment/parallel_async_time_to_quality.py)
- [`experiment/test_small_experiments.py`](../experiment/test_small_experiments.py)

机器可读结果：

- [`parallel-async-time-to-quality-result.json`](parallel-async-time-to-quality-result.json)，正式主实验
- [`parallel-async-incremental-stress-result.json`](parallel-async-incremental-stress-result.json)，增量式压力测试

## 3. 实验设计

### 3.1 公共设置

| 参数 | 值 |
|---|---:|
| train tasks | 8 |
| held-out tasks | 8 |
| repeats | 5 |
| 最大 rollout 预算 | 24，即 `3 rounds x 8 workers` |
| strategy | `AppendRules` |
| self verify | 关闭 |
| held-out evaluator | 确定性、无休眠 |
| async ratio | 3 |
| Python | 3.13.5 |
| 平台 | WSL2，Linux 5.15.167.4，x86_64 |
| 可见 CPU | 28 |

每个 held-out task 对应一个类别 `c0` 到 `c7`。artifact 中出现 `ENABLE_cN` 时，该类别得
1 分，否则得 0 分；质量是 8 个 held-out 类别的平均分。训练 rollout 中的 `sleep` 模拟
会释放 GIL 的模型 API 或工具 I/O 延迟，因此正式代码走的仍是项目真实的 `evolve()` 和
`async_evolve()`，只有外部服务被换成了无网络抖动、无费用的确定性 actor。

这样设计是为了隔离调度器效果。它不能证明某个 LLM 更聪明，也不能替代真实 provider
的限流测试；GLM-5.2 上的算法质量实验见
[`textgrad-openevolve-small-experiments.md`](textgrad-openevolve-small-experiments.md)。

### 3.2 同步并行扩展

- 8 个训练任务的延迟全部为 80 ms。
- 每个任务只产生自己类别的规则，达到 `1.00` 必须学到全部 8 条规则。
- 串行基线仍使用 `n_workers=8`，但设置 `max_concurrency=1`，因此与并行模式的逻辑工作
  单元完全相同。
- 依次比较 `max_concurrency=1/2/4/8`；每次重复旋转模式执行顺序，减少冷热启动和运行
  顺序偏差。

### 3.3 异步消除 barrier

- 6 个快任务各耗时 50 ms，2 个慢任务各耗时 400 ms。
- 任意一个快任务都能发现同一条可迁移规则，该规则覆盖 `c0` 到 `c5`，因此产生后质量
  正好为 `6/8 = 0.75`；慢任务分别覆盖剩余两个稀有类别。
- 同步并行必须等两个慢任务结束才能合并整轮结果。
- `async_evolve()` 的 merger 可以先合并快任务结果，并在达到同一个 `0.75` 阈值时停止。
- 三种模式使用同一组任务、proposal、reward、最大预算和目标阈值。

## 4. 正式结果

### 4.1 并行扩展

所有模式都达到 `1.00`，达到质量阈值时都消耗 8 个 rollout。

| 模式 | 并发 | TTQ 中位数 | TTQ p95 | 相对串行加速 | cost-to-quality |
|---|---:|---:|---:|---:|---:|
| serial | 1 | 0.6744 s | 0.6798 s | 1.00x | 8 |
| sync parallel | 2 | 0.3519 s | 0.3590 s | 1.92x | 8 |
| sync parallel | 4 | 0.1868 s | 0.1933 s | 3.61x | 8 |
| sync parallel | 8 | **0.1115 s** | **0.1133 s** | **6.05x** | 8 |

8 路并行效率为 `6.05 / 8 = 75.6%`。未达到理想 8x 的差额来自线程调度、merger、ledger
提交和 held-out 评估等固定开销；并发增加时这些开销占比更高。

### 4.2 异步 time-to-quality

比较阈值统一为 `0.75`。同步模式在 barrier 后同时拿到全部规则，因此最终 reward 为
`1.00`；异步模式达到目标即停，最终 reward 为 `0.75`。这不是质量退化，而是停止位置
不同；比较的是两者第一次跨过相同阈值的时间。

| 模式 | TTQ 中位数 | TTQ p95 | 相对串行加速 | 相对同步加速 | cost-to-quality | 返回墙钟中位数 |
|---|---:|---:|---:|---:|---:|---:|
| serial | 1.1297 s | 1.1348 s | 1.00x | - | 8 | 1.1416 s |
| sync parallel 8 | 0.4300 s | 0.4407 s | 2.63x | 1.00x | 8 | 0.4401 s |
| async no barrier | **0.0763 s** | **0.0832 s** | **14.81x** | **5.64x** | **6** | 0.4211 s |

逐次配对结果：

| 指标 | 结果 |
|---|---:|
| 异步更快次数 | 5/5 |
| 配对加速中位数 | 5.64x |
| 最小配对加速 | 5.23x |
| 最大配对加速 | 5.87x |
| retired workers | 0 |
| stale discarded | 0 |
| 正式 benchmark 总墙钟 | 18.03 s |

## 5. 增量式压力测试

主实验刻意隔离“可迁移改进已经由快 worker 产出，但同步 barrier 仍在等待慢 worker”的
情形。为了检查结论是否依赖这一条件，压力测试把 proposal 改回每个任务只解决自己的
`1/8` 类别，其他参数保持一致。

| 模式 | TTQ 中位数 | TTQ p95 | cost-to-quality 中位数 | cost 范围 |
|---|---:|---:|---:|---:|
| sync parallel 8 | 0.4317 s | 0.4383 s | 8 | 8-8 |
| async no barrier | **0.2034 s** | 0.4390 s | 15 | 6-17 |

异步中位数仍比同步快 `2.12x`，但只有 `4/5` 次更快，最差一次为 `0.4979 s`，慢于同步。
原因不是 stale discard，本次该指标为 0；原因是 merger 第一次 sweep 可能只收到部分
快 worker 的卡片。其余 worker 在版本刷新前会再次提交已经覆盖过的类别，必须等后续
sweep 才凑齐 6 个不同类别，于是 TTQ 和 rollout 成本出现长尾。

这个压力结果限定了主结论：当一个快结果本身已经达到目标，异步消除 barrier 的收益稳定；
当质量必须由多个细粒度结果拼齐时，异步仍有中位收益，但 intake 批次边界、重复候选去重
和 worker 刷新策略会影响尾延迟。后续优化应优先测 pending-slot 原子预留、按 proposal
身份去重，以及在 merger sweep 前加入很短的收集窗口。

## 6. 复现

正式主实验使用默认参数即可：

```bash
python -m experiment.parallel_async_time_to_quality
```

增量式压力测试：

```bash
python -m experiment.parallel_async_time_to_quality \
  --heavy-tail-pattern incremental \
  --output report/parallel-async-incremental-stress-result.json
```

离线测试：

```bash
pytest -q experiment/test_small_experiments.py
```

当前结果为 `9 passed`。基准不读取 `~/.bashrc`，不需要 API Key，也不会产生模型调用费用。
JSON 保存每次 observation 的 TTQ、cost-to-quality、final reward、返回墙钟、rollout/eval
累计时间、吞吐量、staleness、worker retirement、merge outcomes 和完整质量时间线。

## 7. 结论边界

1. 已证明：在该受控 I/O workload 中，同步并行以相同 8-rollout 成本把 TTQ 缩短 83.46%。
2. 已证明：当慢任务不是达到目标质量所必需时，无 barrier 异步把 TTQ 再缩短 82.26%，
   5 次重复全部胜过同步并行。
3. 没有证明：真实 LLM provider 下也一定得到相同倍数。网络延迟、限流、连接池大小和模型
   输出分布都会改变结果。
4. 当前限制：在途 `run()` 不可取消，函数返回时间不会像内部 TTQ 一样明显下降；增量式
   多卡片聚合也存在尾延迟。两点都已保留在结果中，没有从报告里删除不利运行。
