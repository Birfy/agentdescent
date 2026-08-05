# TextGrad 与 OpenEvolve 并行/异步实验报告

日期：2026-08-05
模型：`glm-5.2`
接口：华为云 ModelArts MaaS OpenAI-compatible API
结论性质：真实算法小规模调度实验，不是合成 workload，也不是论文级 benchmark

## 1. 摘要

这次按新要求，直接用已经实现的 TextGrad 和 OpenEvolve 测量串行、同步并行和异步流水，
不再只依赖 `AppendRules + sleep` 的受控例子。六次 live observation 全部真实请求
`glm-5.2`，总计 48 次模型调用、32,738 tokens、500.24 个模型累计秒数，无 API 失败。

### 1.1 Live 运行

| 算法 | 模式 | 达到共同质量的时间/墙钟 | 最终质量 | Calls | Tokens |
|---|---|---:|---:|---:|---:|
| TextGrad | serial | 122.85 s | 1.000 | 13 | 5,826 |
| TextGrad | sync parallel | **53.81 s** | 1.000 | 13 | 4,465 |
| TextGrad | async pipeline | 60.34 s | 1.000 | 13 | 4,652 |
| OpenEvolve | serial | 48.41 s 完整墙钟，未达目标 | 0.699122 | 3 | 5,835 |
| OpenEvolve | sync parallel | **21.06 s** 完整墙钟，未达目标 | 0.699122 | 3 | 6,072 |
| OpenEvolve | async pipeline | **16.67 s 达标**，21.13 s 完整墙钟 | **0.773900** | 3 | 5,888 |

TextGrad 的同步并行相对串行快 `2.28x`，时间减少 `56.20%`；异步流水相对串行快
`2.04x`，但这次 live 运行比同步并行慢 `12.12%`。这不是应删除的坏数据，而是说明
TextGrad 的 batch update 必须等待全部 prompt gradients，异步能够消除的 barrier 很少，
网络延迟波动足以盖过收益。

OpenEvolve 的同步并行完整墙钟相对串行快 `2.30x`。只有异步 live run 生成了超过目标
`0.704122` 的候选，在 `16.67 s` 首次达标；serial 和 sync 的 GLM 输出不同，候选均未
超过 baseline，因此不能直接用三次 live TTQ 计算公平加速比。

### 1.2 同 trace 配对结果

为消除 temperature 0 下仍存在的模型输出差异，代码从 live JSON 选择一条真实算法
trace，固定其中的回答、gradients、候选程序及每次调用延迟，再计算三种调度方式的关键
路径。配对重放不增加 API 调用，也没有用合成答案替换模型输出。

| 算法与共同目标 | Serial TTQ | Sync TTQ | Async TTQ | Sync/Serial | Async/Sync |
|---|---:|---:|---:|---:|---:|
| TextGrad，validation `1.000` | 100.30 s | 53.81 s | **53.14 s** | 1.86x | **1.013x** |
| OpenEvolve，score `0.704122` | 33.91 s | 21.13 s | **16.66 s** | 1.60x | **1.268x** |

OpenEvolve 配对 trace 达到目标所需候选数为 serial 2、sync parallel 3、async 1。同步模式
即使第一个完成的候选已经有效，也必须等待 generation 中全部 3 个候选；异步模式在高分
候选完成生成和沙箱评估后立刻提交 archive，因此比同步少等 4.47 秒，TTQ 减少 21.14%。

## 2. 产物

代码：

- [`experiment/algorithm_parallel_async_benchmark.py`](../experiment/algorithm_parallel_async_benchmark.py)
- [`experiment/textgrad_prompt_optimization.py`](../experiment/textgrad_prompt_optimization.py)
- [`experiment/openevolve_program_search.py`](../experiment/openevolve_program_search.py)
- [`experiment/test_small_experiments.py`](../experiment/test_small_experiments.py)

结果：

- [`algorithm-parallel-async-result.json`](algorithm-parallel-async-result.json)

辅助的调度器隔离实验仍保留在
[`parallel-async-time-to-quality.md`](parallel-async-time-to-quality.md)，但本报告才是“基于
两个已实现算法”的主验收结果。

## 3. 三种执行语义

### 3.1 Serial

- TextGrad：逐条完成 forward、response gradient、prompt gradient，再处理下一条样本；
  baseline 和 candidate validation 也逐条执行。
- OpenEvolve：逐个完成 mutation、AST 检查、Bubblewrap evaluator 和 archive commit。

### 3.2 Sync parallel

- TextGrad：同一阶段内最多 3 路并发，但 forward 全部结束后才进入 response gradient，
  response gradient 全部结束后才进入 prompt gradient。
- OpenEvolve：3 个 mutation 并发生成，全部生成后才并发 evaluator，全部评估后才统一提交
  archive，保留 generation barrier。

### 3.3 Async pipeline

- TextGrad：每个样本独立执行 `forward -> response gradient -> prompt gradient`，一个样本的
  forward 完成后无需等待其他 forward；但 TGD batch update 仍必须等待全部 gradients。
- OpenEvolve：每个候选独立执行 `mutation -> sandbox evaluation`，通过 `as_completed()`
  在候选完成时立即提交 archive，无需等待同代其他候选。

三种模式使用相同数据、seed、候选预算、验证器和最大并发 3。OpenEvolve 一代内固定同一
parent/best/inspiration，避免 archive 提交时机进一步改变尚未开始的 prompt。

## 4. TextGrad 设置与结果

### 4.1 设置

| 参数 | 值 |
|---|---:|
| 任务 | BBH `word_sorting` |
| train IDs | 0, 1 |
| validation IDs | 50, 51, 52 |
| TGD steps | 1 |
| batch size | 2 |
| validation size | 3 |
| 并发上限 | 3 |
| 共同质量目标 | validation accuracy 1.000 |
| 每种模式高层模型调用 | 13 |

每种模式执行 3 次 baseline forward、每个训练样本 1 次 forward 加 2 次 textual backward、
1 次 batch update，以及 3 次 candidate validation，共 13 次调用。所有模式候选最终都在
3 个 validation 样本上得到 `3/3`，所以可以按共同质量 `1.000` 比较 live 时间。

### 4.2 Live 结果

| 模式 | Baseline | Final | 到 `1.000` | 相对 serial | 模型累计秒数 |
|---|---:|---:|---:|---:|---:|
| serial | 0/3 | 3/3 | 122.85 s | 1.00x | 122.85 s |
| sync parallel | 0/3 | 3/3 | **53.81 s** | **2.28x** | 100.88 s |
| async pipeline | 1/3 | 3/3 | 60.34 s | 2.04x | 118.79 s |

并行模式的模型累计秒数高于墙钟，是多个 API 请求重叠执行的直接证据。async baseline 恰好
答对 1 条，而另两种 baseline 为 0，也说明 provider 在 temperature 0 下并不完全确定；
因此报告使用三者最终都达到的 `1.000`，而不是原始配置中的 `1/3` 阈值。

### 4.3 配对 trace

配对分析使用 sync parallel 运行产生的同一份 13-call trace，共 4,465 tokens。

| 路径组成 | Serial | Sync parallel | Async pipeline |
|---|---:|---:|---:|
| baseline validation | 28.03 s | 10.23 s | 10.23 s |
| 两条 gradient trajectory | 52.03 s | 29.79 s | **29.12 s** |
| batch update | 7.63 s | 7.63 s | 7.63 s |
| candidate validation | 12.61 s | 6.16 s | 6.16 s |
| **合计 TTQ** | **100.30 s** | **53.81 s** | **53.14 s** |

异步只在两条 gradient trajectory 上节约 0.67 秒；batch update 和两次 validation 都仍是
barrier。因此 TextGrad 的主要收益来自并行，异步流水只是 `1.3%` 的小优化。

## 5. OpenEvolve 设置与结果

### 5.1 设置

| 参数 | 值 |
|---|---:|
| 任务 | 官方 function minimization 示例 |
| generations | 1 |
| candidates | 3 |
| trials/candidate | 5 个固定 seeds |
| objective calls/trial | 100 |
| baseline score | 0.699122 |
| 目标 score | 0.704122，即 baseline + 0.005 |
| 并发上限 | 3 |
| 每种模式模型调用 | 3 |

候选仍通过 AST gate、Bubblewrap 网络隔离、资源限制和官方四项指标 evaluator。为了支持
并发 sandbox，在固定 `RLIMIT_NPROC=512` 会被宿主已有线程耗尽的问题上，限制改为“当前
用户 task 数 + 64，且最低 512”；它仍限制候选进程数，但不再随机拒绝 bwrap 自己启动。

### 5.2 Live 结果

| 模式 | 候选分数 | Best | 首次达标 | 完整墙钟 |
|---|---|---:|---:|---:|
| serial | 0.6091 / 0.4328 / 0.4254 | 0.699122 | 未达标 | 48.41 s |
| sync parallel | 0.4254 / 0.4171 / 0.3927 | 0.699122 | 未达标 | **21.06 s** |
| async pipeline | 0.4238 / **0.7739** / 0.7085 | **0.773900** | **16.67 s** | 21.13 s |

sync 与 async 的完整预算墙钟几乎相同，因为两者最终都执行 3 个候选；差别在于 async 在
第二个候选首先完成时就已经产出可用 best，而 sync 的结果要到整代 barrier 后才可提交。

### 5.3 配对 trace

配对分析固定 async live run 的三份真实候选和调用延迟，共 5,888 tokens：

| 固定候选 | Score | Mutation | Evaluator |
|---:|---:|---:|---:|
| 1 | 0.423822 | 17.22 s | 0.023 s |
| 2 | **0.773900** | 16.62 s | 0.023 s |
| 3 | 0.708507 | 21.09 s | 0.023 s |

| 模式 | TTQ | 候选成本到目标 | 完整预算墙钟 | 相对 serial | 相对 sync |
|---|---:|---:|---:|---:|---:|
| serial | 33.91 s | 2 | 55.02 s | 1.00x | - |
| sync parallel | 21.13 s | 3 | 21.13 s | 1.60x | 1.00x |
| async pipeline | **16.66 s** | **1** | 21.13 s | **2.03x** | **1.268x** |

这里 async 的 cost-to-quality 为 1，是指按完成顺序第一个返回的候选已经超过目标；serial
按 slot 顺序要跑到第 2 个，sync 则必须完成整批 3 个才跨过可见的 generation barrier。

## 6. 资源指标

| 算法 | Live runs | Calls | Tokens | 模型累计秒数 | 各模式墙钟之和 |
|---|---:|---:|---:|---:|---:|
| TextGrad | 3 | 39 | 14,943 | 342.52 s | 237.00 s |
| OpenEvolve | 3 | 9 | 17,795 | 157.72 s | 90.60 s |
| **总计** | **6** | **48** | **32,738** | **500.24 s** | **327.60 s** |

- API failures：0。
- Thinking：disabled。
- Temperature：0；输出仍非完全确定。
- 没有估算货币成本，因为当前套餐单价未知；calls 和 tokens 是可复核成本基数。
- Key 只从 `~/.bashrc` 的环境变量读取，未写入代码或 JSON。

## 7. 复现

```bash
bash -ic 'python -m experiment.algorithm_parallel_async_benchmark \
  --provider openai --model glm-5.2 --thinking disabled \
  --max-tokens 1024 --temperature 0 --api-timeout 180 \
  --concurrency 3 --textgrad-batch-size 2 --textgrad-val-size 3 \
  --textgrad-target-accuracy 0.3333333333333333 --textgrad-subset first \
  --openevolve-candidates 3 --openevolve-trials 5 \
  --openevolve-objective-budget 100 --openevolve-min-score-gain 0.005 \
  --openevolve-candidate-timeout 15 --openevolve-archive-size 6 \
  --output report/algorithm-parallel-async-result.json --yes'
```

Dry-run 不需要 Key：

```bash
python -m experiment.algorithm_parallel_async_benchmark --dry-run
```

离线测试：

```bash
pytest -q experiment/test_small_experiments.py
```

结果：`12 passed`。另外单独验证了 3 路 Bubblewrap evaluator 并发，3/3 有效。

## 8. 结论边界

1. 两个真实算法都证明了同步并行能明显缩短时间：TextGrad live `2.28x`，OpenEvolve live
   完整墙钟 `2.30x`。
2. 异步收益取决于算法依赖图。OpenEvolve 的独立候选可以完成即提交，配对 TTQ 比同步快
   `1.268x`；TextGrad 的 batch update 是硬 barrier，配对收益只有 `1.013x`。
3. Live 模式只有 1 次重复、数据和候选预算很小；倍数不能外推为生产 SLA。
4. Trace replay 使用真实输出和真实延迟，但是假设同一调用延迟可以跨调度方式重排，并忽略
   线程启动等小开销；它用于隔离调度因果，不替代 live observation。
5. 最可靠的表述不是“异步永远更快”，而是“并行对两个算法都有显著收益；当候选彼此独立
   且允许完成即提交时，异步能进一步降低 time-to-quality”。
