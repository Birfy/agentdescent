# TextGrad + OpenEvolve 小实验报告

日期：2026-08-04
运行目录：`/home/qiukaixi/agentdescent`
模型：`glm-5.2`
接口：华为云 ModelArts MaaS OpenAI-compatible API
结论性质：机制演示，不是论文全量复现或标准 benchmark 排名

## 1. 摘要

| 实验 | 基线 | 最终/最佳 | 主要提升 | API 调用 | Token | 墙钟时间 |
|---|---:|---:|---:|---:|---:|---:|
| TextGrad / BBH word sorting | 测试 `0/20 = 0%` | 测试 `16/20 = 80%` | `+80` 个百分点 | 140 | 132,499 | 776.85 s |
| OpenEvolve / 函数最小化 | 综合分 `0.933247` | 综合分 `1.477353` | `+58.30%` | 6 | 16,071 | 162.70 s |

两个实验都展示了可测量提升：

1. TextGrad 连续三次接受验证集更好的提示词，第 4 次更新使验证准确率从
   `91.7%` 回落到 `58.3%`，因此被验证回退机制拒绝；最终测试集从 `0%` 提升到
   `80%`。
2. OpenEvolve 在相同的 200 次目标函数调用预算下，把 10 个固定随机种子的平均最优点
   距离从 `1.106502` 降到 `0.037266`，降幅 `96.63%`；最终程序是分层采样加多起点
   Nelder-Mead，而不是增加搜索预算。

最重要的限制也应先说明：TextGrad 使用的官方通用起始提示词要求输出“numerical
value”，它与 word sorting 的字符串列表答案明显不匹配。因此 `0% -> 80%` 很适合展示
文本梯度能否修复提示词，却不能被解释为“把一个已经合理的生产提示词提高了 80 个点”。

## 2. 产物

代码：

- [`experiment/textgrad_prompt_optimization.py`](../experiment/textgrad_prompt_optimization.py)
- [`experiment/openevolve_program_search.py`](../experiment/openevolve_program_search.py)
- [`experiment/_openevolve_runner.py`](../experiment/_openevolve_runner.py)
- [`experiment/_common.py`](../experiment/_common.py)
- [`experiment/test_small_experiments.py`](../experiment/test_small_experiments.py)
- [`experiment/openevolve_best_program.py`](../experiment/openevolve_best_program.py)

机器可读结果：

- [`textgrad-small-result.json`](textgrad-small-result.json)
- [`openevolve-small-result.json`](openevolve-small-result.json)
- [`openevolve-thinking-enabled-partial.json`](openevolve-thinking-enabled-partial.json)，仅用于延迟诊断，不计入正式结果

并行与异步能力另有独立的 time-to-quality 报告：

- [`textgrad-openevolve-parallel-async.md`](textgrad-openevolve-parallel-async.md)，真实算法主实验
- [`parallel-async-time-to-quality.md`](parallel-async-time-to-quality.md)
- [`parallel-async-time-to-quality-result.json`](parallel-async-time-to-quality-result.json)
- [`parallel-async-incremental-stress-result.json`](parallel-async-incremental-stress-result.json)

## 3. 公共运行设置

两个实验都从 `~/.bashrc` 读取 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL`，Key 没有写入代码、
结果或报告。实际请求模型固定为 `glm-5.2`。

GLM-5.2 默认会开启 Thinking；官方文档允许通过
`{"thinking":{"type":"disabled"}}` 关闭。第一次 OpenEvolve 深思考调用耗时
`113.41 s`，产生 5,243 tokens；第二轮一次 180 秒超时后仍未完成。为了让短循环实验可行，
正式实验保持 `glm-5.2` 不变，仅关闭 Thinking。关闭后的健康检查为 `3.89 s / 13
tokens`。[华为云 GLM-5.2 调用示例](https://support.huaweicloud.com/bestpractice-maas/bestpractice_maas_0013_01.html)，
[智谱 Thinking 模式说明](https://docs.bigmodel.cn/cn/guide/capabilities/thinking-mode)

公共设置：

| 参数 | 值 |
|---|---|
| provider | `openai`，指 OpenAI-compatible adapter |
| model | `glm-5.2` |
| thinking | `disabled` |
| seed | `0` |
| API timeout | 180 s |
| API failures in completed runs | 0 |

`seed=0` 固定了数据选择、训练 batch、OpenEvolve 父代选择和程序评估种子；远端模型生成
本身未必完全确定，因此重复执行可能产生不同候选。

## 4. TextGrad 实验

### 4.1 任务和数据

任务使用 BIG-Bench Hard 的 `word_sorting.json`。TextGrad 上游 loader 按位置划分为前
50 条训练、后续 100 条验证、再后续 100 条测试；本实验保留这个身份边界，然后在每个
split 内按输入单词数选最长样本：训练 12 条、验证 12 条、测试 20 条。该选择不依赖
模型答对与否，但它是一个“小型长列表子集”，不可与完整 BBH 分数直接比较。

来源：

- [TextGrad 仓库](https://github.com/zou-group/textgrad)
- [固定 commit 的官方 prompt optimization 入口](https://github.com/zou-group/textgrad/blob/75e912e210864b61999781778cdf756d4468120f/evaluation/prompt_optimization.py)
- [固定 commit 的 BBH loader 和切分](https://github.com/zou-group/textgrad/blob/75e912e210864b61999781778cdf756d4468120f/textgrad/tasks/big_bench_hard.py)
- [BBH word sorting 原始数据](https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/bbh/word_sorting.json)

数据 SHA-256、实际样本 ID 和每条预测都保存在
[`textgrad-small-result.json`](textgrad-small-result.json) 中。

### 4.2 起始提示词

使用上游 `BigBenchHard.get_task_description()` 的原文：

```text
You will answer a reasoning question. Think step by step. The last line of your
response should be of the following format: 'Answer: $VALUE' where VALUE is a
numerical value.
```

这个通用描述没有说明完整字典序，也错误地要求数值答案，因而为提示词优化保留了明确
headroom。

### 4.3 算法过程

每一步 batch size 为 3，严格执行以下链路：

1. 用当前系统提示词对 3 条训练题做 forward。
2. 对每条 response 和标准答案生成一份 textual loss gradient。
3. 把 response gradient 反传到可训练的 system prompt，得到 prompt gradient。
4. 聚合 3 份 prompt gradient，用一次 TGD 更新生成候选提示词。
5. 在 12 条验证集上评估候选；低于当前验证分数就回退。

这对应每步 `3 forward + 3 response backward + 3 prompt backward + 1 update`，再加 12
条验证评估。测试集不参与候选选择，仅在起点和最终提示词各评一次。

为了减少 grader 成本和随机性，本实验以最终 `Answer:` 行的规范化 exact match 代替
TextGrad 官方示例中的 LLM evaluator。这是一个明确偏差，但对 word sorting 有唯一标准
答案，确定性评分更容易审计。

### 4.4 逐步轨迹

| 步骤 | 训练 batch | 候选验证集 | 决策 | 保留验证集 |
|---:|---:|---:|---|---:|
| 0 | - | `0/12 = 0.0%` | 起点 | `0.0%` |
| 1 | `0/3 = 0.0%` | `6/12 = 50.0%` | 接受 | `50.0%` |
| 2 | `1/3 = 33.3%` | `7/12 = 58.3%` | 接受 | `58.3%` |
| 3 | `2/3 = 66.7%` | `11/12 = 91.7%` | 接受 | `91.7%` |
| 4 | `3/3 = 100.0%` | `7/12 = 58.3%` | **拒绝并回退** | `91.7%` |

第 4 步说明训练 batch 的 `100%` 并不代表泛化更好。验证回退挡住了一个 33.4 个百分点
的退化。

### 4.5 最终准确率

| Split | 基线 | 最终 | 绝对提升 | 最终 Wilson 95% CI |
|---|---:|---:|---:|---:|
| validation | `0/12 = 0.0%` | `11/12 = 91.7%` | `+91.7 pp` | `[64.6%, 98.5%]` |
| test | `0/20 = 0.0%` | `16/20 = 80.0%` | `+80.0 pp` | `[58.4%, 91.9%]` |

测试集新增答对 16 条；相对错误率下降 80%。样本量很小，所以置信区间仍然较宽。

最终 4 个错误中，3 个是相邻前缀比较错误：`edgy/eleanor`、`catsup/charm`、
`dandelion/deadlock`；另 1 个输出把 `trammel` 损坏成 `tramm!`。这说明最终提示词已经
解决输出类型、完整性和大部分排序问题，但长列表中的细粒度字符比较仍有提升空间。

### 4.6 最终提示词

```text
You will answer a reasoning question. Think step by step in the body of your
response. Steps may be consolidated for efficiency, but every comparison must
strictly adhere to lexicographic ordering: compare character-by-character at each
position (1st, 2nd, 3rd, etc.) until a definitive order is established for every
pair. Grouping by initial letter is only preliminary; full lexicographic comparison
is required within groups. Before finalizing, verify that each adjacent pair in your
ordered list is correctly sorted. Additionally, verify that the final ordered list
contains exactly the same items as the input, with no omissions, duplicates, or
additions. The last line of your response must be exactly: 'Answer: $VALUE' where
$VALUE is the exact final answer to the question. $VALUE may be a string, list, or
numerical value depending on what is asked—it must never be a meta-property like
item count or step count. If the answer is a list of items, format them exactly as
required by the objective (e.g., space-separated) with no commas, extra punctuation,
or conversational text.
```

它没有复制任何训练题或标准答案，候选泄漏检查通过。

### 4.7 调用与资源指标

| 阶段 | Calls | Prompt tokens | Completion tokens | Total tokens | 模型累计秒数 |
|---|---:|---:|---:|---:|---:|
| forward / 评估 | 112 | 28,146 | 75,701 | 103,847 | 2,314.56 |
| response gradient | 12 | 8,321 | 1,059 | 9,380 | 81.25 |
| prompt gradient | 12 | 10,799 | 3,222 | 14,021 | 153.42 |
| TGD update | 4 | 4,379 | 872 | 5,251 | 37.90 |
| **总计** | **140** | **51,645** | **80,854** | **132,499** | **2,587.12** |

- 墙钟时间：`776.85 s`，约 12 分 57 秒。
- 平均：`946.42 tokens/call`，`10.81 calls/墙钟分钟`。
- 模型累计秒数高于墙钟时间，是因为前向评估以 6 workers 并发，表中时间按请求相加。
- API 返回失败：0。
- 没有估算货币成本，因为当前 Key 对应套餐的实际单价未提供；token 数是可复核的成本基数。

## 5. OpenEvolve 实验

### 5.1 任务

复用 OpenEvolve 官方 function minimization 示例的目标：

```text
f(x, y) = sin(x) * cos(y) + sin(x*y) + (x^2 + y^2) / 20
x, y in [-5, 5]
```

官方 evaluator 使用近似全局最优点 `(-1.704, 0.678)` 和最优值 `-1.519`。本实验保留
它的四项评分和权重。[OpenEvolve 仓库](https://github.com/algorithmicsuperintelligence/openevolve)，
[固定 commit 的官方配置](https://github.com/algorithmicsuperintelligence/openevolve/blob/411fb59c886c18704caaffb611e17cf9e7d824d2/examples/function_minimization/config.yaml)，
[官方 evaluator](https://github.com/algorithmicsuperintelligence/openevolve/blob/411fb59c886c18704caaffb611e17cf9e7d824d2/examples/function_minimization/evaluator.py)，
[官方初始程序](https://github.com/algorithmicsuperintelligence/openevolve/blob/411fb59c886c18704caaffb611e17cf9e7d824d2/examples/function_minimization/initial_program.py)

评分公式：

```text
value_score       = 1 / (1 + abs(avg_value - (-1.519)))
distance_score    = 1 / (1 + avg_distance_to_known_optimum)
reliability_score = successful_trials / 10
base_score        = 0.5*value + 0.3*distance + 0.2*reliability
combined_score    = base_score * basin_multiplier
```

当平均距离 `<0.5` 时 multiplier 为 1.5，`<1.5` 时为 1.2。因此综合分允许高于 1.0。

### 5.2 小实验设置

| 参数 | 值 |
|---|---:|
| LLM mutations | 6 |
| trials per program | 10 个固定 seeds |
| objective calls per trial | 严格上限 200 |
| archive size | 12 |
| islands | 3 |
| exploitation ratio | 0.7 |
| candidate timeout | 15 s |
| max source length | 20,000 chars |

这是一个可审计的紧凑重建：保留“代码是 genome、LLM 变异、外部 evaluator、父代和多样性
inspiration、quality-diversity archive”等核心机制。与上游默认配置相比，本实验使用完整程序
重写而不是 diff，6 轮而不是 10 轮，archive 12 而不是 20，并把初始随机搜索预算从官方
1000 固定为所有程序共同的 200 次，以便在短时间内比较搜索策略而非算力。

### 5.3 程序安全与公平性

模型生成的 Python 不在主进程中执行。执行路径包括：

1. AST 白名单，只允许有限标准库和 `search_algorithm(objective, budget, rng, bounds)`。
2. 拒绝文件、环境、动态执行、dunder introspection 和写死已知最优值。
3. Bubblewrap 只读挂载系统 Python/库，清空环境变量并断开网络，因此候选看不到 API Key。
4. 设置 CPU、内存、文件大小、文件描述符和进程限制，以及墙钟超时。
5. evaluator 重新计算候选返回坐标的目标值，不信任候选自行返回的 value。
6. `BudgetedObjective` 在第 201 次调用前抛错；每轮使用完全相同的 10 个 seeds。

### 5.4 进化轨迹

| 轮次 | 主要策略 | 综合分 | 有效 |
|---:|---|---:|---|
| 0 | 200 点均匀随机搜索 | 0.933247 | 是 |
| 1 | 分层采样 + CMA-ES 风格局部搜索 | 0.946884 | 是 |
| 2 | 多起点坐标/模式搜索 | 0.984726 | 是 |
| 3 | 自适应尺度的 CMA-ES 风格搜索 | 0.960636 | 是 |
| 4 | 八方向模式搜索 | 0.783613 | 是 |
| 5 | 分层采样 + 多起点 Nelder-Mead | **1.477353** | 是 |
| 6 | 坐标下降 + basin hopping | 0.993266 | 是 |

第 4 轮回落并不会覆盖 archive 中的更优程序。第 5 轮是最终最佳，代码位于
[`experiment/openevolve_best_program.py`](../experiment/openevolve_best_program.py)。人工审查
确认它没有写死最优坐标，平均实际使用 `199.6/200` 次 objective calls。

运行时 WSL 曾在第 5、6 轮临时拒绝创建 Bubblewrap namespace。两份候选代码已经生成，
但当时未获得指标；沙箱恢复后使用同一 evaluator/seeds 离线重评，额外模型调用为 0。因
第 5 轮没有及时进入在线 archive，第 6 轮生成时没有以它为父代；这不影响第 5 轮自身的
基线对比，但意味着这 6 轮并不是一次完全无中断的标准 OpenEvolve 轨迹。执行器现已对这
一种基础设施错误增加最多 3 次有限重试。

### 5.5 最终指标

所有指标均为 10 次 trial 汇总；objective value 和距离越低越好，其余分数越高越好。

| 指标 | 基线随机搜索 | 最佳 Nelder-Mead | 变化 |
|---|---:|---:|---:|
| combined score | 0.933247 | **1.477353** | `+0.544106 / +58.30%` |
| value score | 0.870579 | **0.991360** | `+0.120781` |
| distance score | 0.474721 | **0.964072** | `+0.489352` |
| reliability | 1.000000 | 1.000000 | 持平，均为 `10/10` |
| basin multiplier | 1.2 | **1.5** | 进入 `<0.5` 距离区间 |
| average objective value | -1.370340 | **-1.510285** | `-0.139945` |
| objective value stddev | 0.128652 | **0.025203** | 更稳定 |
| average distance | 1.106502 | **0.037266** | `-96.63%` |
| distance stddev | 1.455337 | **0.110211** | 更稳定 |
| best value | -1.502302 | **-1.518686** | 更接近 -1.519 |
| worst value | -1.020609 | **-1.434676** | 最差 seed 也改善 |
| average objective calls | 200.0 | 199.6 | 未增加预算 |

`avg_runtime_seconds` 只测候选函数内部，在本机分别约为 0.16 ms 和 0.24 ms；它不包含
Bubblewrap 启动成本，因此仅用于确认算法没有用运行时间换分，不能当作端到端延迟。

### 5.6 模型调用指标

| Calls | Prompt tokens | Completion tokens | Total tokens | 模型累计秒数 | 墙钟秒数 |
|---:|---:|---:|---:|---:|---:|
| 6 | 10,493 | 5,578 | 16,071 | 163.08 | 162.70 |

- 平均 `2,678.5 tokens/call`。
- `2.21 calls/墙钟分钟`。
- API 返回失败：0。
- 两次离线沙箱重评没有 API 调用，不计入 token。

## 6. 验证

离线测试：

```bash
pytest -q experiment/test_small_experiments.py
```

结果：`12 passed`。覆盖答案解析、官方 positional split、dry-run 零网络、AST 安全门、
官方评分公式、Bubblewrap 确定性执行、无 Key dry-run，以及并行/异步任务构造和
time-to-quality 加速断言，以及两个真实算法的 serial/sync/async 调度入口。

正式运行前后的 dry-run：

```bash
python -m experiment.textgrad_prompt_optimization --dry-run --model glm-5.2
python -m experiment.openevolve_program_search --dry-run --model glm-5.2
```

## 7. 复现命令

`bash -ic` 用于启动会读取 `~/.bashrc` 的交互 shell；命令本身不含 Key。

TextGrad：

```bash
bash -ic 'python -m experiment.textgrad_prompt_optimization \
  --provider openai --model glm-5.2 --thinking disabled \
  --steps 4 --batch-size 3 \
  --train-size 12 --val-size 12 --test-size 20 \
  --workers 6 --subset longest --max-tokens 2048 --yes'
```

OpenEvolve：

```bash
bash -ic 'python -m experiment.openevolve_program_search \
  --provider openai --model glm-5.2 --thinking disabled \
  --iterations 6 --trials 10 --objective-budget 200 \
  --archive-size 12 --islands 3 --max-tokens 2048 --yes'
```

重复执行会覆盖同名结果 JSON；远端生成可能不同，建议需要多 seed 结论时另传 `--output`
保存每次运行。

## 8. 结论与下一步

作为“是否值得移植”的小实验，两种机制都获得正信号：

- TextGrad 很擅长从具体失败反馈中修复任务说明和输出契约，并且验证回退是必要组件。
- OpenEvolve 在固定 evaluator 预算下找到了明显优于随机搜索的程序结构，代码 genome 和
  quality-diversity archive 对 AgentDescent 当前覆盖形成了真正的新机制。

下一阶段若要得到可发表或可合并的结论，应扩展为：TextGrad 使用多个 BBH task、完整
100 条 validation/test 和多个 seed；OpenEvolve 使用上游完整包、无中断 10 至 50 轮、
多个 evolution seed，并同时报告最佳值和跨 seed 均值。当前报告只证明“小规模、固定设置
下确实出现了提升”，不外推到其他任务或模型。
