# 设计文档：meta_evolve —— 演化 `evolve()` 自己的决策插槽

> 目标：用户把一个搜索算法（比如一棵树搜索）插进 `evolve()` 去解一类问题；
> 现在要让这个**算法本身**被演化——在一个数据集上演化，在另一个更新、更难的
> 数据集上验证它是不是真的更好，而不是只在训练地形上更好。
>
> **状态：核心已实现并落地**（P0–P3，见 §6）。本文是设计记录——写的是*为什么*
> 这样切、切在哪、哪些没做。要*怎么用*，看 [Meta-evolution](meta-evolution.md)；
> 一个可离线跑通的实例在 [`examples/metasearch/`](https://github.com/Birfy/agentdescent/tree/main/examples/metasearch)。
>
> 落地的模块：`agentdescent/meta.py`（§3 全部）、`examples/era/era_empirical_software.py`
> 的两处注入口（§3.5）、`examples/metasearch/`（§4 的 stage 0）。
> **未做**：SWE-bench-Science / Terminal-Bench-Science 的 `HarborDomain` 适配器（§4.3，
> 需要容器与 agent，离线测试无法覆盖）；`acceptance` / `conflict` / `fusion` / `promotion`
> / `proposal` 五个插槽的内置冒烟测试与默认种子（§3.3，需要构造 `MergeContext` /
> `Evolvable`，留给使用方传 `smoke=`）。

---

## 0. 结论先行

| 问题 | 结论 |
|---|---|
| port 是怎么插进 `evolve()` 的？ | 两条路：11 个微移植是 `MethodPolicy`，机制走 `engine=Policies(...)`；8 个基准移植直接调 `evolve()`，用 `strategy=` 和 `aggregator_factory=`。树搜索（ERA）是后者：`EraTree` + `FlatPuct`（一个 `SelectionPolicy`）包在 factory 里 |
| "演化这个 policy"里的 policy 指什么？ | `agentdescent.policies.Policies` 的某个字段——引擎的**决策面**。八个插槽：`selection` / `task_sampler` / `acceptance` / `conflict` / `fusion` / `promotion` / `staleness` / `proposal` |
| 之前能演化它吗？ | 不能。`EraTree.__post_init__` 把 `FlatPuct` 写死；而且引擎没有"把插槽值当 artifact"的概念 |
| 方案是什么？ | **把普通引擎往上抬一层**：外层 `evolve()` 的 artifact = 插槽的值；一次外层 rollout = 用候选值跑一整个内层搜索；外层 reward = 内层 held-out 曲线的 AUC；治理 L1 |
| 引擎要改吗？ | **`evolution.py` / `aggregator.py` / `ledger.py` 一行未改**。新增 `agentdescent/meta.py` 一个模块；ERA 例子开两个注入口（`EraTree(policy=)`、`run_agentdescent_era(selection=)`），默认行为不变，上游 FUTS 复现测试仍过 |
| 在哪演化、在哪验证？ | 演化放在**便宜**的内层域（合成地形 → AlgoTune），验证放在**贵但真实**的 2026 科研 agent 基准（SWE-bench-Science、Terminal-Bench-Science）。理由在 §4.1 的成本公式 |
| 离线跑出来什么？ | 一个更贪的选择规则在源地形赢（+0.008，11 胜 4 负），迁移比 0.06——即到了目标地形不灵。**这正是设计要让人看见的那种结果** |

**一句话**：插槽值是文本、能编译成对象、能校验，就能被现有引擎演化——合并、冲突、
gate、ledger、并行都是原来的；新东西只有"值的表示法"（`SlotSpec`）、"内层问题"
（`Problem`）和"内层结果怎么变成一个数"（`MetaOutcome` → `auc`）。

---

## 1. 现状分析（引擎能提供什么）

### 1.1 决策面已经是一等公民

`Policies` 把 `evolve()` 的每个决策做成可替换字段，两条保证使它可信：**没有字段被
静默忽略**（`require_supported` 对不支持的字段直接 raise），**`None` 就是今天的行为**
（`Policies()` 和不传等价）。主分支最近又补了安装钩子：聚合器通过
`install_policy()` 把引擎的 verifier 和 config 交给任何暴露 `bind(verifier)` /
`configure(config)` 的策略对象（[policy-guide §5](policy-guide.md#5-installing-what-the-engine-hands-your-policy)）。
这意味着一个**演化出来的**策略类只要实现这两个可选钩子，就能包住引擎默认规则。

### 1.2 八个 Protocol 全是 `runtime_checkable`

`SelectionPolicy` / `TaskSampler` / `AcceptancePolicy` / `ConflictPolicy` /
`FusionPolicy` / `PromotionPolicy` / `StalenessPolicy` / `ProposalPolicy` 都是结构化
协议。`isinstance(obj, SelectionPolicy)` 是真检查。这是"值可以是任意类的源码"能成立的
前提——门控可以在编译后**结构性地**验证，而不是靠模型自觉。

### 1.3 树搜索是怎么接的，以及卡在哪

`examples/era/era_empirical_software.py`：`EraTree` 持有节点列表，`select_parent()`
把节点包成 `Candidate`、装进 `SelectionContext`、问 `self._policy.select(ctx, 1)`，然后
**在树里**做 visit 预留和沿父链回传。`EraTreeAggregator` 是 `aggregator_factory`，
`run_agentdescent_era()` 把这些接进 `evolve()`。

卡点：`self._policy = FlatPuct(self.c_puct, self.prior_exponent)` 写在 `__post_init__`，
外部无法注入。但也正因为 visit 预留和回传在树里而不在策略里，**策略可以只负责"选谁"**。

### 1.4 自改代码有先例，门控也有先例

SICA 和 Gödel Agent 两个移植把 Python 源码放在 `ValidatedSlot` 里，用 AST 白名单
（`compile_policy`）在 `to_diff` 处一次性校验。校验只在一处、不合格即无 diff 并计数、
不做任何兜底替换——这是仓库的既定规则（`porting-methodpolicy.md`）。本设计沿用。

### 1.5 分层治理已经为"改 harness"准备好了

`blast_radius=0.6` → L1：每次合并强制过 oracle。一个插槽值改变的是"下游一切怎么被搜"，
按定义是 harness。而 `verifier` / `ledger` / `executor` 等机器层字段对应中心类比里的
"训练代码，不可自改"——L0。

---

## 2. 差距清单

| # | 差距 | 影响 |
|---|---|---|
| G1 | 引擎没有"插槽值当 artifact"的表示法：值要能渲染成文本、从提案变 diff、再编译回对象 | 没法把 `Policies(selection=…)` 放进 ledger |
| G2 | 一次 rollout 的产出是"一个答案"，而评估一个搜索规则需要"一整条曲线" | 终值几乎区分不开规则；需要 AUC 这样的过程指标 |
| G3 | `EraTree` 写死 `FlatPuct` | 最需要演化的对象不可注入 |
| G4 | 让模型改写整个 `select()` 太危险：可以永远返回死节点、跳过 visit 预留饿死根、半路 raise | 需要一个"只能在优先级上犯错"的窄面 |
| G5 | 通用性：用户要能演化**任意**插槽，不只 `selection` | 需要一个以 Protocol 为契约的通用门 |
| G6 | 验证：在训练问题上赢不说明什么 | 需要与训练不相交的问题集 + 配对比较 + 迁移比 |
| G7 | 成本：外层每个 rollout 和每次 gate 评估都是一整个内层搜索 | 必须区分"在哪演化"和"在哪验证" |
| G8 | API 文档由 `agentdescent.__all__` 生成，函数对象作默认值会带内存地址 | 公开签名里不能有 callable 默认值（实际踩到，见 §8） |

---

## 3. 设计

### 3.0 总体形状

```
                外层 evolve()  ←── 引擎不变
  artifact   =  插槽的值（SlotSpec 持有：Strategy + compile + describe）
  task       =  一个内层问题 Problem: (value, seed) -> MetaOutcome
  run        =  spec.compile(rendered) → 装进 Policies(<slot>=value) → 跑完整内层搜索 → 曲线
  reward     =  auc(MetaOutcome)      （best-so-far 曲线均值）
  propose    =  slot_reflector(model, spec)：看规则 + 内层轨迹 → 重写规则
  governance =  L1 (blast_radius=0.6)
                    │
                    ▼  内层：Problem 自己决定是什么
        evolve_problem(...)        run_agentdescent_era(selection=value)      landscape_problem(...)
        （内层 evolve）             （ERA 树搜索，EraTree(policy=value)）        （合成地形，毫秒级）
```

| | 内层 | 外层 |
|---|---|---|
| artifact | 程序 / 提示词 | 插槽的值 |
| task | 一个 shard | 一整个搜索问题（一个 seed 实例、一个 AlgoTune 任务、一个 Harbor 任务） |
| reward | 域指标 | `auc`：内层 held-out 曲线 best-so-far 的均值 |
| 治理 | L1（程序是 harness） | L1（规则改的是所有下游） |
| gate | held-out shard | held-out **搜索实例** |

### 3.1 `SlotSpec`：值的表示法（G1）

一个普通 `Strategy`（`initial / render / to_diff`）加两个方法：

```python
def compile(self, rendered: str) -> Any      # 能塞进 Policies(<slot>=...) 的对象；不合格 raise ValueError
def describe(self) -> str                    # 告诉反思模型这个面是什么、提议必须长什么样
```

因为它就是 Strategy，合并、冲突、gate 全走原引擎。校验仍只在 `to_diff` 一处。
三种实现，按"可改的范围"递增：

| spec | 值 | 合并语义 | 门 |
|---|---|---|---|
| `ParamSlot(factory, params, bounds)` | 策略类的数值构造参数（`FlatPuct(c_puct, prior_exponent)`、`Beam(k)`） | 不同参数 union 合并；同一参数矛盾，按 held-out 裁决 | 未知名字、越界值拒绝 |
| `SourceSlot(initial_value, validate, build)` | 由 `build` 编译的源码文本 | 单槽：每轮一次锦标赛 | `validate` raise |
| `priority_selection()` | 树搜索专用：**一个函数** `priority(rank, visits, total, prior, depth, n_nodes)` | 同上 | AST 白名单 + 固定输入网格上必须有限 |
| `policy_source(slot, seed)` | **任意插槽**：满足该插槽 Protocol 的一个类的完整源码 | 同上 | AST 走查 + 受限命名空间构建 + `isinstance(Protocol)` + 冒烟 |

### 3.2 树搜索的窄面：`priority_selection()`（G3 / G4）

为什么不让模型改整个 `select()`：它可以永远返回同一个死节点、跳过 visit 预留饿死根、
或在运行中途 raise。一个"六个数进、一个数出"的函数只可能在**优先级**上犯错——而优先级
恰恰是被搜索的东西。所以：

- 种子是 ERA 上游的 flat PUCT：`rank + c·(1/N)·sqrt(total)/(1+visits)`；
- `PrioritySelection` 是运行它的 `SelectionPolicy`：rank 归一化、prior 归一化、深度、
  沿父链的 visit 预留、平局规则全在包装里；
- 种子源码与 `FlatPuct(c_puct=1.0, prior_exponent=0.0)` **逐步选同一节点**，
  `tests/test_metasearch.py` 与 `tests/test_meta.py` 都钉死这一点；
- 门是 SICA 的 AST 门放宽到打分函数需要的范围（算术、比较、条件、局部变量、`math`），
  再在一个**含根节点（visits=0, total=0）**的固定网格上跑一遍，必须处处有限。
  `rank / visits` 这种规则在提议时就被拒，而不是在根节点炸。

包装把**已评分的 prior**（归一化到和为 1，无评分时均匀）交给规则，候选规则可以用；
种子不用，那是上游的选择。

### 3.3 通用面：`policy_source(slot, seed)`（G5）

值 = 一个名为 `Policy` 的类的完整源码。门分四层：

1. **AST 走查**：`import` 只允许白名单（`math` / `random` / `statistics` / `itertools` /
   `collections` / `functools` / `dataclasses` / `typing` / `enum` / `heapq` / `bisect`，
   以及 `agentdescent.selection/staleness/policies` 里的值类型），禁 dunder 名字与属性，
   禁 `exec` / `eval` / `open` / `getattr` / `type` / `super` 等能触到解释器的调用，禁
   `global` / `nonlocal`；
2. **受限命名空间构建**：只有安全 builtins、白名单模块、引擎值类型；`__import__` 换成
   只回答白名单的版本（方法体内的 `import random` 因此可用，`import os` 不行）；
   无参实例化；
3. **`isinstance(obj, SLOT_PROTOCOLS[slot])`**——协议是 `runtime_checkable`，这是结构性
   检查；
4. **冒烟**：`selection` / `task_sampler` / `staleness` 内置（`select` 必须返回 1..n 个
   来自候选池的对象、单候选时必须返回它；`pick` 必须返回 keys 之一；`decide` 必须返回
   `StaleAction`）；其余五个插槽的输入是 `MergeContext` / `Evolvable`，冒烟由使用方
   `smoke=` 传入。

`seed_source(slot)` 为前三个插槽提供引擎默认行为的源码版本；`describe()` 用 `inspect`
列出 Protocol 的方法签名，反思模型知道什么必须保留。

**边界必须说清楚**：这是 SICA / Gödel Agent 自改代码用的同一级门控，够把模型的重写
限制在"做决策"上，**不是沙箱**，不能跑陌生人的代码。要隔离，用 `ProcessExecutor` /
`SandboxPool` 把内层问题整个放进子进程或容器，门控不变。

### 3.4 内层问题与 meta-reward（G2）

```python
Problem = Callable[[Any, int], MetaOutcome]      # (编译后的插槽值, seed) -> 结果
```

`MetaOutcome` 带 `curve`（每次 sweep 后的 held-out reward）、`final`、`rollouts`、`detail`
（给反思模型看的：树摘要、outcomes、错误）。三个现成的 reward：

| reward | 含义 | 何时用 |
|---|---|---|
| `auc`（默认） | best-so-far 曲线均值 | 选择规则造不出更好的答案，只能**更早**找到；固定预算下终值几乎不区分规则 |
| `final_reward` | 内层终值 | 你真正在乎的只是最终质量 |
| `rollouts_to(target)` | `1/(1+首次达标的 sweep 数)` | time-to-quality |

三种现成的 `Problem`：`evolve_problem(tasks, reward, slot=…, **evolve_kwargs)` 包一次内层
`evolve()`（值装在 `base.merged_with(slot=value)`，seed 透传，`verbose` 强制关）；
`run_agentdescent_era(selection=value)` 包 ERA 树搜索；`landscape_problem(family)` 是
示例里的合成地形。

内层必须**seed 确定**：同值同 seed 同结果。这样 gate 的重复评估能被 `eval_cache` 命中，
配对比较才有意义。

### 3.5 唯一动到的"引擎侧"代码：ERA 的注入口（G3）

```python
EraTree(policy: Optional[SelectionPolicy] = None)       # None → FlatPuct(c_puct, prior_exponent)
run_agentdescent_era(..., selection: Optional[SelectionPolicy] = None)
```

策略只回答"扩展谁"；visit 预留与回传留在树里。`tree.summary()` 多一个 `"selection"`
字段记录用的是哪个类。默认路径逐比特不变，`test_serial_tree_reproduces_upstream_futs`
仍过。

### 3.6 `meta_evolve()` 与 `meta_validate()`（G6）

`meta_evolve(problems, *, slot, spec, propose|model, meta_reward=None, seeds=(0,),
blast_radius=0.6, **evolve_kwargs)`：

- `slot` 必须在 `SLOTS` 里，机器层字段拒绝（L0 线）；
- 每个 `(problem, seed)` 对是一个外层 task，`held_out_frac` 照常切 train / held-out；
- 种子值必须先过自己的门（`spec.compile(spec.render(spec.initial()))`），否则起点就是
  非法的；
- `strategy` / `run` / `reward` / `agent` 是本函数的，传了就 `TypeError`；
- 返回普通 `EvolutionResult`：`spec.compile(result.rendered)` 是演化出的值。

`meta_validate(spec, before, after, problems, seeds)`：在**与外层不相交**的
`(problem, seed)` 上按 seed 配对打分，报告每个问题的 before / after / 增益 / 增益 sd /
胜负；`transfer_ratio(report, source, target)` = 目标增益 / 源增益。读法：接近 1 是更好
的规则；接近 0 且源增益为正是对训练地形的过拟合；负值是用泛化换训练集；源增益为零时
返回 `None`——零比零不是迁移结果。

---

## 4. 在哪演化、在哪验证（G7）

### 4.1 成本公式决定分工

一次外层运行的内层搜索次数约为：

```
rounds × n_workers                     # 外层 rollouts，每个是一整个内层搜索
+ 候选数 × |held_out|                   # 每次 gate 评估
+ 候选数 × cheap_eval_tasks             # 开 tournament 时
```

内层是 AlgoTune 时一次搜索是分钟级；是 SWE-bench-Science 时一次**扩展**就是一个 agent
在容器里跑几分钟。所以：**在便宜的域演化，在贵的基准验证**——验证对每个值只打一次分。

| 阶段 | 内层域 | 一次内层搜索的代价 | 证明什么 |
|---|---|---|---|
| 0 离线（已实现） | `examples/metasearch/_landscape.py`：seed 确定的合成地形，`SOURCE` 用来演化，`TARGET`（更高维、更崎岖、死胡同更多）外层从未见过；树是真实的 `EraTree` | 毫秒 | 机制成立；在分布内赢的规则是否在分布外也赢 |
| 1 在线便宜（接口已开） | AlgoTune（arXiv 2507.15887，155 任务，沙箱计时的加速比；`bench/results/era-algotune-model-prior.md` 已有基线） | 分钟 | 在真实程序搜索、近期难基准上演化出的规则 |
| 2 验证（已设计，未实现） | SWE-bench-Science、Terminal-Bench-Science，作为 ERA `Domain` | 每次扩展一个容器化 agent 运行 | 规则能否迁移到它从没见过的科研 agent 工作 |

### 4.2 为什么是这两个验证集，为什么不是 AIME / GSM-Hard

| 候选 | 结论 | 理由 |
|---|---|---|
| GSM-Hard | 否 | 与 GSM8K test **不是**逐行对齐（committed 样本里只有前两行配得上），"同题换数字"的配对设计不存在；且上限是算术不是推理 |
| AIME 1983–2024 → AIME 2025/2026 | 否 | 整数答案、判分器可复用，但已被刷爆；而且对**程序搜索策略**不是有意义的迁移目标 |
| HMMT / BRUMO 2025–2026 | 否 | 一半答案是 `\frac{1311}{2017}` 这类，需要本仓库自己写符号比较器——`_gsmhard_domain` 拒绝 MATH-500 的同一条线 |
| **SWE-bench-Science**（arXiv 2608.19799） | **是** | 119 任务 / 98 个科研仓库 / 20 个领域，HF `OpenMOSS-Team/SWE-bench-Science`，96 个默认许可任务；Harbor 格式，clean verifier 里程序化判分；Claude Code + Opus 5 pass@1 < 50% |
| **Terminal-Bench-Science 0.1**（2026-08） | **是** | 70 个专家任务、五个自然科学领域，Harbor `terminal-bench-science/terminal-bench-science@latest`，Apache 2.0；最强 agent 30.0% |

两者都晚于任何可能的演化源，迁移数字不是记忆；两者都是"解科研问题的搜索算法"真正
被要求做的事。

### 4.3 Stage 2 的具体形状：Harbor 任务作为 ERA `Domain`

ERA 的搜索对搜索对象无感；`Domain` 就四件事：

| `Domain` 字段 | Harbor 任务 |
|---|---|
| `initial_program` | 对任务基线（`task.toml` + 按 digest 钉住的 Docker 镜像）的空 patch |
| `evaluate(patch, shards)` | 在干净容器里打 patch，跑测试的 **scoring 子集**，返回通过率；held-back 子集只在最后报一次 |
| `prompt(parent)` | 任务的 `instruction.md` + 工作区里已打上的父 patch + scoring 测试的输出，交给 `claude_code()`（或 `openai_compatible`）在容器里干活，产出 `git diff` |
| `test_shards` | held-back 测试 |

这和 ERA 在 shard 上的切分纪律一致。诚实边界：只有一个测试文件的任务没东西可
hold back，搜索信号只剩 agent 自检。适配器需要 Docker / Modal、`harbor` / `pier`
runner 和一个 agent，离线套件跑不了，所以没进仓库；边界写在这里而不是藏起来。

### 4.4 实验协议

1. 每个设置 **3 个外层 seed**，报均值 ± sd。
2. **对照**：`--serial`（单 worker，无合并——上游串行循环）；种子规则（flat PUCT）作
   baseline；在目标上**直接演化**出的规则作上限。
3. **验证用新实例**：与外层训练/gate 过的实例不相交。
4. **读迁移比，不读增益**。离线实例跑出的就是"第二种"：更贪的规则源地形 +0.008
   （11/4），目标 +0.000，迁移比 0.06。

---

## 5. 与主分支近期改动的关系

主分支在本设计进行中合入了两件事，都与本设计相容并被吸收：

- **`bind` / `configure` 安装钩子**（PR #162）：`policy_source` 演化出的类若实现这两个
  钩子，`install_policy()` 会把 verifier 和 config 交给它——演化出的规则可以**包住**
  引擎默认规则而不是从头写。冒烟测试在 bind 之前运行，所以需要 verifier 的候选类
  必须容忍未绑定（否则被 `PolicyUnboundError` 拒在门口，这是想要的行为）。
- **移除一行式入口**（PR #163）：`evolve_skill` 等不再存在，`evolve()` 是唯一入口。
  本设计一开始就只在 `evolve()` 之上叠一层，合并时只需处理 `__init__` 导出与
  `gen_api_docs` 的段落表。

---

## 6. 实施计划与现状

| 阶段 | 内容 | 状态 |
|---|---|---|
| P0 | `EraTree(policy=)` / `run_agentdescent_era(selection=)` 注入口，默认不变 | ✅ |
| P1 | `agentdescent/meta.py`：`MetaOutcome` / `Problem` / `auc` 等 / `ParamSlot` / `SourceSlot` / `priority_selection` / `PrioritySelection` / `meta_evolve` / `meta_validate` / `transfer_ratio` | ✅ |
| P2 | `policy_source(slot, seed)` 通用门 + `seed_source` + `SLOT_PROTOCOLS` | ✅（三个插槽有内置冒烟与种子） |
| P3 | `examples/metasearch/`：合成地形、离线端到端、`--dry-run`、加入 PORTS 契约 | ✅ |
| P4 | AlgoTune 在线跑：3 seed × {seed 规则, 演化规则} × 8 任务，写入 `bench/results/` | 待跑（接口已开，需 API 与沙箱） |
| P5 | `HarborDomain` 适配器（§4.3）+ SWE-bench-Science / TB-Science 验证 | 待做（需容器 + agent） |
| P6 | 其余五个插槽的内置冒烟与默认种子 | 待做 |
| P7 | 多插槽联合演化（`ParamSlot` 的 key 空间天然支持；`SourceSlot` 需要多槽 Strategy） | 开放 |

测试：`tests/test_meta.py`（库）、`tests/test_metasearch.py`（示例）、
`tests/test_example_entrypoints.py`（入口契约）、`tests/test_api_reference.py`（API 页同步）。

---

## 7. 验收标准

1. `PrioritySelection(PRIORITY_SEED)` 与 `FlatPuct(1.0)` 在上游式轨迹和合成地形上**逐步
   选同一节点**（已钉死）。
2. `EraTree()` 不传 policy 时行为逐比特不变，`test_serial_tree_reproduces_upstream_futs` 过。
3. 门拒绝：模块级/方法级越界 import、dunder、`open`/`exec` 类调用、循环（窄面）、在根
   节点除零、返回非有限数、返回不在候选池内的对象、类名不对、带参构造。
4. `meta_evolve` 拒绝机器层插槽、拒绝 `strategy=`/`run=`/`reward=`/`agent=`、没有反思器时
   报错。
5. 离线端到端：脚本化反思器提出的规则被 gate 接受，`result.rendered` 能编译，
   `meta_validate` 报告含源与目标两行，`transfer_ratio` 可读。
6. 公开签名里无 callable 默认值；`python -m tools.gen_api_docs --check` 通过。
7. **迁移验收（P4/P5）**：演化规则在 AlgoTune 训练任务上的增益 > 噪声，在不相交的
   AlgoTune 任务和至少一个科研基准上报告迁移比；迁移比 ≈ 0 是合法且有信息量的结果。

---

## 8. 风险与开放问题

| 风险 | 缓解 |
|---|---|
| **成本**：外层 rollout 与 gate 都是完整内层搜索 | §4.1 分工；小内层预算；少 held-out 实例 + `cheap_eval_tasks`；内层 seed 确定 + `eval_cache=FileCache(...)`；内层 `max_concurrency=1`（外内 worker 数相乘） |
| **门≠沙箱**：`policy_source` 执行模型写的类 | 文档明说；要隔离就把 `Problem` 整个放进 `ProcessExecutor` / 容器；`priority_selection` 是更安全的窄面 |
| **过拟合训练地形** | `meta_validate` 用不相交实例；读迁移比；目标上直接演化的规则作上限对照 |
| **AUC 对内层噪声敏感** | 内层 seed 确定；多 seed 配对；Beta 后验 gate 本身吸收噪声 |
| **五个插槽无内置冒烟** | 明确要求 `smoke=`；`isinstance(Protocol)` 仍然生效；P6 补齐 |
| **API 页非确定**（已踩）：`validate=staticmethod(lambda…)`、`meta_reward=auc` 作默认值时，生成的 api.md 每次带不同内存地址 | 默认值改为 `None`，函数内回退；`tests/test_api_reference` 守住 |
| **外层 `held_out_frac` 切的是问题实例**，不是内层数据 | 内层数据切分由 `Problem` 自己负责；文档在 `meta_evolve` 的 `seeds` 参数处说明 |
| 反思模型不遵守类/函数形状 | 不合格即无 diff 并计数（`invalid_proposals`）；`describe()` 给出 Protocol 签名；示例报告拒绝率 |
