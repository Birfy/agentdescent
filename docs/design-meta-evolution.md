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
> 的两处注入口（§3.5）、`examples/metasearch/`（§4 的 stage 0 与 stage 2 的适配器
> `_harbor.py`）、`bench/metasearch_algotune.py`（stage 1 的跑批脚本）。
> **未做**：AlgoTune 与科研基准的**在线**跑（需要 API key、numpy/scipy 沙箱、Docker
> 守护进程）；容器内的 agent 阶段（那是 `harbor run --agent`，不重造）。

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
4. **冒烟**：八个插槽各有一个内置冒烟（`select` 必须返回 1..n 个来自候选池的对象、
   单候选时必须返回它；`pick` 必须返回 keys 之一；`decide` 必须返回 `StaleAction`；
   `resolve` 必须至少保留一张卡且保留的卡两两不矛盾；`select`(fusion) 不得发明任何 diff
   都没提的 key；`accept` / `observe` / `propose` 必须返回正确类型），`smoke=` 可替换。
   合并侧的冒烟用一个最小 `Evolvable`（`_SmokeArtifact`）和三张卡（两张矛盾、一张不相交）
   构造 `MergeContext` / `MergeReport` / `ProposalContext`。

`seed_source(slot)` 为八个插槽都提供起点：能转写引擎默认规则的转写（`selection` /
`task_sampler` / `staleness` / `fusion` / `promotion`）；默认规则要读 verifier 或
Beta 后验的给最简单的合规规则并在注释里说明（`acceptance` / `conflict`）；`proposal`
是占位形状（引擎默认是 actor 自己的 `propose`，不是策略对象）。`tests/test_meta.py`
把每个种子装进一次真实的内层 `evolve()` 跑通，证明引擎确实安装并调用了它。
`describe()` 用 `inspect` 列出 Protocol 的方法签名，反思模型知道什么必须保留。

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
| 1a 在线最便宜（已跑，`bench/metasearch_slots.py`） | **GSM-Hard / GSM8K 上的指令演进**：一次内层 rollout 是一次完整的内层 `evolve()`，演进的插槽是 `task_sampler`。不需要沙箱、不需要 numpy、不需要容器——只有模型调用 | ~1 分钟、~50–80 次调用 | 元层机制在真实模型与真实数据上成立；采样器在同基准的未见切片与另一个基准上是否迁移 |
| 1b 在线便宜（脚本已就绪，`bench/metasearch_algotune.py`） | AlgoTune（arXiv 2507.15887，155 任务，沙箱计时的加速比；`bench/results/era-algotune-model-prior.md` 已有基线） | 分钟 | 在真实程序搜索、近期难基准上演化出的规则 |
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

这和 ERA 在 shard 上的切分纪律一致。适配器是 `examples/metasearch/_harbor.py`：
`load_task` 读任务目录；`harbor_domain` 把 `reward.json` 的指标映射成 shard（scoring /
held-back；只写 `reward.txt` 的任务只有一个指标、没东西可 hold back，运行计划里会说）；
`harbor_completion` 把 `WorkspaceAgent` 放到 ERA 的 `prompt -> text` 契约后面（物化父
patch、在那里跑 agent、`git diff`，模型不用自己排版 diff）；两个 runner：`LocalRunner`
在宿主机检出上 `git apply` + 跑 `tests/test.sh`（离线测试用真实 git、真实 `test.sh`、
真实 ERA 树搜索端到端跑通），`DockerRunner` 在任务自己的镜像里验证（已写，本机无守护进程
未跑，拒绝路径有测试）。剩下的边界：**容器内的 agent 阶段**是 `harbor run --agent`，不
重造——`LocalRunner` 对"环境就是一个仓库加解释器"的任务是诚实的（SWE-bench-Science 的
多数），对需要镜像工具链的任务不是（Terminal-Bench-Science 的多数）。

### 4.4 实验协议

1. 每个设置 **3 个外层 seed**，报均值 ± sd。
2. **对照**：`--serial`（单 worker，无合并——上游串行循环）；种子规则（flat PUCT）作
   baseline；在目标上**直接演化**出的规则作上限。
3. **验证用新实例**：与外层训练/gate 过的实例不相交。
4. **读迁移比，不读增益**。离线实例跑出的就是"第二种"：更贪的规则源地形 +0.008
   （11/4），目标 +0.000，迁移比 0.06。

### 4.5 在线结果矩阵（P4a，`bench/results/metasearch-slots.md`）

七格：`task_sampler` × {GSM-Hard, AIME, HotpotQA, MGSM-zh, GPQA, BBH} 和
`acceptance` × GSM-Hard。**先跑错了一遍，再修正重跑，两次都留在记录里。**

**第一次（内层 4 次 rollout）的结论是错的**：七格里三格提交，而这三格在同基准未见窗口上
**全部为负**（−0.141 / −0.094 / −0.031），看起来像"方法不迁移"。

**诊断**：内层预算太小到把信号弄反。插桩量到 round-robin 的 4 次 rollout 有 3 次落在
已解出的题上（引擎不向通过的 rollout 索要提案），而会学习的采样器必须先花一次 rollout
才能观察到失败。决定性证据：故意写坏的 `always-first`（永远取 `keys[0]`）在 4 次预算下
**并列第一**，在 12 次预算下**垫底**，规则间跨度从 0.125 涨到 0.208。
**一个能把反例排到第一的预算，测的不是规则。**

**修正后（12 次 rollout，其余不变）**：

| 插槽 | 演进于 | train | **unseen（迁移）** | 4-rollout 时的 unseen |
|---|---|---:|---:|---:|
| `task_sampler` | GSM-Hard | +0.029 (2/0) | **+0.042** (1/0) | −0.141 |
| `task_sampler` | HotpotQA | +0.029 (3/0) | **+0.031** (1/1) | −0.031 |
| `task_sampler` | BBH | +0.023 (2/0) | **+0.021** (1/0) | +0.000 |
| `task_sampler` | GPQA | +0.026 (1/0) | +0.000 | +0.000 |
| `task_sampler` | AIME | +0.000 | +0.000 | +0.000 |
| `task_sampler` | MGSM-zh | +0.000 | +0.000 | −0.094 |
| `acceptance` | GSM-Hard | +0.000 | +0.000 | +0.000 |

**没有一格再出现负迁移**：四格提交，unseen 增益 +0.042 / +0.031 / +0.021 / +0.000，
每个 train 行都是 1–3 胜 0 负。三格仍不提交，原因各不相同：AIME 两个预算下都找不到
胜过 round-robin 的规则；MGSM-zh 的 unseen 窗口在 meta-reward 上已是 **0.958**，无空间可涨；
`acceptance` 打不过引擎自己的 Beta 后验门。

**仍然要说清楚的**：增益只有 +0.02~0.04 AUC，而手写的"重试失败题"采样器在同样窗口上
能到 +0.065；每格只有 1 个验证 seed；GSM-Hard 的跨基准列（−0.052）与同基准 unseen 列
（+0.042）互相矛盾——**同基准内规则站得住，跨基准这批证据说不了话**。

**reflective merge 开与不开数字完全相同**（+0.029 / +0.042 / −0.052）。两次都有 6 个
不同提案、确有可融合的矛盾，两条演进出的规则文本不同却逐格同分——因为这个搜索找到的
每条规则都是同一个想法的不同措辞：**别把 rollout 花在已解出的题上**。这个域上该插槽
大约只有一个可发现的自由度，这也解释了低预算下退化采样器为何能撞上它。

### 4.6 第一次深入的单格结果（12 次 rollout）

`task_sampler` 在 4 个 GSM-Hard 窗口上演进，12 次外层 rollout，L1 下提交 2 次：

| 组 | 种子 | 演进后 | 增益 | 胜/负 |
|---|---|---:|---:|---:|
| 演进过的 4 个窗口 | 0.719 | **0.769** | **+0.050** | 4/1 |
| 同基准未见的 2 个窗口 | 0.713 | 0.725 | +0.013 | 1/1 |
| GSM8K 的 2 个窗口 | 1.000 | 0.900 | −0.100 | 0/1 |

演进出的规则是 UCB 式探索 + **跳过已解出的任务**——引擎不向通过的 rollout 索要提案，
所以落在已解出的题上买不到提案，这正是这个插槽该学的东西。

三行必须分开读。**训练行是站得住的**（配对 seed，内层确定，4 胜 1 负）。**未见行 +0.013、
一胜一负，更像是对训练窗口的拟合而不是更好的采样器**，迁移比 0.25 就是这个样子。
**跨基准那行读不出迁移**：种子规则在两个 GSM8K 窗口上已经是 1.000，唯一能动的方向是往下——
这是验证集选择的缺陷，不是发现，下一轮要把 GSM8K 换成还有空间的基准。手写参照
（"重试失败题"）是 0.784，说明 12 次 rollout 的搜索拿到了可得增益的约四分之三。

**七次空结果各有各的原因**（详见结果页的表）：usage 字段名写错、内层不可复现、
空结果不可读、任务列表按问题分组导致位置切分把训练和评判分到不同问题、
冒烟测试用固定 keys 放过了必崩的采样器、外层 gate 只 hold out 两个窗口。最后一条最值得
带到别的插槽：**只在跨问题平均后才显现的效应，无法被只看一两个问题的 gate 提交**，
而 L1 下打平即否决，于是窄 gate 读起来就像"什么都不work"。

### 4.7 演进树搜索本身（`bench/results/metasearch-tree.md`）

这是最初那句动机的直接答案——"我插了一个树搜索算法解一个问题，我想通过演进 policy
来演进这个算法"。制品是 `priority(rank, visits, total, prior, depth, n_nodes)` 的源码，
作为 `SelectionPolicy` 插进真实的 `EraTree`；一次外层 rollout 就是一次**完整的内层树搜索**
（60 次扩展），元奖励是最优值曲线的 AUC。外层只见 `SOURCE`，`TARGET` 从头到尾没被演进
也没被 gate 过。

**先量天花板，再问搜索找到了多少。** 把种子规则自己的探索常数 `c` 在验证用的同一批
200 个实例上扫一遍：

| `c` | 0.0 | 0.25 | 0.5 | **1.0（种子）** | 2.0 |
|---|---:|---:|---:|---:|---:|
| source | +0.0214 | +0.0202 | +0.0157 | **0.0000** | −0.0552 |
| target | +0.0044 | +0.0103 | +0.0147 | **0.0000** | −0.0434 |

上游 ERA 的 `c = 1` 在两个族上都被所有更小的值打败，所以"少探索一点"是真实可找的方向；
但**两个族要的不是同一条规则**——source 在 `c → 0` 最大，target 在 `c ≈ 0.5` 最大。
于是迁移问题就是"少探索多少"，而只优化 `SOURCE` 的搜索正被拉向贪心那一端。

**三个外层种子，同样 144 次 rollout，只改 gate 的 hold-out 大小：**

| 外层种子 | gate=9 的 source 增益 | gate=60 的 source 增益 | gate=9 的 target | gate=60 的 target |
|---|---:|---:|---:|---:|
| 0 | +0.0213 | +0.0213 | +0.0049 | +0.0087 |
| 1 | **−0.0134（88/109）** | **+0.0251（146/49）** | +0.0040 | +0.0131 |
| 2 | +0.0222 | +0.0216 | +0.0151 | +0.0141 |
| **均值** | **+0.0100** | **+0.0227** | **+0.0080** | **+0.0120** |

均值 +0.0227 / +0.0120 **落在手写规则族的 Pareto 前沿上**：贪心的 source 分数配上
`c = 0.25` 的迁移，参照集里没有任何一条在两个轴上都更好。诚实的说法是——
**搜索没有超出这族手写规则，它找到了这族上的一个点**，而种子本身完全不在前沿上。
三条演进出的规则形状都是 `rank + <收缩的探索项>`，收缩方式各不相同；种子 1 那条把
PUCT 的 `sqrt(total)/(1+visits)` 换成了 UCB1 的 `sqrt(log(total)/(1+visits))`——
不是更小的常数，而是另一种探索时间表，而它恰好是三个里分数最高的。

**这一格最重要的产出是那个失败，不是那个增益。** gate 只 hold out 9 个实例
（24 tasks × 0.4），而这个地形上配对的逐实例 sd 是 0.040–0.062，9 个样本的标准误就是
0.013–0.018——**比可得的全部增益（+0.021）还宽**。种子 1 的 gate 就这样看着 hold-out
从 0.744 涨到 0.748、连提交三次，收下了一条在自己演进的族上 88 胜 109 负的规则；
它的规则写的是 `c = 1.5` 加未访问加成，方向完全相反。而 hold-out 实例在这里是**免费**的：
模型每个 rollout 调一次，rollout 数是 `rounds × workers`，与 `--tasks` 无关；60 次
hold-out 搜索花 0.28 秒。默认已改成 `--tasks 150`。

这是同一个错误在这条线上第三次出现，三个不同层级：

| 层级 | 太小的量 | 症状 | 修法 |
|---|---|---|---|
| 内层预算 | 每次内层 4 次 rollout | **故意写坏的**采样器并列第一 | 12 次 |
| 外层 gate（插槽） | hold out 2 个窗口 | 好提案恰好在这两个上打平，什么都不提交 | 4 个窗口 |
| 外层 gate（树搜索） | hold out 9 个实例 | 在自己族上会输的规则被提交了 | 150 tasks → 60 实例 |

插槽那次是**失败闭合**——窄 gate 读起来像"什么都不 work"；这次是**失败张开**——窄 gate
读起来像"这条 work"，而它并不。L1 下的 oracle 是对同一批数字的第二意见，所以拦不住。

**另一件必须记下的事：前两次跑根本没跑完。** 两次都在墙钟上限处死掉，而 completion
cache 是**空的**——一次模型调用都没返回。示例的反思调用走 `completion_for` 时没带
`thinking`，端点为一次单函数改写生成了几分钟的推理前言。`--thinking disabled` 之后
一个外层 sweep 从 >10 分钟变成 ~5 秒。之所以花了这么久才定位，是因为这个示例在计划行
和总结之间什么都不打印——`bench/metasearch_slots.py` 正是为此长出了 `on_round`，
而这个示例当时没有。**一个在总结之前不报告任何东西的运行，和一个卡死的运行无法区分。**

---

### 4.8 AlgoTune：port 跑通，实验不成立（`bench/results/metasearch-algotune.md`）

Stage 1b 第一次在线跑。**结论是一个量化的否定**，写下来是为了让下一次从噪声问题开始，
而不是从 `pip install numpy` 开始。

**port 本身没问题。** `psd_cone_projection` 一次内层搜索（4 次扩展）36 秒、4 次真实模型
调用、在 Bubblewrap 沙箱里计时拿到 **5.6× 加速**（基线 1.003×）。设计文档里列为阻塞的
东西全是安装问题：装 `bubblewrap`、`numpy`/`scipy`，再补
`cvxpy networkx numba mpmath pot scikit-learn cython`，可用任务从默认的 8 个变成
**123 / 147**（剩下 24 个要 ortools/pysat/sympy/faiss/hdbscan，装上还能更多）。147 个任务
的单位成本也全测了（中位数 2.3 秒，尾部 17.7 秒），存在 `metasearch-algotune-task-cost.json`。

**但奖励看不见规则。** 三条规则在 `psd_cone_projection` 上跨度 **0.0066**，而种子规则和
**它自己**比差 **−0.0055**——84% 是噪声。而这还是好任务：扩大跑的自带噪声检查在
`rbf_interpolation` 上跑种子规则三次得到 0.5680 / 0.5104 / 0.6178，sd **0.0537**，差 20 倍。
0.51 vs 0.62 不是计时抖动，是内层搜索**找到了不同质量的程序**。

**补全缓存关不上这个环**，而且原因在 prompt 里：`mutation_prompt` 有三个块是用实测计时
拼的（`_eval_block` 加速比摘要、`_timing_report`、`_profile_block` 的"最贵 25 行，毫秒"）。
计时抖动 → prompt 变化 → 缓存键就是 prompt → 未命中 → 采样出不同程序 → 又被重新计时。
**实测计时就是内层搜索赖以工作的反馈**，去掉它搜索就瞎了，留着它搜索就是随机的。所以
元奖励是一个对这种随机性的期望，只能靠采样估计——而按 sd 0.054 / 信号 0.007 算，2 SE
分辨需要约 **240 个配对样本**，每个 75 秒，一次验证 10 小时，而外层每次 gate 都要这个分辨率。

**预算还几乎全花在 gate 上。** 扩大到 24 训练 + 24 验证任务（两边都是默认的 6 倍）跑三个
sweep：52 次内层搜索里只有 **4 次是 rollout**，约 92% 的墙钟在评估——因为每个候选提案都
要在整个 held-out 集上打分，而每一次打分就是一次完整 ERA 搜索。

顺带确认了一件对任何有噪声的域都成立的事：`Runtime.eval_one` 按
`cache_key(artifact._signature(), task.id, env_fingerprint)` 记忆化，所以
**每个 (artifact, task) 的第一次噪声抽样会被冻结一整轮**。于是 gate 的比较是"一个冻结抽样
vs 一个新鲜抽样"，配对差的 sd 是 `sd×√2` ≈ 0.076，10 个 held-out 任务平均后 SE ≈ **0.024**,
是 0.007 信号的 **3.4 倍**。sweep 0 的提交和 sweep 2 的 oracle 否决都是抛硬币。
（这也解释了 `held_out` 三轮锁在 0.658：artifact 只变过一次，后两轮全是缓存命中。）

**要改的不是参数，是两件结构性的事**：一，8 次扩展只建出 13 个节点的树，选择规则没有杠杆——
方差主要来自模型写出什么程序；要让规则有杠杆需要几百节点的树，而一次扩展 ~10 秒。
二，奖励是采样期望而不是函数。三条降噪路径（都不免费）：`--test-shards` 调高（内层 ERA
自己的 held-out 只有 2 个 shard）、按稳定性而非成本筛任务、`profile=False` 稳住 prompt。

---

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
| P2 | `policy_source(slot, seed)` 通用门 + `seed_source` + `SLOT_PROTOCOLS` | ✅ |
| P3 | `examples/metasearch/`：合成地形、离线端到端、`--dry-run`、加入 PORTS 契约 | ✅ |
| P4a | GSM 跑批脚本 `bench/metasearch_slots.py`：演进 `task_sampler`，内层是完整的内层 `evolve()`，报告分三组（演进过的 / 同基准未见切片 / 另一个基准）各自的迁移比 | ✅ 脚本 + 离线测试 + **在线跑出结果**（`bench/results/metasearch-slots.md`） |
| P4b | AlgoTune 跑批脚本 `bench/metasearch_algotune.py`（训练/验证任务不相交、新 seed 验证、迁移比、结果 JSON） | ✅ 脚本 + 插桩测试 + **在线跑过**（`bench/results/metasearch-algotune.md`）：port 跑通（5.6× 加速），但**这个域现在测不了选择规则**，噪声是信号的 3.4 倍，见 §4.8 |
| P5 | Harbor 适配器 `_harbor.py`（§4.3）+ SWE-bench-Science / TB-Science 验证 | ✅ 适配器 + `LocalRunner` 离线端到端；`DockerRunner.verify` 已写未在线跑；**基准验证待做**（需 API + Docker + 任务数据） |
| P6 | 其余五个插槽的内置冒烟与默认种子，每个种子在真实内层 `evolve()` 里跑通 | ✅ |
| P7 | 多插槽联合演化（`ParamSlot` 的 key 空间天然支持；`SourceSlot` 需要多槽 Strategy） | 开放 |

测试：`tests/test_meta.py`（库）、`tests/test_metasearch.py`（示例）、
`tests/test_metasearch_gsm.py`（P4a 脚本）、`tests/test_metasearch_algotune.py`（P4b 脚本）、
`tests/test_harbor_domain.py`（P5 适配器）、
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
| **API 页非确定**（已踩）：`validate=staticmethod(lambda…)`、`meta_reward=auc` 作默认值时，生成的 api.md 每次带不同内存地址 | 默认值改为 `None`，函数内回退；`tests/test_api_reference` 守住 |
| **外层 `held_out_frac` 切的是问题实例**，不是内层数据 | 内层数据切分由 `Problem` 自己负责；文档在 `meta_evolve` 的 `seeds` 参数处说明 |
| 反思模型不遵守类/函数形状 | 不合格即无 diff 并计数（`invalid_proposals`）；`describe()` 给出 Protocol 签名；示例报告拒绝率 |
