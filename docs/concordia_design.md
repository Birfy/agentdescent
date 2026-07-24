# Concordia: 面向 RSI 加速的并行自演进框架

**设计文档 v0.2** · 2026-07 · 内部草案

> 工作代号 Concordia(拉丁语"合流、协调"),备选 CREDO(Concurrent Recursive Evolution with Distributed Oversight)。

> **v0.2 修订说明**:结合 2025H2–2026H1 文献做了三处实质更新——(1) 校正并扩充全部外部引用(附录 A);(2) 重写第 2 节相关工作,纳入并区分同期并发工作(FlashEvolve/SkillClaw/CoEvoSkills),把新颖性主张从"三者交点的空白"收敛到可辩护的窄化增量;(3) 修正若干训练类比的精确度(EMA vs SWA、Parameter Server 的版本语义、model soup 的同基座前提)。所有 arXiv 编号已逐条核对。

---

## 1. 问题:RSI 的吞吐瓶颈

现有递归自我改进(RSI)系统——DGM、SICA、AFlow、各类 skill 自演进循环——共享同一个串行模板:

```
while True:
    τ  = rollout(agent, task)        # 执行任务,收集轨迹
    d  = propose(τ)                  # 反思,提出改进 diff
    ok = evaluate(agent + d)         # 评测变体
    if ok: agent = merge(agent, d)   # 采纳
```

单轮迭代时间 T_iter = T_rollout + T_propose + T_eval + T_merge。对科学计算类 agentic 任务,T_rollout 是小时级(工具调用、HPC 排队),T_eval 需要跑评测集,同样昂贵。**串行 RSI 的改进速率上限是 1 diff / T_iter**,即每天个位数次改进机会。这是 RSI 慢的根因——不是提案质量不够,是提案吞吐不够。

> **对该前提的诚实声明**:"吞吐而非提案质量是 RSI 的绑定约束"是一个**有争议的立场**,并非社区共识。近期 RSI 综述(arXiv 2607.07663)恰恰主张绑定约束在别处——**验证器/评测者的可靠性**与**人类方向设定**,而非并行吞吐。本框架不把该前提当作已被引用背书的事实,而是把它作为一个待检验的工程假设:只有当(a) 提案池中确有足量高价值 diff 因串行排队而被浪费,且 (b) 评测者可靠性不先行崩溃(这正是第 6 节 L0 冻结 + 第 5.3 节 AuditScheduler 要守住的),吞吐红利才成立。RQ1/RQ3 的设计即为区分这两种世界。若评测可靠性才是瓶颈,则本框架的价值退化为"更快地把评测噪声放大"——这是必须正面回应的失败模式,而非回避的。

模型训练在十年前面对过同构的问题:单卡 SGD 太慢。答案不是更好的单步优化器,而是**并行化 + 异步化 + 一套处理由此产生的 staleness/长尾/调度问题的系统理论**(DP/TP/PP、parameter server、decoupled PPO、partial rollout)。本文档把这套理论系统性地移植到 RSI:让 N 个 worker 并行产生改进反馈,通过一个聚合器合并进共享的 agent 定义,目标是把改进速率从 O(1/T_iter) 提到 O(N/T_iter) 量级。

**中心类比**:

| 模型训练 | 并行 RSI(本框架) |
|---|---|
| 参数张量 θ | Evolvable artifact 库(skill/prompt/harness/verifier) |
| 梯度 g | diff + evidence card |
| Parameter Server | 版本化 Ledger(git-backed) |
| Optimizer step | Aggregator 的 merge 决策 |
| Learning rate | 接受阈值(随版本退火) |
| Per-param 自适应 lr(Adam) | per-artifact Beta 后验检验 |
| DP / TP / PP | 任务分片 / artifact 结构切分 / 依赖链分段 |
| Staleness / decoupled PPO | per-diff η + rebase 重验 |
| Partial rollout | turn-level checkpoint 续跑 |
| Gradient clipping / trust region | diff 规模上限 + verifier 审计 |
| EMA(权重滑动平均) | stable/dev 双分支(慢分支 = 对 dev 的滑动确认) |
| 训练代码本身(不可被训练修改) | L0 冻结层 |

类比在一处**必然失效**并因此定义了本框架的核心技术问题:梯度可加,diff 不可加。聚合不是求平均,而是**冲突消解 + 统计接受 + 事务提交**。第 4 节专门处理。

**类比的三处精度边界**(避免过度声张同构):

- **Parameter Server → Ledger**:经典 PS(Li et al., OSDI 2014)是"可调一致性的分片可变共享状态",**没有提交历史**。本框架的 git-backed Ledger 额外携带版本向量与 commit 历史——这不是 PS 的性质,而恰恰是 rebase / staleness 处理(第 4.2 节)得以成立的前提。即"多出来的语义"是特性而非包袱,应作为差异点主张,而非当作 PS 的等价物。
- **model soup → 候选融合**:model soup(2203.05482)之所以成立依赖线性模式连通性(所有权重同一预训练初始化)。diff 融合的对应前提是**共享 base_version**;跨 base 的 diff 融合更接近 task arithmetic / 模型合并而非 soup,须先 rebase 到共同 head 再融合(第 4.3 节的融合发生在同桶、同 base 校正之后)。
- **EMA/SWA → 双分支**:采用 **EMA 语义**(慢分支对快分支的指数滑动确认),而非 SWA(SWA 的目标是更宽极小值/泛化,是 SGD 轨迹上的均匀平均,机制不同)。第 4.5 节据此表述。

---

## 2. 与相关工作的关系

### 2.1 三条既有线索(各缺一角)

- **DGM / archive 式开放演进**(2505.22954):并行,但并行的是 *fork*——变体在不断生长的 archive/树上各自分叉,改动不回流合并。算力换多样性,不换收敛速度;且生产系统无法给用户随机分配 fork。这是本框架最直接的对照基线("并行探索但从不合并")。
- **AlphaEvolve**(2506.13131):异步管线成熟(controller / prompt sampler / evaluator pool / program DB 分离,diff-based 变异,program DB 受 MAP-Elites + 岛屿模型启发),但岛屿间只**迁移个体**不合并 diff,且 evaluator 由人给定、**冻结**、不讨论评测者自身演进。其开源复现(OpenEvolve)与后续(ShinkaEvolve 2509.19349;CodeEvolve 2510.14150 引入多亲本 crossover——最接近"跨岛合并"的机制,值得作为融合环节的对照)同样不做并发 worker 的 diff 级合并。
- **组件自演进线**(ExpeL 2308.10144 / AWM 2409.07429 / AFlow 2410.10762 / GEPA 2507.19457 / ACE 2510.04618 / SICA 2504.15228 等):定义了"演进什么"(记忆/工作流/prompt/上下文/自身代码),但全部**单 agent 串行更新**,无并行、无合并、无 staleness。
- **异步 RL 系统**(AReaL 2505.24298 / ROLL Flash 2510.11345 / Kimi partial rollout(k1.5 2501.12599、K2 2507.20534、Kimi-Researcher 技术博客) / APRIL 2509.18521):有完整的 staleness 与长尾理论(decoupled PPO objective、asynchronous ratio、partial rollout),但作用对象是**参数**不是语义 artifact。

### 2.2 同期并发工作(2026):新颖性主张必须收窄

截至 2026H1,已出现与本框架三个支柱分别高度重叠的并发工作。**"三者交点的空白"这一表述已不成立,必须收敛为可辩护的窄化增量,并在投稿中显式区分:**

- **FlashEvolve**(2605.08520,《Accelerating Agent Self-Evolution with Asynchronous Stage Orchestration》):**最具威胁的先行工作**。用异步 worker 池 + 队列取代同步演进循环(rollout / propose / eval / pool-update 各阶段流水重叠),并**明确把异步 RL 的 staleness 理论移植到非参数的文本/代码 artifact**(prompt / context / harness),论点与本框架第 4.2 节几乎同构:文本 artifact 可语义审查,不同于权重空间的不透明 staleness;定义了 Full / Guarded / Reflective 三档 staleness 策略。**这实质上抢先占据了"为语义 artifact 建立 staleness 理论"这一本框架原以为最独特的贡献。** 与本框架的差别:FlashEvolve 做的是**版本门控的 pool 更新 + LLM 反思重放**,**不做 git-backed 的 diff 级合并、冲突消解、也无 parameter-server 化的聚合器**——本框架的差异点须收缩到"并发多 worker 的 diff 级合并 + 冲突消解 + 事务提交"这一层,而非 staleness-for-text 本身。
- **SkillClaw**(2604.08377,《Let Skills Evolve Collectively with Agentic Evolver》):抢先占据"并行 + 可合并"支柱。多用户/多来源的轨迹汇入一个共享 skill 仓库,由 evolver 聚合、去重、交叉授粉后分发回全体——即 fork-free、merge-based 的多源聚合。差别:其聚合是**周期性串行 evolve-server**(定时触发),非并发异步 worker;abstract 未给出 staleness / off-policy / 冲突消解的形式化,也无 git / parameter-server 框架。本框架的增量是"把周期串行聚合替换为并发 + staleness 受控 + 冲突消解的合并"。
- **CoEvoSkills**(2604.01687,《Self-Evolving Agent Skills via Co-Evolutionary Verification》):抢先占据"可演进验证器"支柱。Skill Generator 与信息隔离的 Surrogate Verifier 协同演进、在 SkillsBench 上评测——但显式为**串行单 worker**,无并行、无合并、无 staleness。本框架把该 actor-verifier 共演化嵌入并行受审框架(第 6 节 L1 受审演进 + 第 5.3 节 AuditScheduler)。

### 2.3 收窄后的新颖性陈述(诚实版)

本框架三支柱**逐一都已被同期工作单独覆盖**:并行+合并→SkillClaw;语义 artifact 的 staleness→FlashEvolve;可演进验证器→CoEvoSkills。**真正尚未被占据的,是把三者作为单一系统同时执行的那个特定组合**:并发 N-worker 的**diff 级合并 + 冲突消解**,置于一个**显式 git-backed 版本化 Ledger(parameter-server 语义)** 之上,并对该合并施加**形式化的 per-diff staleness 模型**。检索范围内没有任何单一工作做"并发 worker 之上、带冲突消解、受形式化 staleness 约束的 diff 级 MERGE"。

这是一个**可辩护但窄**的增量——是工程/系统层的综合,而非新的理论原语。相应地,本框架的差异化表面应聚焦在:(1) 聚合器作为"离散空间 optimizer"的冲突消解+统计接受+事务提交管线(第 4 节),(2) git-ledger-of-diffs 的版本向量与 CAS/2PC 提交,(3) 三重长尾的分治(第 5 节)。投稿时**必须显式引用并逐条区分 FlashEvolve / SkillClaw / CoEvoSkills**,不可再主张"空白"。

---

## 3. 系统架构

### 3.1 总览

```
                 ┌──────────────────────────────────────────┐
                 │              TaskScheduler (UCB)          │
                 └───────┬──────────────────────────────────┘
                    lease│tasks
       ┌─────────────────┼─────────────────┐
       ▼                 ▼                 ▼
  Worker 1          Worker 2   ...    Worker N        ← 各持 Ledger 快照 V_i
  (rollout +        (rollout +        (rollout +        中途可热更新(in-flight)
   propose)          propose)          propose)         超时任务 checkpoint 入 ResumeQueue
       │                 │                 │
       └────── diff + evidence card + base_version ──────┐
                                                          ▼
                                              EvidenceBuffer(按 artifact 分桶)
                                                          │
                                              Aggregator(per-artifact 触发)
                                                 │ staleness 过滤 → rebase 重验
                                                 │ 冲突消解 → 候选融合锦标赛
                                                 │ Beta 后验统计检验
                                                 ▼ CAS / 2PC commit
                                              Ledger(git-backed, 版本向量)
                                                 │
                                    增量广播(只推变更 artifact)→ Workers
                                                 │
                                     AuditScheduler(Ĝ)→ Oracle 抽审
```

四个核心组件:**Evolvable** 抽象、**Ledger** 存储、**Aggregator** 聚合器、三合一 **Scheduler**。verifier 体系(rule/learned/oracle 三层)复用 VeriEvo 现有设计,作为 Aggregator 的评测后端。

### 3.2 Evolvable:统一演进单元

任何满足以下接口的 artifact 均可注册进演进循环:

```python
class Evolvable(Protocol):
    id: str
    version: int
    contract: Contract          # 对外接口声明(输入/输出 schema、副作用)
    blast_radius: float         # 影响任务面估计,由依赖图+触发统计自动计算

    def diff(self, other) -> Diff
    def apply(self, diff: Diff) -> "Evolvable"
    def cheap_eval(self, evidence: EvidenceCard) -> float   # rule/learned 层
    def full_eval(self, task_set) -> Metrics                # held-out 评测
```

实例:单个 skill、prompt 模板、few-shot 库、harness 模块(上下文管理器、工具路由)、learned verifier、调度器超参。**"演进什么"由注册决定,不由框架硬编码**——这是通用性的来源。

`blast_radius` 自动分层(见第 6 节),不靠人工标注 artifact 类型:一个被所有任务触发的 skill 会被自动归入慢层,一个只影响单一任务簇的 harness 补丁可以走快层。

### 3.3 Evidence Card:diff 的"梯度元数据"

每个 diff 必须携带:

```yaml
diff_id: ...
base_version: {mol-engine: 47, harness.ctx: 3}   # 版本向量,只含读写到的 artifact
touched: [mol-engine]
contract_breaking: false
evidence:
  - task_id, trajectory_ref, failure_trace
  - before_after_delta: {...}                     # 提案 worker 的本地评测
  - version_annotations: [...]                    # 轨迹各 turn 使用的版本(in-flight 更新时)
cost: {tokens: ..., wallclock: ...}
```

Evidence card 的生命周期比 diff 长:diff 因 staleness 被废弃时,evidence 沉淀回 trajectory pool 供后续复用——这是相对梯度(丢了就丢了)的结构性优势,也是"artifact 空间比参数空间耐 stale"假设(RQ2)的机制基础。

---

## 4. 聚合器:离散空间的 optimizer

### 4.1 触发与分桶

按 artifact 分桶。触发条件 `len(bucket) ≥ B or wait ≥ T_max`:热点 artifact 靠 B 触发(批量聚合),冷点靠 T_max 触发(定期审,防止饿死)。热点聚合期间对该 artifact 加租约锁;冷点无竞争,乐观 CAS。

### 4.2 Staleness:per-diff η 与三段处理

定义 per-diff staleness η(d) = max over touched artifacts (head_version − base_version)。借 ROLL Flash 的 per-sample asynchronous ratio 思路,而非 AReaL 的 batch 平均新鲜度——因为 diff 是离散个体,per-diff 约束更自然。

- **η = 0**:直接进入 merge 评估。
- **0 < η ≤ α**:rebase + 快速重验。语法冲突走 git 三方合并;语义上,把 evidence 中的失败案例在当前 head 上用 rule/learned verifier 廉价重跑,delta 仍成立才保留。这是 AReaL **decoupled PPO objective**(解耦 behavior policy π_behav 与 proximal policy π_prox)的离散对应物:"旧策略采的数据,在当前策略上校正后再用"。注:decoupled objective 本身由更早工作提出,AReaL 的贡献是**有界 staleness 的系统化**(其消融显示 max-staleness η≤8 时性能落在 η=0 oracle 的 ±1 内,η=∞ 显著劣化——这是 RQ2 曲线的直接参照)。近期亦有指出异步系统会丢失历史 logits、破坏 decoupled 校正语义的失败模式分析(2605.12070),本框架的"廉价重验 + 显式版本标注"是其离散侧的规避手段。
- **η > α**:废弃 diff,沉淀 evidence。

α 按 artifact 热度自适应:头部 artifact 版本迭代快,α 给大(~5);尾部几乎不动,α=1。契约破坏型 diff 强制 α=0(跨契约 rebase 的重验成本高于重提)。

### 4.3 冲突消解与候选融合

同桶多 diff 的处理管线:

1. **语法检测**:git hunk 重叠。
2. **语义检测**:LLM 判断是否矛盾(如两个 diff 对同一超参给出反向建议)。矛盾对在共享验证子集上各自评测,类比 PCGrad 投影掉负向者(丢弃或串行化)。
3. **融合**:互补的候选由 merge-LLM 合成融合版,与各原候选一起进 held-out 锦标赛(EvoSkill ELO 机制的直接复用),取最优 commit。类比 model soup:多个局部改进的融合往往优于任一单体。

### 4.4 接受准则:统计检验而非阈值

维护每个 artifact 的 Beta(成功, 失败)后验,merge 条件为 **P(Δ > 0) > 1 − δ**,而非点估计过线。效果等价于 Adam 的 per-parameter 自适应步长:证据稀疏的尾部 artifact 自动获得更保守的有效更新——这是对抗"尾部被单条噪声反馈带偏"的核心机制。附加约束:diff 行数上限(trust region)、δ 随版本号退火(lr decay)。

### 4.5 双分支

stable / dev 两条分支。dev 承接全部通过检验的 commit;stable 周期性吸收 dev 中经受住 K 轮考验(无回归报告)的改动。类比 **EMA**(慢分支对快分支的滑动确认,而非 SWA 的均匀轨迹平均——见第 1 节精度边界):防止单轮噪声污染在线服务版本。生产 worker 挂 stable,探索 worker 挂 dev。

---

## 5. 三重长尾与调度

长尾在本框架里是三个独立问题,分别由三个机制处理:

### 5.1 L-traj:轨迹时长长尾(系统层)

Agentic 任务时长呈重尾(HPC 排队、多轮工具调用)。同步屏障不可行,采用 **turn-level checkpoint 续跑**(Kimi-Researcher / K2 的 partial rollout 移植):任务超时(> 2×P90)在 turn 边界 checkpoint(对话历史 + 外部状态句柄如 job ID),入 ResumeQueue,续跑时加载**最新** Ledger。

**免费 A/B 信号**:续跑轨迹前半段用 V_k、后半段用 V_{k+1},两段成败差本身就是 V_{k+1} 相关 diff 的一条 evidence,显式回流 EvidenceBuffer。轨迹按 turn 打版本向量标注,归因只采信对应版本段落。

### 5.2 L-task:任务分布长尾(数据层)

Artifact 触发频率近似 Zipf。头部 skill 反馈过剩、尾部证据饥饿——这是参数空间没有的问题(梯度自动流向所有参数,diff 只流向被触发的 artifact)。

**TaskScheduler = UCB over (任务簇 × artifact)**:

```
score(s) = 预期学习价值(s) + c·√(ln T / n_s)
```

n_s 为 artifact 近期 evidence 数。尾部自动获得探索加成;头部中近期 delta 趋零者(已收敛)自动降权。叠加难度过滤:全对/全错任务簇无学习信号,降权(对应 GRPO 的 zero-advantage group 过滤)。

**尾部金丝雀集**:held-out 评测强制包含尾部任务样本,防止头部优化(尤其改共享约定的)悄悄回归尾部任务——skill 空间的灾难性遗忘,不设金丝雀测不出来。

### 5.3 L-value:diff 价值长尾(信号层)

绝大多数 diff 是边际修补,极少数是高价值重构。**AuditScheduler** 按 Ĝ 调度分配 oracle 预算:

```
priority(d) ∝ blast_radius(d) × uncertainty_learned_verifier(d) / trust(artifact)
```

头部 artifact 的小修由 rule/learned 层放行;高影响面改动与尾部 artifact 强制过 oracle。聚合器自身的 merge 决策同样被抽审——审计闭环覆盖优化器本身,防自我污染。

---

## 6. 分层治理:blast radius 决定并行度

所有 Evolvable 按 blast_radius 自动落入三层,层决定聚合协议与更新频率(双时间尺度):

| 层 | 内容 | 更新频率 | 协议 |
|---|---|---|---|
| **L2 快层** | 局部 skill、prompt、few-shot 库 | 小时级 | 全异步细粒度合并(第 4 节全套) |
| **L1 慢层** | harness 模块、上下文策略、工具路由、learned verifier | 天/周级 | 串行评估 + 三段发布 |
| **L0 冻结** | oracle 接口、audit 预算机制、merge 权限表、安全约束 | 仅人工 | 演进循环只读 |

**L1 专属机制**:

1. **串行化在途改动**:任何时刻至多一个 L1 diff 在评估中。原因是归因——L2 靠"只有触发该 artifact 的任务受影响"天然隔离,L1 无此隔离,只能时间串行。
2. **离线反事实重放优先**:上线前在冻结的 trajectory pool 上重放(同任务、同 L2 版本、只换 L1 artifact),即对 harness 改动做 off-policy evaluation,成本远低于在线灰度。
3. **三段发布**:离线重放 → 金丝雀(5–10% worker,必须覆盖尾部任务簇)→ 全量。
4. (可选)**PBT 式种群**:算力充裕时,L1 维护 3–5 个变体各带一撮 worker,周期性 exploit+explore,与 ELO 锦标赛同构。

**契约机制**(并发边界):diff 分契约保持型与契约破坏型。前者走各层常规协议;后者必须携带原子适配事务(harness diff + 受影响 skill 的适配 diff,2PC 提交),对应 semver major bump,Ledger 拒收声明依赖旧 major 的新 diff。merge 阈值对契约破坏型加惩罚,使系统结构性偏好"能在实现层解决的不动接口"——类比模型并行中 resharding 的高成本使系统偏好不改并行策略的优化。

**L0 的必要性**:没有冻结层,自指闭环迟早自我污染(verifier 学会给自己放水时不可检测)。AlphaEvolve 冻结 evaluator 是同一判断;本框架把冻结范围最小化到"审计与权限机制",让 verifier 的可学习部分保留在 L1 受审演进——这是与 VeriEvo actor-verifier 共演化的衔接点。

---

## 7. 跨层归因

失败归因(skill / harness / verifier 误判)不能靠 LLM 单独判断——它系统性偏向"怪 skill"(局部 diff 便宜)。机制:对 Ĝ 判定为高价值的失败(反复出现、跨任务簇),在 {旧L1×新L2, 新L1×旧L2} 组合上反事实重放,做最小因子分解。归因预算纳入 AuditScheduler 统一调度。

---

## 8. 并行范式备忘(DP/TP/PP 映射)

- **DP(默认)**:worker 持同一快照,任务分片,diff 回流聚合。本文档主线。
- **TP(可选)**:高热 artifact 按内部结构切分(frontmatter/触发条件/操作步骤/反例库),worker 按 section 授权,按构造无冲突,合并退化为拼接 + 轻量一致性 reviewer(类比 all-reduce 开销)。仅对少数超热 artifact 启用。
- **PP(可选)**:沿 artifact 依赖链(lit-review → mol-engine → hpc-submit)分段,下游失败 blame 回传上游(反向传播穿过 pipeline),各 stage 专属 worker 演化。与第 7 节归因机制共用反事实重放设施。

---

## 9. 实验计划

**RQ1 聚合 vs 分叉**:同算力预算下,merge 式(Concordia)对比 archive 式(DGM 复现)与串行基线,指标为达到目标性能的 wallclock / diff 数。打 P1(聚合问题)。

**RQ2 staleness 容忍度**:α ∈ {0, 1, 5, ∞} 扫描,复现 AReaL 式曲线(适度 staleness 可接受、无界劣化),并检验中心假设——**语义 artifact 空间对 staleness 的容忍度显著高于参数空间**(机制:diff 可 rebase 可重验,梯度不可)。若成立,是 RSI 相对模型 RL 的结构性优势,独立成节。

**RQ3 治理消融**(故意做坏,展示失败模式):
- 去掉 L0 冻结 → 预期 verifier 污染;
- 去掉 L1 串行化 → 预期归因混乱、回归定位失败;
- 去掉尾部保护(UCB 加成 + 金丝雀)→ 预期尾部任务性能坍塌。

负面消融若真实展示出失败模式,论证力强于正向结果。

评测环境:**SkillsBench**(2602.12670,87 任务 / 11 领域的公开 agent-skill 基准)为主评测;**ChemClaw**(公开化学 skills 库,github.com/InternScience/ChemClaw——注意其性质是 skills 集合而非现成基准,须自行在其之上构造任务实例与判分)提供科学计算域的重尾任务源。串行 RSI 基线:**CoEvoSkills**(2604.01687)复现,以及 DGM(archive 式)与单 worker Concordia 消融。(命名提示:公开文献中 "EvoSkill" 2603.02766 与 "CoEvoSkills/EvoSkills" 2604.01687 为两篇不同工作,投稿引用时须消歧;若本框架内部的 EvoSkill/VeriEvo 指自研系统,应在正文首次出现处注明"内部系统",以免与同名公开工作混淆。)

---

## 10. 落地路线(三期)

| 期 | 内容 | 交付 |
|---|---|---|
| 一期(~2周) | git-backed Ledger + per-artifact 版本向量 + 乐观 CAS + 定时 merge,η≤1 硬约束(近似同步 DP) | 最小可用并行 RSI,RQ1 的 merge 侧 |
| 二期(~3周) | 放开 α、rebase 重验、Beta 后验接受、UCB 任务调度、尾部金丝雀 | RQ2 全部,RQ3 尾部消融 |
| 三期(~3周) | turn-level checkpoint 续跑、Ĝ audit 调度、L1 三段发布 + 契约事务 | RQ3 其余消融,完整系统 |

工程基座:OpenClaw-RL / slime-ascend 现有 rollout 设施;Ledger 用裸 git + 轻量元数据 DB;merge-LLM 与语义冲突检测复用现有 verifier 推理端点。

---

## 附录 A:关键参考(v0.2 逐条核对)

所有 arXiv 编号已核对到 abstract 页;标题与关键贡献按原文校正。

### A.1 异步 RL / staleness / partial rollout(本框架移植的源理论)

- **AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning**(arXiv 2505.24298)— **decoupled PPO objective**(解耦 π_behav 与 π_prox);有界 staleness 消融(η≤8 落在 η=0 oracle 的 ±1 内,η=∞ 劣化)。是 RQ2 的主要对照。
- **ROLL Flash: Accelerating RLVR and Agentic Training with Asynchrony**(arXiv 2510.11345)— 两条原则:**asynchronous ratio**(全局滞后预算,决定 generation-pool 大小)+ **fine-grained parallelism**(sample-level 生命周期控制)。⚠ 原文无"per-sample asynchronous ratio"这一合并说法,勿引。
- **APRIL: Active Partial Rollouts in Reinforcement Learning to Tame Long-tail Generation**(arXiv 2509.18521)— 主动 partial rollout,回收未完成响应续跑以驯服长尾(标题为复数 Rollouts)。
- **Kimi k1.5**(arXiv 2501.12599)/ **Kimi K2**(arXiv 2507.20534)— partial rollout;**Kimi-Researcher**("turn-level partial rollout")仅为技术博客(moonshotai.github.io/Kimi-Researcher),**无 arXiv**,应作博客引用。
- *(可选强化 RQ2)* Stable Asynchrony / VCPO(2602.17616)— 由有效样本量动态调 LR + 最小方差基线,治 stale rollout 的重尾 IS 权重;Staleness–LR Scaling Laws for Asynchronous RLHF(2607.01083)— per-step bias ∝ S·η 的两约束稳定性规则,为"适度 staleness 可接受"提供定量背书;A-3PO(2512.06547)— staleness-aware 近似 π_prox;Missing Old Logits in Asynchronous Agentic RL(2605.12070)— decoupled 校正的失败模式(见第 4.2 节)。

### A.2 自演进 / RSI(演进对象一侧)

- **Darwin Gödel Machine**(arXiv 2505.22954,Sakana/UBC)— archive/树式自指演进,**fork 不合并**;本框架最直接对照基线。
- **组件自演进**:ExpeL(2308.10144)、Agent Workflow Memory(2409.07429)、AFlow(2410.10762)、GEPA(2507.19457,prompt 反射演化 + Pareto)、ACE(2510.04618,上下文/playbook 演化)、SICA(2504.15228,自改代码)——均单 agent 串行。
- **综述**:A Survey of Self-Evolving Agents(arXiv 2507.21046)— what/when/how/where 分类 + 组件演进。治理"三律"(Endure/Excel/Evolve:安全 > 性能 > 自主演进)出自**另一篇**综述 A Comprehensive Survey of Self-Evolving AI Agents(arXiv 2508.07407,Fang et al.),v0.1 曾误挂到 2507.21046,已更正。RSI 综述(arXiv 2607.07663)主张瓶颈在验证器可靠性与人类方向设定,是第 1 节前提的反方,须正面回应。

### A.3 同期并发工作(2026,新颖性对照,必须引用并区分)

- **FlashEvolve: Accelerating Agent Self-Evolution with Asynchronous Stage Orchestration**(arXiv 2605.08520)— 异步阶段编排 + 语义 artifact 的 staleness 策略(Full/Guarded/Reflective);最接近的先行工作,见第 2.2 节。
- **SkillClaw: Let Skills Evolve Collectively with Agentic Evolver**(arXiv 2604.08377)— 多源共享 skill 仓库的聚合/去重/交叉授粉(周期串行 evolver)。
- **CoEvoSkills: Self-Evolving Agent Skills via Co-Evolutionary Verification**(arXiv 2604.01687)— generator 与隔离 surrogate verifier 串行协同演化;在 SkillsBench 评测。
- **SkillsBench**(arXiv 2602.12670)— 公开 agent-skill 基准(87 任务 / 11 领域)。

### A.4 演化式程序搜索

- **AlphaEvolve: A coding agent for scientific and algorithmic discovery**(arXiv 2506.13131,DeepMind)— 分布式异步管线、diff-based 变异、program DB 受 MAP-Elites + 岛屿模型启发、evaluator 人给定且冻结。
- **MAP-Elites**(Mouret & Clune,arXiv 1504.04909);岛屿模型(Tanese 1989;Whitley et al. 1999)。
- 后续:**ShinkaEvolve**(arXiv 2509.19349,Sakana,样本高效)、**CodeEvolve**(arXiv 2510.14150,多亲本 crossover ≈ 跨岛合并)、**OpenEvolve**(开源复现,github.com/algorithmicsuperintelligence/openevolve)。

### A.5 训练类比的规范引用(见第 1 节精度边界)

- Parameter Server(Li et al., OSDI 2014)— 注:无版本历史语义。
- Model soups(arXiv 2203.05482)— 需同预训练基座。
- SWA(arXiv 1803.05407)vs EMA(Polyak–Ruppert;Mean Teacher 2017)— 本框架取 EMA 语义。
- PBT(arXiv 1711.09846);PCGrad / Gradient Surgery(arXiv 2001.06782)。
- GRPO zero-advantage(DeepSeekMath,arXiv 2402.03300;DeepSeek-R1,2501.12948)。
- Trust region:TRPO(arXiv 1502.05477)/ PPO(arXiv 1707.06347)— 注:约束的是分布/策略变化,非原始编辑行数;"diff 行数上限"是粗代理。
- Adam(arXiv 1412.6980)— per-parameter 自适应步长,与 per-artifact Beta 后验方向同构、机制不同(梯度方差 vs 贝叶斯证据计数)。
