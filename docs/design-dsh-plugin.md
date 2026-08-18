# 设计文档：把 AgentDescent 做成 DeepSeek Harness 插件

> 目标：让 **DeepSeek Harness（`dsh`）** 装上 AgentDescent 之后，它自己会变好 ——
> 用户平时怎么用它，它就在那条真实轨迹上训练自己的 skill、prompt、preset 乃至插件代码。
>
> **状态：M0–M4 已落地。** 本文写*为什么*这样切、边界划在哪、以及每一步的验收标准。
>
> M4 含 `prompt:<section>`、`preset:<name>` 两个适配器，以及写进 session log 的持久化历史
> （`evolution/commit` / `evolution/run-ended`，log-only，不进模型上下文）。
> Web UI 的 conversation node **装配逻辑**也已落地并有测试（`plugin/src/web/node.ts`）——
> 客户端引擎的契约就是「Definition 是事件上的纯函数」，所以这一半不需要浏览器。
> **只剩那个 keyed React 渲染器**：它要浏览器端 bundle，这个环境跑不了，所以没有盲写。
>
> 已落地的：[`agentdescent.backends.dsh()`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/backends.py)（把 dsh 当 agent 驱动）、
> [`agentdescent.dsh.locate`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/dsh/locate.py)（按 dsh 自己的 rank 顺序解析 skill 根）、
> [`rewards.command()`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/rewards.py)（§8.1 的 B 档最低一级）、
> `LAYOUTS['dsh_skill']`，以及 runner
> [`examples/dsh/evolve_dsh_skill.py`](https://github.com/Birfy/agentdescent/blob/main/examples/dsh/evolve_dsh_skill.py)；
> M1 的 [`plugin/`](https://github.com/Birfy/agentdescent/tree/main/plugin)（`ctx.evolution` seam、
> `skill:<name>` 适配器、双向桥、引擎 provider、`/evolve` 与 `evolve_*` 工具）和
> [`agentdescent/dsh/daemon.py`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/dsh/daemon.py)。
> 进程内 rollout runner（`ctx.agents.create` + `agent.ctx` 作用域绑定 + 拥有到 idle 的区间）
> 也已落地。
>
> 另有：L1 提交的**人审队列**（`/evolve pending|approve|reject`）、
> 五个 scorer（`gold` / `command` / `judge` / `replay-pairwise` / `efficiency`）、
> 以及可挂载的 bundle 行本身（`plugin/src/plugin.ts` + `cordis.patch.yml`）。
>
> **演化 dsh 插件自身**：Python 侧
> [`agentdescent.dsh.plugin`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/dsh/plugin.py)
> + [`examples/dsh/evolve_dsh_plugin.py`](https://github.com/Birfy/agentdescent/blob/main/examples/dsh/evolve_dsh_plugin.py)，
> TS 侧 `plugin:<pkg>` 适配器。冻结集（`cordis.patch.yml` / `package.json` / 测试）
> **两侧各判一次**，且调用方的 `frozen=` 只能**追加**不能替换 —— 详见 §9.1。
>
> 对照阅读：[演化一个目录](directory-evolution.md)（本设计复用的底座）、
> [治理 L0/L1/L2](governance.md)、[Where rollouts run](execution.md)、
> [Agents & LLMs](agents.md)。

---

## 0. 结论先行

| 决策 | 结论 | 一句话理由 |
|---|---|---|
| **主目标（已定）** | **让用户自己那台机器上的 dsh 自我演化** —— 参数是用户的 profile，不是 benchmark | 这一条决定了后面所有取舍：数据和目标都来自用户自己，而不是来自某个榜（§2） |
| **TS 包放哪（已定）** | **本仓库开 `plugin/`（monorepo）** | 一处版本、一处 CI、一次 PR 改两边；代价是分发只能走 npm / tarball（§11） |
| 做成什么形态 | **一个 npm bundle（TS）+ 一个 Python sidecar**，不是纯 MCP、不是纯 Python 脚本 | 只有原生 Cordis 插件才够得着 `ctx.skills` / `ctx.systemPrompt` / `ctx.approval` |
| 引擎跑在哪 | **Python 侧**（ledger / aggregator / staleness / governance 一行不动） | 16k 行经过测量的优化器，没有任何理由重写成 TS |
| rollout 跑在哪 | **默认跑在 harness 进程内**（fork session + `agent.ctx` 作用域注册），重活切到 SDK 子进程 | 要测的就是那个真实 harness：真工具、真沙箱、真审批、真 prompt 装配 |
| 参数是什么 | dsh 里**已经是插件的那些东西**：skill、prompt section、preset、插件代码 | dsh 把 agent 每一层都做成了可替换模块，等于第一次把「agent 的参数」变成**可寻址**的 |
| 奖励从哪来 | **用户说了算**：自带数据集 + gold ▸ 自定义目标（命令退出码 / judge 提示 / Python callable）▸ 都没有才退到重放对照 + 效率 | 自我演化**不等于**没有标注；兜底路径只是默认值，不是设计中心（§2.2） |
| 安全边界 | L0 冻结集在 **Python 和 TS 两侧各判一次**，L1 走 `ctx.approval` 人审 | 自改 harness 的系统，冻结集不能只有一道门 |
| 先做哪一步 | **M0：离线自我演化**，对着用户真实 profile 跑一轮，纯 Python 零 TS | 三天出第一个真实数字，且已经是产品的缩微版 |

---

## 1. 两边各是什么

**DeepSeek Harness**（2026-08-13 developer preview，MIT，`deepseek-ai/deepseek-harness`）的口号是
「model + harness = agent」和「everything is a plugin」。底下是 Cordis：插件向共享
`ctx` 贡献**服务、类型化事件、可撤销的 effect**。模型适配器、工具注册表、session log、
agent loop 本身，全都是插件行，全都能从配置里换掉。一个跑起来的 `dsh` 是启动时按
profile → bundle → patch 层序组合出来的插件树。

**AgentDescent** 的口号是「gradient descent，但参数是 agent」。N 个 worker 并行提 diff，
aggregator 做 staleness 过滤 → 冲突消解 → fusion 锦标赛 → Beta 后验接受 → 事务提交。
`evolve()` 一个入口，`Strategy` 说明*什么在演化*，`run`/`reward`/`propose` 说明*演化规则*。

**它们互补的点只有一个，但很关键**：优化器需要参数是可寻址、可替换、可回滚的。
在别的 agent 框架里，「harness」是一坨写死的代码，你只能演化最外面那层 prompt。
dsh 把每一层都做成了带 disposer 的插件行 —— 于是 skill、system prompt section、
工具集、subagent preset、甚至一个插件包的源码，全都成了**可以下降的参数**。

反过来，dsh 缺的正是 AgentDescent 有的：一个把「这次改动到底有没有更好」这件事
做成统计判定而不是感觉判定的优化器。

> 一句话定位：**`model + harness + descent = 一个会变好的 agent`**。

---

## 2. 目标已定：让用户的 dsh 自我演化

这一节是整篇设计的约束源。目标不是「用 dsh 当 rollout runtime 去刷分」，而是
**用户机器上那一份 profile 越用越好**。后面每一个取舍都从这里推出来。

### 2.1 这个目标改变了什么

| | 刷 benchmark（未选） | **自我演化（已选）** |
|---|---|---|
| 任务从哪来 | 现成公开数据集，量大、分布固定 | **用户给什么就是什么**：自带数据集、真实会话、合成题，或三者混合；分布会漂 |
| 奖励怎么算 | benchmark 自带的指标 | **用户给什么就用什么**：gold、自定义判据，都没给才兜底（§2.2） |
| held-out | 几千条，统计功效充足 | 自带数据集时正常；只靠真实会话时可能只有几十条（§2.3） |
| 交付形态 | 跑完一个 run，出一张表 | **常驻、空闲触发、不打断人**（异步运行时从收尾功能升级为核心） |
| 首要 artifact | 随 benchmark 定 | `skill:*` 和 `prompt:*` —— 用户 harness 实际会积累的东西 |
| 风险重心 | 成本 | **同意、可撤销、改坏了怎么办**（§2.5、§9） |

一个直接后果：**里程碑重排**。原来放在最后的异步运行时提到 M3，原来靠前的
L1 插件代码演化推到 M4 —— 「后台悄悄变好」比「能改自己的代码」更早成为产品（§12）。

### 2.2 奖励从哪来：用户说了算

**自我演化不等于没有标注。** 这里最容易犯的错，是替用户断定他没有目标 —— 实际上
大多数人是有的，只是形式不是 benchmark。三档，能用高的就别用低的：

| 档 | 用户给了什么 | 怎么算 reward | 信号强度 |
|---|---|---|---|
| **A** | **自带数据集（有 gold）** —— QA 对、回归用例、「以前答错过的清单」 | 直接复用 `agentdescent.rewards`：`exact_match` / `contains` / `last_number` / `numeric_close`，或 `skill.SCORERS` 里的名字 | 最强 |
| **B** | **自定义目标（无 gold，但有判据）** —— 「SQL 得在我库上跑过」「别问我问题、别动 migrations」「守我们的代码规范」 | 三种表达，门槛从低到高：**命令退出码 → judge 提示 → Python callable**（§8.1） | 强 |
| **C** | **什么都没给** | 兜底：重放对照 + 效率 + 隐式人类判决 | 弱 |

A 档和刷 benchmark 在技术上没有区别，区别只在数据是用户自己的 —— 引擎那边一行都不用改。
B 档是真实场景里**最常见**的一档，因为人们想要的往往不是「答对」而是「按我的方式做对」。

C 档是**默认值，不是设计中心**：它保证「装上就能跑」，但 UI 上必须明说信号更弱、
接受会更保守。下面单独讲它，因为三档里只有它需要新机制。

**合成任务是补量手段，不是一档信号。** `examples/r_zero`、`examples/absolute_zero`、
`examples/agent0` 三个端口的 challenger/solver 可以在用户领域里造题 —— A/B/C 三档
都能拿它扩 held-out（§2.3）。

#### C 档的三路信号

| 信号 | 怎么拿 | 额外成本 | 噪声 | 定位 |
|---|---|---|---|---|
| **重放对照** | 历史那一轮的完整轨迹就是参照系；候选跑同一任务，**成对比较** | 一次 rollout + 一次判定 | 中 | **主力** |
| **效率** | 步数、工具错误数、token 数 | 零（日志里就有） | 低 | 乘数项（A/B 档也用） |
| **隐式人类判决** | `turn/end` 的 kind、是否被 `steer`/`interrupt` 打断、审批被拒、`dsh-feedback` 的显式反馈 | 零 | 高 | 当**采样权重**，不当 reward |

**为什么重放对照是 C 档的主力。** dsh 有一条硬不变量 —— *model-visible means logged*，任何
进入模型请求的东西都必须能从 log 重建，运行时有断言 —— 加上 `sessions.create(id, {seed})`
重放。于是**每一条历史轨迹都是一个免费的参照系**：那次实际做了什么、走了几步、
怎么结束的，全在。候选 artifact 跑同一个任务，拿两条轨迹做**成对比较**。

成对比较这件事必须强调：无参照的绝对打分恰恰是 LLM judge 最差的用法，而
「A 和 B 哪个更好」是它最可靠的用法。自我演化场景恰好天然提供了 B。

**效率项必须是词典序，不能加权求和 —— 这条三档通用。** 否则「少走两步」能买下「答错了」：

```py
def reward(task, run):
    if not passed(run):            # 成功门在前，且是硬门
        return 0.0
    return 0.5 + 0.5 * (1.0 - normalised_cost(run))   # 成功之后才比省
```

奖励值域仍然是 `[0, 1]`，Beta 后验接受不用改。

**隐式判决为什么不直接当 reward。** 「用户打断了」既可能是答得烂，也可能是他临时
改主意。把它当 reward 是在学噪声；把它当采样权重 —— 被打断过的那类任务多采一点 ——
是在把注意力放对地方。`agentdescent.sampling` 的 `TaskSampler` 正是这个挂点。

### 2.3 held-out 太小，以及为什么「拒绝大多数提案」是对的

**这一节只在用户没有自带数据集时才尖锐**（§2.2 的 B/C 档）。A 档用户手上有几百上千条
的话，直接跳过。

只靠真实会话时，用户可能只有几十条轮次。这时候 Beta 后验接受会**拒掉绝大多数提案**。

这是正确行为，不是 bug。N 小的时候接受，接受的就是噪声。所以：

1. **不要调低接受阈值让 demo 好看。** 这是本设计唯一一条「宁可不出成果」的红线。
2. **用合成任务补量**，而不是降标准。三个现成的端口可以直接复用。
3. **跨时间累积。** transcript 越用越多，held-out 自然长大 —— 这个系统天然是
   「你用得越多，它越会改自己」。这恰好也是自我演化最好的叙事。

### 2.4 分布漂移

`staleness.py` 管的是**diff 相对 head 的陈旧**，不是**任务分布的漂移**。两件事，
别混。用户换了个项目、换了门语言，之前学到的 skill 可能一夜之间不相关。

应对两条，都很轻：

- 采样器按**近因加权**（在 `DifficultyWeighted` 旁边加一个 recency 权重）。
- 用 `AuditScheduler` 周期性把 head 拿到**最近**的任务上重评。head 的 held-out 在近期
  任务上掉下来，就是该重新开跑的信号 —— 也是该提醒用户「这条 skill 可能过时了」的信号。

### 2.5 同意、成本、可撤销

这是用户自己的机器、自己的历史、自己的 harness 被改写。四条非协商项：

- **transcript 采集默认关**，首次开启时明确告知用什么、存哪、怎么关。
- **诚实说清一件事**：重放会把历史内容**再发一次**给模型提供方。虽然是当初已经发过的
  同一批内容，但这是一次新的发送，不能含糊过去。
- **预算上限（token）在 sidecar 侧强制**，UI 在启动按钮旁边给估值。默认沿用
  `skilldir.py` 已选好的 `self_verify=False` 和 `cheap_eval_tasks=4`。
- **一切可撤销**：profile 目录进 git；`/evolve off` dispose provider 的 effect，所有
  作用域注册随之解绑（Cordis 的 effect 本来就是可撤销的，这是框架保证，不是我们额外实现）。

---

## 3. 形态选型：为什么是原生插件

| 形态 | 做法 | 能拿到什么 | 判断 |
|---|---|---|---|
| **A. 原生 Cordis 插件 + Python sidecar** | npm bundle 注册服务/工具/命令/skill provider，驱动 Python 子进程做优化器 | 全部 seam：`ctx.skills`、`ctx.systemPrompt`、`ctx.tools`、`ctx.commands`、`ctx.jobs`、`ctx.approval`、`session/event`、HMR 热替换 | ✅ **采用** |
| **B. Python 侧 backend** | `agentdescent.backends.dsh()` 用 `deepseek-harness-sdk` 把 dsh 当子进程驱动 | 一个很好的 rollout runtime；但产物落在 Python 手里，进不了用户那个活着的 harness | ✅ **作为 A 的一条腿**，也是 M0 |
| **C. MCP server** | Python 起 MCP server，dsh 一个插件一个 server 地挂上来 | 只有工具。没有 skill provider、没有 prompt section、没有 session 事件、没有审批钩子 | ❌ 不值得做 |

C 被淘汰的理由要说清楚：dsh 自己的 cookbook 写着「A "native hook" is an ordinary
Cordis plugin on an interception point; it needs no external protocol.」在一个把所有
东西都做成插件的框架里，走 MCP 等于自愿放弃 90% 的接触面。

---

## 4. 总架构

```text
┌─────────────────────────── dsh 进程（Cordis 插件树）───────────────────────────┐
│                                                                               │
│  用户 / 模型                                                                   │
│    │  /evolve skill:sql              ctx.commands                             │
│    │  evolve_start(...)              ctx.tools                                │
│    ▼                                                                          │
│  ┌────────────────────┐   ctx.evolution（Service Definition：dsh-evolution）   │
│  │  consumers          │   start / status / cancel / head / onCommit          │
│  │  tool · command · UI│                                                      │
│  └─────────┬──────────┘                                                       │
│            ▼                                                                  │
│  ┌───────────────────────────────────────────┐                                │
│  │ dsh-evolution-agentdescent（Provider）     │                                │
│  │  · sidecar 监管（ctx.subprocess）           │                                │
│  │  · RolloutRunner  · ScorerRegistry         │                                │
│  │  · TaskSource（session-query）             │                                │
│  │  · ArtifactAdapters · GovernanceGuard      │                                │
│  └───┬─────────────────────────────────┬─────┘                                │
│      │ ① rollout.execute               │ ③ artifact.publish                    │
│      │                                 ▼                                      │
│      │                    ctx.skills.registerProvider ─┐                       │
│      │                    ctx.systemPrompt.section  ───┼─→ 活着的 harness 变了 │
│      │                    profile 文件树 + HMR      ───┘   （effect 可撤销）    │
│      ▼                                                                        │
│  fork session → agent.ctx 作用域注册候选 → followup(task) → 到 idle            │
│      │                                                                        │
│      └──→ session/event ──→ ② Scorer（重放对照 + 效率）──→ reward ∈ [0,1]      │
│                                                                               │
└───────────────────┬───────────────────────────────────────────────────────────┘
                    │  NDJSON-RPC over stdio（双向）
                    ▼
┌─────────────────────────── Python sidecar（agentdescent）────────────────────┐
│  evolve(tasks, reward, run=…, strategy=…, blast_radius=…, asynchronous=True)  │
│    ledger（git）→ workers → aggregator（staleness→冲突→fusion→Beta）→ commit  │
│  Executor 实现：DshRolloutExecutor —— 每个 rollout 反向调用 ①                 │
└──────────────────────────────────────────────────────────────────────────────┘
```

数据流是一个闭环：**harness 产生轨迹 → 轨迹变成任务和参照系 → 优化器提 diff →
提交后的产物注册回 harness → 下一轮轨迹在新产物上产生。**

---

## 5. 三个包，一条 seam

dsh 的架构文档把「加一个能力」定义成必须同时设计三个角色：Service Definition、
Service Provider、Consumer —— 「one role alone is not a seam」。照做：

### 5.1 `dsh-evolution` —— Service Definition

只有接口，不认识 AgentDescent。

```ts
// ctx key: 'evolution'
export interface EvolutionRuntime {
  start(spec: EvolutionSpec): Promise<RunId>
  status(id: RunId): RunStatus | undefined
  cancel(id: RunId, reason?: string): Promise<void>
  head(artifactId: ArtifactId): ArtifactHead | undefined
  history(artifactId: ArtifactId, limit?: number): Promise<CommitRecord[]>

  /** 什么可以被演化：把 ledger head 映射成一次活的注册 */
  registerArtifact(adapter: ArtifactAdapter): () => void
  /** 奖励怎么算：(task, 该次 rollout 的 session 事件, 可选参照轨迹) -> [0,1] */
  registerScorer(scorer: Scorer): () => void
  /** 任务从哪来：transcript、合成、用户给的清单 */
  registerTaskSource(source: TaskSource): () => void

  onCommit(listener: (c: CommitRecord) => void): () => void
}
```

三个注册表都用 dsh 已有的分层作用域语义（host 层 + per-scope 层，就近覆盖），
和 `ctx.skills` / `ctx.tools` 保持一致。

事件：`evolution/run-started`、`evolution/round`、`evolution/commit`、`evolution/run-ended`。
`evolution/commit` 之前插一个 waterfall `evolution/pre-commit`，返回
`{kind:'accept'} | {kind:'ask'} | {kind:'deny', reason}` —— 治理层挂在这儿（§9）。

### 5.2 `dsh-evolution-agentdescent` —— Service Provider

实现 `ctx.evolution`：监管 Python sidecar、把 rollout 请求翻译成 harness 里的子会话、
把提交翻译成注册表更新。**这是唯一知道 AgentDescent 存在的包。**

### 5.3 Consumers

- `dsh-tool-evolve` —— 模型可调：`evolve_start` / `evolve_status` / `evolve_accept`。
  意味着 agent 自己可以说「这类任务我总做错，开一轮训练」。
- `dsh-command-evolve` —— 人可调：`/evolve <artifact>`、`/evolve status`、`/evolve off`、
  `/evolve data on|off`（transcript 采集开关，默认 off）。
  **`/evolve` 第一件事是问目标**：有数据集就指过去，有判据就写一条命令或一段 rubric，
  都没有才落到兜底档 —— 并且当场告诉用户「兜底档信号弱，接受会保守」（§2.2）。
  目标一旦定下就随 artifact 存进 ledger，下次不用重问。
- `dsh-evolution-web` —— Web Client 的一个 `ConversationNodeDefinition`，把轮次、
  接受/拒绝、held-out 曲线、已花 token 画出来。

### 5.4 bundle 清单

```json
{
  "name": "dsh-plugin-agentdescent",
  "version": "0.1.0",
  "type": "module",
  "files": ["lib", "cordis.patch.yml"],
  "dsh": { "bundle": { "patch": "./cordis.patch.yml" } },
  "peerDependencies": { "@deepseek-ai/dsh-base": "^0.1.0" }
}
```

```yaml
# cordis.patch.yml
- insert:
    - id: evolution
      name: dsh-evolution
    - id: evolution-agentdescent
      name: dsh-evolution-agentdescent
      inject: [evolution, subprocess, agents, sessions, sessionQuery, skills, systemPrompt, approval, jobs]
      config:
        python: python3
        home: !!js dshHomePath('agentdescent')
        maxConcurrency: 4
        autoMergeMaxBlastRadius: 0.2
        harvestTranscripts: false        # §2.5：默认关
        tokenBudget: 2000000
    - id: evolution-tool
      name: dsh-tool-evolve
      inject: [tools, evolution]
    - id: evolution-command
      name: dsh-command-evolve
      inject: [commands, evolution]
```

安装：`dsh plugin --profile default add dsh-plugin-agentdescent`。

---

## 6. 参数表：被演化的东西 ↔ dsh 的注册点

左边是 AgentDescent 的 artifact，右边是它在活着的 harness 里以什么形式存在。
顺序就是自我演化目标下的优先级。

| Artifact id | dsh 注册点 | 提交后如何生效 | blast radius | 复用的入口 |
|---|---|---|---|---|
| `skill:<name>` | `ctx.skills.registerProvider` | provider 的 `invalidate()` → `skills/change` → 下次 `snapshot()` 拿到新版 | 0.2（L2，自动合并） | `evolve_skill_dir()` |
| **`skills:<root>`（技能库）** | 同上，但**一个 provider 服务整个库** | 替换 provider ⇒ 服务的集合**恰好等于**提交的集合（增删都生效） | 0.2 | `evolve_skill_library()` |
| `prompt:<section>` | `ctx.systemPrompt.section({name, order})` | 重新注册（旧 effect dispose） | 0.2，但**只在 `turn/end` 提交** | `evolve_skill()` / `AppendRules` |
| `prompt:<section>` | `ctx.systemPrompt.section()` | **等到 turn 边界**再换（KV cache） | 0.2 | `evolve_skill()` |
| `preset:<name>` | persona 全局 + `ctx.tools.restrict()` **只能按 agent** | persona 立即；工具集只对用了该 preset 的 agent | 0.6（L1，oracle + 人审） | `evolve_agent_dir()` |
| `plugin:<pkg>` | profile 目录下的文件树 + Cordis HMR | 写文件 → HMR 重载该行 | 0.6（L1 + 测试门） | `evolve_agent_code()` |

### 6.1 演化「一个 skill」和演化「技能库」不是同一件事

`skill:<name>` 只能**改进一个已经有人决定要有的 skill**。技能库是另一个 artifact，
不是更大的那个 —— 状态是整个根目录，于是一轮 run 可以：

* **加一个没人写过的 skill**（这是单 skill artifact 结构上做不到的事）；
* **退役**一个从来没帮上忙的；
* 把一条经验从一个 skill 挪到另一个。

引擎一行都不用改，原因值得写下来：state 是扁平的 `{path: content}`，aggregator 只问
「两个 diff 有没有碰同一个 key」。以根目录为 artifact 时，**两个 worker 各写一个新
skill = 互补 diff = fuse**；**两个改同一个 skill = 冲突 = 按 held-out 裁决**。
「新增一个文件」就是「一个之前不存在的 key」。

真正不同的只有两处：rollout 把候选摊在**根目录本身**（`LAYOUTS["dsh_skill_library"]`），
以及 harness 侧的一个设计决定 ——

> **一个 provider 服务整个库。** 每个 skill 一个 provider 的写法会让「新增」生效而
> 「删除」**静默失效**：被删掉的 skill 仍由那个没人 dispose 的注册继续服务，模型还
>看得见一个 ledger 说已经删了的 skill，held-out 分数于是不再描述 harness 的实际行为。
> 换掉一个 provider，服务的集合就**恰好等于**提交的集合 —— 增删都对。

`max_files_per_diff` 在库这一档是 3 而不是 2：最小的有意义提案是「新 skill + 一个
reference + 在既有 skill 里提一句」，两个文件装不下。它仍然是信任域，只是按库提案的
实际大小取的。

**为什么这四个几乎是免费的**：`filetree.py` 已经确立了「state 的 key 就是相对路径」
这个转换 —— 于是两个 worker 改不同文件 = 互补 diff = fuse，改同一文件 = 冲突 =
按 held-out 分数裁决。dsh 的 skill 目录、prompt 包、preset 目录、插件包，**全都是文件树**。
适配器只需要 `load_tree(path) -> state` 进去、`materialize(state, ws)` 出来。

```ts
export interface ArtifactAdapter {
  id: ArtifactId                       // 'skill:sql'
  blastRadius: number                  // 0.2 / 0.6
  load(): Promise<Record<string, string>>              // 当前状态（路径 -> 内容）
  /** 把一份候选状态作用域绑定到某个 agent —— rollout 用 */
  bind(agentCtx: Context, state: Record<string, string>): () => void
  /** 提交：把新状态变成全局生效的注册 */
  publish(state: Record<string, string>): Promise<() => void>
}
```

`bind` 和 `publish` 是同一件事的两个作用域，这正是 dsh 的设计允许的：架构文档明写
「Scope a registration to one agent → use that agent's `agent.ctx`」。于是
**8 个 worker 可以在同一个进程里各自看到自己那份候选 skill，互不污染** —— 不需要 8 个
容器，也不需要 8 个 harness 进程。这是进程内 rollout 最大的收益。

---

## 7. Rollout：harness 就是 runtime

```ts
async function rollout(spec: RolloutSpec): Promise<RolloutResult> {
  const child = await ctx.sessions.fork(spec.baseSession ?? undefined)
  const agent = await ctx.agents.create(child)
  using _bound = adapter.bind(agent.ctx, spec.candidateState)   // ← 候选只对这个 agent 可见

  const events: SessionEvent[] = []
  const off = ctx.on('session/event', (s, e) => { if (s.id === child) events.push(e) })

  const receipt = await agent.followup(createUserMessage({ content: spec.task.prompt }))
  const end = await waitForIdle(agent, receipt, spec.timeoutMs)   // 拥有「receipt → idle」这段区间

  off()
  return { output: lastAssistantText(events), events, finishReason: end.kind, cost: meter(events) }
}
```

三个必须说清楚的点：

1. **区间所有权**。Python SDK 的 `Session.run()` 已经把语义写死了：一次 run 拥有
   「durable inbox receipt → 下一次 whole-agent idle」这段区间，`final_response` 是这段
   区间里最后一条 root session 的 assistant 文本。进程内版本必须用同一套语义，否则
   两条路径测出来的 reward 不可比。
2. **并发上限归 harness 管**。sidecar 的 `max_concurrency` 由插件配置注入，不能让
   AgentDescent 的调度器和 harness 的调度器互相打架。背压走 RPC：TS 侧对
   `rollout.execute` 做有界队列。
3. **候选绝不放宽沙箱**。子 agent 继承父作用域的 sandbox / approval 组合，适配器往
   `agent.ctx` 里注册的东西只有候选 artifact 本身，不含任何新能力。

### 7.1 另一条腿：SDK 子进程 rollout

`executor.py` 里已经写过为什么需要进程：**被演化的代码是模型写的，段错误、OOM、
失控分配是家常便饭，线程里每一样都会带走整个 run**。演化插件代码（L1，M4）正是这个场景。

```py
from deepseek_harness import DeepSeekHarness

with DeepSeekHarness(provider="deepseek-official", model="deepseek-v4-flash",
                     cordis=candidate_cordis_path, cwd=workspace) as h:
    result = h.run(task.prompt)          # RunResult(session_id, final_response, finish_reason, events, ...)
```

这条路径顺带就是 **M0 路径**：不需要装插件，一个 Python 脚本就能跑完整演化。

### 7.2 跨进程时的 spec 形状

`workspec.py` 已经解决了这个问题：闭包不能过进程边界（`rewards.last_number()`、
`reflector(model)`、`lambda` 全都 pickle 不了），所以工作被描述成 `Ref`（点分目标 +
JSON 配置，在对面用对面自己的代码解析，import 前缀白名单）。TS↔Python 的桥直接
沿用这套词汇：**桥上跑的是 `RolloutSpec`，不是函数**。

---

## 8. 奖励与任务的实现层

```ts
export interface Scorer {
  name: string
  /** reference 是历史那一轮的轨迹；没有就退化成无参照打分 */
  score(task: EvolutionTask, run: RolloutResult, reference?: RolloutResult): Promise<number>
}
```

对应 §2.2 的三档，首发五个 scorer，**注册顺序就是选择顺序**：

| scorer | 档 | 实现 |
|---|---|---|
| `gold` | A | 复用 `agentdescent.rewards` 的 `exact_match` / `contains` / `last_number` / `numeric_close` |
| `command` | B | 在 rollout 工作区跑一条用户给的命令，退出码 0 = 1.0 |
| `judge` | B | 用户给的判据文本 + 候选轨迹 → judge subagent |
| `replay-pairwise` | C | 拿 reference 轨迹和候选轨迹做成对判定（先比客观项：终态工具效果是否一致、步数、工具错误数；不可判再上 judge） |
| `efficiency` | 全部 | 词典序：成功门 → `0.5 + 0.5*(1-归一化成本)`，作为上面任何一个的乘数项 |

（`tests` 不是第六个 —— 它就是 `command`，命令是仓库自己的测试命令，复用
`runners.code_runner` 的 `TEST_FAILURE_MARKER` 解析。L1 代码演化的门用的就是这条。）

### 8.1 自定义目标的三种表达

B 档要好用，门槛必须能降到「不写代码也能用」。三种表达，同一个 `Scorer` 接口：

```yaml
# 1. 命令退出码 —— 门槛最低，也最硬
objective:
  kind: command
  run: "psql -f {output} --quiet"      # {output} 是候选这次的产出落到工作区的路径
  timeoutSeconds: 30

# 2. judge 提示 —— 说得清但判不了的目标
objective:
  kind: judge
  rubric: |
    好的回答必须：用我们的 snake_case 表名；不 SELECT *；
    不动 migrations/ 下的任何文件。
  model: deepseek-v4-flash

# 3. Python callable —— 完全的表达力
objective:
  kind: python
  ref: "myproj.scorers:sql_reward"     # workspec.Ref，白名单前缀，JSON 配置
```

第三种直接落在 `workspec.Ref` 上（点分目标 + JSON 配置 + import 前缀白名单），
所以它天然能过进程边界，不需要为自定义目标另造一套序列化。

三种可以叠加：`objective` 收一个列表时，取**最小值**（每一条都是必须过的门），
而不是加权平均 —— 理由和 §2.2 的词典序是同一条。

### 8.2 任务来源：transcript 必须走 TS 侧

**不要在 Python 里解析 dsh 的磁盘 session 格式。** 查过了：默认是
`<root>/--<normalized-cwd>--/<encoded-id>/session.jsonl.zstd`，而且开着 `packChunks`——
连续同块的 `assistant/chunk` 会被压成 `text-chunks` / `reasoning-chunks` 这类打包行，
用的是 `packChunkRuns`/`decodeStorageRecord` 那套白名单编解码。在 Python 里重写一个
解码器，等于给自己接一条每次 dsh 发版都可能断的兼容面 —— 何况还要额外拖一个 zstd 依赖
进这个「核心零依赖」的包。

所以 `TaskSource` 一律在 TS 侧，走 `ctx.sessionQuery` / `deriveMessages()` 这些受支持的
接口，把结果以普通 JSON 递给 sidecar。三个来源：

- `transcript` —— 用户真实轮次（默认关，§2.5）。prompt = 那条用户消息，
  reference = 那一轮的实际轨迹。
- `synthetic` —— 复用 `examples/r_zero`、`examples/absolute_zero`、`examples/agent0`
  的 challenger/solver，在用户领域里造题补 held-out 的量。
- `manual` —— 用户自己给一份代表性 prompt 清单。**M0 用的就是这个**（不需要任何采集）。

---

## 9. 治理：自改 harness 的那道门

AgentDescent 的 L0/L1/L2 直接映射到 dsh 的真实执行轴上，**不是靠约定，是靠代码**。

| 层 | 内容 | dsh 侧的执行 |
|---|---|---|
| **L0 冻结** | 审批/权限行、`ctx.sandbox` 后端、credentials、scorer/oracle 自身、`dsh-evolution-*` 自己那几行 | 双重判定：Python 侧 `governance.FROZEN_IDS`，TS 侧 `ctx.tools.guard()` 的单调终局否决 |
| **L1 慢层**（`blast_radius ≥ 0.5`） | preset、插件代码、工具集 | `evolution/pre-commit` 返回 `ask` → `ctx.approval` 弹给人，附 diff + evidence card + held-out 差值；同时 `AuditScheduler.force_oracle` |
| **L2 快层**（`< 0.5`） | skill、prompt section | 自动合并，但每次提交都是 profile 目录里的一个 git commit |

四个必须成立的性质：

1. **冻结集是手写清单，不是启发式**。`governance.py` 的注释已经把理由写死了：
   「a verifier that learns to pass itself is exactly what an *estimated* layer would
   fail to catch」。artifact 能不能改判它自己的那个东西，是结构事实，不是测量结果。
2. **两侧各判一次**。sidecar 被攻破或有 bug 时，TS 侧的 guard 仍然拦得住 —— 单边
   信任在一个会自我修改的系统里不成立。
3. **可回滚**。ledger 本来就是 git backed。profile 目录也进 git，于是
   `dsh --profile x --dump-config` 能看差异，`git revert` 能退回去。
4. **总开关是真的**。`/evolve off` dispose provider 的 effect，所有作用域注册随之解绑。

### 9.1 演化插件时，冻结的是「能改规则的文件」

`governance.FROZEN_IDS` 冻的是 **artifact id**，它表达不了「这个 artifact 可以演化，
但它的某几个文件不行」。演化一个 dsh 插件时，恰好有三类文件属于后者 —— 每一类都不是
「重要」，而是**能改写规则本身**：

| 冻结的路径 | 它能做什么 |
|---|---|
| `cordis.patch.yml` | 决定**哪些行被插进插件树** —— 包括审批门、沙箱后端、以及演化引擎自己那几行。能改自己的 patch，就等于能把审自己的东西摘掉 |
| `package.json` | 带 `prepare` / `postinstall`（安装期执行，在 agent 的沙箱之外），以及 `dsh.bundle` 指针 |
| `test/**`、`*.test.ts` | 就是那道门。能改门的候选可以自己放自己过 |

三条工程约束：

1. **两侧各判一次**。Python 侧 `FROZEN_PLUGIN_PATHS`，TS 侧 `isFrozen()` —— 后者是
   **重复实现而不是跨桥引用**：只在「提出改动的那一侧」执行的治理规则，只在那一侧
   没出错的前提下成立。
2. **调用方的 `frozen=` 只能追加，不能替换**。这是这里唯一一个**没有可见症状**的错误：
   替换掉之后 run 照样跑完、照样提交，只是它已经被允许去改那些决定「有没有东西在审它」
   的行了。
3. **`npm ci --ignore-scripts`**。否则一个能加 `postinstall` 的候选，会在任何一个测试
   跑起来之前先在你机器上执行代码 —— 而门本来应该是最先跑的那个。

L1 强制 oracle 这一条顺带是免费的：oracle 评的是刚被接受性检验评过的同一份 artifact、
同一个 held-out 集，评估缓存直接命中。

---

## 10. 桥协议

NDJSON-RPC over stdio，**双向**，带版本握手。方向反过来的那半边才是重点：Python 侧
反向调用 harness 做 rollout，这正是「harness 是 runtime」的落实。

**TS → Python**

| 方法 | 语义 |
|---|---|
| `run.start(EvolutionSpec)` | 起一轮演化，立即返回 `runId` |
| `run.status(runId)` / `run.cancel(runId)` | 状态 / 取消 |
| `ledger.head(artifactId)` / `ledger.history(...)` | 读 |
| `commit.decide(commitId, verdict)` | 人审结果回灌 |

**Python → TS**

| 方法 | 语义 |
|---|---|
| `rollout.execute(RolloutSpec) -> RolloutResult` | 在 harness 里跑一次 rollout |
| `reward.score(task, run, reference?) -> number` | 走 ScorerRegistry |
| `tasks.fetch(sourceName, n, filter) -> Task[]` | 走 TaskSource（含 reference 轨迹） |
| `approval.ask(diff, layer, evidence) -> verdict` | L1 人审 |
| `artifact.publish(artifactId, state, version)` | 提交后注册回活的 harness |
| `log.progress(runId, round, stats)` | 进度，喂给 `ctx.jobs` 的 notice 和 Web UI |

**失败语义**（必须写死，否则演化跑一半的状态无法收敛）：

- sidecar 死 → job 失败，但 ledger 在磁盘上是完整的 git 仓库，`run.start` 带
  `resumeFrom` 可续。
- harness 死 → sidecar 读到 stdin EOF 自杀，不留孤儿进程。
- 单次 `rollout.execute` 超时 → 该 rollout 记 `error`，不杀整轮。
- 版本不匹配 → 握手阶段拒绝启动并明确报出两边版本，不做兼容猜测。
- **token 预算耗尽** → sidecar 停止发起新 rollout，当前轮跑完即收，状态标
  `budget-exhausted`。预算是硬上限，不是建议值。

长运行挂 `ctx.jobs`：一轮演化是一个 job，`job_*` 工具可读可杀，进度走 notice。
空闲触发：监听 `turn/end`，whole-agent idle 持续超过阈值才启动，任何新用户输入立刻让路。

---

## 11. Monorepo 落点

```
agentdescent/                  Python 包（发 PyPI，不变）
└── dsh/
    ├── backend.py     dsh() —— 用 deepseek-harness-sdk 把 dsh 当 Completion/WorkspaceAgent
    ├── daemon.py      sidecar：NDJSON-RPC server，把 evolve() 包成 run.start/status/cancel
    ├── executor.py    DshRolloutExecutor —— 实现 Executor，每个 rollout 反向 RPC
    └── bridge.py      RolloutSpec ↔ JSON（复用 workspec.Ref 的白名单解析）
plugin/                        TS 包（发 npm，名 dsh-plugin-agentdescent）
├── package.json               dsh.bundle
├── cordis.patch.yml
├── src/{index,engine,rollout,scorer,tasks,tools,command,job}.ts
│   └── artifacts/{skill,prompt,preset,plugin-tree}.ts
└── test/
examples/dsh/
└── evolve_dsh_skill.py        M0 的可跑示例（--dry-run 无需 API key）
```

monorepo 的四个具体后果，每一条都要落到文件：

1. **打包不受影响。** `pyproject.toml` 的 `[tool.setuptools.packages.find] include =
   ["agentdescent*"]` 已经把 `plugin/` 挡在 wheel 外，没有 `MANIFEST.in`，sdist 也不会
   带 TS 源。不用改 —— 但要加一条测试守住它，别哪天有人放宽了 include。
2. **CI 要加一个 job。** 现在的 `tests.yml` 是纯 Python 三版本矩阵。加一个
   `plugin` job（pnpm + `tsc --noEmit` + vitest），用 `paths` 过滤，别让只改 Python 的
   PR 去装 node。
3. **版本不绑死。** Python 版本单一来源是 `agentdescent/__init__.py.__version__`
   （`pyproject` 用 `attr:` 取，注释里记着它以前漂过）。npm 侧**不共用这个号**：
   插件的破坏性变更多半来自 dsh 而不是引擎，绑死会逼出一堆无意义的 major。
   插件在 `peerDependencies` 里锁 agentdescent 的**最小版本**，握手时再校验一次（§10）。
4. **分发只能走 npm 或 tarball。** 这是选 monorepo 唯一实打实的代价：
   `dsh plugin add github:Birfy/agentdescent` 装不了 —— pnpm 的 git 依赖只认仓库根，
   而插件在 `plugin/` 子目录；就算能装，还会撞上 pnpm ≥10 拒绝跑 git 依赖 `prepare`
   的那道坎。所以需要一个 `publish-plugin.yml`（发布时构建 `lib/` 再 `pnpm publish`），
   和现有的 PyPI `publish.yml` 并列。

   顺带一提，那道坎本来也不该让用户去开：`allowBuilds` 的语义是「允许这个包在你机器
   上、在沙箱之外执行代码」，对一个会自我修改的插件来说，这个口子开得没道理。

---

## 12. 里程碑

按自我演化这个目标重排过 —— 「后台悄悄变好」比「能改自己的代码」更早成为产品。

| | 内容 | 验收 | 估时 |
|---|---|---|---|
| **M0** ✅ | **离线自我演化。** 纯 Python 零 TS：读用户真实 profile 的 skill 目录，任务和判据由用户给（A 档数据集，或 B 档一条命令），rollout 走 `deepseek-harness-sdk` 子进程，产物 `EvolutionResult.write_to()` 落回去 | 在**真实 profile** 上跑完，给出 before/after 两个数，产物能被 dsh 正常加载 | 3 天 |
| **M1** ✅ | 插件最小可用：`plugin/` 骨架 + sidecar + `ctx.evolution` seam + skill 适配器 + 进程内 rollout + `/evolve` + `evolve_start` 工具。只有 L2 | `dsh plugin add ./plugin` 之后 `/evolve skill:x` 跑完，新 skill 在**同一个会话里**生效 | 1 周 |
| **M2** ✅ | 奖励闭环：ScorerRegistry + A/B 档（`gold` / `command` / `judge`）+ C 档兜底（`replay-pairwise` + `efficiency`）+ transcript TaskSource（TS 侧 `session-query`） | 三档各跑通一个真实例子；**兜底档接受不了时能说清是因为 N 太小**，而不是静悄悄没结果 | 1–1.5 周 |
| **M3** ✅ | 常驻形态：`ctx.jobs` + 空闲触发 + 异步运行时（`asynchronous=True, async_ratio=3`）+ Web UI 节点 + 预算上限 | 正常用 harness 的同时后台演化，任何用户输入立刻让路，花了多少 token 看得见 | 1 周 |
| **M4** ◑ | L1：prompt section / preset / 插件代码，测试门 + `ctx.approval` 人审 + 冻结集双 guard | 一次 L1 提交必须弹审批；冻结行的补丁在两侧都被拒（各写一个测试） | 1–2 周 |

M0 单独就值得做完再决定要不要继续 —— 它给出这条路线的第一个真实数字，而且已经是
产品的缩微版：真实 profile、真实 harness、真实产物落盘。

---

## 13. 风险与未决

**dsh 还在 developer preview**，官方明说「compatibility may change」。对策：bundle 里
锁 dsh 版本区间；所有 harness API 只在适配器文件里出现，一次 break 只改一处。
§8.2 那条「不在 Python 里解析磁盘格式」是同一个对策的具体化。

**奖励噪声**，在兜底档（§2.2 的 C）是头号风险：没有判据，rollout 方差又比纯 completion 大。
A/B 档没有这个问题 —— 所以产品上要**尽量把用户推到 A/B**，而不是把兜底档做得更聪明。
Beta 后验接受本来就是干这个的 —— §2.3 那条红线（不调阈值）是这份设计里唯一
「宁可不出成果」的约束。held-out 和提案集的严格隔离由 ledger 保证。

**成本**。每一次 rollout 都是真实 harness 里的真实模型调用，而重放对照还要多跑一次
参照比较。默认 `self_verify=False` + `cheap_eval_tasks=4`，预算硬上限在 sidecar 侧
强制，UI 上启动前给估值。

**KV cache**。dsh 把 prompt 前缀稳定性当一等公民（每个包的 README 都有
「KV Cache effect」一节）。prompt section 在会话中途换掉会让该会话的缓存失效。
所以 `prompt:*` 的提交只在 `turn/end` 落地，并且批量落。

**未决，需要拍板**：

1. **`ctx.evolution` 这条 seam 要不要试着上游**。如果 dsh 官方愿意收，它就是标准
   接口；不收就完全活在我们自己的 bundle 里 —— 也能跑，只是别人换不掉优化器。
2. **transcript 采集的默认值**。本文按「默认关」写。要不要在首次装插件时用
   `ctx.approval` 主动问一次？（倾向要，但那是一次打断。）
3. **预算默认值**。`tokenBudget: 2000000` 是个占位数字，需要 M0 跑完拿真实
   token 消耗来定 —— 一轮自我演化到底烧多少，现在只有估计没有测量。
