# 设计文档：把 AgentDescent 做成 DeepSeek Harness 插件

> 目标：让 **DeepSeek Harness（`dsh`）** 装上 AgentDescent 之后，它自己会变好 ——
> 用户平时怎么用它，它就在那条真实轨迹上训练自己的 skill、prompt、preset 乃至插件代码。
>
> **状态：设计中**，尚未实现。本文写*为什么*这样切、边界划在哪、以及每一步的验收标准。
>
> 对照阅读：[演化一个目录](directory-evolution.md)（本设计复用的底座）、
> [治理 L0/L1/L2](governance.md)、[Where rollouts run](execution.md)、
> [Agents & LLMs](agents.md)。

---

## 0. 结论先行

| 决策 | 结论 | 一句话理由 |
|---|---|---|
| 做成什么形态 | **一个 npm bundle（TS）+ 一个 Python sidecar**，不是纯 MCP、不是纯 Python 脚本 | 只有原生 Cordis 插件才够得着 `ctx.skills` / `ctx.systemPrompt` / `ctx.approval`；MCP 只能给你工具 |
| 引擎跑在哪 | **Python 侧**（ledger / aggregator / staleness / governance 一行不动） | 16k 行经过测量的优化器，没有任何理由重写成 TS |
| rollout 跑在哪 | **默认跑在 harness 进程内**（fork session + `agent.ctx` 作用域注册），重活可切到 SDK 子进程 | 要测的就是那个真实 harness：真工具、真沙箱、真审批、真 prompt 装配 |
| 参数（被演化的东西）是什么 | dsh 里**已经是插件的那些东西**：skill、prompt section、preset、插件代码 | dsh 把 agent 的每一层都做成了可替换模块，等于第一次把「agent 的参数」变成**可寻址**的 |
| 数据从哪来 | **用户自己的 session log** | dsh 有硬不变量「model-visible means logged」+ 可重放，这是 AgentDescent 自己没有的轨迹存储 |
| 安全边界 | L0 冻结集在 **Python 和 TS 两侧各判一次**，L1 走 `ctx.approval` 人审 | 自改 harness 的系统，冻结集不能只有一道门 |
| 先做哪一步 | **M0：`agentdescent.backends.dsh()`**，纯 Python，两三天，不写一行 TS | 它本身就有用，同时是插件形态的 rollout 底座 |

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

## 2. 形态选型：为什么是原生插件

| 形态 | 做法 | 能拿到什么 | 判断 |
|---|---|---|---|
| **A. 原生 Cordis 插件 + Python sidecar** | npm bundle 注册服务/工具/命令/skill provider，驱动一个 Python 子进程做优化器 | 全部 seam：`ctx.skills`、`ctx.systemPrompt`、`ctx.tools`、`ctx.commands`、`ctx.jobs`、`ctx.approval`、`session/event`、HMR 热替换 | ✅ **推荐** |
| **B. Python 侧 backend** | `agentdescent.backends.dsh()` 用 `deepseek-harness-sdk` 把 dsh 当子进程驱动 | 一个很好的 rollout runtime；但产物落在 Python 手里，进不了用户那个活着的 harness | ✅ **先做，作为 A 的底座**，本身也独立有用 |
| **C. MCP server** | Python 起 MCP server，dsh 一个插件一个 server 地挂上来 | 只有工具。没有 skill provider、没有 prompt section、没有 session 事件、没有审批钩子、没有 job | ❌ 不值得做产品，顶多当过渡 |

C 被淘汰的理由要说清楚：dsh 自己的 cookbook 写着「A "native hook" is an ordinary
Cordis plugin on an interception point; it needs no external protocol.」在一个把所有
东西都做成插件的框架里，走 MCP 等于自愿放弃 90% 的接触面。

A 和 B 不是二选一：**B 是 A 的一条腿**。A 的进程内 rollout 是默认路径，B 的子进程
rollout 是 L1 代码演化和 CI 场景的逃生口（理由见 §7）。

---

## 3. 总架构

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
│      └──→ session/event ──→ ② Scorer ──→ reward ∈ [0,1]                       │
│                                                                               │
└───────────────────┬───────────────────────────────────────────────────────────┘
                    │  NDJSON-RPC over stdio（双向）
                    ▼
┌─────────────────────────── Python sidecar（agentdescent）────────────────────┐
│  evolve(tasks, reward, run=…, strategy=…, blast_radius=…, asynchronous=…)     │
│    ledger（git）→ workers → aggregator（staleness→冲突→fusion→Beta）→ commit  │
│  Executor 实现：DshRolloutExecutor —— 每个 rollout 反向调用 ①                 │
└──────────────────────────────────────────────────────────────────────────────┘
```

数据流是一个闭环：**harness 产生轨迹 → 轨迹变成任务和奖励 → 优化器提 diff →
提交后的产物注册回 harness → 下一轮轨迹在新产物上产生。**

---

## 4. 三个包，一条 seam

dsh 的架构文档把「加一个能力」定义成必须同时设计三个角色：Service Definition、
Service Provider、Consumer —— 「one role alone is not a seam」。照做：

### 4.1 `dsh-evolution` —— Service Definition

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
  /** 奖励怎么算：(task, 该次 rollout 的 session 事件) -> [0,1] */
  registerScorer(scorer: Scorer): () => void
  /** 任务从哪来：数据集、transcript、合成 */
  registerTaskSource(source: TaskSource): () => void

  onCommit(listener: (c: CommitRecord) => void): () => void
}
```

三个注册表都用 dsh 已有的分层作用域语义（host 层 + per-scope 层，就近覆盖），
和 `ctx.skills` / `ctx.tools` 保持一致。

事件：`evolution/run-started`、`evolution/round`、`evolution/commit`、`evolution/run-ended`。
`evolution/commit` 之前插一个 waterfall `evolution/pre-commit`，返回
`{kind:'accept'} | {kind:'ask'} | {kind:'deny', reason}` —— 治理层就挂在这儿（§9）。

### 4.2 `dsh-evolution-agentdescent` —— Service Provider

实现 `ctx.evolution`，内容是：监管 Python sidecar、把 rollout 请求翻译成 harness 里的
子会话、把提交翻译成注册表更新。**这是唯一知道 AgentDescent 存在的包。**

### 4.3 Consumers

- `dsh-tool-evolve` —— 模型可调：`evolve_start` / `evolve_status` / `evolve_accept`。
  意味着 agent 自己可以说「这类任务我总做错，开一轮训练」。
- `dsh-command-evolve` —— 人可调：`/evolve <artifact>`、`/evolve status`、`/evolve off`。
- `dsh-evolution-web` —— Web Client 的一个 `ConversationNodeDefinition`，把轮次、
  接受/拒绝、held-out 曲线画出来。

### 4.4 分发

一个 bundle 把上面的行插进去：

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
      inject: [evolution, subprocess, agents, sessions, skills, systemPrompt, approval, jobs]
      config:
        python: python3
        home: !!js dshHomePath('agentdescent')
        maxConcurrency: 4
        autoMergeMaxBlastRadius: 0.2
    - id: evolution-tool
      name: dsh-tool-evolve
      inject: [tools, evolution]
    - id: evolution-command
      name: dsh-command-evolve
      inject: [commands, evolution]
```

安装：`dsh plugin --profile default add dsh-plugin-agentdescent`。

> **注意 npm 上的构建陷阱**：从 git 装的话 pnpm ≥10 默认拒绝跑 `prepare`，TS 包会
> 因为没有 `lib/` 而加载失败。要么发 npm（发布时构建好 `lib/`），要么发 tarball。
> 别让用户去 `allowBuilds` —— 那是「允许这个包在你机器上、在沙箱之外执行代码」。

---

## 5. 参数表：被演化的东西 ↔ dsh 的注册点

这是整个设计的核心表。左边是 AgentDescent 的 artifact，右边是它在活着的 harness 里
以什么形式存在。

| Artifact id | dsh 注册点 | 提交后如何生效 | blast radius | 复用的 AgentDescent 入口 |
|---|---|---|---|---|
| `skill:<name>` | `ctx.skills.registerProvider` | provider 的 `invalidate()` → `skills/change` → 下次 `snapshot()` 拿到新版 | 0.2（L2，自动合并） | `evolve_skill_dir()` |
| `prompt:<section>` | `ctx.systemPrompt.section({name, order})` | 重新注册（旧 effect dispose） | 0.2，但**只在 `turn/end` 提交** | `evolve_skill()` / `AppendRules` |
| `preset:<name>` | `agent.ctx` 上的 `ctx.tools.restrict()` + persona section | 下一个会话生效 | 0.6（L1，oracle + 人审） | `evolve_agent_dir()` |
| `plugin:<pkg>` | profile 目录下的文件树 + Cordis HMR | 写文件 → HMR 重载该行 | 0.6（L1 + 测试门） | `evolve_agent_code()` |

**为什么这四个几乎是免费的**：`filetree.py` 已经确立了「state 的 key 就是相对路径」
这个转换 —— 于是两个 worker 改不同文件 = 互补 diff = fuse，改同一文件 = 冲突 =
按 held-out 分数裁决。dsh 的 skill 目录、prompt 包、preset 目录、插件包，**全都是文件树**。
所以适配器只需要做两件事：`load_tree(path) -> state` 进去，`materialize(state, ws)` 出来。

`ArtifactAdapter` 的形状：

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

## 6. Rollout：harness 就是 runtime

一次 rollout 的完整生命周期：

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

1. **区间所有权**。Python SDK 的 `Session.run()` 已经把这个语义写死了：一次 run 拥有
   「durable inbox receipt → 下一次 whole-agent idle」这段区间，`final_response` 是这段
   区间里最后一条 root session 的 assistant 文本。进程内版本必须用同一套语义，否则
   两条路径测出来的 reward 不可比。
2. **并发上限归 harness 管**。sidecar 的 `max_concurrency` 由插件配置注入，不能让
   AgentDescent 的调度器和 harness 的调度器互相打架。背压走 RPC：TS 侧对
   `rollout.execute` 做有界队列，满了就不 ack。
3. **候选绝不放宽沙箱**。子 agent 继承父作用域的 sandbox / approval 组合，适配器往
   `agent.ctx` 里注册的东西只有候选 artifact 本身，不含任何新能力。

### 6.1 另一条腿：SDK 子进程 rollout

`executor.py` 里已经写过为什么需要进程：**被演化的代码是模型写的，段错误、OOM、
失控分配是家常便饭，线程里每一样都会带走整个 run**。演化插件代码（L1）正是这个场景。

所以 provider 支持第二种 executor：Python 侧直接用 `deepseek-harness-sdk` 起独立
harness 进程。

```py
from deepseek_harness import DeepSeekHarness

with DeepSeekHarness(provider="deepseek-official", model="deepseek-v4-flash",
                     cordis=candidate_cordis_path, cwd=workspace) as h:
    result = h.run(task.prompt)          # RunResult(session_id, final_response, finish_reason, events, ...)
```

这条路径顺带就是 **CI 路径**：不需要装插件，一个 Python 脚本就能在 GitHub Actions 里
跑完整演化。也是 M0 的全部内容。

### 6.2 跨进程时的 spec 形状

`workspec.py` 已经解决了这个问题：闭包不能过进程边界（`rewards.last_number()`、
`reflector(model)`、`lambda` 全都 pickle 不了），所以工作被描述成 `Ref`（点分目标 +
JSON 配置，在对面用对面自己的代码解析，import 前缀白名单）。TS↔Python 的桥直接
沿用这套词汇：**桥上跑的是 `RolloutSpec`，不是函数**。

---

## 7. Reward：从 session log 里长出来

dsh 有一条硬不变量：**model-visible means logged** —— 任何进入模型请求的东西都必须
能从 log 重建，运行时有断言。加上 `deriveMessages()` 投影和 `sessions.create(id, {seed})`
重放，这给了 AgentDescent 一个它自己没有的东西：**一个持久、可重放、归因精确的轨迹存储**。

`Scorer` 的形状故意比 `reward(task, output)` 宽一格：

```ts
export interface Scorer {
  name: string
  score(task: EvolutionTask, run: RolloutResult): Promise<number>   // [0, 1]
}
```

首发四个：

| scorer | 怎么算 | 用在哪 |
|---|---|---|
| `gold` | 复用 `agentdescent.skill.SCORERS`（`exact` / `f1` / …）对 `final_response` | 有标注数据集 |
| `tests` | 在 rollout 工作区跑仓库自己的测试命令，复用 `runners.code_runner` 的 `TEST_FAILURE_MARKER` | L1 代码演化的门 |
| `efficiency` | 从事件里算：`turn/end` 的 `kind`、工具错误率、step 数、token 数 | 「同样做对，但更省」 |
| `judge` | 起一个 subagent 打分 | 没有 gold 的开放任务 |

### 7.1 用自己的 transcript 当训练集

这是这个插件独有的东西，也是最该谨慎的东西。

`TaskSource` 可以从 `session-query` 里把用户真实用过的轮次捞出来变成 `Task`：
prompt = 那条用户消息，gold = 人类当时接受的结果。于是 **harness 在「你实际怎么用它」
上训练**，而不是在某个 benchmark 上。

约束（默认全开）：只读本机、只用当前 profile、默认排除任何带凭据来源的会话、
入库前跑一次 secret 扫描、`/evolve data off` 一键关掉。这个能力必须是 opt-in，
并且在首次开启时明确告诉用户「你的历史会话会被用作训练数据，全程不出本机」。

---

## 8. 治理：自改 harness 的那道门

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
4. **总开关是真的**。`/evolve off` dispose provider 的 effect —— Cordis 的 effect
   本来就是可撤销的，所有作用域注册随之解绑。这不是我们额外实现的功能，是框架保证。

L1 强制 oracle 这一条顺带是免费的：oracle 评的是刚被接受性检验评过的同一份 artifact、
同一个 held-out 集，评估缓存直接命中。

---

## 9. 桥协议

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
| `reward.score(task, rolloutResult) -> number` | 走 ScorerRegistry |
| `approval.ask(diff, layer, evidence) -> verdict` | L1 人审 |
| `artifact.publish(artifactId, state, version)` | 提交后注册回活的 harness |
| `log.progress(runId, round, stats)` | 进度，喂给 `ctx.jobs` 的 notice 和 Web UI |

**失败语义**（必须写死，否则演化跑一半的状态无法收敛）：

- sidecar 死 → job 失败，但 ledger 在磁盘上是完整的 git 仓库，`run.start` 带
  `resumeFrom` 可续。
- harness 死 → sidecar 读到 stdin EOF 自杀，不留孤儿进程。
- 单次 `rollout.execute` 超时 → 该 rollout 记 `error`，不杀整轮（AgentDescent 的
  worker 本来就容忍单个失败）。
- 版本不匹配 → 握手阶段拒绝启动并明确报出两边版本，不做兼容猜测。

长运行挂 `ctx.jobs`：一轮演化是一个 job，`job_*` 工具可读可杀，进度走 notice。

---

## 10. 落点与包结构

```
agentdescent/                          （本仓库，Python）
└── dsh/
    ├── backend.py     dsh() —— 用 deepseek-harness-sdk 把 dsh 当 Completion/WorkspaceAgent
    ├── daemon.py      sidecar：NDJSON-RPC server，把 evolve() 包成 run.start/status/cancel
    ├── executor.py    DshRolloutExecutor —— 实现 Executor，每个 rollout 反向 RPC
    └── bridge.py      RolloutSpec ↔ JSON（复用 workspec.Ref 的白名单解析）
examples/dsh/
    └── evolve_dsh_skill.py            M0 的可跑示例（--dry-run 无需 API key）

dsh-plugin-agentdescent/               （新仓库，TS/npm）
├── package.json                       dsh.bundle
├── cordis.patch.yml
└── src/{index,engine,rollout,scorer,tools,command,job}.ts
    └── artifacts/{skill,prompt,preset,plugin-tree}.ts
```

**为什么 Python 侧住在本仓库**：`backends.py` 里已经有 `openhands()` 和 `claude_code()`，
`dsh()` 是第三个同类，本身独立有用。依赖方向也只有这一种是诚实的 —— 插件依赖
AgentDescent，反过来不行。

---

## 11. 里程碑

| | 内容 | 验收 | 估时 |
|---|---|---|---|
| **M0** | `agentdescent.backends.dsh()` + 一个示例。纯 Python，零 TS | 在一个真实数据集上跑完 `evolve_skill(..., agent=LLMAgent(dsh()))`，报出 baseline → evolved 两个数 | 2–3 天 |
| **M1** | 插件最小可用：sidecar + `ctx.evolution` seam + skill 适配器 + 进程内 rollout + `/evolve` + `evolve_start` 工具。只有 L2 | `dsh plugin add` 之后，`/evolve skill:x` 能跑完并让新 skill 在**同一个会话里**生效 | 1 周 |
| **M2** | ScorerRegistry + transcript 数据源 + `ctx.jobs` + Web UI 节点 | 用自己的历史会话训出一个 skill，全过程可在 UI 里看 | 1 周 |
| **M3** | L1：prompt section / preset / 插件代码，测试门 + `ctx.approval` 人审 + 冻结集双重 guard | 一次 L1 提交必须弹审批；冻结行的补丁在两侧都被拒 | 1–2 周 |
| **M4** | 异步运行时（`asynchronous=True, async_ratio=3`） | 演化在你正常用 harness 的同时进行，无轮次栅栏 | 1 周 |

M0 单独就值得做完再决定要不要继续 —— 它给出这条路线的第一个真实数字。

---

## 12. 风险与未决问题

**dsh 还在 developer preview**，官方明说「compatibility may change」。对策：bundle 里
锁 dsh 版本区间；所有 harness API 只在适配器文件里出现，一次 break 只改一处。

**成本**。每一次 rollout 都是一次真实 harness 里的真实模型调用。默认沿用
`skilldir.py` 已经选好的两个默认值 —— `self_verify=False`（否则每个提案的轨迹跑两遍）
和 `cheap_eval_tasks=4`（否则排序阶段拿全量 held-out 给每个候选打分）—— 并在 UI 上
把预估 token 数摆在启动按钮旁边。

**KV cache**。dsh 把 prompt 前缀稳定性当一等公民（每个包的 README 都有
「KV Cache effect」一节）。prompt section 在会话中途换掉会让该会话的缓存失效。
所以 `prompt:*` 的提交只在 `turn/end` 落地，并且批量落。

**奖励噪声**。真实 harness 的 rollout 方差比纯 completion 大。Beta 后验接受本来就是
干这个的 —— 不要为了 demo 好看去调低接受阈值。held-out 集必须和提案集严格隔离，
这一点 ledger 已经保证。

**未决，需要拍板**：

1. **主要目标是哪个**：(a) 让用户的 dsh 自我进化（参数 = 用户自己的 harness 配置），
   还是 (b) 把 dsh 当作 AgentDescent 的高质量 rollout runtime 去刷 benchmark？
   两者共用底座，但 M1 之后的优先级完全不同。
2. **TS 包放哪**：独立仓库，还是本仓库开一个 `plugin/` 目录（monorepo）？
   独立仓库对 npm 发布和 `dsh plugin add github:…` 更顺，代价是两处版本要对齐。
3. **`ctx.evolution` 这条 seam 要不要试着上游**。如果 dsh 官方愿意收，它就是标准
   接口；如果不收，就完全活在我们自己的 bundle 里 —— 后者也能跑，只是别人换不掉
   优化器。
