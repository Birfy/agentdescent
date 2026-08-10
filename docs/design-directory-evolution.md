# 设计文档：演化「目录形态」的 Skill / Agent，并由真实 Agent 执行

> 目标：用户给出一个**目录**（一个 skill 目录、一个 agent 目录，或一份 agent 代码），
> AgentDescent 演化它；每一次 rollout 由**真实 agent**（Claude Code / Codex /
> OpenHands / 任意 CLI agent）在一个装载了当前候选目录的工作区里执行任务。
>
> **状态：已实现并落地**（P0–P5）。本文保留为设计记录 —— 它写了*为什么*这样做，
> 以及复核时发现并修掉的三个真问题。要*怎么用*，看
> [演化一个目录（skill / agent / 代码）](directory-evolution.md)。
>
> 落地的模块：`agentdescent/filetree.py`（§3.1）、`agentdescent/treestrategy.py`（§3.2/§3.3）、
> `agentdescent/runners.py`（§3.4）、`agentdescent/skilldir.py`（§3.7），
> 加上 `evolution.py` 里的两处改动（§3.6 的删除语义、`EvolutionResult.write_to`）。
> **P6（EvoSkill 迁移，§5.2）也已完成** —— 见下方 5.2 的结论。
> 未做（当时）：`SectionedFileTree`、Docker 沙箱、ADAS 真执行。其中 Docker 沙箱后续已以 `ContainerProvider`/`SandboxPool` 落地（见 [sandboxes](sandboxes.md)）（§5.3）。

---

## 0. 结论先行

| 能力 | 现状 | 说明 |
|---|---|---|
| 演化「一段文本 skill」（system prompt / 指令 / playbook） | ✅ 已支持 | `SingleSlot` / `AppendRules` / `KeyedRules` + `evolve_skill()` |
| 用**真实 agent**（Claude Code / Codex / OpenHands）跑 rollout | ✅ 已支持 | `cli_agent(["claude","-p"])`、`claude_code()`、`codex()`、`openhands()`，全部是 `Completion` |
| 把材料**落到磁盘**让 agent 用工具去读 | ⚠️ 有先例但只覆盖单文件 | `WorkspaceAgent.in_workspace(path)` + `document_agent()`（[backends.py:102](https://github.com/Birfy/agentdescent/blob/main/agentdescent/backends.py#L102)）只写一个 `document.txt` |
| 演化一个 **skill 目录**（`SKILL.md` + `references/` + `scripts/`） | ❌ 缺 → ✅ **已实现** | `evolve_skill_dir()`；缺的是「目录 ↔ state」装载/物化与多文件提案协议 |
| 演化一个 **agent 目录**（`.claude/agents/*.md`、子 agent 定义、harness 配置） | ❌ 缺 → ✅ **已实现** | `evolve_agent_dir()`（L1）+ 路径级冻结 |
| 演化 **agent 代码本身**（可执行的 Python/TS） | ❌ 缺 → ✅ **已实现** | `evolve_agent_code()`：一次性工作区真执行 + pristine overlay 测试门 |
| 把演化结果**装回**用户目录 | ❌ 缺 → ✅ **已实现** | `EvolutionResult.write_to()`（默认先备份，不删多余文件） |
| 现有算法示例复用这套底座 | ✅ **EvoSkill 已迁移** | `SkillLibraryTree(FileTree)`；prompt 逐字节不变，12 个原有测试全绿（§5.2） |

**一句话**：引擎层（并行 worker、聚合器、ledger、治理、异步运行时）**完全够用且不需要动**；
缺的是**一层「目录适配层」**——把 `Dict[str, str]` 状态和磁盘目录互相翻译，并在每次 rollout
时把候选目录物化到一次性工作区里交给真实 agent。

实测下来估算是准的：新增 4 个模块（`filetree` / `treestrategy` / `runners` / `skilldir`），
`evolution.py` 只动了两处（§3.6 的删除语义，加上 `EvolutionResult.write_to`），
`aggregator` / `ledger` / `async_evolve` / `parallel` / `governance` 一行未改。

---

## 1. 现状分析（引擎能提供什么）

### 1.1 演化的最小契约

`evolve()` 只要求三个可调用对象（[evolution.py:1025](https://github.com/Birfy/agentdescent/blob/main/agentdescent/evolution.py#L1025)）：

```python
run(rendered: str, task: Task) -> str                       # 用当前 artifact 做一次 rollout
reward(task: Task, output: str) -> float                    # [0, 1]
propose(rendered, task, output, reward) -> Optional[str]    # 反思出一个改进提案
```

`rendered` 是 `Strategy.render(state)` 的结果，`state` 是**扁平的 `Dict[str, str]`**
（[evolution.py:228](https://github.com/Birfy/agentdescent/blob/main/agentdescent/evolution.py#L228) `Strategy` 协议）。
**关键洞察：`state` 的 key 完全可以是相对文件路径，value 是文件内容。**
引擎从不解释 key 的含义，它只做：

- **冲突检测**：`diffs_contradict` = 同 key 不同 value（[aggregator.py:289](https://github.com/Birfy/agentdescent/blob/main/agentdescent/aggregator.py#L289)）
- **融合**：`fuse_diffs` = 多个 diff 的 `ops` 字典合并（[aggregator.py:297](https://github.com/Birfy/agentdescent/blob/main/agentdescent/aggregator.py#L297)）
- **接受**：held-out 分数 + Beta 后验，事务性提交到 git ledger

也就是说：**两个 worker 改不同文件 → 自动融合；改同一文件 → 判为矛盾，按 held-out 分数择优。**
这正是我们想要的目录级语义，不用写一行聚合器代码。

### 1.2 真实 agent 已经是一等公民

```python
Completion = Callable[[str], str]          # agents.py:31

class WorkspaceAgent(Protocol):            # agents.py:113
    def __call__(self, prompt: str) -> str: ...
    def in_workspace(self, path: str) -> Completion: ...
```

`cli_agent(["claude","-p"]).in_workspace("/tmp/w")` 会以 `cwd=/tmp/w` 起子进程
（[agents.py:147](https://github.com/Birfy/agentdescent/blob/main/agentdescent/agents.py#L147)）。`document_agent()` 已经演示了完整套路：
`mkdtemp` → 写文件 → `in_workspace(dir)(prompt)`。我们要做的就是把「写一个文件」
换成「物化整棵目录树」。

### 1.3 治理层已经为「改 harness」准备好了

- `blast_radius <= 0.30` → L2 快层（skill）；`> 0.30` → L1 慢层（harness/agent 代码），
  每次合并强制走 oracle（[governance.py:46](https://github.com/Birfy/agentdescent/blob/main/agentdescent/governance.py#L46)）。
- `FROZEN_IDS` = L0 只读（oracle / 安全约束）——**但它是 artifact 级的 id 列表，不是路径级**。

### 1.4 已有的两个「演化 agent」示例，以及它们的诚实边界

- `examples/adas/adas_meta_agent_search.py`：演化 agent 的**控制流**，但为了安全用一个
  受限 DSL 解释器**替代了 `exec` 模型写的代码**（文件头已注明）。
- `examples/dgm/dgm_self_improve.py`：演化 coding agent 的 harness，但目标函数是
  **capability 覆盖的代理指标**，不是真跑 SWE-bench（文件头「Honesty boundary」已注明）。

结论：**「演化代码 + 真跑」这条链路在仓库里从未闭合过。**

---

## 2. 差距清单（要做的事，按重要性）

| # | 差距 | 影响 |
|---|---|---|
| G1 | 无「目录 → state」装载器与「state → 工作区」物化器 | 无法喂入/产出目录 |
| G2 | `run` 只拿到一个字符串，没有工作区生命周期 | 真实 agent 无法「用」这个 skill 目录 |
| G3 | 提案是单个字符串，无多文件编辑协议 | 反思模型没法说「改这三个文件」 |
| G4 | **`apply()` 只做 `state.update(ops)`，无法删除 key**（[evolution.py:464](https://github.com/Birfy/agentdescent/blob/main/agentdescent/evolution.py#L464)） | 无法删除/重命名文件 |
| G5 | 无路径级冻结 | agent 可以改自己的测试/评分器，指标自我作弊 |
| G6 | 无候选代码的执行沙箱与测试门 | 演化代码不安全、不可信 |
| G7 | `_signature()` = `render()` 作为评估缓存 key（[evolution.py:470](https://github.com/Birfy/agentdescent/blob/main/agentdescent/evolution.py#L470)） | `render` 若不是状态的**无损**序列化，不同目录会共用缓存分数 |
| G8 | 信任域按「op 个数 ≤ 6、单 value ≤ 32k 字符」计（[aggregator.py:86](https://github.com/Birfy/agentdescent/blob/main/agentdescent/aggregator.py#L86)） | 对文件树含义变成「一次最多改 6 个文件、单文件 32k」——需显式设定而非默认 |
| G9 | `run`/`propose` 会被多线程并发调用（`max_concurrency` worker 线程 + `eval_concurrency=8` 评估线程） | 工作区必须**每次调用独立**，不能复用固定路径 |
| G10 | 无「装回用户目录」的输出路径 | 结果只能是 JSON |
| G11 | ledger 把整个 state 存成 `artifacts/<id>.json`（[ledger.py:211](https://github.com/Birfy/agentdescent/blob/main/agentdescent/ledger.py#L211)） | 大目录/二进制文件会撑爆 commit；需要大小上限与 ignore 规则 |

---

## 3. 设计

### 3.0 总体形状

```
用户目录 ~/.claude/skills/pdf-audit/
        │  load_tree()            （新增）
        ▼
   state: {"SKILL.md": "...", "references/rules.md": "...", "scripts/check.py": "..."}
        │  FileTree strategy（新增：render / to_diff / keys / frozen_paths）
        ▼
   evolve()  ← 引擎不变
        │  每次 rollout：materialize() 到一次性 workspace（新增）
        ▼
   /tmp/ad-ws-xxxx/                          ← layout 决定目录结构
       .claude/skills/pdf-audit/SKILL.md
       .claude/skills/pdf-audit/references/rules.md
       <task fixtures…>
        │  claude_code().in_workspace(ws)(prompt)
        ▼
   agent 输出 → reward() → 反思 → 多文件提案 → Diff(ops={path: content})
        │
        ▼
   result.write_to("~/.claude/skills/pdf-audit")   （新增）
```

### 3.1 新模块一：`agentdescent/filetree.py` — 目录 ↔ state

```python
@dataclass(frozen=True)
class TreeSpec:
    """哪些文件属于这棵可演化的树。"""
    include: Sequence[str] = ("**/*.md", "**/*.py", "**/*.txt", "**/*.json", "**/*.yaml")
    exclude: Sequence[str] = ("**/.git/**", "**/__pycache__/**", "**/node_modules/**")
    #: 必须 <= AggregatorConfig.trust_region_chars（默认 32_000），否则任何超限文件
    #: 一旦被修改，diff 必然被判 OVERSIZED 丢弃 —— 一条永远走不通的路。
    #: 留出余量给序列化开销；要放宽就同时传
    #: agg_config=AggregatorConfig(trust_region_chars=N)。
    max_file_bytes: int = 28_000
    max_files: int = 200
    max_total_bytes: int = 2_000_000

    def validate_against(self, trust_region_chars: int) -> None:
        """装载前校验，而不是等到第 5 轮才在 outcomes() 里看到一堆 oversized。"""

def load_tree(path: str, spec: TreeSpec = TreeSpec()) -> Dict[str, str]:
    """目录 → {相对路径: 文本内容}。二进制/超限文件报错而非跳过。"""

def materialize(state: Mapping[str, str], dest: str, *,
                prefix: str = "", mode_map: Mapping[str, int] = ...) -> None:
    """state → 磁盘。prefix 决定装到 dest 的哪个子路径下。
    路径安全：拒绝绝对路径、`..`、符号链接逃逸。scripts/*.py 自动 chmod +x。"""

def canonical(state: Mapping[str, str]) -> str:
    """无损、稳定的序列化（用于 render 与评估缓存 key，见 G7）。"""
```

**设计要点**

- `load_tree` 对二进制/超限文件**报错**而不是跳过：静默跳过会让「演化后的目录」
  与原目录不等价，装回去就丢文件。仓库既有风格（`document_agent` 的截断告警、
  `openai_compatible` 的空 content 归一化）也是「绝不静默丢数据」。
- `materialize` 的路径校验是**安全边界**：state 里的 key 来自模型提案，必须假定敌意。

### 3.2 新模块二：`agentdescent/treestrategy.py` — `FileTree` Strategy

```python
@dataclass
class FileTree:
    """artifact 就是一棵文件树；state 的 key 是相对路径。"""
    initial_files: Dict[str, str]
    editable: Sequence[str] = ("**",)      # 允许模型改的路径 glob
    frozen: Sequence[str] = ()             # 只读：测试、评分器、安全约束（G5）
    max_files_per_diff: int = 4

    def keys(self) -> Sequence[str]:       # → TensorParallel 可按文件切分工人
        return [p for p in self.initial_files if _match(p, self.editable)]

    def initial(self) -> Dict[str, str]:
        return dict(self.initial_files)

    def render(self, state) -> str:
        return canonical(state)            # 无损，满足 G7

    def to_diff(self, state, proposal, author, base_version, target) -> Optional[Diff]:
        edits = parse_edits(proposal)      # §3.3
        edits = {p: c for p, c in edits.items()
                 if _match(p, self.editable) and not _match(p, self.frozen)
                 and state.get(p) != c}
        if not edits or len(edits) > self.max_files_per_diff:
            return None                    # 越界提案被丢弃并计入 outcomes
        return Diff(diff_id=f"{author}:{stable_hash(...)}:{base_version}",
                    target=target, ops=edits, author=author)
```

**`render` 的两难与取舍**：`render()` 同时被用作
(a) 传给 `run` 的内容、(b) 评估缓存 key（`_signature`）。对文件树来说，(a) 其实不需要
——真实 agent 从磁盘读文件，不从 prompt 读。所以 `render` 返回**规范序列化**，
`run` 收到它之后先 `parse` 回 state 再物化。这样缓存 key 天然无损，且不必改引擎。
（备选：给 `Strategy` 加可选的 `signature(state)` 钩子，改动更干净但要动 `evolution.py`。）

### 3.3 提案协议（G3）

反思模型返回一个带围栏的 JSON：

```
<EDITS>
{"rationale": "SKILL.md 没说明表格跨页时要合并", 
 "edits": [{"path": "SKILL.md", "content": "...完整新内容..."}]}
</EDITS>
```

- **整文件替换**而非 unified diff：模型产出的 patch 经常对不上上下文行，
  失败是静默的；整文件替换要么合法要么解析失败，可观测。代价是 token，
  用 `max_file_bytes` 和「把大 skill 拆成多文件」来控。
- 解析失败 → `to_diff` 返回 `None`（引擎已有的正常路径）；**不抛异常**，
  因为那是后端质量问题不是调用者 bug。
- 配套一个 `tree_reflector(complete, spec)`：把当前树 + 失败轨迹 + reward
  渲染进模板，要求只输出上述块。

### 3.4 新模块三：`agentdescent/runners.py` — 让真实 agent「用」这棵树

```python
LAYOUTS = {
    "claude_skill":  lambda name: f".claude/skills/{name}/",
    "claude_agent":  lambda name: f".claude/agents/",
    "agent_repo":    lambda name: "",          # 树就是仓库根
}

def tree_runner(agent: WorkspaceAgent, *, layout: str, name: str,
                prompt_template: str, fixtures: Optional[Callable[[Task], Dict[str, str]]] = None,
                keep_failed: bool = False, timeout: float = 600.0) -> Run:
    """返回一个 run(rendered, task) -> str。

    每次调用（G9：可能并发）：
      1. mkdtemp 一个**独立**工作区
      2. materialize(parse(rendered), ws, prefix=layout_prefix(layout, name))
      3. 写入 task 的 fixtures（task.meta 里带的输入文件）
      4. agent.in_workspace(ws)(prompt_template.format(task=..., name=name))
      5. 读取 agent stdout（或约定的 ws/ANSWER.txt）作为 output
      6. 清理工作区（失败时按 keep_failed 保留，便于排查）
    """
```

**为什么必须一次一个工作区**：`evolve(max_concurrency=N)` 用线程池跑 worker，
`score()` 另开 `eval_concurrency=8` 个线程评估（[evolution.py:500](https://github.com/Birfy/agentdescent/blob/main/agentdescent/evolution.py#L500)）。
共用固定目录会让两个候选互相覆盖，产生**看起来正常但完全错误的分数**。

**agent 代码演化的额外一层**：`code_runner(...)` 在物化后先跑
`setup_cmd`（如 `pip install -e .`）与 `test_cmd`，任一失败则 rollout 直接得 0 分并
把 stderr 写进 output 供反思使用——「改坏了」于是变成一个可学习的负信号而不是崩溃。

### 3.5 治理与安全（G5 / G6）

| 演化对象 | `blast_radius` | 层 | 门禁 |
|---|---|---|---|
| skill 目录（纯 Markdown/参考资料） | `0.2` | L2 | held-out 分数即可 |
| agent 目录（子 agent 定义、工具配置、harness 提示） | `0.6` | L1 | 每次合并走 oracle（引擎已有行为） |
| agent 代码（可执行） | `0.6` + `frozen` + 测试门 | L1 | oracle + 测试必须通过 + 沙箱执行 |

三道硬约束：

1. **路径级冻结，且必须是运行时不变量而不只是提案过滤**：
   `FileTree(frozen=["tests/**", "eval/**", "SAFETY.md"])`。
   这是文件世界里的 L0。没有它，演化 agent 代码等于允许 agent 改自己的考卷——
   `governance.py` 的模块文档已经点明这正是「估计出来的层级」抓不住的结构性事实。

   **只在 `to_diff` 里拒绝触碰 frozen 路径的提案是不够的**：它只挡住了「直接改考卷」。
   候选代码仍可在**运行期**动手脚——新增一个 `conftest.py`、改 `sys.path`、
   monkeypatch 断言、让 `pytest` 提前 `exit(0)`。真正的门是两条叠加：

   ```
   物化候选树 → 用 **原始副本** 覆盖所有 frozen 路径（overlay，后写胜） 
              → 测试/评分从树 **外部** 调用，工作目录只读挂载给被测代码
   ```

   即「提案过滤」防误改，「pristine overlay + 外部调用」防作弊。后者才是安全边界，
   前者只是让反思模型少浪费一次提案。
2. **执行隔离**：候选代码只在一次性工作区里、以子进程 + 超时运行。
   第一版用 `subprocess` + `timeout` + 独立 `TMPDIR`；文档里明确写清
   「这不是安全沙箱，只是隔离」，需要真沙箱时接 Docker/`bwrap`（留 `sandbox=` 钩子）。
3. **网络与凭据**：给 runner 传裁剪过的 `env`（`cli_agent` 已支持 `env=`），
   默认不透传 `ANTHROPIC_API_KEY` 之外的密钥。

### 3.6 需要动引擎的唯一一处：删除语义（G4）

`EvolvingArtifact.apply` 是 `new_state.update(diff.ops)`，key 只能新增/覆盖，
不能删除，所以**删文件、改文件名都表达不了**。三个选项：

| 方案 | 改动 | 代价 |
|---|---|---|
| A. 墓碑值：`ops[path] = DELETED` 哨兵，`materialize` 时跳过 | 0 行引擎改动 | state 越来越脏；`render` 要过滤；`canonical` 需处理 |
| B. `apply` 支持 `ops[k] is None` 表示删除 | `evolution.py` 约 3 行 | 影响所有 strategy（但 `None` 目前是非法 value，向后兼容） |
| C. 自定义 `Evolvable`（不用 `EvolvingArtifact`） | 大 | 要重做 `score`/`cheap_eval`/序列化 |

**推荐 B**：`apply` 遇到 `None` 时 **pop 掉这个 key**（而不是把 `None` 存进 state），
于是 state 里永远不出现 `None`，`render` / `canonical` / ledger 序列化都不用改。
配套核实过的三处：`fuse_diffs` 的 `ops.update` 语义天然正确；
信任域检查 `len(str(v))` 对 `None` 是 4，不会误判 oversized；
ledger 的 JSON 往返也不受影响（`None` 只存在于 `Diff.ops`，不进 state）。

第一版也可以先只支持「新增 + 覆盖」，把删除列为已知限制。

### 3.7 一行式入口（`agentdescent/skilldir.py`）

```python
from agentdescent import evolve_skill_dir

result = evolve_skill_dir(
    "~/.claude/skills/pdf-audit",          # 用户目录
    tasks,                                  # Task(prompt=..., meta={"fixtures": {...}, "gold": ...})
    reward=my_scorer,
    agent=claude_code(),                    # 真实 agent 执行
    reflect_with=openai_compatible(model="deepseek-v4-flash"),   # 便宜模型做反思
    layout="claude_skill",
    frozen=["tests/**"],
    n_workers=4, max_concurrency=4, rounds=8,
)
print(result.final_reward, result.outcomes())
result.write_to("~/.claude/skills/pdf-audit", backup=True)   # 装回，先备份
```

三个变体，共用同一套底座：

- `evolve_skill_dir(path, ...)` → `blast_radius=0.2`，layout `claude_skill`
- `evolve_agent_dir(path, ...)` → `blast_radius=0.6`，layout `claude_agent`
- `evolve_agent_code(path, ..., test_cmd="pytest -q")` → `blast_radius=0.6` + 测试门 + `frozen=["tests/**"]`

`EvolutionResult` 增加 `write_to(path, *, backup=True, dry_run=False)`：
默认先把原目录复制成 `<path>.bak-<version>`，`dry_run` 打印将写/将删的文件清单。
**这是本设计里唯一会写用户目录的操作，必须显式调用，`evolve` 本身永不写回。**

### 3.8 成本模型（真实 agent 下，这是首要设计约束）

一次 rollout = 一次真实 agent 调用（Claude Code 量级 30–90 秒）。数清楚每轮跑几次：

| 来源 | 次数 | 控制项 |
|---|---|---|
| worker rollout | `n_workers` | — |
| self-verify 复跑 | `n_workers`（每个提案一次） | **`self_verify=False`** |
| 冲突解析 + 融合锦标赛的排序评估 | `候选数 × cheap 集大小` | **`cheap_eval_tasks=4`** |
| Beta 接受检验（`eval_counts`，胜出者 base+cand） | `1 × |held_out|` | 无（这是提交安全线，不该省） |
| L1 oracle 门 | **0** | 见下 |
| 每轮 held-out 测量 | 0（已被缓存覆盖） | — |

两个反直觉的点，都已对代码核实：

1. **oracle 门是免费的。** `oracle_eval` 和 `eval_counts` 调同一个 `eval_fn`、同一个
   held-out 集，而 `_EvalCache` 按 `(render(), task.id)` 记忆化
   （[evolution.py:531](https://github.com/Birfy/agentdescent/blob/main/agentdescent/evolution.py#L531)），所以 L1 的强制审计只消耗
   `oracle_budget` 计数器，不产生任何额外 agent 调用。**演化 agent 目录/代码
   （L1）并不比演化 skill（L2）贵。**
2. **默认配置是最贵的配置。** `cheap_eval_tasks=None` 时廉价层被钉死成全量 held-out
   （[evolution.py:996](https://github.com/Birfy/agentdescent/blob/main/agentdescent/evolution.py#L996) 的注释已经点明「eval_fn RUNS THE
   AGENT」这个场景），排序阶段于是变成成本主项；`self_verify=True` 又给每个提案加一次。
   目录演化的入口函数应把 `cheap_eval_tasks=4`、`self_verify=False` 设为**默认值**，
   而不是留给调用者去发现。

**超时**：靠 `cli_agent(timeout=...)`（`subprocess.run` 会真正杀掉进程），不要靠
`round_timeout` —— 后者只放弃等待，线程和子进程还在跑
（[evolution.py:1466](https://github.com/Birfy/agentdescent/blob/main/agentdescent/evolution.py#L1466) 的注释即此意）。

### 3.9 评估噪声：缓存把单样本变成了真值

`_EvalCache` 让每个 `(树, 任务)` 组合**终生只评估一次**。对确定性 scorer 这是纯赚；
对真实 agent 则意味着**单次采样被 Beta 后验当作真值**，方差被系统性低估，噪声会
被当成改进接受。三个缓解手段，按代价排序：

1. `temperature=0`（provider 支持时），把随机性压到最低；
2. `reward` 内部做 k 次多数表决（成本 ×k，但只作用在你真正在意的门上）；
3. 加大 held-out —— 引擎在 `< 4` 时已经会告警（[evolution.py:935](https://github.com/Birfy/agentdescent/blob/main/agentdescent/evolution.py#L935)），
   但对随机 agent，4 个任务远远不够。

这不是可以「以后再说」的实现细节：它决定了报出来的 `final_reward` 是否可信。

---

## 4. 并行语义（免费获得的部分）

- **数据并行**（默认）：每个 worker 拿一份任务分片，各自提案。
- **张量并行**：`FileTree.keys()` 返回所有可编辑文件路径 → `TensorParallel(n_sections=4)`
  给每个 worker 一个**互斥的文件子集**，越界提案被计为 `section-violation`
  （[evolution.py:588](https://github.com/Birfy/agentdescent/blob/main/agentdescent/evolution.py#L588) `_resolve_sections` 已经会校验这个配对）。
  对「一个 skill 目录里 SKILL.md / references / scripts 各由一个 worker 负责」是天然匹配。

  **但 TP 下无法新建文件。** 违规判定是 `section_map.get(k) != unit.section`
  （[evolution.py:1436](https://github.com/Birfy/agentdescent/blob/main/agentdescent/evolution.py#L1436)），而 `section_map` 由
  `strategy.keys()` 在**首轮之前**一次算定；一个新路径的 `section_map.get(...)` 是
  `None`，必然 `!= section`，于是每个新建文件的提案都被计为 `section-violation`。
  三个选项：(a) TP 只用于「精修已有文件」阶段，新建文件走 DataParallel；
  (b) `FileTree(planned_paths=[...])` 预先声明将来可能出现的空文件位，让它们进入 key 空间；
  (c) 默认就用 DataParallel（**推荐**，除非文件数明显大于 worker 数且都已存在）。
- **异步**：`asynchronous=True` 直接可用（记得显式给 `max_seconds`，否则默认 20 秒）。
- **融合的粒度问题**：以整文件为 key 意味着两个 worker 对同一个 `SKILL.md` 的
  *互补*修改会被判为矛盾、只留一个。缓解办法是**鼓励把 skill 拆成多文件**
  （`SKILL.md` 只放路由，细则进 `references/*.md`），这本来也是写 skill 的最佳实践。
  更细的粒度（文件内小节做 key）留作后续：`SectionedFileTree`，key 形如
  `SKILL.md#错误处理`。

---

## 5. 把现有算法示例迁移到这套底座

这一节既是路线图，也是**对抽象的检验**：如果 `FileTree` 只能服务新功能、表达不了仓库
里已有的六个算法端口，那说明抽象选错了。实际检查下来，它能吃掉其中四个自定义 Strategy。

### 5.1 现状：每个例子都自己写了一个 Strategy

| 例子 | 自定义 Strategy | state 形状 | 本质 |
|---|---|---|---|
| EvoSkill | `SkillLibraryStrategy` → **`SkillLibraryTree`** | `{skill_name: 正文}` → `{skills/<name>/SKILL.md: 正文}` | **就是一个多 skill 目录** |
| ADAS | `AgentDesignStrategy` | `{design, name, thought}` | 一份 agent 程序（DSL JSON）+ 元数据 |
| DGM | `HarnessStrategy` | `{capabilities: "a,b,c"}` | agent 能力集的代理表示 |
| ACE / GEPA / SkillOpt | 各自的单槽变体（现已统一到 `SingleSlot`） | `{value: 文本}` | 单文件 |

### 5.2 EvoSkill —— 第一梯队，几乎是零语义改动

`SkillLibraryStrategy`（已替换为 `SkillLibraryTree`）的 state 是 `{skill 名: SKILL.md 正文}`
（[evoskill_skill_discovery.py:379](https://github.com/Birfy/agentdescent/blob/main/examples/evoskill/evoskill_skill_discovery.py#L379)），
它的 Generator 提示词里甚至直接写着「Output a SKILL.md body」。这**已经是**一个
skill 目录，只是活在内存里、靠 `render_skills()` 拼进 prompt。迁移只是换个 key：

```
{"treasury-tables": "..."}   →   {"skills/treasury-tables/SKILL.md": "..."}
```

顺带一个值得注意的印证：EvoSkill 里的 `_parse_rendered_skills(rendered)` 正是
「`render` 无损 → 在 `run` 里解析回来」——也就是 §3.2 提出的做法。**这不是我发明的模式，
是这个仓库已有的惯例**，这让 `FileTree.render = canonical` 的选择更有底气。

**迁移的实质收益不在代码整洁，而在能力**：EvoSkill 的 OfficeQA 路径已经在用
`document_agent(claude_code())`，也就是已经有一个真实工作区了
（[backends.py:104](https://github.com/Birfy/agentdescent/blob/main/agentdescent/backends.py#L104)）。把 skill 树一并物化进去之后，
agent 可以**按需读取**它需要的那个 skill 文件，而不是把整个 skill 库塞进 prompt。
这才是「skill 目录」相对于「prompt 拼接」的实质差别（渐进披露），也是当前实现拿不到的东西。

EvoSkill 自带的聚合器与 `FileTree` 正交，原样保留。（写下这条时 EvoSkill 有两个：
`TopKFrontierAggregator` 和按 `asynchronous` 切换的 `SgdSkillAggregator`。后者已被
删除 —— 按调度切换聚合器等于让 async 那一格测量的是另一个算法，见
[port fidelity](port-fidelity.md#evoskill-automated-skill-discovery)。）

### 5.3 ADAS / DGM —— 第三梯队，高价值但也是最大的安全面

这两个例子的文件头都写着自己的诚实边界：ADAS 用受限 DSL **替代了 `exec` 模型写的代码**，
DGM 用 **capability 覆盖的代理指标**替代真实 SWE-bench。迁到 `FileTree` + `code_runner`
之后，这两条边界**正好是本设计要兑现的东西**：

- ADAS：`{"design": <DSL JSON>}` → `{"agent.py": <真实代码>}`，在一次性工作区 + frozen
  overlay + 超时下**真执行**；
- DGM：`{"capabilities": "a,b,c"}` → 真实 agent 代码树 + `test_cmd`，把代理指标换成真实通过率。

但要说清楚：**这不是免费的**。两者的核心是各自的聚合器（keep-all archive、
sigmoid×novelty 父代选择、staged 升级），换 substrate 不改算法，但需要一个真实评测环境
（DGM 是 SWE-bench 的 Docker harness，~1.5h/任务）。所以这一梯队的门槛是**评测基础设施**，
不是本设计的这几个模块。建议：先做 ADAS（MGSM 上跑真实 `agent.py`，秒级任务，可离线验证），
DGM 保持代理指标不动，只在文档里指明接入点。

### 5.4 ACE / GEPA / SkillOpt —— 不迁移

单文本槽用 `SingleSlot` 表达最清楚；套一层文件树只会增加概念负担，没有收益。
`FileTree` 是 `SingleSlot` 的超集这件事，写在文档里就够了。

### 5.2.1 迁移之后的结论（已验证）

迁移做完了，而且**比设计时预计的便宜**：我原以为「FileTree 的 render 必须无损」和
「EvoSkill 的 render 是 prompt 文本」是硬冲突，只能二选一（改 prompt 格式 / 改 Strategy 协议）。
实际不是 —— **`run` 才是把 artifact 变成 prompt 的地方**（框架自己的文档就是这么说的：
"The framework never injects the artifact into your prompt — you do"）。所以：

- `render()` = canonical（无损，当缓存 key）；
- `run()` 里 `render_skills(skills_of(parse_tree(rendered)))` 把它变回
  `### skill: <name>` 格式 —— **prompt 逐字节不变**，有测试守着；
- `to_diff` 覆写成接受 repo 原本的 `name :: body`，不用 `<EDITS>` JSON ——
  EvoSkill 忠实的部分是两角色 Proposer/Generator induction，不是名字和正文之间的分隔符；
  换协议会改 Generator 的 prompt，那才是真的改变了被测量的东西。

12 个原有测试一行没改、全绿；新增 3 个测试守住「state key 是真路径」「prompt 不变」
「一次一个 skill」。**抽象验收通过。**

真正拿到的新能力：`document_agent` 新增 `skill_files=` —— 工具型 backend
（`--backend claude-code` / `openhands`）现在把 skill 库物化进 agent 的工作区，
prompt 里只放一句指路，agent 自己按需打开。以前是每个问题都把整个 skill 库塞进 prompt。

### 5.5 迁移的排序原则

先迁**能证明新能力**的（EvoSkill：按需读取 skill 文件），
再迁**能兑现已声明边界**的（ADAS：真执行），
不迁**只会变复杂**的（ACE/GEPA/SkillOpt）。
每一步都保留原例子的算法与聚合器不动 —— 换的是 substrate，不是算法。

---

## 6. 实施计划

| 阶段 | 内容 | 交付 | 规模 |
|---|---|---|---|
| P0 | `filetree.py`：`load_tree` / `materialize` / `canonical` + 路径安全测试 | 目录能进能出 | ~180 行 + 测试 |
| P1 | `FileTree` strategy + `parse_edits` + `tree_reflector` | 多文件提案跑通 | ~200 行 + 测试 |
| P2 | `runners.py::tree_runner` + `evolve_skill_dir` + `result.write_to` | **skill 目录端到端**（`echo()` 假 agent 离线可测） | ~200 行 + 测试 |
| P3 | 真实 agent 打通：`claude_code()` 冒烟示例 `examples/skill_dir_evolution.py` | 真跑一遍 | ~120 行 |
| P4 | `evolve_agent_dir` + 路径冻结 + L1 治理接线 | agent 目录 | ~80 行 |
| P5 | `code_runner`（setup/test 门 + 隔离 + 超时）+ `evolve_agent_code` | agent 代码 | ~200 行 + 测试 |
| P6 ✅ | **迁移 EvoSkill 到 `FileTree`**（§5.2）：skill 库 → skill 目录，工具型 backend 改为按需读取 | 证明抽象够用 | 实际 ~90 行 |
| P7 | 删除语义（§3.6 方案 B）、`SectionedFileTree`、Docker 沙箱钩子 | 收尾 | — |
| P8 | （可选，取决于评测基建）ADAS 迁到真实 `agent.py` 执行（§5.3） | 兑现文件头的诚实边界 | — |

P0–P3 是最小可用闭环（约 1–2 天）；**P6 建议紧跟 P3**——它是对抽象的验收，
如果 EvoSkill 迁不过去，说明 `FileTree` 的形状要改，越早发现越好。
P5 是风险最高的一段，建议在 P3 的真实数据上先看清楚反思模型对多文件提案的服从度再动。

**不需要改动**：`aggregator.py`、`ledger.py`、`async_evolve.py`、`parallel.py`、
`governance.py`。**需要小改**：`evolution.py`（仅 §3.6 的删除语义，且可延后）、
`__init__.py`（导出）。

---

## 7. 验收标准

1. `load_tree(materialize(state))` 对任意合法 state 恒等（round-trip 属性测试）。
2. `materialize` 拒绝 `../etc/passwd`、`/etc/passwd`、符号链接逃逸（安全测试）。
3. 用 `echo()` 假 agent + 合成任务，`evolve_skill_dir` 在 8 轮内把 held-out
   分数从 0 提到 1，且 `outcomes()["committed"] > 0`（离线、无 API key、CI 可跑）。
4. 两个 worker 改不同文件 → `outcomes()` 出现融合提交；改同一文件 → 矛盾被解析，
   分高者胜（复用 `domains/router.py` 的验证套路）。
5. `frozen=["tests/**"]` 下，任何触碰 `tests/` 的提案都不产生 diff（**必须有专门测试**）。
6. `max_concurrency=8` 时并发跑 100 次 rollout，无工作区串扰（每次 workspace 唯一）。
7. `write_to(dry_run=True)` 精确列出将写/将删文件；`backup=True` 产出可回滚副本。
8. **抽象验收**：EvoSkill 迁到 `FileTree` 后，离线测试结果与迁移前一致
   （同 seed 同分数），且删掉 `SkillLibraryStrategy`。迁不过去 = 抽象形状要改。
9. **作弊验收**：给 `evolve_agent_code` 一个「把测试改成 `assert True`」的人造提案，
   确认它既进不了 diff（提案过滤），即便手工注入 state 也因 pristine overlay 而无效。

---

## 8. 风险与开放问题

| 风险 | 缓解 |
|---|---|
| **成本**：每个 rollout 起一个真实 coding agent | §3.8 的成本表；入口函数默认 `cheap_eval_tasks=4`、`self_verify=False`；反思用便宜模型；`Usage` 计量并在示例里打印 |
| **评估噪声被缓存固化成真值** | §3.9：`temperature=0` / k 次多数表决 / 加大 held-out |
| 反思模型不遵守 `<EDITS>` 协议 | 解析失败即丢弃并计数；示例里报告服从率；必要时降级为「只允许改一个文件」 |
| 整文件替换的 token 开销 | `max_file_bytes=28k`（与信任域对齐）+ 鼓励拆文件；超限文件标为不可编辑 |
| 演化代码的安全性 | 一次性工作区 + 子进程 + 超时 + 裁剪 env；文档明说「隔离≠沙箱」；`sandbox=` 钩子留给 Docker |
| 指标自我作弊（agent 改测试） | 双层：`frozen` 提案过滤 + **pristine overlay / 树外调用测试**（§3.5），后者才是安全边界 |
| **临时工作区泄漏**：引擎只回收自己的 scratch git repo，不管 runner 的工作区 | `try/finally` 删除 + 仿 `_reap_stale_scratch_repos` 写一个按前缀+age 的回收器；`keep_failed=True` 时只保留失败的 |
| **`claude -p` 非交互权限**：默认会因工具授权卡住或拒绝 | `claude_code(extra_args=["--permission-mode", "acceptEdits", "--allowedTools", ...])`；先跑一次冒烟确认不会挂 |
| **实验效度**：skill 可能根本没被加载，测到的是「agent 会不会去找它」 | prompt 模板显式指向 skill 路径；跑一组「空 skill 目录」对照，确认分数确实有差 |
| ledger 膨胀（每个版本存整棵树的 JSON，[ledger.py:211](https://github.com/Birfy/agentdescent/blob/main/agentdescent/ledger.py#L211)） | `TreeSpec` 的 `max_total_bytes`；git 自身会 delta 压缩；必要时改存内容哈希 + blob |
| **过拟合 held-out**：目录级 artifact 表达能力强，容易记住答案 | 保留 `held_out_frac`，并在示例里额外留一个从不参与任何门禁的 test 集（EvoSkill/ADAS 的 50/25/25 划分已是仓库惯例） |

**待你确认的四个决定**

1. **首要目标是哪一个**：skill 目录（最简单、收益直接）、agent 目录、还是 agent 代码
   （最难、最需要沙箱）？建议按 P0→P3 先把 skill 目录闭环，P6 立刻用 EvoSkill 验收抽象。
2. **执行 agent 用哪个**：`claude_code()`（本机 CLI）还是 `openhands()`（需 Python ≥3.12）？
   前者零依赖，建议默认。
3. **删除语义**是先按「已知限制」搁置，还是这一版就做 §3.6 方案 B？
4. **ADAS 要不要真执行**（§5.3）：这会把它从「安全 DSL」变成「真跑模型写的代码」，
   收益是兑现文件头声明的边界，代价是引入本仓库目前没有的执行安全面。
   我的建议是**做**，但必须先有 P5 的 `code_runner`（一次性工作区 + frozen overlay +
   超时 + 裁剪 env），且默认不联网。
