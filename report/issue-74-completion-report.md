# Issue #74 完成报告

日期：2026-08-04
Issue：<https://github.com/Birfy/agentdescent/issues/74>
状态：本地实现、调试与 PR 创建完成，等待上游 review
PR：<https://github.com/Birfy/agentdescent/pull/85>

## 1. 完成结论

Issue #74 要求的公共入口契约、移植模板、忠实度 checklist、候选算法清单和
CONTRIBUTING 最短验证路径均已在本地完成。

这次改动没有修改 `agentdescent/` 核心引擎。六个算法原有的 Strategy、Aggregator、
reward、数据集选择和真实运行路径保持不变，改动集中在 CLI 入口、严格离线 dry-run、
测试与文档。

## 2. Issue 验收项

| 验收项 | 状态 | 实现位置 |
|---|---|---|
| 公共 parser | 完成 | `examples/_common.py` 的 `add_standard_args()` |
| 六个现有移植接入公共 parser | 完成 | 六个算法 example 的 `build_parser()` |
| DGM 补齐 `--yes` | 完成 | 公共 parser；DGM 仅在设置 `--model` 时确认 API 调用 |
| 保留上游迭代术语 | 完成 | ACE/GEPA=`rounds`，EvoSkill=`iterations`，SkillOpt=`steps`，ADAS/DGM=`generations` |
| `--dry-run` 零网络、零 API Key | 完成 | 六个入口均在 loader 和模型 adapter 前返回 |
| 可复制的代码模板 | 完成 | `examples/_TEMPLATE.py` |
| 配套离线测试桩 | 完成 | `tests/_TEMPLATE_example.py` |
| 20 行以内的 checklist | 完成 | `docs/porting-checklist.md`，共 16 行 |
| 按机制整理候选算法 | 完成 | `docs/self-evolution-examples.md` |
| 记录现有移植作者 | 完成 | 六个现有移植均记录为 `chendanyang` |
| CONTRIBUTING 最短验证路径 | 完成 | clone、install、pytest、run_demo、改一行重跑 |

## 3. 公共 CLI 契约

新增 `examples/_common.py`，集中定义：

- `--provider`
- `--model`
- `--seed`
- `--async`
- `--async-ratio`
- `--max-seconds`
- `--dry-run`
- `--yes`

`add_standard_args()` 允许移植保留自己的 `model_default` 和
`max_seconds_default`。因此 DGM 仍默认 `model=None`，不会因为标准化而变成默认付费模型
调用；各算法原有异步时间预算也没有被错误统一。

六个入口都新增了可测试的 `build_parser()` 和 `main(argv=None)`。公共参数只由 helper
注册一次，算法专属参数仍保留在原文件中。

## 4. 零网络 dry-run

原来的 dry-run 会先加载数据集，只是不调用模型。在冷缓存环境中仍可能访问 Hugging
Face、GitHub raw 文件或 gated dataset，不符合 issue 的“零网络”要求。

现在六个入口统一采用以下顺序：

```text
解析和校验纯 CLI 参数
-> 打印算法、数据集身份和参数计划
-> dry-run 立即返回
-> 加载数据集
-> 创建模型并运行
```

因此 dry-run 不再展示真实样例或依赖数据量才能计算的精确预算，但可以在空缓存、无
API Key、无网络的机器上稳定执行。相关 README 和 docs 中旧的“dry-run 会加载数据集”
说明已经同步修正，并明确区分不属于这六个 faithful ports 的其他 examples。

## 5. 移植模板

`examples/_TEMPLATE.py` 是可 import、可执行的模板，包含：

- 公共 parser 的正确使用方式；
- 保留上游迭代词汇的显式 hook；
- 通过 `agentdescent.dataloader.hf_rows` 加载数据；
- `AppendRules` Strategy，并提示可选的 `KeyedRules`、`SingleSlot`、`FileTree`；
- `agentdescent.rewards.exact_match` reward；
- `LLMAgent`、`evolve()` 和 usage 统计；
- `rendered`、`final_reward`、`outcomes()`、`stop_reason`、`error` 输出；
- loader/model 之前的 dry-run 返回；
- 上游 released code、数据集、作者和偏离记录占位符。

`tests/_TEMPLATE_example.py` 展示 Strategy、reward、数据整形和离线 dry-run 测试。文件名
以下划线开头，不会被 pytest 自动当作正式第七个移植收集。

## 6. 入口契约测试

新增 `tests/test_example_entrypoints.py`，覆盖：

1. 公共参数的类型、字段名和可覆盖值。
2. 六个入口的默认 provider/model/seed/async/max-seconds。
3. 每个入口恰好调用一次公共 helper。
4. 公共参数不会被本地 parser 重复声明。
5. 六个移植各自的迭代术语没有被统一错。
6. 六个 `main(["--dry-run"])` 不调用数据 loader、模型 adapter、`urllib` 或 socket。
7. 删除 `OPENAI_API_KEY` 和 `ANTHROPIC_API_KEY` 后 dry-run 仍能执行。
8. checklist 始终不超过 20 行。
9. 模板可 import，parser 和 dry-run 可运行。

六个原有算法测试文件全部保留，继续负责算法忠实度和纯逻辑测试。

## 7. 文档与候选算法

新增的候选表按机制组织，而不是按论文热度组织：

| 机制 | 候选 | 当前覆盖 |
|---|---|---|
| 进化 / 程序搜索 | AlphaEvolve/OpenEvolve、PromptBreeder、AFlow | 无 |
| 反思 / 文本梯度 | TextGrad、Reflexion、Self-Refine | GEPA 部分覆盖 |
| 技能 / 终身学习 | Voyager、SkillWeaver | EvoSkill 接近 |
| 自博弈 / 无标注数据 | Absolute Zero、R-Zero、Agent0 | 无，最高优先级 |
| 自改代码 | SICA、Godel Agent | DGM |

候选项 owner 当前为 `TBD`，避免虚构负责人。现有六个移植增加作者列，方便后续忠实度
判断出现问题时找到最初做过上游代码对照的人。

`docs/porting-checklist.md` 已加入 `mkdocs.yml` 导航，避免孤立页面和链接检查失败。

## 8. 第二轮 debug 记录

第二轮独立调试中检查了两个疑点：

1. 最初调试断言把 GEPA 误认为应使用 `--generations`。重新读取 issue 原文后确认 GEPA
   上游术语就是 `--rounds`，当前实现正确，因此只修正了调试断言，没有错误修改源码。
2. 发现 `docs/install.md` 的新说明已经限定为“六个 faithful ports”，但示例仍包含旧的
   `examples.skill_evolution`。该入口不属于六个标准化移植，现已换成
   `examples.ace_context_evolution`，并重新通过文档测试和严格构建。

除这处文档不一致外，第二轮没有发现新的实现缺陷。

## 9. 验证结果

### 9.1 全量与针对性测试

| 检查 | 结果 |
|---|---|
| `pytest --collect-only -q` | 在最新 `origin/main` 基线上收集 820 个测试 |
| `pytest -q` | 退出码 0，0 failed，11 skipped |
| 六个算法及 fidelity 针对性测试 | 通过 |
| 最终入口、模板、docs/API 针对性测试 | 38 passed |
| `python -m compileall -q examples tests` | 通过 |
| `python -m pip check` | `No broken requirements found` |

pytest 的 warning 均来自仓库已有的小 held-out 集合提示和异步轮次语义提示，没有新增异常。

### 9.2 零网络验证

使用 `env -i`、临时空 HOME、临时空缓存且不提供任何 API Key，实际启动六个模块：

- 六个 `--dry-run` 全部退出码 0；
- 六个 `--help` 全部退出码 0；
- 所有标准参数均存在；
- 各算法迭代参数与 issue 原文一致。

随后使用 `strace -f -e trace=network` 监听六个 dry-run 和模板 dry-run。所有进程均退出码
0，且没有报告任何网络 syscall。这同时验证了测试 mock 之外的真实进程行为。

### 9.3 模板真实路径

模板还使用四条内存伪数据和一个本地伪 completion 真正执行了一轮 `evolve()`，成功输出：

- rendered artifact；
- `final_reward=1.000`；
- outcomes；
- stop reason；
- error 状态；
- usage summary。

这证明模板不仅可以 import 和 dry-run，替换占位数据后真实控制流也能运行。

### 9.4 文档和工作区

| 检查 | 结果 |
|---|---|
| `mkdocs build --strict` | 通过 |
| `git diff --check` | 通过 |
| checklist 行数 | 16，满足不超过 20 行 |
| staged 文件 | 无 |
| 核心 `agentdescent/` 变更 | 无 |

MkDocs 唯一额外输出是 Material for MkDocs 关于未来 MkDocs 2.0 的上游提示，不是本项目
文档 warning，也没有导致 strict build 失败。

## 10. 文件变更

新增：

- `examples/_common.py`
- `examples/_TEMPLATE.py`
- `tests/_TEMPLATE_example.py`
- `tests/test_example_entrypoints.py`
- `docs/porting-checklist.md`

修改入口：

- `examples/ace_context_evolution.py`
- `examples/gepa_prompt_evolution.py`
- `examples/evoskill_skill_discovery.py`
- `examples/skillopt_skill_training.py`
- `examples/adas_meta_agent_search.py`
- `examples/dgm_self_improve.py`

修改文档：

- `CHANGELOG.md`
- `CONTRIBUTING.md`
- `README.md`
- `docs/agents.md`
- `docs/algo-adas.md`
- `docs/dataloader.md`
- `docs/install.md`
- `docs/results.md`
- `docs/self-evolution-examples.md`
- `docs/usage.md`
- `mkdocs.yml`

## 11. 当前 Git 与 PR 状态

- 当前分支：`feat/74-port-entry-contract`
- 当前 HEAD：`e5d8790`（`feat: standardize faithful port entrypoints`）
- 基线：最新 `origin/main`，分支相对上游为 0 behind / 1 ahead
- fork：`cyanneko/agentdescent`
- 已推送分支：`cyanneko:feat/74-port-entry-contract`
- PR：<https://github.com/Birfy/agentdescent/pull/85>
- PR base/head：`Birfy:main` <- `cyanneko:feat/74-port-entry-contract`
- PR 状态：OPEN、非 draft、MERGEABLE；创建时 CI 尚未上报状态
- 没有 staged 内容
- 唯一未提交 tracked 变更仍是用户原有的 `.gitignore`

`.gitignore` 中的 `report` 是用户此前已有的本地修改，本次实现没有改动该规则。因为
`report/` 被忽略，本报告不会出现在普通 `git status` 中。

## 12. 审阅重点

提交前建议按以下顺序审阅：

1. `examples/_common.py` 是否符合项目希望长期维护的 CLI 契约。
2. 六个入口的 dry-run 提前返回是否接受“不再加载真实样例和精确预算”的行为变化。
3. `examples/_TEMPLATE.py` 和 checklist 是否足以约束下一位移植作者。
4. 候选算法的机制分类、优先级和 `TBD` owner 是否符合项目安排。
5. 文档中现有六个移植作者统一记录为 `chendanyang` 是否需要改成 GitHub handle。

PR 已按用户确认创建。后续修改、处理 review 或合并前仍应先核对 CI 与上游反馈。
