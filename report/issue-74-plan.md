# Issue #74 实施方案

日期：2026-08-04
Issue：<https://github.com/Birfy/agentdescent/issues/74>
文档性质：编码前方案，以下内容保留当时的分析与实施思路
当前状态：实现已提交为 [PR #85](https://github.com/Birfy/agentdescent/pull/85)，详见
[Issue #74 完成报告](issue-74-completion-report.md)

## 1. 结论

这个 issue 应当被当成“六个算法移植的入口契约建设”，而不是一次核心引擎重构。

我会把改动限制在 `examples/`、对应测试、文档和 `CONTRIBUTING.md`。`agentdescent/`
里的 `evolve()`、Aggregator、Ledger、Strategy 协议都不需要修改。

完成后的目标是：

1. 六个移植共享同一套通用 CLI 参数定义。
2. 每个移植仍保留上游自己的迭代术语和算法专属参数。
3. `--dry-run` 在空缓存、无 API Key 的环境中也绝不访问网络。
4. 新移植可以从一个可复制的模板和一份短 checklist 开始。
5. 候选算法、机制空白和移植负责人有一个明确的记录位置。

## 2. 当前代码的实际情况

六个目标入口是：

- `examples/ace_context_evolution.py`
- `examples/gepa_prompt_evolution.py`
- `examples/evoskill_skill_discovery.py`
- `examples/skillopt_skill_training.py`
- `examples/adas_meta_agent_search.py`
- `examples/dgm_self_improve.py`

它们都重复声明了 `--provider`、`--model`、`--seed`、`--async`、
`--async-ratio`、`--max-seconds` 和 `--dry-run`。前五个还有 `--yes`，DGM
没有。

算法自身的迭代参数刻意不同，必须保留：

| 移植 | 迭代参数 | model 默认值 | max-seconds |
|---|---|---:|---:|
| ACE | `--rounds` | `claude-haiku-4-5` | 30 |
| GEPA | `--rounds` | `claude-haiku-4-5` | 45 |
| EvoSkill | `--iterations` | `claude-haiku-4-5` | 40 |
| SkillOpt | `--steps` | `claude-haiku-4-5` | 40 |
| ADAS | `--generations` | `claude-haiku-4-5` | 60 |
| DGM | `--generations` | `None`，允许确定性离线运行 | 15 |

目前六个 `main()` 都在判断 `args.dry_run` 之前加载数据。数据虽然统一经过
`agentdescent.dataloader`，但冷缓存时会使用 `urllib`、Hugging Face 或 gated
dataset 访问网络。因此现在的真实语义只是“不调用模型 API”，不满足 issue 要求的
“零网络”。

另一个现状是 parser 写在 `main()` 内部，测试无法直接取得 parser，只能修改
`sys.argv` 或启动子进程。这会让公共入口契约很难被精确测试。

## 3. 公共 parser 的设计

新增 `examples/_common.py`，核心接口严格使用 issue 指定的名字：

```python
PROVIDER_CHOICES = ("claude", "openai", "glm")
DEFAULT_MODEL = "claude-haiku-4-5"

def add_standard_args(
    parser,
    *,
    model_default=DEFAULT_MODEL,
    max_seconds_default=30.0,
):
    ...
    return parser
```

该函数统一添加：

- `--provider`
- `--model`
- `--seed`
- `--async`，目标字段仍为 `args.asynchronous`
- `--async-ratio`
- `--max-seconds`
- `--dry-run`
- `--yes`

Issue 正文把前七个称为“七个公共 flag”，但又把 DGM 缺少 `--yes` 当成重复定义已经
漂移的直接证据。我的处理是把 `--yes` 也放进公共 helper；否则只能统一七个参数，却
不能解决 issue 自己指出的缺陷。将来 #73 的 `--serial` 也只需要在这个 helper 增加
一次。

`model_default` 和 `max_seconds_default` 是必要的窄参数，不建立额外配置对象。这样参数
的名字、类型、choices 和帮助文字只有一个定义，同时保留各算法已经公开的默认行为，
尤其不能把 DGM 的 `model=None` 改成默认付费调用模型。

## 4. 六个入口如何改

每个文件增加一个很小的 `build_parser()`，算法专属参数仍留在自己的文件：

```python
def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_standard_args(parser, max_seconds_default=45.0)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--workers", type=int, default=3)
    return parser

def main(argv=None):
    args = build_parser().parse_args(argv)
    ...
```

`main(argv=None)` 保持命令行行为不变，又允许测试直接传参数，不再污染全局
`sys.argv`。这不是公共库 API 变化，因为 `examples` 本来就不随 wheel 发布。

DGM 会得到 `--yes`。只有设置了 `--model`、确实准备访问 API 时才显示确认；不传
`--model` 的确定性运行不能突然出现确认提示。

不会统一这些内容：

- `--rounds` / `--generations` / `--iterations` / `--steps`
- worker、frontier、archive、dataset、hard mode 等算法专属参数
- 每个算法的计划输出、Strategy 和 Aggregator

## 5. `--dry-run` 的零网络保证

仅抽 parser 不能保证零网络，必须同时调整控制流。我会让六个入口在任何 dataset
loader、模型 adapter 和 API probe 之前处理 `--dry-run`。

推荐顺序：

```text
parse args
-> 校验纯 CLI 参数
-> 打印算法、数据集身份和基于参数可计算的计划
-> dry-run 立即返回
-> 加载真实数据集
-> 打印样例和精确 split/call budget
-> 用户确认
-> 创建模型并运行
```

这意味着 dry-run 不再承诺“下载并展示真实样例”，而是承诺更强且可验证的“展示运行
计划，不碰网络”。相关 help、模块 docstring 和文档中“load dataset”的旧措辞需要一起
修正。

我不会给 dataloader 增加全局 offline 模式，也不会为六个 benchmark 各造一套假数据。
前者扩大核心 API，后者容易让 dry-run 输出被误认为真实 benchmark；对这个 issue 来说，
在加载前返回更简单也更诚实。

## 6. 模板

新增 `examples/_TEMPLATE.py`，它应当是语法有效、可复制的骨架，明确展示：

- `add_standard_args(parser)` 的使用方式。
- 一个保留上游术语的迭代参数 hook，例如模板中的 `--iterations`。
- 通过 `agentdescent.dataloader` 加载和切分数据，而不是自行写 HTTP。
- 从 `AppendRules`、`KeyedRules`、`SingleSlot` 或 `FileTree` 选择一个 Strategy。
- 优先使用 `agentdescent.rewards.exact_match`、`contains`、`last_number` 或
  `numeric_close`。
- `--dry-run` 必须位于 dataset loader 前。
- `evolve()` 调用和最终结果中的 `error`、`stop_reason`、`outcomes()` 报告。
- 需要记录上游 released code、数据集和已知偏离的位置。

同时新增 `tests/_TEMPLATE_example.py` 作为不会被 pytest 自动收集的测试桩，展示至少三类
测试：Strategy diff、reward/数据整形纯逻辑、dry-run 网络禁用。模板文件用下划线开头，
不会被误认为第七个正式算法移植。

## 7. 测试方案

新增 `tests/test_example_entrypoints.py`，把入口约定本身当作公共契约测试。

测试内容：

1. 六个 `build_parser()` 都接受八个标准参数，并映射到相同字段和类型。
2. 每个参数只出现一次，避免 helper 与本地 parser 重复注册。
3. 各移植原有默认值保持不变，特别是六个不同的 `max_seconds` 和 DGM 的
   `model is None`。
4. 迭代词汇不被错误统一：ACE/GEPA 是 rounds，ADAS/DGM 是 generations，EvoSkill
   是 iterations，SkillOpt 是 steps。
5. DGM 接受 `--yes`；没有 `--model` 时不会要求 API 确认。
6. 参数化调用六个 `main(["--dry-run"])`，同时把 dataset loader、
   `urllib`/socket 入口、`claude` 和 `openai_compatible` 替换为“一旦调用就失败”的
   sentinel。测试必须在没有 API Key 时通过。
7. 模板本身可 import，parser 可构建，dry-run 路径可执行。

第 6 条是关键验收：不能只断言输出里出现了 `dry-run`，必须证明所有可能产生外部访问
的边界都没有被调用。

保留六个现有 `tests/test_<algo>_example.py`，它们继续验证算法本身的忠实逻辑；新测试只
负责入口协议，不把两类职责混在一起。

## 8. 文档和候选清单

新增 `docs/porting-checklist.md`，正文控制在 20 行以内，逐项覆盖 issue 给出的七条验收
标准，并补一项“记录移植作者/维护人”。把它加入 `mkdocs.yml` 的
“Self-evolution algorithms”导航，否则现有 docs 测试会把它判为 orphan page。

候选算法表不塞进 20 行 checklist，而是放到 `docs/self-evolution-examples.md` 当前
“Not yet ported”位置，按机制记录：

| 机制 | 候选 | 当前覆盖 | owner |
|---|---|---|---|
| 进化 / 程序搜索 | AlphaEvolve/OpenEvolve、PromptBreeder、AFlow | 无 | TBD |
| 反思 / 文本梯度 | TextGrad、Reflexion、Self-Refine | GEPA 部分覆盖 | TBD |
| 技能 / 终身学习 | Voyager、SkillWeaver | EvoSkill 接近 | TBD |
| 自博弈 / 无标注数据 | Absolute Zero、R-Zero、Agent0 | 无，优先级最高 | TBD |
| 自改代码 | SICA、Godel Agent | DGM | TBD |

现有六个移植的 Git 历史作者目前都是 `chendanyang`。我会在正式文档使用名字而不是根据
邮箱猜 GitHub handle；如果项目希望记录 handle，应在提交前确认一次。未来候选没有认领
者时明确写 `TBD`，不虚构 owner。

`docs/port-fidelity.md` 当前还不存在，因此 checklist 不会添加一个会触发链接测试失败的
内部链接；可以先链接 #73，等该页面落地后再换成内部链接。

## 9. CONTRIBUTING 和 Changelog

在 `CONTRIBUTING.md` 开头附近加入一条真正连续的最短验证路径：

```text
clone -> pip install -e ".[dev]" -> pytest -q
-> python -m examples.run_demo
-> 修改 run_demo 中一个可观察参数并重跑，确认输出变化
```

建议用 `rounds = 40` 改成较小值作为“改一行”练习，因为变化直观、不涉及算法正确性，
并提醒完成后恢复该行。

`CHANGELOG.md` 的 Unreleased 下记录：公共 example parser、移植模板、严格离线 dry-run
契约和 porting checklist。README 不新增算法，因此不伪造新表格行；checklist 会明确规定
以后每新增一个正式移植时同时更新 README 和总览表。

## 10. 预计文件变更

新增：

- `examples/_common.py`
- `examples/_TEMPLATE.py`
- `tests/_TEMPLATE_example.py`
- `tests/test_example_entrypoints.py`
- `docs/porting-checklist.md`

修改：

- 六个算法 example 的 parser 和 dry-run 控制流
- `docs/self-evolution-examples.md`
- `mkdocs.yml`
- `CONTRIBUTING.md`
- `CHANGELOG.md`

不修改：

- `agentdescent/` 核心包
- 六个算法的 Strategy、Aggregator、reward 和 fidelity 常量
- 数据集选择和真实运行路径

## 11. 实施顺序

1. 先写入口契约测试，让当前重复 parser、DGM 缺 `--yes` 和联网 dry-run 按预期失败。
2. 实现 `_common.py` 和六个 `build_parser()`，保住所有现有默认值。
3. 重排六个 dry-run 分支，直到网络/API sentinel 测试通过。
4. 增加代码模板和测试模板。
5. 写 checklist、候选表、作者记录、nav、CONTRIBUTING 和 changelog。
6. 跑针对性测试，再跑完整测试与文档构建。

## 12. 最终验收命令

```bash
pytest -q tests/test_example_entrypoints.py
pytest -q tests/test_ace_example.py tests/test_gepa_example.py \
  tests/test_evoskill_example.py tests/test_skillopt_example.py \
  tests/test_adas_example.py tests/test_dgm_example.py
pytest -q
mkdocs build --strict
python -m examples.run_demo
```

还会逐个检查六个命令的 `--help` 和 `--dry-run`。真实模型调用不属于这个 issue 的测试
条件，也不会为了验证 CLI 消耗用户的 API 额度。

## 13. 主要风险

1. **dry-run 输出变化。** 提前返回后无法展示真实样例和精确 split；需要同步更新文档，
   并确保输出仍足以让用户确认 provider、model、迭代预算和数据集身份。
2. **默认值被意外统一。** 公共 flag 不等于公共默认值，测试必须逐个锁定。
3. **DGM 的特殊语义。** `model=None` 是真实能力，不可因公共 parser 变成默认 API 调用；
   `--yes` 只在传入模型时生效。
4. **模板变成第二套规范。** checklist、模板和测试必须引用同一 helper；不能在模板里再次
   手写公共参数。
5. **文档孤页或超过 20 行。** 现有测试会检查 nav 和链接；另外单独测试 checklist 行数，
   把“20 行以内”变成机器可执行的约束。

这套方案的核心判断是：#74 的价值不在少写几十行 `argparse`，而在把“什么叫合格移植”
从六个文件中的默契，变成新移植无法轻易绕过的入口、模板、测试和文档契约。
