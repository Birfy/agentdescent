# EvoSkill — Automated Skill Discovery

> **Skill-library self-evolution.** Discover reusable `SKILL.md` skills from
> execution failures, governed by a bounded top-K frontier. Runs through
> [`evolve()`](evolution.md) with a custom `Strategy` + `aggregator_factory`.
> Example: [`examples/evoskill_skill_discovery.py`](https://github.com/Birfy/concordia/blob/main/examples/evoskill_skill_discovery.py).

| | |
|---|---|
| **Paper** | *EvoSkill: Automated Skill Discovery for Coding Agents* — Alzubi et al., 2026 ([arXiv:2603.02766](https://arxiv.org/abs/2603.02766)) |
| **Repo** | [`sentient-agi/EvoSkill`](https://github.com/sentient-agi/EvoSkill) |
| **Dataset** | **OfficeQA** (U.S. Treasury Bulletins), deterministic numeric scorer |
| **Layer** | L2 skill (`blast_radius=0.2`) |

## The algorithm (faithful to the code, not just the paper)

Traced from the repo (`src/loop/runner.py`, `src/registry/manager.py`,
`src/evaluation/reward.py`):

* **Failure-driven skill induction.** Sample train items, run the base agent,
  collect failures (an item fails when its multi-tolerance score `< 0.8`). A
  **Skill Proposer** analyses failure *patterns* → a **Skill Generator** writes
  one `SKILL.md`.
* **Bounded top-K aggregate frontier — NOT per-instance Pareto.** Despite the
  paper's framing, `manager.py:update_frontier` is a leaderboard on a single
  scalar (mean validation accuracy): admit if the frontier has room, else replace
  the worst member iff strictly greater. Parent for the next round = the best.
* The unit-aware numeric scorer and the exact tolerance ladder
  (`[0.05, 0.01, 0.1, 0.0, 0.025]`, weight `1/(1+20·tol)`) are ported.

!!! note "Fidelity is to the released code"
    The paper claims per-instance Pareto selection and joint skill+prompt
    mutation; the code has neither. This example follows the **code**.

## How it plugs into `evolve()`

* `strategy=SkillLibraryStrategy()` — a proposed `name :: body` becomes a `Diff`
  that appends a skill to the library.
* `propose` — failure-driven Proposer + Generator (two LLM calls).
* `aggregator_factory` → `TopKFrontierAggregator` — the bounded top-K frontier;
  it scores candidates on held-out and commits the **best frontier member** as
  the dev head, so the next round extends it (`selection_strategy="best"`).

## Plug-ins implemented

In [`examples/evoskill_skill_discovery.py`](https://github.com/Birfy/concordia/blob/main/examples/evoskill_skill_discovery.py)
(+ [`concordia/backends.py`](https://github.com/Birfy/concordia/blob/main/concordia/backends.py)):

| Plug-in | `evolve()` slot | What it does |
|---|---|---|
| **`SkillLibraryStrategy`** | `strategy=` | a proposed `SKILL.md` (`name :: body`) becomes a `Diff` on the skill library |
| **`TopKFrontierAggregator`** + **`Frontier`** | `aggregator_factory=` | the bounded top-K aggregate frontier; commits the best member as the dev head |
| `make_propose(...)` | `propose=` | failure-driven Skill Proposer + Skill Generator (writes one `SKILL.md`) |
| **`openhands_backend` / `tool_loop_backend`** (`concordia.backends`) | the base agent | real OpenHands tool agent, a grep/read ReAct loop, or the default keyword retriever — selected by `--backend` |

## The base agent — `--backend` (this is what makes it work)

OfficeQA answers are figures buried in **200 KB – 1.2 MB financial tables**, often
needing *grep + computation* (e.g. summing the monthly "national defense" rows for
a calendar year → `2,602`). A single LLM call with a keyword excerpt scores
**0.000** — the bottleneck is document navigation, not a learnable skill. So the
base agent is pluggable ([`concordia.backends`](dataloader.md)):

| `--backend` | Base agent | Runs where |
|---|---|---|
| `retrieval` (default) | passive 40-line keyword excerpt | anywhere; **too weak for OfficeQA** |
| `toolloop` | dependency-free `grep`/`read` ReAct loop (any Completion) | anywhere |
| `openhands` | **real OpenHands agent** (terminal + file_editor tools) via the OpenHands SDK | Python ≥ 3.12 + `pip install openhands-ai` |

The OpenHands backend is the faithful fix (real EvoSkill uses Read/Grep/Bash). With
`deepseek-v4-pro` it autonomously `grep`s the tables, `view`s the right rows, and
**computes** the answer — solving questions the retriever never could.

```bash
# real OpenHands agent, DeepSeek endpoint (needs Python 3.12 env + openhands-ai)
OPENAI_BASE_URL=https://api.deepseek.com OPENAI_API_KEY=... \
  python -m examples.evoskill_skill_discovery --provider glm \
    --model deepseek-v4-pro --backend openhands
```

```python
from concordia.backends import openhands_backend, tool_loop_backend
backend = openhands_backend(model="openai/deepseek-v4-pro",
                            base_url="https://api.deepseek.com")   # or tool_loop_backend(completion)
answer = backend.answer(question, document_text, skills=rendered_skills)
```

## Empirical results — real OpenHands agent + DeepSeek on OfficeQA

To validate the tool-using base agent, we ran EvoSkill's skill-discovery loop on
OfficeQA (the [official `sentient-agi/EvoSkill`](https://github.com/sentient-agi/EvoSkill),
`skill_only`, iterations=3, `multi_tolerance` scorer) with a **real OpenHands
agent** (terminal + file_editor tools) driven by **DeepSeek** (`deepseek-v4-flash`,
OpenAI-compatible endpoint via LiteLLM: `model="openai/deepseek-v4-flash"` +
`OPENAI_BASE_URL=https://api.deepseek.com`). Base agent = the model; the accuracy
is the held-out `val` multi-tolerance score.

| Questions | train / val | Baseline (no skills) | After discovery | Skill learned |
|---|---|---|---|---|
| **100** | **40 / 29** | **66.7%** | **79.7% (+13.0)** | **`answer-verification-and-final-validation`** |

On 29 held-out val questions of genuinely hard multi-step financial math (VaR,
Macaulay duration, moving averages, dispersion indices), the baseline is **66.7%**
and EvoSkill lifts it **+13 points to 79.7%** by discovering an *answer-verification*
skill (re-check the multi-step computation before answering). It does **not** reach
100% because several questions are genuinely hard. The agent does work the passive
keyword-retriever cannot: it `grep`s the tables, `view`s the right rows, and
**computes** (e.g. summing the monthly "national defense" rows for 1940 → **2,602**).

**Parallel + async.** EvoSkill evaluates `val` via `asyncio.Semaphore(concurrency)`
+ `gather` — set `concurrency=8` and the 29 val questions evaluated in **~7 min**
(vs ~40–60 min serial). This mirrors the framework's own
[parallel/async execution](evolution.md#parallelism-async-the-frameworks-core).

**Two portability gotchas** (documented, not hidden):

* *Structured output.* DeepSeek's API rejects OpenAI strict
  `response_format:{type:"json_schema", strict:true}` (HTTP 400 *"response_format
  type is unavailable"*); the answer-extraction re-ask must use `{type:"json_object"}`
  with the schema in the prompt. The official `claude` harness uses native
  structured output and needs no shim.
* *Environment.* The real OpenHands SDK needs **Python ≥ 3.12** (a `uv`-managed
  venv works, no Docker/admin); the OpenAI-compatible base URL routes DeepSeek
  through LiteLLM.

## Dataset caveat

The full OfficeQA is HF-**gated** (`databricks/officeqa`, set `HF_TOKEN`); absent
that the example loads the repo's **bundled 12-row sample**.

## Run it

```bash
python -m examples.evoskill_skill_discovery --dry-run
python -m examples.evoskill_skill_discovery --model claude-haiku-4-5 --backend toolloop
```

Offline tests: `tests/test_evoskill_example.py`, `tests/test_backends.py`.
