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

## Dataset caveat

The full OfficeQA is HF-**gated** (`databricks/officeqa`, set `HF_TOKEN`); absent
that the example loads the repo's **bundled 12-row sample**.

## Run it

```bash
python -m examples.evoskill_skill_discovery --dry-run
python -m examples.evoskill_skill_discovery --model claude-haiku-4-5 --backend toolloop
```

Offline tests: `tests/test_evoskill_example.py`, `tests/test_backends.py`.
