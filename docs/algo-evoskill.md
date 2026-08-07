# EvoSkill — Automated Skill Discovery

> **Skill-library self-evolution.** Discover reusable `SKILL.md` skills from
> execution failures, governed by a bounded top-K frontier. Runs through
> [`evolve()`](evolution.md) with a custom `Strategy` + `aggregator_factory`.
> Example: [`examples/evoskill/evoskill_skill_discovery.py`](https://github.com/Birfy/agentdescent/blob/main/examples/evoskill/evoskill_skill_discovery.py).

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

* `strategy=SkillLibraryTree()` — a proposed `name :: body` becomes a `Diff`
  that appends (or edits) a skill in the library. It is a
  [`FileTree`](directory-evolution.md) subclass, so the library **is a directory**
  (`skills/<name>/SKILL.md`) rather than a name→text dict: with a tool-using
  backend the skills are written into the agent's workspace and it reads the ones
  it needs, instead of every skill riding along in every prompt. The repo's
  `name :: body` protocol is kept rather than `FileTree`'s `<EDITS>` JSON — what
  is faithful here is the two-role Proposer/Generator induction, not the
  separator, and switching protocols would change the Generator's prompt.
* `propose` — **batch-level** failure-driven Proposer + Generator: it accumulates
  a batch of `batch_size` failures (shared across the concurrent workers) and then
  induces **one** `SKILL.md` from their shared pattern (two LLM calls) — matching
  the repo's per-iteration induction, not one skill per trajectory.
* `aggregator_factory` — **two optimizers, picked by path**:
    * **sync** (`asynchronous=False`) → `TopKFrontierAggregator`: the strict
      bounded top-K frontier faithful to `registry/manager.py` — scores **every**
      candidate on held-out, commits the best frontier member as the dev head.
    * **async** (`asynchronous=True`) → `SgdSkillAggregator`: SGD-style skill
      descent — apply each skill update, validate on held-out only every
      `val_every` steps, **roll back** the mini-batch on no gain. Amortises the
      held-out eval ~`val_every`× (the [async optimizer variant](aggregator.md#the-async-optimizer-variant-sgd-style-descent)).
* `self_verify=False` — the repo scores the *child* on the validation set and
  never re-runs the sampled task, so the async worker skips its per-trajectory
  re-run rollout.

## Plug-ins implemented

In [`examples/evoskill/evoskill_skill_discovery.py`](https://github.com/Birfy/agentdescent/blob/main/examples/evoskill/evoskill_skill_discovery.py)
(+ [`agentdescent/backends.py`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/backends.py)):

| Plug-in | `evolve()` slot | What it does |
|---|---|---|
| `FrontierBest` | selection ([seam](selection.md)) | best-of-frontier parent rule as a named policy |
| **`SkillLibraryTree`** | `strategy=` | a proposed `SKILL.md` (`name :: body`) becomes a `Diff` on the skill library — a [`FileTree`](directory-evolution.md), so the library is a real directory of `SKILL.md` files |
| **`TopKFrontierAggregator`** + **`Frontier`** | `aggregator_factory=` (**sync**) | the bounded top-K aggregate frontier; scores every candidate on held-out, commits the best member as the dev head |
| **`SgdSkillAggregator`** | `aggregator_factory=` (**async**) | SGD-style skill descent: apply updates, validate every `val_every` steps, checkpoint + roll back on no held-out gain |
| `make_propose(...)` | `propose=` | **batch-level** failure-driven Skill Proposer + Generator — one `SKILL.md` per `batch_size` failures (shared across workers) |
| `self_verify=False` | async runtime | skip the per-trajectory re-run — the repo scores the child on val only |
| **`openhands_backend` / `tool_loop_backend`** (`agentdescent.backends`) | the base agent | real OpenHands tool agent, a grep/read ReAct loop, or the default keyword retriever — selected by `--backend` |

## The base agent — `--backend` (this is what makes it work)

OfficeQA answers are figures buried in **200 KB – 1.2 MB financial tables**, often
needing *grep + computation* (e.g. summing the monthly "national defense" rows for
a calendar year → `2,602`). A single LLM call with a keyword excerpt scores
**0.000** — the bottleneck is document navigation, not a learnable skill. So the
base agent is pluggable ([`agentdescent.backends`](dataloader.md)):

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
  python -m examples.evoskill.evoskill_skill_discovery --provider glm \
    --model deepseek-v4-pro --backend openhands
```

```python
from agentdescent.backends import openhands_backend, tool_loop_backend
backend = openhands_backend(model="openai/deepseek-v4-pro",
                            base_url="https://api.deepseek.com")   # or tool_loop_backend(completion)
answer = backend.answer(question, document_text, skills=rendered_skills)
```

## Empirical results — real OpenHands agent + DeepSeek on OfficeQA

We ran this example through `evolve()` on **100 OfficeQA questions** (70 train /
30 held-out val) with a **real OpenHands agent** (terminal + file_editor tools)
driven by **DeepSeek** (`deepseek-v4-flash`, OpenAI-compatible via LiteLLM:
`model="openai/deepseek-v4-flash"` + `OPENAI_BASE_URL=https://api.deepseek.com`).
Accuracy is the held-out `val` multi-tolerance score.

| Questions | train / val | Baseline (no skills) | After discovery | Gate | Skills |
|---|---|---|---|---|---|
| **100** | **70 / 30** | **58.0%** | **65.7% (+7.7)** | 2 admitted / 15 tried | `dataextractionandanalysis`, `financial-and-statistical-analysis` |

On 30 held-out questions of genuinely hard multi-step financial math (VaR,
Macaulay duration, moving averages, dispersion indices), the baseline is **58.0%**
and EvoSkill lifts it **+7.7 points to 65.7%**. The agent does what the passive
keyword-retriever cannot: it `grep`s the tables, `view`s the right rows, and
**computes** (e.g. summing the monthly "national defense" rows for 1940 → **2,602**).

**The gate is what makes it work — and is the whole point of the aggregator.**
Every proposed skill is validated on held-out and admitted only if it improves
(`TopKFrontierAggregator`); across 15 candidates the gate **rejected 13** and kept
2, so the best never regresses. The contrast is stark: with the **same base agent
and skills but no gate** — apply every skill and score once at the end — the
prescriptive skills *degrade* the already-strong agent to **55.7% (−9.3)**. The
per-candidate validation is not overhead; it is the mechanism that turns "skills
the model wrote" into "skills that actually help".

**Async is not needed here — the run is val-bound.** When every candidate must be
validated on the full held-out set and validations run one at a time, the
barrier-free async runtime gives no speedup (workers just queue behind the
merger's held-out eval). So this uses the **synchronous** path; parallelism stays
where it pays — the held-out eval runs concurrently (`eval_concurrency`) and a
round's workers run concurrently (`max_concurrency`). Wall-clock was ~3 h,
essentially `15 candidates × 30-item val`. The OpenHands backend's setup, the
DeepSeek structured-output shim, and the Python ≥ 3.12 requirement are documented
in [Connecting agents & LLMs → Tool-using agent backends](agents.md#running-the-openhands-backend).

## Dataset caveat

The full OfficeQA is HF-**gated** (`databricks/officeqa`, set `HF_TOKEN`); absent
that the example loads the repo's **bundled 12-row sample**.

## Datasets — `--dataset officeqa|finqa`

OfficeQA is **HF-gated** (`databricks/officeqa`: an accepted licence plus
`HF_TOKEN`). Without that access the example uses **FinQA**
(`dreamerdeo/finqa`, ungated) — the same shape, a financial document plus a
numeric answer to locate and compute, at 60 items with ~4 KB documents that a
model without tools can read directly.

```bash
python -m examples.evoskill.evoskill_skill_discovery --dataset finqa \
    --provider openai --model deepseek-v4-flash --iterations 5 --yes
```

Measured: val **0.487 → 0.573**, held-out **test 0.617**, one skill discovered.
The skill it induced is about numeric presentation, which is what the scorer
rewards:

> *"When a percentage appears in a table, round your answer to the same number of
> decimal places shown in that table... compute the unrounded value first, then
> round once at the end to the required precision."*

The run header states which dataset it loaded.

!!! note "FinQA does not reproduce the retrieval challenge"
    OfficeQA's difficulty is finding one figure inside a 272 KB bulletin, which is
    what makes a *tool-using* agent worth having. FinQA's documents fit in a
    prompt, so it exercises the discovery loop but not the retrieval problem. For
    that, use OfficeQA with `--backend openhands|toolloop|claude-code`.

## Run it

```bash
python -m examples.evoskill.evoskill_skill_discovery --dry-run
python -m examples.evoskill.evoskill_skill_discovery --model claude-haiku-4-5 --backend toolloop
```

Offline tests: `tests/test_evoskill_example.py`, `tests/test_backends.py`.
