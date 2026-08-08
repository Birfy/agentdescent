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

## Empirical results — Claude Code as the base agent, on FinQA

Barrier-free (`--async`), 4 workers, **120 rollouts pinned**, `--reflective-merge`,
one seed. The base agent is the **Claude Code CLI** (`--backend claude-code`),
which gets a workspace per question with the document at `document.txt` and the
learned library materialised under `.claude/skills/` — it reads the skills as
files with its own tools, which is the shape upstream runs. The model behind the
CLI is `deepseek-v4-flash` through an Anthropic-compatible endpoint
(`CLAUDE_CODE_SIMPLE=1` forces API-key auth so the CLI uses it).

Dataset is **FinQA**, not OfficeQA — see [Dataset caveat](#dataset-caveat).
60 items → 30 train / 15 val / 15 test.

**Three runs, changing only what happens to a diff proposed against a head the
merger has since moved:**

| `--staleness` | stale discarded | frontier | skills on head | val | test | wall |
|---|---:|---:|---:|---:|---:|---:|
| `guarded` (default) | **12 / 12 (100%)** | 3 / 5 | 5 | 0.547 → 0.680 | 0.633 | ~20 min |
| `reflective` | 0 | — | — | 0.627 → 0.633 | — | stopped: 4 sweeps in 30 min |
| **`full`** | **0 / 0** | **5 / 5** | **12** | 0.527 → **0.707** | 0.633 | ~19 min |

**A discarded card is a whole induction batch thrown away** — four failures
collected, a Proposer call and a Generator call spent — and the default discards
all of them here. Every proposal commits under this port, so head moves on every
sweep and the three-version lag budget is exhausted immediately. The frontier
filled 3 of its 5 slots, which means the *bounded* part of "bounded top-K" never
came into play.

**`reflective` fixes the discarding and pays for it on the wrong thread.** Its
rebase branch re-verifies each card on the card's own trajectories — cheap when a
rollout is one API call, and *two Claude Code invocations* (~50 s) when it is an
agent. Serial, on the merger's critical path: 30 minutes bought four sweeps.

**`full` is the one that works here.** It rebases onto the current head and skips
the per-card re-verification, leaving the frontier's own validation sweep as the
only gate — which is not a weakening, because `--reflective-merge` fuses the
sweep's diffs into one candidate that is then scored across the full val split
before admission. Nothing unverified reaches the head; it is verified once, in
aggregate, instead of once per card. Discards go to zero, the frontier fills for
the first time (0.707 / 0.660 / 0.640 / 0.613 / 0.593 — an actual leaderboard),
and val rises 18 points against `guarded`'s 13, from a lower baseline.

!!! warning "Read val with the variance, and test with suspicion"
    Baselines across these runs on the **same split and seed** were 0.727, 0.547,
    0.627 and 0.527 — the Claude Code agent does not answer identically twice, and
    15 val items make one item worth 6.7 points. Only the *within-run* movement
    means anything; the cross-run absolute values do not.

    And test does not follow val: **0.633 under both `guarded` and `full`**, while
    val moved 13 and 18 points. The induced skills say why — several of the twelve
    are restatements of one rule:

    ```
    ### percentage-answer-formatting
    - Round percentage answers to one decimal place for general Treasury statistics
    ### percentage-rounding-for-financials
    - Report percentage values to one decimal place (e.g., 12.4%), not more
    ### round-percentages-to-one-decimal
    - Always present final percentage answers rounded to exactly one decimal place
    ```

    Two things are visible there. The library learns **one lesson several times**:
    grow-and-refine's near-duplicate check is lexical, and these are lexically
    distinct restatements of the same rule. And the lessons target the **scorer's
    surface** rather than the task — under a multi-tolerance numeric match,
    failures present as rounding and unit mismatches, so that is what the
    Reflector generalises. Both are why val moves and test does not.

**Two counts, not one.** `frontier` is what induction produced; `skills on head`
is what displaced the seed. They used to be reported as a single number, which
said "the Proposer produced nothing" about a run that produced six candidates and
admitted every one of them — opposite diagnoses from the same figure.

## Dataset caveat

The full OfficeQA is HF-**gated** (`databricks/officeqa`, set `HF_TOKEN`); absent
that the example falls back to **FinQA**, which is what the results above were
measured on. The earlier fallback was the repo's bundled 12-row sample, and it
split into 5 train / 3 val / 2 test — too small to measure anything, so every run
reported 0.000 and read as a broken algorithm rather than a missing dataset.

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

# the `full` row of the table above
CLAUDE_CODE_SIMPLE=1 ANTHROPIC_MODEL=deepseek-v4-flash \
python -m examples.evoskill.evoskill_skill_discovery --yes --seed 0 \
    --backend claude-code --workers 4 --budget-rollouts 120 --iterations 9999 \
    --reflective-merge --staleness full --async --async-ratio 3 \
    --max-seconds 3600 --eval-concurrency 4 --model deepseek-v4-flash
```

`--iterations 9999` so the port's own default (6) does not stop the run before
the rollout budget does. `CLAUDE_CODE_SIMPLE=1` makes the CLI use
`ANTHROPIC_API_KEY` instead of its stored OAuth credential, which is what lets it
drive a non-Anthropic endpoint.

Offline tests: `tests/test_evoskill_example.py`, `tests/test_backends.py`.
