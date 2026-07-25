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

## Dataset caveat

The full OfficeQA is HF-**gated** (`databricks/officeqa`, set `HF_TOKEN`); absent
that the example loads the repo's **bundled 12-row sample**. EvoSkill's Read/Grep
doc tools are approximated by a keyword line-retriever. With one non-tool LLM on
272 KB bulletins accuracy is low — the value is the faithful *loop*.

## Run it

```bash
python -m examples.evoskill_skill_discovery --dry-run
python -m examples.evoskill_skill_discovery --model claude-haiku-4-5
```

Offline tests: `tests/test_evoskill_example.py`.
