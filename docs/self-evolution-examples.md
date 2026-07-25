# Self-evolution algorithms — faithful ports

Concordia is a *general* engine for parallel, merge-based evolution. To show it
is faithful to the field — not a toy — this page ports a set of the most
representative **skill self-evolution** and **harness self-evolution** algorithms
from the literature, each as one runnable example, each faithful to the original
paper/repo's **algorithm** and **dataset choice**.

Two categories, mirroring the two governance layers:

* **Skill self-evolution** — evolve a *skill / prompt / context* (an **L2**
  artifact, `blast_radius=0.2`): ACE, GEPA, EvoSkill, SkillOpt.
* **Harness self-evolution** — evolve the *agentic system / coding agent itself*
  (an **L1** artifact, `blast_radius=0.6`, oracle-gated): ADAS, DGM.

**All six run through the one entry point, [`evolve()`](evolution.md)** — each is
just a custom `strategy=` (how a proposal becomes a `Diff`) and/or a custom
`aggregator_factory=` (the selection/acceptance optimizer). No example bypasses
the engine; they differ only in those two plug-ins and the blast radius. Each has
a dedicated page:

| Algorithm | Kind | Dataset (faithful) | `evolve()` plug-ins | Page |
|---|---|---|---|---|
| **ACE** (Agentic Context Engineering) | skill / context | FiNER-139 (XBRL tagging) | `strategy=ACEPlaybook`; Curator = default aggregator | [→](algo-ace.md) |
| **GEPA** (Reflective Prompt Evolution) | skill / prompt | HotpotQA (EM) | `aggregator_factory=` Pareto optimizer | [→](algo-gepa.md) |
| **EvoSkill** (Automated Skill Discovery) | skill library | OfficeQA (Treasury) | `strategy` + `aggregator_factory=` top-K frontier | [→](algo-evoskill.md) |
| **SkillOpt** (ReflACT) | skill document | SearchQA (EM/F1) | `strategy` (edits) + `aggregator_factory=` strict gate | [→](algo-skillopt.md) |
| **ADAS** (Meta Agent Search) | harness (L1) | MGSM | `strategy` + `aggregator_factory=` keep-all archive | [→](algo-adas.md) |
| **DGM** (Darwin Gödel Machine) | harness (L1) | SWE-bench Verified | `strategy` + `aggregator_factory=` archive + selection | [→](algo-dgm.md) |

Every example takes `--dry-run` (load the dataset, print the plan, **no API
calls**) and has an offline test suite (`tests/test_<name>_example.py`) exercising
its pure logic. Datasets load through the shared
[**`concordia.dataloader`**](dataloader.md) data layer — dependency-free
(`urllib` only), cached under `~/.cache/concordia/`, from each benchmark's
canonical source. Where a paper's full setup needs heavy infrastructure, the
boundary is documented in the example's module docstring — never hidden.

---

## Skill self-evolution

### ACE — Agentic Context Engineering

*Paper* arXiv:2510.04618 · *repo* `ace-agent/ace` · *dataset* FiNER-139.

ACE evolves a **context playbook** (accumulated lessons) with three roles that
map exactly onto `evolve()`:

* **Generator** → `LLMAgent.solve` (solve a task using the playbook),
* **Reflector** → `LLMAgent.propose` (distil one *delta bullet* from a failure),
* **Curator** → the **aggregator** — deterministic, non-LLM merge (dedup +
  statistical acceptance).

The custom `ACEPlaybook` strategy keeps the two ACE invariants: **incremental
delta updates** (only ever append a new content-addressed bullet — never a
monolithic rewrite, so "context collapse" cannot happen) and **grow-and-refine
de-dup** (near-duplicate bullets pruned at insert). ACE's per-bullet
helpful/harmful counters become the aggregator's per-diff **Beta-posterior
acceptance** — a bullet commits only if it raises held-out reward.

```bash
python -m examples.ace_context_evolution --dry-run
python -m examples.ace_context_evolution --model claude-haiku-4-5
```

### GEPA — Reflective Prompt Evolution

*Paper* arXiv:2507.19457 · *repo* `gepa-ai/gepa` · *dataset* HotpotQA.

GEPA's distinctive mechanism is **per-instance Pareto candidate selection**
(Algorithm 2): instead of greedily mutating the single best-*average* prompt, it
samples the next parent from the per-instance Pareto frontier, weighted by how
many instances a candidate uniquely wins — keeping complementary specialists
alive. This is realised by a custom `ParetoAggregator` plugged into `evolve()`
through `aggregator_factory=` (the sanctioned "swap the whole optimizer" hook).
The optimizer sets the dev head to the sampled Pareto parent, so `evolve()`'s
next round mutates *it*, not the greedy best. Reflective mutation (the LLM
rewriting the instruction from execution trace + NL feedback) is the propose step.

```bash
python -m examples.gepa_prompt_evolution --dry-run
python -m examples.gepa_prompt_evolution --model claude-haiku-4-5
```

### EvoSkill — Automated Skill Discovery

*Paper* arXiv:2603.02766 · *repo* `sentient-agi/EvoSkill` · *dataset* OfficeQA.

Faithful to what the **repo code** does (which differs from some paper claims):
**failure-driven skill induction** (collect items scored `< 0.8`, a Skill
Proposer analyses failure *patterns* → a Skill Generator writes one `SKILL.md`)
governed by a **bounded top-K aggregate frontier** — *not* a per-instance Pareto
frontier (`registry/manager.py:update_frontier` is a leaderboard on mean
validation accuracy). The unit-aware numeric scorer and the exact tolerance
ladder (`[0.05, 0.01, 0.1, 0.0, 0.025]`, weight `1/(1+20·tol)`) are ported.

*Dataset note:* the full OfficeQA is HF-**gated** (`databricks/officeqa`, set
`HF_TOKEN`); absent that the example loads the repo's **bundled 12-row sample**.
EvoSkill's Read/Grep doc tools are approximated by a keyword line-retriever. With
one non-tool LLM on 272 KB bulletins accuracy is low — the value is the faithful
*loop*.

```bash
python -m examples.evoskill_skill_discovery --dry-run
```

### SkillOpt — ReflACT

*Paper* arXiv:2605.23904 · *repo* `microsoft/SkillOpt` · *dataset* SearchQA.

Trains the **skill document as the external state of a frozen agent**. All four
load-bearing invariants are reproduced faithfully from the repo:

1. **Bounded string edits** on one markdown doc — ops `{append, insert_after,
   replace, delete}`;
2. **strict held-out accept gate** — a candidate is accepted only if it strictly
   improves the validation hard-EM over the *current* skill (greedy, like
   `evolve()`);
3. **textual learning-rate budget** — an integer edit cap per step (Concordia's
   `trust_region_ops` analogue);
4. **rejected-edit buffer** — rejected edits are remembered in-epoch and fed back
   to the optimizer (Concordia's "settled evidence survives").

```bash
python -m examples.skillopt_skill_training --dry-run
python -m examples.skillopt_skill_training --model claude-haiku-4-5
```

---

## Harness self-evolution

### ADAS — Meta Agent Search

*Paper* arXiv:2408.08435 · *repo* `ShengranHu/ADAS` · *dataset* MGSM.

Evolves the **agentic system itself** — a *harness* change, so the artifact is
**L1** (`classify()` prints the layer). A **meta-agent**, conditioned on the
entire **archive** of prior agents + their fitness, proposes the next agent, with
two Reflexion refinement rounds; fitness is a bootstrap-CI mean; the archive is
**keep-all**. The seven ADAS MGSM seeds (CoT, Self-Consistency, Reflexion,
Debate, Step-back, Quality-Diversity, Role-Assignment) are the starting archive.

*Safety substitution (documented):* ADAS `exec`s model-written Python `forward()`
functions. To avoid arbitrary code execution, an agent here is a **composable
control-flow program** in a small validated DSL run by a safe interpreter — the
Meta Agent Search loop, seeds, MGSM scoring, and keep-all archive are faithful;
only the agent *substrate* is a safe DSL. `--select dgm` swaps archive
conditioning for the DGM parent-selection rule.

```bash
python -m examples.adas_meta_agent_search --dry-run
python -m examples.adas_meta_agent_search --model claude-haiku-4-5 --generations 6
```

### DGM — Darwin Gödel Machine

*Paper* arXiv:2505.22954 · *repo* `jennyzzt/dgm` · *dataset* SWE-bench Verified.

The archetypal harness self-evolution: a coding agent that **edits its own
codebase**, keeping every variant in an open-ended **archive**. Faithful to
`DGM_outer.py`: keep-all archive, staged empirical validation (small=10 →
medium=50 if score > 0.4 → big=140), and the exact parent-selection rule
`p_i ∝ sigmoid(10·(score−0.5)) · 1/(1+children_i)` (favour high performers,
discount already-explored parents). The agent is an **L1** harness.

*Honesty boundary:* DGM's real objective runs each candidate patch inside the
**SWE-bench Docker harness** (per-task containers, real test suites, arbitrary
code execution) — out of scope for a dependency-free example. The objective here
is a **transparent surrogate** (each real SWE instance has a latent
required-capability set an agent must cover), so the DGM *algorithm* runs and is
tested offline while the *scores* are simulated, not SWE-bench results. Pass a
real `evaluate_fn` to `run_dgm` to plug in the actual harness.

```bash
python -m examples.dgm_self_improve                      # runs offline (surrogate)
python -m examples.dgm_self_improve --generations 12 --archive keep_all
```

---

## Not yet ported

* **CoEvoSkills** (arXiv:2604.01687, "Self-Evolving Agent Skills via
  Co-Evolutionary Verification") — the Skill Generator + co-evolving Surrogate
  Verifier + opaque pass/fail oracle is a compelling fit for Concordia's
  aggregator, but the **official code is unreleased** ("coming soon") and its
  benchmark (SkillsBench) requires a Claude Code / Codex agent harness. With no
  original repo to be faithful to, it is intentionally deferred until the authors
  release code, rather than reconstructed and mislabelled as faithful.

## Fidelity principles

1. **Faithful to the repo, not just the paper.** Where the released code diverges
   from the paper's claims (e.g. EvoSkill's frontier is top-K aggregate, not
   per-instance Pareto), the example follows the **code** and says so.
2. **Faithful dataset identity.** Each example uses the paper's actual benchmark,
   loaded from its canonical source.
3. **Documented boundaries.** Where the full setup needs heavy infra (AppWorld,
   SWE-bench Docker, gated data), the example states the boundary and, when
   needed, substitutes a clearly-labelled surrogate — the algorithm stays
   faithful and runnable; nothing is passed off as a benchmark result it is not.
