# Self-evolution algorithms — faithful ports

AgentDescent is a *general* engine for parallel, merge-based evolution. To show it
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

**All six are parallel — and can run async.** Each passes `max_concurrency=n_workers`,
so a round's workers run **concurrently** (overlapping LLM rollouts) with the
aggregator merge as the barrier (*synchronous data-parallelism*). Add **`--async`**
and the same example runs **barrier-free** through
[`async_evolve()`](evolution.md#the-barrier-free-runtime-async_evolve) — workers
never wait for the merge, and the staleness policy rebases/discards stale diffs.
Their custom optimizers keep shared state thread-safe, so both modes work
unchanged.

```bash
python -m examples.ace_context_evolution --model claude-haiku-4-5           # synchronous DP
python -m examples.ace_context_evolution --model claude-haiku-4-5 --async   # barrier-free
```

| Algorithm | Port author | Kind | Dataset (faithful) | `evolve()` plug-ins | Page |
|---|---|---|---|---|---|
| **ACE** (Agentic Context Engineering) | chendanyang | skill / context | FiNER-139 (XBRL tagging) | `strategy=ACEPlaybook`; Curator = default aggregator | [→](algo-ace.md) |
| **GEPA** (Reflective Prompt Evolution) | chendanyang | skill / prompt | HotpotQA (EM) | `aggregator_factory=` Pareto optimizer | [→](algo-gepa.md) |
| **EvoSkill** (Automated Skill Discovery) | chendanyang | skill library | OfficeQA (Treasury) | `strategy` + `aggregator_factory=` top-K frontier (sync) / SGD descent (async) | [→](algo-evoskill.md) |
| **SkillOpt** (ReflACT) | chendanyang | skill document | SearchQA (EM/F1) | `strategy` (edits) + `aggregator_factory=` strict gate | [→](algo-skillopt.md) |
| **ADAS** (Meta Agent Search) | chendanyang | harness (L1) | MGSM | `strategy` + `aggregator_factory=` keep-all archive | [→](algo-adas.md) |
| **DGM** (Darwin Gödel Machine) | chendanyang | harness (L1) | SWE-bench Verified | `strategy` + `aggregator_factory=` archive + selection | [→](algo-dgm.md) |

Every example takes `--dry-run`, which prints its configuration and returns with
**zero network access and no API key**, and has an offline test suite
(`tests/test_<name>_example.py`) exercising its pure logic. Real runs load their
datasets through the shared
[**`agentdescent.dataloader`**](dataloader.md) data layer — dependency-free
(`urllib` only), cached under `~/.cache/agentdescent/`, from each benchmark's
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

*Documented deviation:* GEPA's Algorithm 1 admits a child by comparing means on a
feedback minibatch of size *b*. `evolve()` rolls out **one** task per worker per
round, so that comparison is a single Bernoulli draw — for a binary reward like
HotpotQA EM, exactly `{-1, 0, +1}`. The admission test is therefore "did not
regress" rather than "improved": requiring the one sampled instance to flip
wrong→right threw away prompts that help broadly but do not fix *that* question,
and a rejected candidate never enters the pool, never gets a score row, and so can
never reach the frontier — which is precisely the complementary specialist the
frontier exists to keep alive. Algorithm 2 itself (per-instance frontier,
domination pruning, win-frequency sampling) is scored on the full `D_pareto` row
and is unaffected.

```bash
python -m examples.gepa_prompt_evolution --dry-run
python -m examples.gepa_prompt_evolution --model claude-haiku-4-5
```

### EvoSkill — Automated Skill Discovery

*Paper* arXiv:2603.02766 · *repo* `sentient-agi/EvoSkill` · *dataset* OfficeQA.

Faithful to what the **repo code** does (which differs from some paper claims):
**batch-level failure-driven skill induction** (collect items scored `< 0.8`, a
Skill Proposer analyses a *batch* of failure patterns → a Skill Generator writes
one `SKILL.md`) governed by a **bounded top-K aggregate frontier** — *not* a
per-instance Pareto frontier (`src/registry/manager.py:update_frontier` is a
leaderboard on mean validation accuracy, while the paper's abstract says "a Pareto
frontier of agent programs governs selection"). The unit-aware numeric scorer and the
exact tolerance ladder (`src/loop/runner.py:79` — `[0.05, 0.01, 0.1, 0.0, 0.025]`,
weight `1/(1+20·tol)`, pass threshold `0.8` at `:319`) are ported, as is the
frontier bound (`src/registry/manager.py:379`, `max_size=5`). On the **sync** path this strict per-candidate frontier
(`TopKFrontierAggregator`) runs verbatim; on the **async** path it switches to
`SgdSkillAggregator` — SGD-style skill descent that validates every `val_every`
steps and rolls back on no gain, amortising the held-out eval
([why](aggregator.md#the-async-optimizer-variant-sgd-style-descent)).

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
3. **textual learning-rate budget** — an integer edit cap per step (AgentDescent's
   `trust_region_ops` analogue);
4. **rejected-edit buffer** — rejected edits are remembered in-epoch and fed back
   into the reflection prompt, so the optimizer stops re-proposing them. The
   example implements this itself: the core `settle()` pool retains discarded
   evidence but [nothing reads it back](concepts.md#33-staleness-policies-flashevolve-full-guarded-reflective), so this
   is the worked example of what that pool is *for*.

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
two Reflexion refinement rounds; fitness is a bootstrap-CI mean (upstream `_mgsm/utils.py` resamples 100 000
times; this example uses 2 000 so it runs in seconds — a stated deviation, not a
silent one); the archive is **keep-all**. The seven ADAS MGSM seeds (CoT, Self-Consistency, Reflexion,
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

## Candidate ports

The backlog is organised by missing mechanism, not paper popularity. Assign an
owner before implementation so fidelity questions have a person who read the
released code.

| Mechanism | Candidates | Existing coverage | Owner |
|---|---|---|---|
| Evolution / program search | AlphaEvolve (OpenEvolve), PromptBreeder, AFlow | none | TBD |
| Reflection / textual gradients | TextGrad, Reflexion, Self-Refine | partial (GEPA is reflective, not textual-gradient) | TBD |
| Skills / lifelong learning | Voyager, SkillWeaver | adjacent (EvoSkill) | TBD |
| Self-play / unlabeled data | Absolute Zero, R-Zero, Agent0 | **none; highest priority** | TBD |
| Self-modifying code | SICA, Gödel Agent | DGM | TBD |

The unlabeled path is the most important gap: all six current ports derive reward
from a benchmark with gold labels, although `evolve()` only requires a score in
`[0, 1]` and does not require labels.

### Deferred pending released code

* **CoEvoSkills** (arXiv:2604.01687, "Self-Evolving Agent Skills via
  Co-Evolutionary Verification") — the Skill Generator + co-evolving Surrogate
  Verifier + opaque pass/fail oracle is a compelling fit for AgentDescent's
  aggregator, but the **official code is unreleased** ("coming soon") and its
  benchmark (SkillsBench) requires a Claude Code / Codex agent harness. With no
  original repo to be faithful to, it is intentionally deferred until the authors
  release code, rather than reconstructed and mislabelled as faithful.

## Fidelity principles

Use the short [porting checklist](porting-checklist.md) before adding a row here.

1. **Faithful to the repo, not just the paper.** Where the released code diverges
   from the paper's claims (e.g. EvoSkill's frontier is top-K aggregate, not
   per-instance Pareto), the example follows the **code** and says so.
2. **Faithful dataset identity.** Each example uses the paper's actual benchmark,
   loaded from its canonical source.
3. **Documented boundaries.** Where the full setup needs heavy infra (AppWorld,
   SWE-bench Docker, gated data), the example states the boundary and, when
   needed, substitutes a clearly-labelled surrogate — the algorithm stays
   faithful and runnable; nothing is passed off as a benchmark result it is not.
