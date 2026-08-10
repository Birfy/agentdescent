# Self-evolution algorithms — faithful ports

AgentDescent is a *general* engine for parallel, merge-based evolution. To show it
is faithful to the field — not a toy — this page ports a set of the most
representative **skill self-evolution**, **program evolution**, and **harness
self-evolution** algorithms from the literature, each as one runnable example,
each faithful to the original paper/repo's **algorithm** and **dataset choice**.

Three artifact categories, spanning the two governance layers:

* **Skill self-evolution** — evolve a *skill / prompt / context* (an **L2**
  artifact, `blast_radius=0.2`): ACE, GEPA, EvoSkill, SkillOpt.
* **Harness self-evolution** — evolve the *agentic system / coding agent itself*
  (an **L1** artifact, `blast_radius=0.6`, oracle-gated): ADAS, DGM.
* **Program evolution** — evolve executable search code (an **L1** artifact,
  `blast_radius=0.6`, sandbox-evaluated): OpenEvolve.

**All eighteen run through the AgentDescent evolution engines.** The seven
benchmark-faithful ports above are each a custom `strategy=` and/or a custom
`aggregator_factory=`, with their parent/gate rules extracted as named policy
classes at the standard seams ([selection](selection.md),
[acceptance](acceptance-policies.md)). The eleven newer ports are declarative
[`MethodPolicy`](policies.md) definitions over a shared runner — their
mechanisms plug in as `Policies(...)` fields, their artifacts as shared
[strategies](strategies.md), and the [runtime matrix](matrix-overview.md)
measures them under all three schedulers. No example bypasses the engine. Each
has a dedicated page:

**All eighteen are parallel — and can run async.** In synchronous mode their workers
run **concurrently** (overlapping LLM rollouts) with the aggregator merge as the
barrier (*synchronous data-parallelism*). Add **`--async`** and the same example
runs **barrier-free** through
[`async_evolve()`](evolution.md#the-barrier-free-runtime-async_evolve) — workers
never wait for the merge, and the staleness policy rebases/discards stale diffs.
Their custom optimizers keep shared state thread-safe, so both modes work
unchanged.

```bash
python -m examples.ace.ace_context_evolution --model claude-haiku-4-5           # synchronous DP
python -m examples.ace.ace_context_evolution --model claude-haiku-4-5 --async   # barrier-free
```

| Algorithm | Port author | Kind | Dataset (faithful) | `evolve()` plug-ins | Page |
|---|---|---|---|---|---|
| **ACE** (Agentic Context Engineering) | chendanyang | skill / context | FiNER-139 (XBRL tagging) | `strategy=ACEPlaybook`; Curator = default aggregator | [→](algo-ace.md) |
| **GEPA** (Reflective Prompt Evolution) | chendanyang | skill / prompt | HotpotQA (EM) | `aggregator_factory=` Pareto optimizer | [→](algo-gepa.md) |
| **EvoSkill** (Automated Skill Discovery) | chendanyang | skill library | OfficeQA (Treasury) | `strategy` + `aggregator_factory=` top-K frontier (sync) / SGD descent (async) | [→](algo-evoskill.md) |
| **SkillOpt** (ReflACT) | chendanyang | skill document | SearchQA (EM/F1) | `strategy` (edits) + `aggregator_factory=` strict gate | [→](algo-skillopt.md) |
| **ADAS** (Meta Agent Search) | chendanyang | harness (L1) | MGSM | `strategy` + `aggregator_factory=` keep-all archive | [→](algo-adas.md) |
| **DGM** (Darwin Gödel Machine) | chendanyang | harness (L1) | SWE-bench Verified | `strategy` + `aggregator_factory=` archive + selection | [→](algo-dgm.md) |
| **OpenEvolve** (Program Evolution) | cyanneko | program (L1) | Function minimization | `strategy` + `aggregator_factory=` MAP-Elites islands | [→](algo-openevolve.md) |

### The shared command line

Eight flags have one definition, in
[`examples/_common.py`](https://github.com/Birfy/agentdescent/blob/main/examples/_common.py),
and the behaviour behind each lives there too — a flag declared centrally and
honoured locally is how a port grows a `--yes` it never reads.

| flag | what it does | honoured by |
|---|---|---|
| `--provider` / `--model` | pick Claude or any OpenAI-compatible endpoint | `completion_for` |
| `--seed` | the run's seed | the port |
| `--async` / `--async-ratio` / `--max-seconds` | barrier-free runtime and its lag budget | the port |
| `--dry-run` | print the plan; zero network, zero API key | the port's early return |
| `--yes` | skip the confirmation before real API calls | `confirm` |
| `--serial` | **the upstream algorithm's own semantics**: one worker, nothing to merge | `worker_count` |
| `--budget-rollouts` | total rollouts, held fixed as workers vary | `budget_kwargs` |

The iteration count is deliberately *not* standardised: `--rounds` (ACE, GEPA),
`--generations` (ADAS, DGM), `--iterations` (EvoSkill, OpenEvolve) and `--steps`
(SkillOpt) each keep their upstream vocabulary, which is part of being a faithful
port.

`--serial` is the control every one of these ports was missing. They all
parallelise an algorithm that was published as a serial loop, and until this flag
existed none of them could run that loop — so every claim about parallelising
them had no baseline in the repository at all. It is refused together with
`--async`: the barrier-free runtime's concurrency *is* `n_workers`, so
`--serial --async` is a one-worker *asynchronous* run whose diffs can still go
stale against a moved head, and staleness in the control arm is the one thing a
control must not have.

**`--serial` on its own is still not a comparison.** Six of these seven ports pass
a fixed iteration count and let the worker count multiply it, so an `N=8` run does
eight times the rollouts of the serial one — measured on the engine at
`rounds=24`: 192 rollouts against 24. A wall-clock read across that gap is eight
times the model spend reported as parallel efficiency. `--budget-rollouts N` pins
both arms to the same total; pass it to both, at the same value, and report
`result.rollouts` rather than the budget, because the synchronous path checks at
the round barrier and can overshoot by up to `n_workers - 1`.

OpenEvolve is the exception and needed no fixing: it derives
`rounds = iterations // workers`, so its total work was already fixed and the flag
simply sets `--iterations`.

Every example takes `--dry-run`, which prints its configuration and returns with
**zero network access and no API key**, and has an offline test suite
(`tests/test_<name>_example.py`) exercising its pure logic. Ports that need an
external dataset load it through the shared
[**`agentdescent.dataloader`**](dataloader.md) data layer — dependency-free
(`urllib` only), cached under `~/.cache/agentdescent/`, from each benchmark's
canonical source. Where a paper's full setup needs heavy infrastructure, the
boundary is documented in the example's module docstring — never hidden.

### The MethodPolicy command line

The eleven [`MethodPolicy`](policies.md) ports share one `main`, so they share
one parser: `build_parser()` in
[`examples/_method_runner.py`](https://github.com/Birfy/agentdescent/blob/main/examples/_method_runner.py),
which layers the flags below onto `add_standard_args`.

| flag | what it does here |
|---|---|
| `--budget-rollouts N` | mapped onto `--candidates`: the proposals the run may spend in total |
| `--workers N` | worker count, and the batch size the merge is sized to (`candidates // workers` rounds in the synchronous modes) |
| `--async` / `--async-ratio N` | barrier-free runtime and its lag budget. **`--async-ratio` defaults to 1 here**, not the shared 3 — see below |
| `--eval-concurrency N` | held-out evaluations in flight at once; wall-clock only. Left off, the runner's own rule applies: **1** under `--serial`, the worker count otherwise |
| `--eval-cache DIR` | memoise the gate to a directory two processes can share ([`FileCache`](api.md)). Off by default: a cache that outlives the run makes a rerun return the first run's numbers |
| `--staleness` | what to do with a diff proposed against a head the merger has since moved |
| `--reflective-merge` / `--no-reflective-merge` | override the method's own `reflective` declaration in either direction |
| `--val-cap` | **not offered.** These methods freeze train/held-out/test in `build()`, before the parser is consulted, so there is no gate split left to cap |

`--reflective-merge` is absent from every reproduce command on the algorithm
pages, and that is not an omission: nine of the eleven declare `reflective=True`
and Voyager and SkillWeaver declare `False`, matching what each measured row
recorded. The declaration is a fidelity statement, so the flag is for a control
arm that needs to vary exactly it.

!!! warning "Four of these reached this runner and were dropped — and one of them cost a number"
    `--async-ratio`, `--eval-concurrency` and `--eval-cache` were declared by
    the shared parser and never passed to `run_port`; `--val-cap` was accepted
    and could not be honoured at all. A run that set all four was byte-identical
    to a run that set none. All four are now wired or withdrawn, and
    `tests/test_method_runner_flags.py` enumerates the parser and fails on a
    flag with nowhere recorded that reads it.

    **`--async-ratio` is the expensive one.** Every async row under
    `bench/results/` recorded `async_ratio: 2` — the value its command line
    asked for — and ran at `run_port`'s default of **1**, because the flag never
    arrived. Those files now record 1, with a note, because that is what
    happened.

    So the default here is **1, not the shared 3**. Adopting 3 while fixing the
    flag would have made every documented `--async` command mean something new
    and left fifteen measured rows unreproducible from the command line.
    `--async-ratio 3` is one argument away, and it is a real change: the lag
    budget bounds both how far a worker's snapshot may drift behind head *and*
    how many cards may sit un-merged ahead of the merger.

    `--eval-concurrency` defaults to unset for the same class of reason. The
    shared default is 8, and taking it would have made `--serial` — the control
    arm, the upstream algorithm's own one-at-a-time loop — score its gate eight
    ways at once. That is the confound `bench/matrix_run.py` already documents
    for the other seven ports.

Reproducing one of the measured rows on the algorithm pages has this shape; each
page's **Run it** section gives the exact command:

```bash
python -m examples.<method>.<module> --yes --seed 0 \
    --budget-rollouts 80 --workers 8 \
    --async --async-ratio 1 --max-seconds 3600 \
    --staleness full --temperature 0.7 --no-thinking \
    --provider claude --model deepseek-v4-flash
```

`--max-seconds` is the one number the results files do not record. Any value
comfortably above the `engine_s` in the row's own cell leaves `--budget-rollouts`
as the binding stop, which is what those runs hit.

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
python -m examples.ace.ace_context_evolution --dry-run
python -m examples.ace.ace_context_evolution --model claude-haiku-4-5
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
python -m examples.gepa.gepa_prompt_evolution --dry-run
python -m examples.gepa.gepa_prompt_evolution --model claude-haiku-4-5
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
python -m examples.evoskill.evoskill_skill_discovery --dry-run
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
python -m examples.skillopt.skillopt_skill_training --dry-run
python -m examples.skillopt.skillopt_skill_training --model claude-haiku-4-5
```

---

## Program evolution

### OpenEvolve — function-minimization program search

*Repo* `algorithmicsuperintelligence/openevolve` · *task* bundled function
minimization.

OpenEvolve evolves Python source with model mutations, MAP-Elites feature grids,
island-local selection, and ring migration. The port maps source replacement to
a `Strategy`, the archive to a custom `Aggregator`, and concurrency to
`evolve()` / `async_evolve()`. Generated programs run behind an AST gate and a
sandbox with no network and explicit resource limits -- Bubblewrap on Linux,
Seatbelt (`sandbox-exec`) on macOS, and a refusal to run where neither exists.

The [dedicated page](algo-openevolve.md) documents the pinned upstream revision,
the intentional substitutions, and a live run in which the evolved program
reached the evaluator's ceiling on held-out seeds.

```bash
python -m examples.openevolve.openevolve_program_evolution --dry-run
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
python -m examples.adas.adas_meta_agent_search --dry-run
python -m examples.adas.adas_meta_agent_search --model claude-haiku-4-5 --generations 6
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
python -m examples.dgm.dgm_self_improve                      # runs offline (surrogate)
python -m examples.dgm.dgm_self_improve --generations 12 --archive keep_all
```

---

## Mechanism coverage

Every mechanism family in the original backlog now has at least one
implemented port. Fidelity differs by port and is recorded on each page (see
[port fidelity](port-fidelity.md)):

| Mechanism | Benchmark-faithful | Microports / analogues |
|---|---|---|
| Evolution / program search | [OpenEvolve](algo-openevolve.md) | [PromptBreeder](algo-promptbreeder.md), [AFlow](algo-aflow.md) |
| Reflection / refinement | [GEPA](algo-gepa.md) (reflective) | [Reflexion](algo-reflexion.md), [Self-Refine](algo-self-refine.md) |
| Skills / lifelong learning | [EvoSkill](algo-evoskill.md), [SkillOpt](algo-skillopt.md), [ACE](algo-ace.md) | [Voyager](algo-voyager.md), [SkillWeaver](algo-skillweaver.md) |
| Self-play / unlabeled data | — | [Absolute Zero](algo-absolute-zero.md), [R-Zero](algo-r-zero.md), [Agent0](algo-agent0.md) |
| Self-modifying code / harness | [DGM](algo-dgm.md), [ADAS](algo-adas.md) | [SICA](algo-sica.md), [Gödel Agent](algo-godel-agent.md) |

The label-free path now exists — the three self-play ports derive reward from a
grounded local verifier with no gold labels — but only as inference analogues:
a benchmark-faithful label-free port (real RL updates, real task domains)
remains open. TextGrad remains unimplemented in the reflection family.

### Deferred pending released code

* **CoEvoSkills** (arXiv:2604.01687, "Self-Evolving Agent Skills via
  Co-Evolutionary Verification") — the Skill Generator + co-evolving Surrogate
  Verifier + opaque pass/fail oracle is a compelling fit for AgentDescent's
  aggregator, but the **official code is unreleased** ("coming soon") and its
  benchmark (SkillsBench) requires a Claude Code / Codex agent harness. With no
  original repo to be faithful to, it is intentionally deferred until the authors
  release code, rather than reconstructed and mislabelled as faithful.

## Fidelity principles

Use the short [porting checklist](porting-checklist.md) before adding a row here,
and record the result in [port fidelity](port-fidelity.md) after — the checklist
is the standard, that page is what each port was actually found to do.

1. **Faithful to the repo, not just the paper.** Where the released code diverges
   from the paper's claims (e.g. EvoSkill's frontier is top-K aggregate, not
   per-instance Pareto), the example follows the **code** and says so.
2. **Faithful dataset identity.** Each example uses the paper's actual benchmark,
   loaded from its canonical source.
3. **Documented boundaries.** Where the full setup needs heavy infra (AppWorld,
   SWE-bench Docker, gated data), the example states the boundary and, when
   needed, substitutes a clearly-labelled surrogate — the algorithm stays
   faithful and runnable; nothing is passed off as a benchmark result it is not.
