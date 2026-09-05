# Evolving the search algorithm: the selection rule as the artifact

> "I plugged a tree search into `evolve()` to solve a problem. Now I want to
> evolve the algorithm itself — on one dataset, and validate it on a newer,
> harder one."

Everything in this repository evolves *what an agent reads* — a skill, a prompt,
a memory, a program. This example evolves *how the search is conducted*. It is a
two-level (meta) evolution, and the whole design is about which level owns what.

## How a port plugs a search algorithm in today

Nineteen ports reach `evolve()` two ways. The eleven declarative ones are a
`MethodPolicy` whose *mechanism* rides in `engine=Policies(...)` — `selection`,
`task_sampler`, `acceptance`, `conflict`, `fusion` — over a shared runner
(`examples/_method_runner.py`). The eight benchmark-faithful ones pass a custom
`strategy=` and/or the `aggregator_factory=` exit. A tree search is the second
kind: ERA's flat-PUCT tree is `EraTree` + `FlatPuct` behind an
`EraTreeAggregator` factory, and `FlatPuct` is a
`SelectionPolicy` — the engine's seam for *which candidate the next batch starts
from*.

So "the policy" a tree search is plugged in through is literally
`agentdescent.selection.SelectionPolicy`, and it used to be hard-wired:
`EraTree.__post_init__` built `FlatPuct(c_puct, prior_exponent)` with no way to
hand it anything else. That is the seam this example opens:

```python
EraTree(policy=my_selection_policy)              # default: FlatPuct, unchanged
run_agentdescent_era(..., selection=my_policy)    # the live ERA search, any domain
```

`tests/test_era_example.py` still pins the default tree against a transcription
of upstream `futs.search`, so a run that names no policy is the port upstream
ships.

## The API: `meta_evolve()`

The outer loop is a library call, [`agentdescent.meta`](../../docs/meta-evolution.md),
and this example is one instantiation of it:

```python
from agentdescent import meta_evolve, meta_validate, priority_selection, slot_reflector

spec = priority_selection()                       # the `selection` slot, as a gated priority rule
result = meta_evolve({"source": landscape_problem(SOURCE)},   # (value, seed) -> MetaOutcome
                     slot="selection", spec=spec,
                     propose=slot_reflector(model, spec), seeds=range(20),
                     rounds=6, n_workers=4)
policy = spec.compile(result.rendered)            # -> EraTree(policy=...), Policies(selection=...)
meta_validate(spec, spec.render(spec.initial()), result.rendered,
              {"source": ..., "target": ...}, seeds=fresh)
```

`slot` is any field of the decision plane (`SLOTS`: selection, task_sampler,
acceptance, conflict, fusion, promotion, staleness, proposal); the machinery
fields are refused. A `SlotSpec` is a strategy that also compiles: `ParamSlot`
holds a policy class's numeric constructor keywords (different parameters
union-merge), `SourceSlot` holds gated source (every round is a tournament).
An inner problem is anything `(value, seed) -> MetaOutcome`; `evolve_problem`
wraps an inner `evolve()` and `run_agentdescent_era(selection=...)` wraps an
ERA search.

## The two levels

| | inner | outer |
|---|---|---|
| artifact | a program (ERA node) | the **selection rule** `priority(rank, visits, total, prior, depth, n_nodes)` |
| task | one shard of one problem | one **whole search problem** (a landscape instance, an AlgoTune task, a Harbor task) |
| `run` | score a program on a shard | compile the rule, plug it into the real `EraTree`, run a whole inner search at a fixed budget, return the trace |
| `reward` | the domain metric | the inner search's **AUC**: mean best-so-far over the budget |
| `propose` | rewrite the program | a model reads the rule and the trace and rewrites the rule |
| governance | L1 (a program is a harness) | L1 (`blast_radius=0.6`) — a rule changes how *everything* is searched |
| gate | held-out shards | held-out **search instances** |

Two decisions worth defending:

**The surface is one scoring function, not `select`.** A whole `select` can
return the same dead node forever, skip the visit reservation and starve the
root, or raise mid-run. A function of six numbers returning one number can only
be wrong about *priority* — the thing being searched for. Rank normalisation,
prior normalisation, back-propagation and the tie-break stay in the tree and the
wrapper (`EvolvedSelection`). Validation lives once, in the strategy's `to_diff`
(`SearchPolicySlot`): SICA's AST gate widened to arithmetic, comparisons,
conditionals and `math`, then the compiled rule is run over a fixed grid of
inputs — the root before any expansion included — and must be finite everywhere.
A rule that divides by `visits` is refused at proposal time, not at the root.

**The reward is AUC, not the final best.** A selection rule cannot make a better
program exist; it can only find one sooner. At a fixed expansion budget the final
best barely separates rules; the area under the best-so-far curve is what
selection controls.

## Where to evolve, where to validate

The outer loop runs a whole inner search per rollout *and per held-out instance
at every gate*. That multiplies the inner cost by hundreds, so the inner domain
you evolve on has to be cheap, and the benchmark you validate on can be
expensive because it is scored once per rule.

| stage | inner domain | cost per inner search | what it establishes |
|---|---|---|---|
| 0 — offline | `_landscape.py`: seeded synthetic landscapes, `SOURCE` to evolve on, `TARGET` (higher-dim, ruggeder, deadlier) never seen by the outer loop | milliseconds | the machinery, and whether a rule that wins in-distribution wins out of it |
| 1 — live, cheap | **AlgoTune** (arXiv:2507.15887): sandboxed speedup over a reference; `run_agentdescent_era(domain=algotune_domain(...), selection=EvolvedSelection(src))`, reward `era_auc(run.result.history)` | minutes (a model call + timing per expansion) | a rule evolved on real program search, on a recent, hard benchmark with a baseline already measured in `bench/results/era-algotune-model-prior.md` |
| 2 — validate | **SWE-bench-Science** (arXiv:2608.19799, 119 tasks / 98 scientific repos, HF `OpenMOSS-Team/SWE-bench-Science`) and **Terminal-Bench-Science 0.1** (70 expert tasks, Harbor `terminal-bench-science/terminal-bench-science@latest`) | an agent run in a container per expansion | whether the evolved rule transfers to agentic scientific work it never saw |

Why those two for validation, and not AIME or GSM-Hard: both are Harbor-format
container tasks released in August 2026, scored 0/1 by task-specific tests in a
clean verifier, and the strongest published agent resolves under half of either
(Claude Code + Opus 5: below 50% on SWE-bench-Science, 30.0% on TB-Science
0.1). They post-date every source the rule could be evolved on, so a transfer
number there is not memorisation, and they are what "a search algorithm that
solves scientific problems" is actually asked to do.

### Stage 2, concretely: a Harbor task as an ERA `Domain`

ERA's search is indifferent to what it searches over; a `Domain` is four things.
For a Harbor task:

| `Domain` field | Harbor task |
|---|---|
| `initial_program` | the empty patch against the task's baseline (`task.toml` + Docker image pinned by digest) |
| `evaluate(patch, shards)` | apply the patch in a clean container, run the **scoring subset** of the task's tests, return the pass fraction; the held-back subset is reported once at the end |
| `prompt(parent)` | the task's `instruction.md`, the parent patch applied in the workspace, the scoring tests' output — handed to `claude_code()` (or `openai_compatible`) which works in the container and returns `git diff` |
| `test_shards` | the held-back tests |

That is the same split discipline ERA uses on its shards, and it is where the
honest boundary sits: a task with one test file has nothing to hold back, and
its search signal is then the agent's own checks. The adapter is not in this
repository yet — it needs Docker or Modal, the `harbor`/`pier` runner and an
agent, none of which the offline suite can exercise — and the boundary is stated
here rather than hidden, as every port's is.

## Protocol

1. **Three seeds** of the outer run per setting; report mean ± sd.
2. **Controls.** `--serial` (one worker, nothing to merge — the upstream serial
   loop); the seed rule (flat PUCT) as the baseline; and, on the target, the
   rule evolved *on the target itself* as the ceiling.
3. **Fresh instances for validation.** `validate()` scores the seed rule and the
   evolved rule on instances disjoint from everything the outer loop trained or
   gated on, for both `SOURCE` and `TARGET`, and reports gain, its sd, wins /
   losses, and the **transfer ratio** (target gain / source gain).
4. **Read the ratio, not the gain.** A ratio near 1 is a better search rule; near
   0 with a positive source gain is a fit to the landscape it was evolved on; a
   negative one is a rule that trades generality for its training set. The
   offline run in `tests/test_metasearch.py` produces the second kind on purpose
   (a greedier rule: +0.008 on `SOURCE`, 11/4 wins; +0.000 on `TARGET`), which
   is exactly the result this design exists to make visible.

## Run

```bash
python -m examples.metasearch.evolve_search_policy --dry-run
python -m examples.metasearch.evolve_search_policy --provider openai \
    --model deepseek-v4-flash --rounds 6 --workers 4 --tasks 20 --yes --out out/metasearch.json
python -m examples.metasearch.evolve_search_policy --serial --rounds 12 --yes   # the control
```

Offline tests: `pytest tests/test_metasearch.py` — the gate, the seed rule's
bit-for-bit equivalence with `FlatPuct` on both the upstream-style trace and the
landscape, the end-to-end outer loop with a scripted proposer, and the report.
