# Meta-evolution — evolving the slots of `evolve()`

*Module:* [`agentdescent.meta`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/meta.py)
· *Example:* [`examples/metasearch/`](https://github.com/Birfy/agentdescent/tree/main/examples/metasearch)
· *Design record:* [design-meta-evolution.md](design-meta-evolution.md)

Every decision `evolve()` makes is a field of [`Policies`](policies.md), and
every algorithm port plugs its mechanism into one of those fields — a tree
search into `selection` (behind the [`aggregator_factory` exit](aggregator-factory.md)
when it needs a tree of its own). `meta_evolve()` evolves **the field itself**:

```python
from agentdescent import meta_evolve, meta_validate, priority_selection, slot_reflector

spec = priority_selection()               # the `selection` slot as a gated priority rule
result = meta_evolve(
    {"algotune": algotune_problem},        # inner problems: (value, seed) -> MetaOutcome
    slot="selection", spec=spec,
    propose=slot_reflector(model, spec),   # the reflector rewrites the value
    seeds=range(8), rounds=6, n_workers=4)

policy = spec.compile(result.rendered)    # -> Policies(selection=policy), or EraTree(policy=...)
report = meta_validate(spec, spec.render(spec.initial()), result.rendered,
                       {"swe_science": swe_problem, "tb_science": tb_problem}, seeds=range(4))
```

It is the ordinary engine one level up. The **artifact** is the slot's value,
held by a `SlotSpec` — a [strategy](strategies.md) that also knows how to
*compile* its rendering into the object `Policies` takes. A **task** is one
inner `Problem`: run a whole inner search with the candidate value installed,
at a fixed budget, from a seed, and return a `MetaOutcome` (the held-out curve,
the final reward, the rollouts spent, whatever the reflector should see). The
**reward** is a function of that outcome — `auc` by default, the mean
best-so-far, because a decision rule cannot make a better answer exist, only
find one sooner. **Governance** is L1: a slot value changes how everything
downstream is searched, so every merge also passes the oracle.

## What may evolve

`SLOTS` is the decision plane: `selection`, `task_sampler`, `acceptance`,
`conflict`, `fusion`, `promotion`, `staleness`, `proposal`. The machinery
fields — `verifier`, `ledger`, `executor`, `evaluator`, `eval_cache`,
`aggregator_factory`, `sandbox_*` — are the training code and are refused,
which is where the [central analogy](concepts.md) draws the L0 line.

## Two representations of a value

| spec | value | merge behaviour | gate |
|---|---|---|---|
| `ParamSlot(factory, params, bounds)` | the numeric constructor keywords of any policy class (`FlatPuct(c_puct, prior_exponent)`, `Beam(k)`, …) | different parameters union-merge; the same one contradicts and is resolved on held-out | unknown names and out-of-bounds values are refused |
| `SourceSlot(initial_value, validate, build)` | source text compiled by `build` | one slot: every round is a tournament | `validate` raises `ValueError` |

**`policy_source(slot, seed)` is the general one**: the value is the source of a
class satisfying the slot's own Protocol (`SLOT_PROTOCOLS`, all
`runtime_checkable`). The gate refuses imports outside a fixed allowlist, dunder
access and the calls that reach the interpreter, builds the class in a
namespace of safe builtins, checks it with `isinstance` against the Protocol,
and runs a smoke test — one shipped per slot, replaceable via `smoke=`.
`seed_source(slot)` provides a starting value for every slot: the engine's
default rule transcribed where a seed can carry it (`selection`,
`task_sampler`, `staleness`, `fusion`, `promotion`), the simplest contract-
satisfying rule where the default reads the verifier or a posterior
(`acceptance`, `conflict`), and a placeholder shape for `proposal`. It is the
gate SICA and Gödel Agent run their self-edits behind,
not a sandbox: enough to keep a model's rewrite to *deciding*, not enough for
code from a stranger.

`priority_selection()` is the narrower, safer `SourceSlot` for a tree search: the value
is one function, `priority(rank, visits, total, prior, depth, n_nodes)`, seeded
with upstream ERA's flat PUCT, and `PrioritySelection` is the `SelectionPolicy`
that runs it — rank normalisation, prior normalisation, the visit reservation up
the parent chain and the tie-break stay in the wrapper, so a rule can only be
wrong about *priority*. The gate is an AST whitelist plus a run over a fixed
grid of inputs that includes the root before any expansion: a rule that divides
by `visits` is refused at proposal time.

## What the outer loop actually does

`meta_evolve` is not a new optimiser. It is [`evolve()`](api.md) — the same
parallel, merge-based loop the rest of this project uses — with three things
substituted:

| `evolve()` term | what `meta_evolve` puts there |
|---|---|
| the **artifact** | the slot's value: for `priority_selection()`, the source of one function |
| a **rollout** | compile the candidate, then run **one whole inner search** with it and return the best-so-far curve |
| the **reward** | a `MetaOutcome -> [0, 1]` summary of that curve, `auc` by default |
| the **task set** | one task per `(inner problem, inner seed)` pair, interleaved *seed-major* so every problem lands on both sides of the train/held-out cut |

One outer rollout is therefore an entire inner search, which is what makes this
expensive and what decides where to evolve and where to validate.

The loop per round:

1. **Split.** `held_out_frac` cuts the tasks into train and held-out, as
   `evolve()` always does. The held-out side is never rolled out on.
2. **Roll out.** `n_workers` workers each take a train task in parallel, compile
   the current rule through the slot's gate, and run the inner search.
3. **Propose.** For a rollout that scored below target, `slot_reflector` makes
   **one model call**. The prompt is only four things: the slot's own
   `describe()` (arguments and the syntax contract — no advice about what to
   change), the current value, the inner outcome as JSON (`curve`, `final`,
   `outcomes`), and the reward. The model replies with a whole new function.
4. **Merge.** The workers' proposals go through the ordinary conflict and fusion
   policies — by default `reflective_merge`, so contradicting proposals are
   fused rather than one being discarded.
5. **Gate.** The merged candidate is scored on the **held-out** tasks and
   committed only if it does not lose ground there.
6. **Oracle.** `blast_radius=0.6` is `HARNESS_BLAST_RADIUS`, so the slot is
   governed at **L1**: every merge is also forced through the oracle, which
   scores the same artifact on the same held-out set. A tie is a veto.

Two gates sit in front of the model's output, and they are what make an
arbitrary LLM rewrite safe to execute:

* **an AST whitelist** — exactly one function of the six named arguments, built
  from arithmetic, comparisons, conditionals, locals, `min`/`max`/`abs` and
  `math.sqrt/log/log1p/exp/tanh/pow`. No imports, loops, or other calls.
* **a fixed input grid** — the rule must return a finite number everywhere on
  it, including the root before any expansion. A rule that divides by `visits`
  is refused at proposal time rather than crashing at the root.

The wrapper keeps rank normalisation, prior normalisation, the visit reservation
up the parent chain and the tie-break, so an evolved rule can only be wrong
about *priority* — never about the tree's bookkeeping.

### The configuration the real-data result used

```python
meta_evolve(problems, slot="selection", spec=priority_selection(),
            propose=slot_reflector(complete, spec),
            seeds=[0], rounds=8, n_workers=3, held_out_frac=0.4)
```

with an inner search of 8 expansions, `deepseek-v4-pro` at temperature 0.7,
thinking disabled, and completions cached so an inner run is a function of the
rule.

**It asked for 8 rounds and got 2**, stopping on a `max_seconds=5400` budget:
6 rollouts, 2 commits, 0 rejections, 40 model calls, 197k tokens, 2.7 hours.
A rollout is one whole inner search, so a sweep of three workers costs about
half an hour — budget the outer loop in *sweeps of wall clock*, not in rounds.

**`seeds=[0]`, not `[0, 1, 2]`.** The recorded run passed three seeds and they were three copies of one comparison: on a domain where the inner run is a function of the value, the seed randomises nothing. Spend that budget on more *problems* instead — the reasoning is in [the result page](https://github.com/Birfy/agentdescent/blob/main/bench/results/metasearch-selection-srbench.md).

## From the CLI and the host plugins

`kind: "policy_slot"` puts all of this behind the ordinary spec surface, so
`plan`, `evolve`, `status` and `watch` work on it unchanged — it is still an
`evolve()` call. `show` prints the evolved rule, and `apply` needs a
destination: the artifact is source text, but `target` names a slot rather than
a file, so `agentdescent apply <run> --to rule.py` is how the rule comes out. `agentdescent init selection --kind policy_slot`
writes a starter:

```json
{
  "kind": "policy_slot",
  "target": "selection",
  "data": {"problems": "mypkg.problems:build", "seeds": [0]},
  "score": "auc",
  "agent": {"ref": "openai_compatible", "model": "deepseek-v4-flash"},
  "evolve": {"rounds": 4, "n_workers": 2, "max_seconds": 5400}
}
```

Two fields do not mean what they mean for every other kind, and both are the
nature of the thing rather than an inconsistency:

* **`target` is a slot name, not a path.** `selection`, `task_sampler`, … — the
  artifact is a decision rule of the optimiser, and there is no file to point at.
* **`data` holds refs, not rows.** An inner problem is a callable
  `(value, seed) -> MetaOutcome`; no row format can express one. `data.problems`
  is a `module:attribute` ref to a zero-argument builder returning a mapping of them,
  inside the spec's import allowlist. Pass `--data mypkg.problems:build` to
  `init` to choose that ref; make the module importable and include its package
  in the generated spec's `allow` list before planning.

`score` takes `auc` (the default), `final_reward`, `rollouts_to`, or a ref to
any `MetaOutcome -> float`.

**`plan` is worth reading here rather than skipping.** Because a rollout is a
whole inner search, it reports how many outer tasks exist and how many of them
the gate actually gets — and it says so when that number is 0 or very small,
which is the defect this line of work hit at four different levels and which has
failed both ways: committing nothing, and silently committing a rule that lost
on its own family.

## Where to evolve, where to validate

The outer loop runs a whole inner search per rollout and again per held-out
problem at every gate, so evolve where the inner problem is cheap and validate
where it is expensive — that is scored once per value. Three experiments ship, cheapest first:

| script | slot | inner problem | needs |
|---|---|---|---|
| [`bench/metasearch_slots.py`](https://github.com/Birfy/agentdescent/blob/main/bench/metasearch_slots.py) | `task_sampler` | a whole inner `evolve()` evolving an instruction on a slice of GSM-Hard / GSM8K | a model, nothing else |
| [`bench/metasearch_algotune.py`](https://github.com/Birfy/agentdescent/blob/main/bench/metasearch_algotune.py) | `selection` | an ERA tree search on one AlgoTune task, scored in speedup | a model, a sandbox |
| [`examples/metasearch/_harbor.py`](https://github.com/Birfy/agentdescent/tree/main/examples/metasearch) | `selection` | an ERA tree search over patches to a Harbor science task | a model, a container, an agent |

The example itself does stage 0 offline on a synthetic landscape (a seeded
family to evolve on, a harder one never seen), and its boundary is stated in
[`examples/metasearch/README.md`](https://github.com/Birfy/agentdescent/blob/main/examples/metasearch/README.md).

`meta_validate` scores the value before and after on problems disjoint from the
outer run, paired by seed, and `transfer_ratio` reads target gain over source
gain: near 1 is a better rule, near 0 with a positive source gain is a fit to the
training landscape, negative is a rule that traded generality for it.

**A seed is only a replicate if it moves the run.** On a domain where the
evaluator is deterministic and completions are cached, an inner run is a
function of the value — which is the property that makes the domain measurable
at all, and it makes `seeds=[0, 1, 2]` three copies of one comparison. Replicate
across problems or across the data split, and check the numbers differ before
counting them as independent.

### What one evolved rule looks like

Seed (upstream ERA's flat PUCT) against the rule two outer sweeps on
LLM-SRBench `lsr_synth` committed:

```python
# before
def priority(rank, visits, total, prior, depth, n_nodes):
    c = 1.0
    return rank + c * (1.0 / n_nodes) * math.sqrt(total) / (1 + visits)

# after
def priority(rank, visits, total, prior, depth, n_nodes):
    c, d = 1.0, 0.1
    exploration = c * math.sqrt(math.log1p(total) + 1.0) / (1.0 + visits)
    prior_term = prior * math.sqrt(total + 1.0) * (1.0 - rank)
    return rank + prior_term + exploration - d * depth
```

`1 / n_nodes` is **deleted**, so exploration stops decaying as the tree grows;
`(1 - rank)` explicitly favours low-ranked nodes; `- 0.1 * depth` prefers
breadth. Measured at nine nodes it moves a node **9.5x** further off its raw
rank than the seed does, and in the opposite direction as the tree grows.

The same seed and the same reflector on a *synthetic* landscape go the other way
— every one of three runs **shrinks** exploration. Nothing in the prompt says
which way to go; the domains want opposite things and the search finds each.

## What has been measured

| result | what it says |
|---|---|
| [Evolving the tree search itself](https://github.com/Birfy/agentdescent/blob/main/bench/results/metasearch-tree.md) | **positive, synthetic.** +0.0227 source / +0.0120 target over 3 seeds; lands on the Pareto front of the hand-tuned PUCT family, from a seed that is off it |
| [A search rule on real scientific data](https://github.com/Birfy/agentdescent/blob/main/bench/results/metasearch-selection-srbench.md) | **positive, real data.** LLM-SRBench `lsr_synth`: 5 wins / 0 losses / 4 ties over nine held-out paired comparisons, sign test p = 0.031 — and the rule moves in the *opposite* direction to the synthetic one, as predicted in advance |
| [Choosing a domain to evolve on](https://github.com/Birfy/agentdescent/blob/main/bench/results/metasearch-domain-selection.md) | **the method, and every null.** Four properties a domain must supply, three checks that cost one inner run apiece, and what each domain tried here actually supplied — AlgoTune (irreparable: the timing *is* the feedback) and SWE-bench-Science (three properties out of four; the fourth needs the agent phase) included |
| [A benchmark x slot matrix](https://github.com/Birfy/agentdescent/blob/main/bench/results/metasearch-slots.md) | **mixed.** Seven cells on `task_sampler` and `acceptance`; four commit, unseen gains +0.02 to +0.04, none negative once the inner budget is adequate |

## The product surface: `meta_run`

`meta_evolve` returns a raw `EvolutionResult`; wiring the value back and
judging the transfer is the caller's job. `meta_run()` is that job, done — one
call that evolves, validates on problems the outer loop never saw, and decides:

```python
from agentdescent import meta_run, ParamSlot, DifficultyWeighted

report, result = meta_run(
    {"train-a": problem_a, "train-b": problem_b},   # the outer loop's train set
    {"val-z": problem_z},                           # never seen by the outer loop
    slot="task_sampler",
    spec=ParamSlot(factory=lambda c: DifficultyWeighted(c=c),
                   params={"c": 0.2}, bounds={"c": (0.05, 1.0)}),
    propose=offline_propose,                        # or slot_reflector(model, spec)
    seeds=(0, 1), val_seeds=(20, 21), rounds=3, n_workers=1)

print(report.verdict, report.reason)                # "adopt" / "reject", with why
value = spec.compile(report.value)                   # -> Policies(task_sampler=value)
```

`MetaRunReport` bundles the rendered value (`report.value`; compile with
`spec.compile(...)`), the per-problem validation (`before` / `after` / `gain`),
the transfer ratio (mean val gain / mean train gain) and an `adopt` / `reject`
verdict: *adopt* when the mean gain on unseen problems clears `min_gain` and
the transfer is not below `transfer_floor`, *reject* otherwise, with the
reason. The report rides `result.meta`, so a CLI or a plugin shows the decision
without recomputing it.

**The verdict is honest.** The caller must keep validation problems and seeds
disjoint from the training set (`meta_run` raises on overlap), so a verdict is
never measured on the problems that shaped the value. The measured matrix
(below) found transfer of +0.02 to +0.04 on unseen benchmarks — real, positive,
and modest. `meta_run` does not dress that up: a rule that does not transfer is
reported as `reject`, which is a result, not an error.

## Continuous meta-learning: the archive drives the next session

`meta_run` decides and the run dies with the process; `continuous_meta_run`
makes the decision durable. It is `meta_run` wired to the
[consolidation archive](consolidation.md), so the next session starts where the
last one was validated rather than from the seed:

```python
from agentdescent import ConsolidationStore, continuous_meta_run

store = ConsolidationStore("~/.agentdescent/archive")

# session 1: the empty archive is an ordinary meta_run
report, _ = continuous_meta_run(
    store, slot="task_sampler", spec=slot(), propose=offline_propose,
    tasks_for=lambda rec: tasks_for_config(rec.config),   # caller's data, read through the archive
    run=run, reward=reward, inner_propose=propose, inner_strategy=SingleSlot("cap:0"),
    config=CONFIG, run_id="run-1",
    problems={"a": problem_a, "b": problem_b},            # first session: caller's problems
    val_problems={"z": problem_z},
    inner={"rounds": 5, "n_workers": 1}, rounds=3, n_workers=1)

# session 2: the archive seeds the search and supplies the problems
report, _ = continuous_meta_run(
    store, slot="task_sampler", spec=slot(), propose=offline_propose,
    tasks_for=lambda rec: tasks_for_config(rec.config),
    run=run, reward=reward, inner_propose=propose, inner_strategy=SingleSlot("cap:0"),
    config=CONFIG, run_id="run-2", problems=None, val_problems={"z": problem_z})
```

The loop per session:

1. **Problems come from the archive.** The top `train_runs` archived rules'
   problem descriptors become the train set, the next `val_runs` the validation
   set — rebuilt by `problem_from_archive` from each record's descriptor plus
   `tasks_for` and the inner domain functions, so the outer loop never validates
   on a problem it trained on. An archive too thin to fill a set falls back to
   `problems` / `val_problems` — the first session is an ordinary `meta_run`.
2. **The search starts where the archive left off.** The best archived rule for
   the slot, spec and configuration becomes the session's `initial_state` (same
   configuration first, then measured reward), so the evolution *refines* last
   session's rule rather than rediscovering it. Use a fresh `repo_path` per
   session: the engine ignores `initial_state` when it is resuming a repo that
   already has state.
3. **The gain is measured twice.** `mean_gain` is the evolved value against the
   prior this session had to beat; `fresh_gain` is it against the fresh seed —
   the honest cross-session number, the rule you have now against the rule you
   had before the first session ever ran.
4. **Adoption is archived.** On `adopt`, `archive_rule` writes the new value
   back with a `problem` descriptor (the slot and the JSON-safe inner settings;
   the tasks stay with the caller — the "link, don't copy" rule applied to a
   problem instead of a trajectory). The next session starts from a better rule,
   and the report's `seeded_from` / `archive_count` say so.

`archive_rule(store, report, result, *, run_id, slot, spec, config, problem)`
writes an adopted rule to the archive; `problem_from_archive(store, run_id, *,
tasks_for, spec, run, reward, propose, strategy)` rebuilds a runnable
`Problem` from a record's descriptor, refusing a record archived by a different
spec (its state would be misread, exactly the guard `seed_from_archive` runs).
`ConsolidationRecord.problem` is a new optional field — an archive written
before it existed reads back as `None` and stays seedable.

Offline demo (two sessions, the second seeded by the first, each archiving a
strictly better rule):

```bash
python -m examples.meta.continuous_meta_demo
```

## Meta-audit: the trust layer for the rule-about-rules

The audit package verifies the *inner* reward — `AuditedReward` pairs a cheap
verifier against ground truth on a probability sample. Nothing verified the
**meta reward**: `meta_run` decides adopt/reject on an `auc` scored over inner
outcomes, and that number was the one unverified number in the loop. `meta_audit`
is `AuditedReward` applied one level up:

```python
from agentdescent import MetaAuditStore, meta_run_audited, final_reward

store = MetaAuditStore("~/.agentdescent/audit.jsonl")
report, result = meta_run_audited(
    {"train-a": problem_a, "train-b": problem_b},
    {"val-z": problem_z}, slot="task_sampler", spec=slot(),
    propose=offline_propose, oracle=final_reward, store=store,
    audit_rate=0.1, rounds=3, n_workers=1)
# result.meta: {"verdict": ..., "audited_n": 12, "oracle_gain": 0.04, "gain_bias": 0.3, ...}
```

The contract is the audit layer's own:

* **A probability sample.** Every `(problem, seed)` pair recorded carries its
  `inclusion_prob` — without it nothing estimates anything, and it cannot be
  reconstructed later.
* **The decision path never sees the oracle.** `meta_reward` scores every
  verdict; the oracle is called only on sampled pairs and only lands in the
  store. Auditing can be turned on mid-fleet without changing which rules are
  adopted. An oracle that cannot answer yet returns `None`: the pair is stored
  pending and `MetaAuditStore.resolve()` attaches truth later.
* **The verdict is durable.** Sampled evidence and the decision it served are
  both in the JSONL store, append-only, reconciled by last occurrence, so
  "why was this rule adopted" is answerable after the run is gone.
* **The bias is estimable.** Over the resolved sample, `verifier_gain -
  oracle_gain` is the residual, and `MetaAuditStore.gain_bias()` is its Hajek
  estimate. If the meta reward is systematically optimistic, that shows up as a
  number instead of adopting rules in silence. `rectified_gain()` is the honest
  gain estimate over the same sample.

The measured case in the tests: a meta reward that reads an `auc` of `+0.65`
where truth reads `+0.0` — `gain_bias` reports the `+0.65`, the verdict is
still computed (and recorded) on the verifier, and the store holds the pairs
with both readings.

## Meta-population: the archive seeds a population, the population promotes

`continuous_meta_run` seeds one session from the archive's best rule;
`meta_population_run` is the parallel version — the archive's top rules each
seed an island, every island searches from its own starting rule, and the best
adopt island's rule is promoted back:

```python
from agentdescent import ConsolidationStore, meta_population_run

report, _ = meta_population_run(
    store, slot="task_sampler", spec=slot(), propose=offline_propose,
    tasks_for=lambda rec: tasks_for_config(rec.config),
    run=run, reward=reward, inner_propose=propose, inner_strategy=SingleSlot("cap:0"),
    config=CONFIG, islands=3, train_runs=3,
    problems={"ta": helpful, "tb": helpful}, val_problems={"va": helpful_val},
    rounds=3, n_workers=1)
for island in report.islands:          # one row per starting rule
    print(island.index, island.seeded_from, island.mean_gain, island.verdict)
print(report.promoted_from, report.reason)
```

The loop, in the same shape as the rest of the meta surface:

1. **Islands are sessions.** The archive's rule records for the slot/spec/config
   are ranked as `continuous_meta_run` ranks them; the top `islands` distinct
   rules each seed a `meta_evolve` via `initial_state`. Every island trains on
   the same problems and validates the same `val_problems` at `val_seeds`
   disjoint from training, so the islands differ only in *where they started*.
2. **Each island runs in its own fresh repo.** A shared `repo_path` would make
   the engine resume the ledger and silently ignore the islands' distinct
   `initial_state` — a population of clones. `repo_dir=` names the base
   directory for the per-island repos.
3. **Promotion is evidence-based and fail-closed.** Only an island whose
   verdict is `adopt` (positive mean gain on unseen problems, non-negative
   transfer) can be promoted; the best such island — highest mean gain, then
   best transfer — is written back via `archive_rule`. An island that errored
   is reported and never promoted, without stopping the others.
4. **The report is per-seed.** `MetaPopulationReport.islands` is one
   `PopulationIsland` per starting rule (seeded_from, value, mean_gain,
   transfer, verdict, reason); `promoted_from` / `promoted_run_id` /
   `archive_count` say what the population changed.

An empty archive is refused with a pointer to run `continuous_meta_run` once
first — a population needs something to diversify from. Offline demo (two
sessions seed the archive, a population of two islands promotes the one that
gained):

```bash
python -m examples.meta.population_meta_demo
```

## Meta-curriculum: the meta loop generates the problems it is judged on

`continuous_meta_run` trains on the archive's problems and validates on the
caller's. The problem distribution is the least-examined part of the design —
the same gap `curriculum_evolve` closed one level down for tasks.
`curriculum_meta_run` closes it for problems: a challenger generates frontier
*problems*, the rule climbs toward them, and the validation set never moves:

```python
from agentdescent import ConsolidationStore, MetaProblemChallenger, curriculum_meta_run

challenger = MetaProblemChallenger(
    generate=lambda spec, seed: parse(spec),        # a candidate problem config
    validate=build_problem,                          # config -> Problem; raises on invalid
    propose=advance_difficulty,                      # (descriptor, report) -> next descriptor
    spec="d:0.5")

report, result = curriculum_meta_run(
    store, slot="task_sampler", spec=slot(), challenger=challenger,
    propose=offline_propose,
    problems={"ta": problem_a, "tb": problem_b},    # the fixed curriculum
    val_problems={"va": problem_z},                 # never touched by the challenger
    rounds=6, n_workers=1)
print(report["frontier"])   # generated_total, invalid_total, learned, stuck, r0_mean
```

The loop:

1. **The challenger is the distribution's own solver-challenger.** Between outer
   rounds it draws a candidate problem configuration, runs it through a
   mandatory validity gate (`validate` must build a runnable `Problem`; a
   payload that cannot is counted, never crashed over), and installs the
   validated problems into a `FrontierSampler` that mixes them with the seed
   problems.
2. **Learnability is two-phase and measured online.** `record` sees the meta
   reward; a generated problem the rule already solves is dropped on its first
   attempt, one it never learns is dropped as stuck, one it learns is counted —
   the frontier keeps turning toward the rule's current edge, and
   `propose(descriptor, report)` pushes the generation spec outward on the
   measured `learned` count.
3. **The challenger never touches the final evaluation.** The validation set is
   fixed from `val_problems` and the held-out split is computed from the seed
   tasks alone — what the run reports is always measured on problems the
   challenger never generated.
4. **Continuity is the archive's.** The search starts from the best archived
   rule for the slot/spec/config (in a fresh ledger repo, so the seed is
   honoured), and on `adopt` the rule is written back with its problem
   descriptor — the next session starts where this one was validated, with a
   problem distribution the meta loop itself has been pushing outward.

Offline demo (a challenger pushes the difficulty outward while the rule climbs):

```bash
python -m examples.meta.curriculum_meta_demo
```

## Self-exam: adoptions are provisional until the oracle says so

`archive_rule` writes an adopted rule into the archive, and `_archived_rules`
keeps seeding the next search from it. `exam_archive` is the rung that makes
that adoption **provisional**: it re-validates each rule against the problems
its own descriptor describes, using an independent oracle, and stamps the
outcome on the record:

```python
from agentdescent import ConsolidationStore, exam_archive, final_reward

report = exam_archive(
    store, slot="task_sampler", spec=slot(), oracle=final_reward,
    tasks_for=lambda rec: tasks_for_config(rec.config),
    run=run, reward=reward, propose=propose, strategy=SingleSlot("cap:0"),
    config=CONFIG, min_gain=0.0, max_flags=2)
print(report.passed, report.flagged, report.demoted)
```

The ladder:

* **`passed`** — the oracle gain on the rebuilt problems clears `min_gain`
  (flags reset). The rule keeps seeding the search.
* **`flagged`** — the oracle disagrees; the rule stays seedable and the flag
  count climbs.
* **`demoted`** — a rule that fails truth `max_flags` times in a row is retired:
  `_archived_rules` skips it, so `continuous_meta_run` and population selection
  never start a search from a rule truth has rejected.

`ConsolidationRecord.exam` / `exam_flags` / `exam_note` carry the stamp
(optional, backwards compatible), and `ConsolidationStore.mark_exam` rewrites
atomically. The meta loop's verdicts are thereby provisional until examined,
and the archive self-corrects: what the loop believed is checked with evidence,
and what fails the check stops steering the search.

This is the audit rung made *actionable*: `meta_run_audited` records the
probability-sampled evidence, and `exam_archive` turns resolved oracle truth
into a seeding decision.

## Trajectory-driven curriculum: the problems come from the run's own failures

`curriculum_meta_run` (#429) gives the meta loop a challenger that generates
frontier problems; where those problems come from is the challenger's
hardest question. `trajectory_challenger` answers it with the run's own
memory — every archived rule links its evidence through
`ConsolidationRecord.trajectory_store` (#334), and the trajectory ledger (#248)
records every rollout, including the ones that failed:

```python
from agentdescent import ConsolidationStore, trajectory_challenger, problem_from_tasks, curriculum_meta_run

challenger = trajectory_challenger(
    store,
    tasks_for=lambda ids: tasks_by_id(ids),            # task_id -> Task
    build_problem=lambda tasks: problem_from_tasks(    # tasks -> Problem
        tasks, slot="task_sampler", run=run, reward=reward,
        propose=propose, strategy=SingleSlot("cap:0"), rounds=5, n_workers=1),
    slot="task_sampler", failure_threshold=0.5)

report, result = curriculum_meta_run(
    store, slot="task_sampler", spec=slot(), challenger=challenger,
    propose=offline_propose,
    problems={"ta": problem_a, "tb": problem_b}, val_problems={"va": problem_z},
    rounds=6, n_workers=1)
```

The challenger's generation spec is an **archived run_id**: `generate` reads
that rule's trajectory store, takes the rollouts whose reward fell below
`failure_threshold` (a corrupt record is skipped; a rule with no trajectories —
or no failures to mine — is counted invalid, never crashed over), and `propose`
advances across the archive's rules that recorded trajectories, so the frontier
turns across the framework's failures instead of one rule's. `problem_from_tasks`
is the natural problem builder: an `evolve_problem` over exactly the failed
tasks.

The #429 guarantee is inherited unchanged: the validation set stays fixed and
the held-out split is computed from the seed tasks alone, so a problem mined
from a failure can only ever appear in the training frontier. Memory becomes
the exam — what the meta loop got wrong is what it is asked next.
