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
`plan`, `evolve`, `status`, `watch`, `show` and `apply` work on it unchanged —
it is still an `evolve()` call. `agentdescent init selection --kind policy_slot`
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
  is a `module:attribute` ref to a mapping of them (or a zero-argument builder),
  inside the spec's import allowlist.

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
| [Choosing a domain to evolve on](https://github.com/Birfy/agentdescent/blob/main/bench/results/metasearch-domain-selection.md) | **the method.** Four properties a domain must supply, each named by the null that taught it, and three checks that cost one inner run apiece |
| [A benchmark x slot matrix](https://github.com/Birfy/agentdescent/blob/main/bench/results/metasearch-slots.md) | **mixed.** Seven cells on `task_sampler` and `acceptance`; four commit, unseen gains +0.02 to +0.04, none negative once the inner budget is adequate |
| [AlgoTune](https://github.com/Birfy/agentdescent/blob/main/bench/results/metasearch-algotune.md) | **negative, structural.** The port works; the experiment cannot, because the measured timing *is* the feedback the search runs on |
| [SWE-bench-Science](https://github.com/Birfy/agentdescent/blob/main/bench/results/metasearch-swe-bench-science.md) | **domain qualified, experiment blocked.** Deterministic verifier, 3-31 reward levels, 5.9 s median — but nothing beats the root without the agent phase, on a weak model or a strong one |
