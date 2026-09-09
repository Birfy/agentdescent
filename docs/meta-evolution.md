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

## What has been measured

| result | what it says |
|---|---|
| [Evolving the tree search itself](https://github.com/Birfy/agentdescent/blob/main/bench/results/metasearch-tree.md) | **positive, synthetic.** +0.0227 source / +0.0120 target over 3 seeds; lands on the Pareto front of the hand-tuned PUCT family, from a seed that is off it |
| [A search rule on real scientific data](https://github.com/Birfy/agentdescent/blob/main/bench/results/metasearch-selection-srbench.md) | **positive, real data.** LLM-SRBench `lsr_synth`: 5 wins / 0 losses / 4 ties over nine held-out paired comparisons, sign test p = 0.031 — and the rule moves in the *opposite* direction to the synthetic one, as predicted in advance |
| [Choosing a domain to evolve on](https://github.com/Birfy/agentdescent/blob/main/bench/results/metasearch-domain-selection.md) | **the method.** Four properties a domain must supply, each named by the null that taught it, and three checks that cost one inner run apiece |
| [A benchmark x slot matrix](https://github.com/Birfy/agentdescent/blob/main/bench/results/metasearch-slots.md) | **mixed.** Seven cells on `task_sampler` and `acceptance`; four commit, unseen gains +0.02 to +0.04, none negative once the inner budget is adequate |
| [AlgoTune](https://github.com/Birfy/agentdescent/blob/main/bench/results/metasearch-algotune.md) | **negative, structural.** The port works; the experiment cannot, because the measured timing *is* the feedback the search runs on |
| [SWE-bench-Science](https://github.com/Birfy/agentdescent/blob/main/bench/results/metasearch-swe-bench-science.md) | **domain qualified, experiment blocked.** Deterministic verifier, 3-31 reward levels, 5.9 s median — but nothing beats the root without the agent phase, on a weak model or a strong one |
