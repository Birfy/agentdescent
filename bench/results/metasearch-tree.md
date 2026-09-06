# Evolving the tree search itself

> "I plugged a tree search into `evolve()` to solve a problem. Now I want to
> evolve the search algorithm."

The artifact is the source of `priority(rank, visits, total, prior, depth,
n_nodes)` — the rule ERA's flat-PUCT tree uses to pick which node to expand
next. It plugs into the real `EraTree` as its `SelectionPolicy`, so an outer
rollout is *one whole inner tree search*: compile the candidate rule, run 60
expansions on a seeded landscape instance, return the best-so-far curve. The
meta-reward is that curve's AUC — a selection rule cannot make a better program
exist, only find one sooner.

Governance is L1 (`blast_radius=0.6`): the rule is a harness, so every merge
also passes the oracle. Produced by
[`examples/metasearch/evolve_search_policy.py`](../../examples/metasearch/evolve_search_policy.py).

**Validation is a different landscape.** The outer loop only ever sees `SOURCE`
(dim 4, step 0.45, 25% dead ends, no ruggedness). `TARGET` is dim 6, step 0.35,
45% dead, ruggedness 0.35 — it is never evolved on, never gated on, and is the
transfer column throughout.

## The ceiling: what is actually available here

Before asking whether a search finds a better rule, measure what "better" is
worth. Sweeping the seed rule's own exploration constant over the same 200
instances per family that the report validates on:

| `c` | 0.0 | 0.1 | 0.25 | 0.5 | **1.0 (seed)** | 2.0 |
|---|---:|---:|---:|---:|---:|---:|
| source | +0.0214 | +0.0214 | +0.0202 | +0.0157 | **0.0000** | −0.0552 |
| target | +0.0044 | +0.0044 | +0.0103 | +0.0147 | **0.0000** | −0.0434 |

Two things follow, and both shape how the run below should be read.

**The seed is on the wrong side of both optima.** Upstream ERA's `c = 1` is
beaten by every value below it on both families, so "explore less" is a real,
findable direction — which is what makes this a fair test rather than a search
for something that isn't there.

**The two families do not want the same rule.** Source is maximised as `c → 0`
(pure greedy; everything under 0.25 is indistinguishable). Target peaks near
`c = 0.5`, where greedy gives up two thirds of the gain. So the whole
transfer question is *how much* less to explore — and a search that optimises
`SOURCE` alone is being pulled toward the greedy end, where transfer is worst.
**The available gain is about +0.021 on source and +0.015 on target, and no
single rule gets both.**

`worst-first` (`return -rank`) is in the reference set to be bad: −0.37 source,
−0.18 target. If it ever fails to come last, the measurement is broken, not the
rules.

## The run

24 outer sweeps, 6 workers, 144 rollouts, inner budget 60 expansions,
`deepseek-v4-flash`, thinking disabled, completion cache on so the inner run is
a function of the rule. Scored paired against the seed rule on 200 fresh
instances per family that the outer loop never touched.

### The gate fix, seed by seed

The same three outer seeds, twice: once with `--tasks 24` (gate held-out = 9
instances) and once with `--tasks 150` (60 instances). Nothing else changed --
same rounds, workers, inner budget, model, and the same 200 fresh validation
instances per seed.

| outer seed | commits | source gain (w/l) | target gain (w/l) | | commits | source gain (w/l) | target gain (w/l) |
|---|---:|---:|---:|---|---:|---:|---:|
| | **gate = 9** | | | | **gate = 60** | | |
| 0 | 1 | +0.0213 (135/59) | +0.0049 (116/80) | | 2 | +0.0213 (140/55) | +0.0087 (118/78) |
| 1 | 3 | **-0.0134 (88/109)** | +0.0040 (95/99) | | 3 | **+0.0251 (146/49)** | +0.0131 (113/85) |
| 2 | 2 | +0.0222 (134/57) | +0.0151 (128/68) | | 1 | +0.0216 (131/59) | +0.0141 (124/73) |
| **mean** | | **+0.0100** | **+0.0080** | | | **+0.0227** | **+0.0120** |

**Every seed improved, and the one that was negative is now the best of the
three.** Seed 1 went from a rule that loses 88/109 on its own family to +0.0251,
144 rollouts either way. The mean source gain more than doubled.

### Against the ceiling

Mean over the three seeds, each scored on its own 200 fresh instances:

| rule | source | target |
|---|---:|---:|
| seed (flat-PUCT `c = 1`) | 0.0000 | 0.0000 |
| PUCT `c = 0.5` | +0.0175 | **+0.0138** |
| PUCT `c = 0.25` | +0.0214 | +0.0120 |
| greedy (`c -> 0`) | **+0.0227** | +0.0100 |
| **evolved (3 seeds, gate = 60)** | **+0.0227** | **+0.0120** |
| worst-first | −0.3729 | −0.1886 |

The evolved rule **lands on the Pareto front of the hand-tuned family**: greedy's
source score with `c = 0.25`'s transfer. Nothing in the reference set beats it on
both axes -- `c = 0.5` transfers better only by giving up a quarter of the source
gain. That is the honest claim: *the search did not exceed the hand-tuned family,
it found a point on it*, from a seed that sits off the front entirely, having
only ever seen `SOURCE`.

It is worth being clear about how much that is. The whole axis is 0.02 of AUC,
and the search covers it in 144 rollouts (~3 minutes, 130k-210k tokens). What it
did not do is discover anything outside the exploration/exploitation trade-off
the reference family already parameterises.

### What the rules look like

All three evolved rules keep the shape `rank + <shrinking exploration term>` and
every one of them shrinks it, three different ways:

```python
# seed 0: 1/n_nodes -> 1/(1+n_nodes), and a 1/(1+0.5*depth) decay
return rank + 1.0 * (1.0 / (1.0 + n_nodes)) * math.sqrt(total) / (1 + visits) * (1.0 / (1.0 + 0.5 * depth))

# seed 1: sqrt(log(total)/visits) instead of sqrt(total)/visits -- UCB1, not PUCT
explore = 1.5 * prior * math.sqrt(math.log(total + 1) / (1 + visits)) / (1.0 + depth * 0.1)

# seed 2: sqrt(total/(1+n_nodes)) over log1p(visits), plus a first-visit depth bonus
explore = prior * math.sqrt(total / (1.0 + n_nodes)) / (1.0 + math.log1p(visits))
```

Seed 1's is the interesting one: it replaced PUCT's `sqrt(total)/(1+visits)` with
UCB1's `sqrt(log(total)/(1+visits))`, which is a different exploration schedule
rather than a smaller constant -- and it is the seed that scored highest. None of
these is a rule I would have written, and all three land where the tuned constant
does.

## What the gate could see, and why it matters

The first arm ran with `--tasks 24`, so the outer gate held out **9 instances**
(24 × `held_out_frac=0.4`). The paired per-instance sd on this landscape is
0.040–0.062, which puts the standard error of a 9-instance mean at
**0.013–0.018** — wider than the entire gain available (+0.021).

That is not a subtle degradation. Seed 1's gate watched held-out reward rise
0.744 → 0.748 across three commits and accepted a rule that **loses 88/109 on
the family it was evolved on** (−0.013). Its rule reads like the reason:
`c = 1.5`, an unvisited-node bonus — it moved *up* the exploration axis, the
opposite of the known direction, and nine instances could not tell.

Held-out instances here are free. The model is called once per *rollout*, and
rollouts are `rounds × workers` whatever `--tasks` is; 60 held-out searches at
budget 60 cost 0.28 s, one gate. The default is now `--tasks 150` — 60 held-out
instances, SE 0.005–0.008.

This is the third time the same mistake has shown up in this line of work, at
three different levels:

| level | too small | symptom | fix |
|---|---|---|---|
| inner budget | 4 rollouts per inner run | a *deliberately bad* sampler tied for best | 12 rollouts (`bench/metasearch_slots.py`) |
| outer gate (slots) | 2 held-out windows | good proposals tied the seed on exactly those two; nothing committed | 4 windows |
| outer gate (tree) | 9 held-out instances | a rule that loses on its own family was committed | 150 tasks → 60 instances |

The slot experiment's version of this failed *closed* — a narrow gate reads as
"nothing works". This one failed **open**: a narrow gate reads as "this works"
for a rule that doesn't. Under L1 the oracle is a second opinion on the same
numbers, so it does not catch it.

## The other thing that was wrong: the run never finished

Two earlier attempts died at the wall-clock limit with an **empty completion
cache** — not one model call had returned. The example built its reflector
through `completion_for` with no `thinking` option, so the endpoint spent
minutes generating a reasoning preamble for a one-function rewrite. With
`--thinking disabled` an outer sweep takes ~5 s instead of >10 minutes.

It took as long as it did to find because the example printed nothing between
its plan line and its summary. `bench/metasearch_slots.py` grew an `on_round`
hook for exactly this reason and this example had not; it has one now. **A run
that reports nothing until its summary cannot be told from a stalled one.**

## The result files

[`metasearch-tree-seed0.json`](metasearch-tree-seed0.json),
[`seed1`](metasearch-tree-seed1.json), [`seed2`](metasearch-tree-seed2.json) --
the `--tasks 150` arm, one per outer seed. Each carries the evolved rule, the
paired validation, the reference column, the outcome histogram, and the model
configuration (`model.thinking` included, since it changes the reply).

## Run it

```bash
python -m examples.metasearch.evolve_search_policy --dry-run
python -m examples.metasearch.evolve_search_policy \
    --provider openai --model deepseek-v4-flash --thinking disabled \
    --rounds 24 --workers 6 --tasks 150 --validate-seeds 200 \
    --completion-cache .cache/metasearch-tree --yes
```

`--thinking disabled` changes the reply and not only the latency, so it is
printed in the plan line and recorded in the result JSON next to the model id.
It has to be identical across arms of any comparison.
