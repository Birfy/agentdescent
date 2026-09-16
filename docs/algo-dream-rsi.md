---
description: Dream-RSI (recursive self-improvement through evolving worlds) on AgentDescent - a finished discovery run replayed as a simulator, so an exploration policy is scored without running one.
---

# Dream-RSI — Replaying a Finished Run

> **The run you already paid for can score the policies you never tried.** A
> completed discovery run recorded every attempt it made and where each one
> started; an alternative exploration policy walks that same tree, taking a
> different subset of branches in a different order with different parallel
> groupings, and every outcome it needs is already on disk. Runs through
> [`meta_evolve()`](meta-evolution.md) with **no engine change** — the artifact
> is the `selection` slot, and only the *problem* is new. Module:
> [`agentdescent.dream`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/dream.py).

| | |
|---|---|
| **Paper** | *Dream-RSI: Recursive Self-Improvement through Evolving Worlds* — Tong Zheng, Xidong Wu, Zheng Zhang et al., Google / Google DeepMind / UMD / UVA, 2026 |
| **Upstream code** | **none released.** The authors' repository carries the paper and the project page; the code, discovered programs and reproduction scripts are listed as in preparation |
| **Module** | [`agentdescent/dream.py`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/dream.py) |
| **Example** | [`examples/dreamrsi/dream_rsi_worlds.py`](https://github.com/Birfy/agentdescent/blob/main/examples/dreamrsi/dream_rsi_worlds.py) |
| **Domain** | a synthetic discovery family with saturating branches (`_world.py`), `SOURCE` to dream on and `TARGET` to deploy on |
| **Layer** | L1 (`blast_radius=0.6`) — an exploration policy changes how everything downstream is searched, so every merge also passes the oracle |
| **Fidelity** | `mechanism_microport` — [what the classes mean](port-fidelity.md) |

!!! note "Why this is not in the ports table"
    [Self-evolution algorithms](self-evolution-examples.md) and
    [port fidelity](port-fidelity.md) account for twenty published algorithms,
    each with a domain and a measured before-and-after. Dream-RSI is not one of
    them and is deliberately not counted there: it is a **meta-level** method,
    like [`meta_evolve()`](meta-evolution.md) whose machinery it reuses
    wholesale, it ships as a library module rather than only an example, and it
    reproduces no benchmark. It lives with meta-evolution, which is where its
    code lives.

!!! warning "This port follows a paper, not a repository"
    Every other port on this site follows the authors' released code where it
    disagrees with their paper, because [that is the rule](port-fidelity.md).
    Dream-RSI has no released code, so there is nothing to follow: this is a
    port of the equations in §3 and the policy contract in Appendix B.2, and
    where those two disagree the appendix wins and the page says where. None of
    the paper's results is reproduced, and no number below corresponds to one.

## The problem it solves, in this repository's own terms

[`meta_evolve()`](meta-evolution.md) already evolves a decision rule of
`evolve()`, and its documentation states the price without flinching: *"One
outer rollout is therefore an entire inner search."* The measured run on that
page asked for eight rounds and got two, stopping at a 5 400-second wall clock —
six rollouts, 2.7 hours.

That is the bottleneck Dream-RSI attacks, and its observation is that the price
has already been paid. A discovery run records, for every attempt, where it
started and what it scored. Arrange those into a tree and an alternative
exploration policy can be *run* over it: choosing a branch reveals a node whose
outcome is already stored. No agent call, no evaluator, no sandbox.

| | `meta_evolve` over a live search | Dream-RSI, dreaming |
|---|---|---|
| one outer rollout is | a whole inner search | a walk over a recorded tree |
| its cost | the inner search's model + evaluator budget | dictionary lookups |
| its transition | stochastic — the agent may answer differently | deterministic — the outcome is on disk |
| what it can discover | anything the inner search could do | a better walk over **what was already explored** |

The last row is the trade, and it is the paper's own framing: the simulator is
*"a grounded model of the portion of the discovery space that has already been
observed."* Dreaming finds a cheaper, better-ordered, better-batched walk. It
cannot find a direction nobody took.

## The mechanism

### The discovery tree

A rollout builds a rooted tree. The root is the initial workspace. Every other
node is one `CONTINUE(v)`: the discovery agent resumed `v`'s workspace, produced
a candidate, and the evaluator scored it. Continuing the root opens a new
branch; continuing a leaf advances that branch, and the new child becomes the
leaf. So **only the root has several children** — `DiscoveryTree.paper_shaped`
is that invariant, and `unreachable` counts what a tree violating it loses.

The legal set is `A(T) = {root} ∪ {leaves}`, and batches are
`A(T; W) = {C ⊆ A(T) : |C| ≤ W}`. One decision round selects a batch and
observes its completed outcomes. An empty batch terminates.

### Replay

Replay starts from the root alone and reveals `Child(v; T)` for each `v` in the
batch. The policy sees the revealed subtree and the legal set; it never sees an
unrevealed node's score, and that is enforced by construction rather than asked
for — `_candidates()` builds the context from the revealed set, and the count of
recorded children still waiting behind a node is deliberately *not* exposed even
though the simulator obviously knows it.

Replay ends on an empty batch, when a round reveals nothing, or at `K2` rounds.
Every policy version starts from the root again: a version does not inherit the
subtree an earlier one revealed.

### Equation 1

```
V = max score revealed  −  β₁·N  +  β₂·N / max(1, k)
```

`N` is the revealed non-root nodes — the generation–evaluation requests the
trajectory represents — and `k` the completed decision rounds, so the third term
is the mean batch size.

**The paper publishes no coefficients**, so `ReplayObjective()` defaults both to
zero, which makes `V` the best score attained and nothing else.
`ReplayObjective.scaled(max_nodes=, n_workers=)` is the constructor to use, and
the one thing it gets right is that the two weights must **not** be equal:

```
−β₁·N + β₂·N/k  =  N · (β₂/k − β₁)
```

vanishes at `k = β₂/β₁` for **any** `N`. Equal weights put that break-even at
`max_nodes / n_workers` — exactly the round budget — so every policy that runs
its rounds out is scored on discovery quality alone however many continuations
it spent, and the entire cost story disappears.

Measured on one world of the example domain (`max_nodes=24`, `n_workers=4`):

| policy | quality | N | k | `V` at 0.25/0.25 | `V` at 0.25/0.10 |
|---|---|---|---|---|---|
| seed (parallel refining) | 0.818 | 24 | 6 | **0.8181** | 0.6681 |
| the example's evolved policy | 0.818 | **16** | 6 | **0.8181** | **0.7181** |

Same quality, a third of the discovery budget saved — and at equal weights that
saving is worth exactly nothing. The default is `0.25 / 0.10`, which puts the
break-even at `0.4 · max_nodes / n_workers` and leaves parallelism as a
tie-break between policies of equal cost. `tests/test_dream_rsi.py` pins both
halves of that.

### The loop

```python
from agentdescent import ReplayObjective, dream_rsi, exploration_policy

spec = exploration_policy()                    # the policy, as gated source
result = dream_rsi(
    continue_fn,                               # (tree, parent, attempt) -> Attempt
    spec=spec, model=model,
    objective=ReplayObjective.scaled(max_nodes=24, n_workers=4),
    rounds=4, n_workers=4, online_rounds=6)
policy = spec.compile(result.rendered)         # -> Policies(selection=policy)
```

Per outer iteration `t`:

1. **Online explore** — `explore()` deploys the current policy against the real
   discovery agent for at most `K1` decision rounds and records a
   `DiscoveryTree`.
2. **Construct** — the tree joins the `SimulatorPool`, which is `H_t`.
3. **Dream** — every world in the pool becomes a
   [`Problem`](meta-evolution.md), and `meta_evolve()` runs over them exactly as
   it runs over live inner searches: the same held-out split, the same conflict
   and fusion policies, the same L1 oracle.
4. **Select** — `π_{t+1}` is whichever of {current, evolved} has the higher mean
   replay value on the **whole** fixed history.

Step 4 is not the same check as step 3's gate, and it is there because the two
can disagree. `meta_evolve` commits on `held_out_frac` of the pool; the
selection step re-scores the committed value on all of it. Because the candidate
set contains the current policy, `V(π_{t+1}) ≥ V(π_t)` by construction — the
paper's one formal guarantee, and cheap enough here to be worth enforcing rather
than inferring.

It earns its place, and the case is measured rather than hypothetical. On eight
worlds of the example domain, a pruning policy that ranks branches by their
**latest** score is *committed* by the gate on its three held-out worlds — and
scores **0.7337** on the whole pool against the seed's **0.7584**. The gate
alone would have redeployed a regression; the argmax keeps it out.
`tests/test_dream_rsi.py` pins that case.

## Why this needed no new engine

The paper's exploration policy maps a discovery tree to a batch of nodes to
continue. That is [`SelectionPolicy`](selection.md) — `select(ctx, n)` over
`ctx.candidates`, with `n` the worker count — so:

| Dream-RSI | what it already is here |
|---|---|
| exploration policy | a `SelectionPolicy`, i.e. the `selection` slot of [`Policies`](policies.md) |
| policy code, revised by an LLM | a [`SourceSlot`](meta-evolution.md) behind `compile_policy_source`'s AST gate |
| `M` revisions per offline phase | `meta_evolve(rounds=…, n_workers=…)` |
| replay worlds | the outer task set, one task per world |
| Equation 1 | a `MetaReward` |
| policy-development agent | a `propose`, here `replay_reflector` |
| "redeploy the improved policy online" | `Policies(selection=spec.compile(rendered))` |

The whole port is therefore one new *problem type* plus the simulator that makes
it cheap. `docs/api.md` lists it under its own module because it is a method,
not because the engine grew a seam for it.

### What the policy is handed

`ctx.candidates` is the revealed subtree — every revealed node, not only the
legal ones, because Appendix B.2 gives the policy both `observed()` and
`legal_actions()` and a policy that only saw its legal moves could not do what
that prompt spends two sections demanding: reconstruct a branch's trajectory and
tell a repairable failure from a dead direction.

| `Candidate` field | what it carries |
|---|---|
| `version` / `parent` | the node index and where it was continued from (`None` = the root) |
| `score` | the recorded score, or `None` when the attempt produced no number — a failure, which is not a low score |
| `state["legal"]` | `"1"` when this node may be continued this round |
| `state[…]` | `valid`, and the domain's own strings — `fail_class`, `error` |
| `per_task` | `score`, `delta_vs_parent`, `delta_vs_baseline`, `depth`, `branch` |
| `selected` | how many of this node's children are already revealed |

An empty return stops the rollout. That is why the slot needs a smoke test of
its own: the shipped `selection` smoke requires `select` to return at least one
candidate, and for an exploration policy **stopping is a decision**. Everything
else about the gate — the import allowlist, the `isinstance` check, the
restricted namespace — is [`agentdescent.meta`](meta-evolution.md)'s, unchanged.

## Paper says / appendix says / this port follows

* **Batch multiplicity.** §3 defines a batch as `C ⊆ A(T)`, so the root enters a
  batch at most once and opening `W` parallel branches takes `W` decision
  rounds. Appendix B.2 says a batch *"may contain several roots"*, because
  there the grid has one cell per unopened branch — and the paper's own seed
  policy is described as one that *"launches multiple independent exploration
  workspaces in parallel"*, which the §3 reading cannot do in round one.
  **This port follows the appendix**: the root carries a multiplicity of `W`,
  one cell per branch; every other node carries one. The two readings agree on
  every batch that does not touch the root.
* **Averaging across worlds.** §3 averages raw `V` over the history. Raw `V` is
  not comparable across worlds — one whose best recorded node scores 0.9 and one
  whose best scores 0.2 put the same policy at different heights for reasons
  that have nothing to do with the policy — and `evolve()`'s held-out gate
  averages rewards *across tasks*. **This port normalises each world's `V` by
  that world's own attainable range** (root score at maximum price, to best
  recorded node for free at full parallelism), and keeps the raw value in the
  outcome's `detail`. It is the one place an equation is changed rather than
  read.
* **A barren legal move.** `A(T)` does not consult the record, so a leaf whose
  branch ended is legal and reveals nothing. **This port keeps it legal** and
  charges the policy a worker for it, counted as `barren` rather than `illegal`
  — the honest simulation of a worker that would have been spent online. A round
  that reveals nothing ends the replay, which is §3's *"no recorded continuation
  remains available"*.
* **Per-episode policy state.** Appendix B.2's policies keep state and start each
  episode with `question.reset()`. Ours are ordinary objects whose class the gate
  guarantees takes no constructor arguments, so **a new instance is the reset**,
  taken for every replay and every online rollout. Without it a policy would
  carry "I already opened four branches" into the next world, and
  [`meta_validate`](meta-evolution.md)'s paired comparison would stop being
  paired.
* **`β` as a swept knob.** Appendix B.2 has the policy read one scalar `beta` in
  `__init__`, fixed within an episode and swept on a grid offline. **Not
  ported.** The sweep is a second optimiser on top of the one here, and the
  objective it sweeps against (`pareto.reward = pareto.auc − λ·parallel_penalty`)
  is a different functional from Equation 1. Equation 1 is what this implements.
* **Rollouts per iteration.** The paper deploys the policy once per iteration, so
  `H_1` is a single tree. `evolve()` refuses fewer than four tasks, and more to
  the point, dreaming on one world while the gate holds out that same world is
  precisely the fit-to-the-training-landscape failure
  [`meta_validate`](meta-evolution.md) exists to catch. **This port deploys
  `online_repeats` times per iteration** (default 4) and skips dreaming, with
  `stop_reason="pool-too-small"` recorded, until the pool holds `min_worlds`.
  `online_repeats=1` follows the paper and starts dreaming at iteration 4.

## What the example demonstrates, and what it does not

[`examples/dreamrsi/`](https://github.com/Birfy/agentdescent/tree/main/examples/dreamrsi)
runs the whole loop with no model and no sandbox. Its discovery domain is
synthetic, and the one property it has to get right is the one the method turns
on: **branches saturate.** Each branch has a hidden ceiling and approaches it
geometrically, so past a point further attempts on it buy nothing. Without that,
"reveal everything" is the optimal replay, the recording policy wins by
construction, and dreaming can discover nothing. A branch's ceiling is not
visible in its opening attempt, which is Appendix B.2's *"shallow weak scores are
not enough to discard a branch"* made true rather than asserted.

```bash
python -m examples.dreamrsi.dream_rsi_worlds --dry-run
python -m examples.dreamrsi.dream_rsi_worlds --offline --yes
python -m examples.dreamrsi.dream_rsi_worlds --provider openai \
    --model deepseek-v4-flash --thinking disabled --rounds 4 --yes
```

`--offline` scripts the policy-development agent instead of calling a model, so
what it shows is the **mechanism**, not a model's ability to find a policy. On
one three-iteration offline run (`W=4`, `K1=6`, 8 rollouts per iteration):

| round | worlds in pool | online agent calls | mean replay value | |
|---|---|---|---|---|
| 1 | 8 | 192 | 0.7584 → **0.7790** | redeployed |
| 2 | 16 | **128** | 0.7822 | kept |
| 3 | 24 | **128** | 0.7868 | kept |

and the redeployed policy, deployed on 200 **fresh** worlds it never dreamt in:

| family | policy | best found | agent calls | |
|---|---|---|---|---|
| `source` (dreamt on) | seed | 0.7965 | 24.0 | |
| | evolved | 0.7704 | **16.0** | **1.50×** fewer calls, −0.026 quality |
| `target` (never seen) | seed | 0.6735 | 24.0 | |
| | evolved | 0.6339 | **16.0** | **1.50×** fewer calls, −0.040 quality |

Read the quality column, not only the calls. At these coefficients the policy
that dreaming selects **buys a third fewer agent calls with a small quality
loss**, and the loss is 1.5× larger on the family it was never dreamt on than on
the one it was — which is the transfer gap, visible because the report deploys on
both. That is the honest shape of the result, and it is not the paper's claim
(which is competitive or better quality at lower cost, on real benchmarks).

Two things the scripted proposal itself had to get right, both of which cost a
run to find and both of which Appendix B.2 states in prose:

* **Prune, do not stop.** A policy that simply stops after three waves scores
  *below* the seed on the worlds the seed recorded — 0.755 against 0.758 over
  eight of them. It wins five of the eight and loses the mean: a branch's ceiling
  is not yet visible at depth three, and the worlds where it guesses wrong cost
  more than the continuations it saved were worth.
* **Rank a branch by its best score, not its latest.** Ranking on the frontier
  node's own score drops any branch whose last attempt happened to fail — 15% of
  them here. Anchoring on the branch's best takes the same policy from 0.734 to
  0.779 on those eight worlds, and from 4 wins to 7. The losing variant is the
  one quoted above under step 4: the dreaming gate commits it, and only the
  pool-wide argmax refuses it.

## Boundaries

* **No result of the paper is reproduced.** The Lasso regularisation path,
  circle packing, the autocorrelation inequalities and KernelBench are all
  out of scope here; the discovery domain is synthetic and costs microseconds.
  What is ported is the method.
* **A world only holds what was explored.** Dreaming reorders, subsets and stops;
  it cannot try a direction the recording policy never took. This is why the pool
  keeps every world instead of only the newest, and it is the reason a policy's
  advantage on worlds *it* recorded is not evidence about its advantage online —
  hence the example's separate deployment column.
* **The discovery agent is the caller's.** `Continuation` is
  `(tree, parent, attempt) -> Attempt`; wiring it to a real coding agent, a
  container and an evaluator is the same work every other port on this site does
  in its own example, and is not done here.
* **`β` sweeping is not ported**, as above.
* **The online phase does not run through `evolve()`.** `explore()` is its own
  short driver, because `evolve()`'s ledger holds one live `dev` head and
  [refuses a policy that names several starting points](selection.md) — a
  discovery tree is exactly such a policy. The decision interface is shared
  between the two phases (`_eligible` is called by both), which is the paper's
  own structural claim: *"Both phases use the same tree-based decision interface.
  Their difference lies in how a continuation produces its next observation."*

## Tests

`pytest tests/test_dream_rsi.py` — 44 offline tests, no network. The legal set
against §3's definition, the prefix-only rule, the batch constraints (cap,
multiplicity, illegal and barren picks), Equation 1 on a hand-checked world, the
break-even identity that decides the default coefficients, determinism and the
fact that the seed argument is inert, policy state not leaking between replays,
the gate, the recording policy replaying its own world node for node, and the
`V(π_{t+1}) ≥ V(π_t)` guarantee end to end.
