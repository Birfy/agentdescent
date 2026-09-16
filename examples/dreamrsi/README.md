# Dream-RSI: the run you already paid for, replayed as a simulator

> "Evaluating an exploration policy costs a whole discovery run. I have already
> run one. Can it score the policies I did not try?"

[`examples/metasearch/`](../metasearch/) evolves a search rule by running a
whole inner search per rollout, and its own page says what that costs: a
measured run asked for eight outer rounds and finished two in 2.7 hours. This
example is the other half of that problem. A finished discovery run recorded
every attempt and where each one started; an alternative exploration policy can
be *scored* by walking that record, because every outcome it would need is
already stored.

Full design record: [`docs/algo-dream-rsi.md`](../../docs/algo-dream-rsi.md).
Library: [`agentdescent/dream.py`](../../agentdescent/dream.py).

## The three stages

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

| stage | object | cost |
|---|---|---|
| **1. online explore** | `explore()` deploys the current policy against the real discovery agent for `K1` decision rounds, recording a `DiscoveryTree` | the whole cost of the method |
| **2. construct** | the tree joins a `SimulatorPool` — the history `H_t` | nothing |
| **3. dream** | every world becomes a `Problem`; `meta_evolve()` runs over them unchanged, scored by Equation 1 | dictionary lookups |
| **4. select & redeploy** | `π_{t+1} = argmax` over {current, evolved} on the whole pool, so `V(π_{t+1}) ≥ V(π_t)` | one replay per candidate |

## The two levels

| | inner (discovery) | outer (dreaming) |
|---|---|---|
| artifact | whatever the discovery agent writes | the **exploration policy** `select(ctx, n)` |
| task | one continuation | one **recorded discovery tree** |
| `run` | resume a workspace, generate, evaluate | replay the tree under the candidate policy |
| `reward` | the domain metric | Equation 1: quality − β₁·N + β₂·N/k |
| `propose` | the discovery agent's own proposal | a model reads the replay *trajectory* and rewrites the policy |
| governance | the domain's | L1 (`blast_radius=0.6`) — a policy changes how everything is searched |
| gate | the evaluator | held-out **worlds**, then the argmax on the whole pool |

## Why the synthetic domain saturates

`_world.py` gives each branch a hidden ceiling it approaches geometrically. That
is not decoration — it is the property the method needs to be interesting at
all. If every extra attempt on a branch bought more score, then "reveal
everything" would be the optimal replay, the recording policy would win by
construction, and dreaming could discover nothing. Real discovery does not look
like that: a direction is worked until it stops paying, which is what
Dream-RSI's own exploration prompt spends a section on ("flattening returns").

A branch's ceiling is also **not visible in its opening attempt**, which makes
Appendix B.2's rule true rather than merely asserted: *"shallow weak scores are
not enough to discard a branch: deeper attempts can recover."*

`SOURCE` is what the loop dreams on. `TARGET` fails twice as often, saturates
more slowly, and spreads its ceilings wider; the loop never sees it, and it is
the column that says whether an evolved policy is a better exploration policy or
a fit to one generator.

## Run

```bash
python -m examples.dreamrsi.dream_rsi_worlds --dry-run
python -m examples.dreamrsi.dream_rsi_worlds --offline --yes
python -m examples.dreamrsi.dream_rsi_worlds --provider openai \
    --model deepseek-v4-flash --thinking disabled --rounds 4 --yes
```

`--offline` scripts the policy-development agent, so the run exercises the whole
loop with no API key. It demonstrates the **mechanism** — that a better policy is
scored above the seed on worlds the seed itself recorded, is selected, and then
costs less online — and not a model's ability to find one.

Three flags are worth reading rather than accepting:

* `--repeats 8` — online rollouts per iteration. The paper deploys once, which
  leaves `H_1` a single tree; `evolve()` refuses fewer than four tasks, and
  dreaming on one world while the gate holds out that same world is the
  fit-to-the-record failure this whole design exists to see. Eight here because
  a rollout on a synthetic domain is free. On a real domain a rollout *is* the
  cost of the method, and 1–2 is what you can afford.
* `--parallel-weight 0.10`, below `--cost-weight 0.25` — the two terms of
  Equation 1 are together `N·(β₂/k − β₁)`, which vanishes at `k = β₂/β₁` for any
  `N`. Equal weights put that at the round budget itself, so the objective goes
  **blind to how many continuations a full-length rollout spent**. Measured on
  one world: the seed reaches 0.818 on 24 continuations, the evolved policy
  reaches the same 0.818 on 16, both in 6 rounds — and at `0.25/0.25` they score
  the identical `V = 0.8181`. A third of the discovery budget, worth nothing.
* `--validate-family target` — the deployment column is on a family the loop
  never dreamt in. A policy that wins on replay value and loses here was fitted
  to the record.

## What one offline run reports

Three iterations, `W=4`, `K1=6`, 8 rollouts each:

```
[round 1] worlds=8  online_calls=192  V 0.7584 -> 0.7790  redeployed
[round 2] worlds=16 online_calls=128  V 0.7822            kept
[round 3] worlds=24 online_calls=128  V 0.7868            kept
```

and on 200 fresh worlds the loop never saw:

| family | policy | best found | agent calls |
|---|---|---|---|
| `source` | seed | 0.7965 | 24.0 |
| | evolved | 0.7704 | **16.0** (1.50×) |
| `target` | seed | 0.6735 | 24.0 |
| | evolved | 0.6339 | **16.0** (1.50×) |

A third fewer agent calls for a small quality loss, and the loss is 1.5× larger
on the family it never dreamt in. That is the trade Equation 1 encodes at these
coefficients — not a free lunch, and not the paper's claim.

## Tests

`pytest tests/test_dream_rsi.py` — the legal set against the paper's definition,
the prefix-only rule, the batch constraints, Equation 1 on a hand-checked world,
the break-even identity behind the default coefficients, the recording policy
replaying its own world node for node, and the `V(π_{t+1}) ≥ V(π_t)` guarantee
end to end.
