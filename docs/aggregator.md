# The aggregator (the optimizer)

> **Plugs into [`evolve`](evolution.md) via** `agg_config=` (tune the reference
> aggregator) or `aggregator_factory=` (replace it entirely).

The aggregator is the framework's **optimizer** — the discrete-space analogue of
an optimizer step. It's the one place the training analogy breaks: *gradients
add, diffs do not*, so aggregation is not averaging but **conflict resolution +
statistical acceptance + transactional commit**. Every accepted change to the
shared ledger goes through it.

---

*Module:* [`agentdescent.aggregator`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/aggregator.py)
· *API:* [`Aggregator`, `AggregatorConfig`, `MergeOutcome`, …](api.md#the-aggregator-the-optimizer)
· *Neighbours:* [verifier](verifier.md) scores candidates · [ledger](ledger.md) commits them ·
[staleness](staleness.md) decides what to do with an out-of-date diff ·
[governance](governance.md) decides how hard to gate.

## Replacing a decision

The seven decisions a merge makes are objects, and each can be swapped without
touching `Aggregator`:

```python
from agentdescent import Policies, evolve
from agentdescent.policies import AcceptDecision

class AcceptEverything:
    def accept(self, ctx):
        return AcceptDecision(True, "committed")

evolve(tasks, reward, agent=agent, policies=Policies(acceptance=AcceptEverything()))
```

| decision | default | what it is given |
|---|---|---|
| task sampling | `RoundRobin` | the shard's task ids and the round index |
| proposal generation | your `propose=` callable | a `ProposalContext` |
| conflict | `DefaultConflict` | the artifact and the surviving cards |
| fusion | `DefaultFusion` | the artifact and the kept diffs |
| staleness | `GuardedStaleness` | `eta`, `alpha`, whether the diff breaks a contract |
| acceptance | `DefaultAcceptance` | a `MergeContext` |
| promotion | `DefaultPromotion` | this round's `MergeReport`s |

Two things the defaults know that a replacement should be told:

* **Acceptance reads the full held-out set, never the cheap layer.**
  `MergeContext` carries both (`base_counts` vs `base_cheap`) because the
  regression guard once read the cheap one, which `cheap_eval_tasks`
  sub-samples -- so a four-task sample could veto a commit the full-set test had
  just approved.
* **Promotion counts rounds *survived*, not commits.** Counting commits inverts
  the incentive: an artifact that converges stops committing and so can never be
  promoted, while one that thrashes promotes every K commits.

Not replaceable, deliberately: the audit gate. It asks whether the cheap layer is
still trustworthy -- a question about the measuring instrument, which belongs to
the infrastructure rather than the algorithm.


## What it does (per artifact bucket)

Evidence cards are bucketed by artifact; a bucket fires on batch size `B` or a
`T_max` timeout. Then, in order:

1. **Staleness filter** — per-diff `η` vs `α`; the [staleness policy](evolution.md#6-staleness-staleness_policy) decides `ACCEPT / REBASE / DISCARD`.
2. **Conflict resolution** — contradictory diffs (same key, different value) are projected out PCGrad-style; keep the better of the pair, iterating until no surviving pair contradicts. Key *overlap* alone is not a conflict — identical proposals are duplicates and dedupe.
3. **Fusion tournament** — complementary diffs are fused (model-soup style) and run against the singles on held-out; the best wins.
4. **Audit gate** — the candidate is submitted to the `AuditScheduler`; a high-blast-radius / low-trust merge is forced through the oracle, which can **veto it outright** (`oracle-rejected`) before the acceptance test runs. *The optimizer audits itself.* This is a blocking gate on the accept path, not a post-commit spot-check.
5. **Statistical acceptance** — commit only if `P(Δ > 0) > 1 − δ` under a Beta posterior comparison (not a point threshold); `δ` anneals with version.
6. **Commit** — compare-and-swap on `dev`, one artifact per merge. The `Ledger` *also* offers `commit_atomic` (2PC across several artifacts, for a contract-breaking diff that must land with its adapters), but the reference aggregator buckets by artifact and never needs it — no engine path calls it today.
7. **Dual-branch promotion** — `dev → stable` after *K* **regression-free rounds** on dev. One round is one `step()`. A commit restarts the clock (the new version has survived nothing yet) and so does an oracle rejection. So a *converged* artifact — one that stopped committing because nothing beats it — is the one most likely to be promoted, which is the point. A run that ends cleanly also publishes its head via `finalize()`, so stopping on `target_reward` does not leave `stable` a confirmation short.

Deep dive on the *why*: [concepts §4](concepts.md#4-the-aggregator-a-discrete-space-optimizer).

### Reading step 3: did fusion actually help?

The tournament is the whole answer to "two local improvements might be worse
together than either alone", so it is worth knowing whether it ever fires and
whether it wins. `RoundStat.fused` counted **committed** fusions, which cannot
tell you: the tournament only commits a fusion that won, so that count is a tally
of successes with the denominator missing.

The shipped `FusionPolicy` records a `FusionTrial` per tournament — it was already
computing the scores to rank the candidates — and `result.fusion_stats()` reads
them back:

```python
stats = result.fusion_stats()
print(stats.summary())
# fusion: won 12/31 (39%), mean gain -0.004, 9 losses (worst -0.070,
#         1 below baseline), 10 ties
```

| field | what it answers |
|---|---|
| `trials` / `contested` | how many tournaments ran, and how many had a fusion in them at all |
| `single_candidate` / `contradiction` | why the rest did not — one survivor, or survivors that contradicted |
| `win_rate` | fused wins over `contested`; `None` when nothing was contested, so "never ran" cannot be read as "always lost" |
| `mean_gain` | mean `fused − best single` |
| `negative` / `mean_loss` / `worst_loss` | the losing tail — the number the objection is actually about |
| `below_baseline` | fusions worse than the artifact they started from, as opposed to merely ranked below the best single |
| `ties` | fusion exactly matching the best single; high here with an empty tail means the cheap layer cannot separate the candidates |

Ties do not count as fusion wins: `max` keeps the first of equal scores, so a
fusion that merely matches the best single loses. That is the conservative reading
and it matters — counting ties as wins would inflate the rate on exactly the
workloads where the held-out set is too small to tell the candidates apart.

A replaced `FusionPolicy` is not obliged to keep `trials`; then `stats.trials` is
`0` and `win_rate` is `None`, which reads as "not instrumented" rather than as a
verdict.

---

## Tuning — `agg_config=` (`AggregatorConfig`)

Keep the reference pipeline, change its knobs:

```python
from agentdescent import AggregatorConfig

evolve(tasks, reward, agent=agent, agg_config=AggregatorConfig(
    batch_trigger=2,      # fire a merge once this many proposals collect for an artifact
    max_wait_rounds=1,    # ...or after this many rounds (so cold artifacts don't starve)
    base_delta=0.5,       # acceptance risk: commit iff P(Δ>0) > 1-δ, annealed by version
    alpha_head=5,         # staleness tolerance for hot artifacts
    alpha_tail=1,         # ...and for cold ones
    trust_region_ops=6,   # max edits per diff
    promote_after_k=3,    # dev -> stable after K regression-free rounds (EMA)
))
```

| Field | Controls |
|---|---|
| `batch_trigger` / `max_wait_rounds` | when a bucket fires (size vs timeout) |
| `base_delta` | acceptance strictness (`1 − δ` threshold), annealed by version |
| `alpha_head` / `alpha_tail` | staleness tolerance `α` (hot vs cold artifacts) |
| `trust_region_ops` | diff-size cap (the trust region) |
| `promote_after_k` | dev→stable after this many regression-free rounds (EMA) |
| `anneal_half_life` | how fast the acceptance threshold tightens with version |
| `accept_samples` | Monte-Carlo draws behind each acceptance decision |

!!! note "`anneal_half_life` sets the shape of a long run"
    `base_delta` was exposed; the half-life that turns it into the actual
    threshold was a default argument inside `stats.annealed_delta`, reachable from
    nothing a caller touches. It decides how quickly committing gets harder:

    | artifact version | `P(Δ>0)` must exceed |
    |---|---|
    | 1 | 0.505 |
    | 64 | 0.750 |
    | 128 | 0.875 |
    | 256 | 0.969 |
    | 400+ | 0.990 (floor) |

    Version counts *commits*, so a run with many small accepted diffs reaches the
    floor much sooner than one with a few large ones.

!!! tip "Making the cheap layer actually cheap — `evolve(cheap_eval_tasks=)`"
    The aggregator scores candidates twice for two different reasons, and only one
    of them needs to be exact:

    | | what it decides | cost |
    |---|---|---|
    | **cheap layer** | which candidate to *put forward* — conflict resolution, the fusion tournament | once **per candidate** |
    | **acceptance test** (`eval_counts`) | whether to *commit* it | once per merge |

    `evolve()` used to pin the cheap layer to the whole held-out set, so rule /
    learned / oracle were one full sweep wearing three names — and on an LLM
    workload `eval_fn` **runs the agent**, so a round paid a full sweep for every
    candidate it merely wanted to rank. `oracle_budget` capped nothing either: its
    documented fallback (`rule_eval`) returned the very value it was trying to
    avoid buying.

    `cheap_eval_tasks=N` scores N held-out tasks for ranking. The acceptance test
    still uses the full set — as does the regression guard beside it — so this
    trades ranking precision, never commit safety.
    The sample is **fixed for the run** — it used to be redrawn on every call,
    which is harmless only while the "sample" is the whole set, and silently scores
    candidate A on `{1,3,5}` against candidate B on `{2,4,6}` the moment it is not.
    Default is `None` (exact), so nothing changes unless you opt in.

---

## Replacing — `aggregator_factory=` (`AggregatorProtocol`)

To change the *logic*, not just the knobs, plug in your own aggregator. The
contract is two methods:

```python
from typing import Protocol, List
from agentdescent import EvidenceCard
from agentdescent import MergeReport

class AggregatorProtocol(Protocol):
    def ingest(self, card: EvidenceCard) -> None: ...     # a worker's diff + evidence
    def step(self) -> List[MergeReport]: ...              # decide what to merge now
```

!!! note "Your aggregator is checked, and its mistakes are not hidden"
    The factory's result must have callable `ingest` and `step` — missing either
    raises before the first rollout, naming what is absent. `step()` must return a
    list of `MergeReport`; returning `None` or a list of something else raises
    `AggregatorContractError` naming your class, instead of surfacing as
    `'NoneType' object is not iterable` from inside the driver.

    That error, `RewardContractError` and `ProposalContractError` all derive from
    **`ContractError`**, and both engines let it propagate. A *backend* failure (a
    rate limit, a dead endpoint) is absorbed and reported through `result.error` so
    a long run keeps its partial results — a broken contract in your own code is
    not, because the run is meaningless either way and hiding it wastes the budget.

`evolve` builds the aggregator through a **factory** that receives the runtime
deps it owns — `(ledger, verifier, audit, config, staleness_policy)` — and
returns any `AggregatorProtocol`:

```python
from agentdescent import Aggregator

class StrictAggregator(Aggregator):
    def _tournament(self, artifact, diffs):
        # e.g. never fuse -- evaluate only single diffs
        return super()._tournament(artifact, [diffs[0]] if diffs else diffs)

def factory(ledger, verifier, audit, config, staleness_policy):
    return StrictAggregator(ledger, verifier, audit, config,
                            staleness_policy=staleness_policy)

evolve(tasks, reward, agent=agent, aggregator_factory=factory)
```

### Override points on the reference `Aggregator`

The easiest customization is subclassing and overriding one decision; each stage
above is a method:

| Method | Stage you're changing |
|---|---|
| `_staleness_filter(artifact, head, cards)` | which stale diffs survive / rebase |
| `_resolve_conflicts(artifact, cards)` | how contradictions are dropped |
| `_tournament(artifact, diffs)` | fusion + candidate selection |
| `_process(artifact_id)` | the acceptance test / commit block |
| `ingest` / `step` | buffering + when merges fire |

### From scratch

You don't have to subclass — anything with `ingest` + `step` works. A trivial
"accept-everything, no merge" aggregator (for a baseline):

```python
class NaiveAggregator:
    def __init__(self, ledger, verifier, audit, config, staleness_policy):
        self.ledger, self._pending = ledger, []
    def ingest(self, card): self._pending.append(card)
    def step(self):
        # ... apply each pending diff to the ledger head, no conflict/acceptance ...
        self._pending.clear()
        return []

evolve(tasks, reward, agent=agent, aggregator_factory=NaiveAggregator)
```

Use this to A/B your own merge/acceptance policy against the reference optimizer
while keeping the rest of the loop (agents, strategy, parallelism, governance)
unchanged.

## The async optimizer variant — SGD-style descent

On the [barrier-free async path](evolution.md#the-barrier-free-runtime-async_evolve)
the expensive step is usually the **held-out eval** (an agent rollout per
validation item). Validating *every* candidate — the reference greedy hill-climb
and most frontier optimizers — makes held-out the wall-clock bottleneck when
workers propose faster than one full eval completes.

An aggregator can **amortise** it, exactly like mini-batch SGD amortises the
validation pass over many gradient steps:

1. **Apply** each incoming diff as a cheap *update step* (`ingest` accumulates,
   `step` commits the moved head so workers immediately build on it) — **no eval**.
2. **Validate every `N` steps.** Score the accumulated head on held-out once per
   *N* applied updates, not once per update.
3. **Keep or roll back.** If the mini-batch improved held-out, checkpoint it;
   otherwise **roll back** the head to the last validated checkpoint.

This costs ~`N`× fewer held-out evals. It is a *different* acceptance rule from
the per-candidate frontier — a deliberate async acceleration — so a faithful port
keeps the strict per-candidate optimizer on the **sync** path and switches to the
SGD variant only when `asynchronous=True`. [EvoSkill](algo-evoskill.md)'s
`SgdSkillAggregator` is the worked example (`val_every=N`, checkpoint + rollback);
its sync path keeps the strict `TopKFrontierAggregator`.

```python
class SgdMerger:                       # apply-then-periodically-validate, roll back on no gain
    def __init__(self, ledger, verifier, ctx, artifact_id):
        self.ledger, self.verifier, self.ctx, self.aid = ledger, verifier, ctx, artifact_id
        self.cards, self.checkpoint, self.ckpt_score, self.steps = [], {}, 0.0, 0
    def ingest(self, card): self.cards.append(card)
    def step(self):
        head = self.ledger.snapshot(Ledger.DEV).get(self.aid)
        cards, self.cards = self.cards, []
        for c in cards:                                    # 1. apply updates, no eval
            head = head.apply(c.diff); self.steps += 1
        self._commit(head.state)                           #    move the head; workers build on it
        if self.steps >= self.ctx.val_every:               # 2. validate every N steps
            score = self._eval(head)
            if score > self.ckpt_score:                    # 3. keep ...
                self.checkpoint, self.ckpt_score = dict(head.state), score
            else:                                          #    ... or roll back to checkpoint
                self._commit(self.checkpoint)
            self.steps = 0
        return [...]
```

Because `apply()` only *merges* ops, a rollback that must **drop** skills added
since the checkpoint commits a full replacement artifact (exact state), not a
diff. The pending-intake [lag budget](evolution.md#the-barrier-free-runtime-async_evolve)
keeps the mini-batch bounded so one `step()` never faces an unbounded pile.

## Example optimizers (from the algorithm ports)

The [self-evolution examples](self-evolution-examples.md) are, at heart, custom
`aggregator_factory=` optimizers — each swaps the reference greedy hill-climb for
a paper's own selection/acceptance rule. They are all `AggregatorProtocol`
implementations you can read and reuse:

| Aggregator | Example | Selection / acceptance rule |
|---|---|---|
| `ParetoAggregator` | [GEPA](algo-gepa.md) | per-instance **Pareto frontier** sampling (Algorithm 2); commits the sampled Pareto parent as the dev head |
| `TopKFrontierAggregator` | [EvoSkill](algo-evoskill.md) | bounded **top-K aggregate frontier** (sync path); commits the best member as the head |
| `SgdSkillAggregator` | [EvoSkill](algo-evoskill.md) | **async SGD-style descent**: apply skill updates, validate every `val_every` steps, roll back on no held-out gain |
| `StrictGateAggregator` | [SkillOpt](algo-skillopt.md) | **strict held-out-EM gate** + rejected-edit buffer + integer LR budget |
| `MetaSearchAggregator` | [ADAS](algo-adas.md) | **keep-all archive** with bootstrap-CI fitness (L1 harness) |
| `DGMArchiveAggregator` | [DGM](algo-dgm.md) | **keep-all archive** + staged eval + `sigmoid(perf)×1/(1+children)` parent selection (L1) |

They share one trick: the archive/frontier/gate sets the dev head to the
*selected* parent each `step()`, so `evolve()`'s next round mutates it — that is
how non-greedy selection rides the greedy loop.
