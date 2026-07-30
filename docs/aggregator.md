# The aggregator (the optimizer)

> **Plugs into [`evolve`](evolution.md) via** `agg_config=` (tune the reference
> aggregator) or `aggregator_factory=` (replace it entirely).

The aggregator is the framework's **optimizer** — the discrete-space analogue of
an optimizer step. It's the one place the training analogy breaks: *gradients
add, diffs do not*, so aggregation is not averaging but **conflict resolution +
statistical acceptance + transactional commit**. Every accepted change to the
shared ledger goes through it.

---

## What it does (per artifact bucket)

Evidence cards are bucketed by artifact; a bucket fires on batch size `B` or a
`T_max` timeout. Then, in order:

1. **Staleness filter** — per-diff `η` vs `α`; the [staleness policy](evolution.md#6-staleness-staleness_policy) decides `ACCEPT / REBASE / DISCARD`.
2. **Conflict resolution** — contradictory diffs (same key, different value) are projected out PCGrad-style; keep the better of the pair, iterating until no surviving pair contradicts. Key *overlap* alone is not a conflict — identical proposals are duplicates and dedupe.
3. **Fusion tournament** — complementary diffs are fused (model-soup style) and run against the singles on held-out; the best wins.
4. **Statistical acceptance** — commit only if `P(Δ > 0) > 1 − δ` under a Beta posterior (not a point threshold); `δ` anneals with version.
5. **Commit** — compare-and-swap on `dev`, one artifact per merge. The `Ledger` *also* offers `commit_atomic` (2PC across several artifacts, for a contract-breaking diff that must land with its adapters), but the reference aggregator buckets by artifact and never needs it — no engine path calls it today.
6. **Dual-branch promotion** — `dev → stable` every *K* **accepted commits** (EMA-style confirmation). Note it counts commits, not rounds: a round that merges nothing does not advance the counter, and there is no separate regression check — the guard is the acceptance test each commit already passed.
7. **Audit** — the merge is submitted to the `AuditScheduler`; high-blast-radius / L1 merges are forced through the oracle. *The optimizer audits itself.*

Deep dive on the *why*: [concepts §4](concepts.md#4-the-aggregator-a-discrete-space-optimizer).

---

## Tuning — `agg_config=` (`AggregatorConfig`)

Keep the reference pipeline, change its knobs:

```python
from agentdescent.aggregator import AggregatorConfig

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
| `promote_after_k` | dev→stable EMA confirmation rounds |

---

## Replacing — `aggregator_factory=` (`AggregatorProtocol`)

To change the *logic*, not just the knobs, plug in your own aggregator. The
contract is two methods:

```python
from typing import Protocol, List
from agentdescent.evolvable import EvidenceCard
from agentdescent.aggregator import MergeReport

class AggregatorProtocol(Protocol):
    def ingest(self, card: EvidenceCard) -> None: ...     # a worker's diff + evidence
    def step(self) -> List[MergeReport]: ...              # decide what to merge now
```

`evolve` builds the aggregator through a **factory** that receives the runtime
deps it owns — `(ledger, verifier, audit, config, staleness_policy)` — and
returns any `AggregatorProtocol`:

```python
from agentdescent.aggregator import Aggregator

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
