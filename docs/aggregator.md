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
2. **Conflict resolution** — contradictory diffs (same key, different value) are projected out PCGrad-style; keep the better of the pair.
3. **Fusion tournament** — complementary diffs are fused (model-soup style) and run against the singles on held-out; the best wins.
4. **Statistical acceptance** — commit only if `P(Δ > 0) > 1 − δ` under a Beta posterior (not a point threshold); `δ` anneals with version.
5. **Commit** — compare-and-swap on `dev` (2PC for contract-breaking multi-artifact diffs).
6. **Dual-branch promotion** — `dev → stable` after *K* regression-free rounds (EMA).
7. **Audit** — the merge is submitted to the `AuditScheduler`; high-blast-radius / L1 merges are forced through the oracle. *The optimizer audits itself.*

Deep dive on the *why*: [concepts §4](concepts.md#4-the-aggregator-a-discrete-space-optimizer).

---

## Tuning — `agg_config=` (`AggregatorConfig`)

Keep the reference pipeline, change its knobs:

```python
from concordia.aggregator import AggregatorConfig

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
from concordia.evolvable import EvidenceCard
from concordia.aggregator import MergeReport

class AggregatorProtocol(Protocol):
    def ingest(self, card: EvidenceCard) -> None: ...     # a worker's diff + evidence
    def step(self) -> List[MergeReport]: ...              # decide what to merge now
```

`evolve` builds the aggregator through a **factory** that receives the runtime
deps it owns — `(ledger, verifier, audit, config, staleness_policy)` — and
returns any `AggregatorProtocol`:

```python
from concordia.aggregator import Aggregator

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
