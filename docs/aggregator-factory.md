# The `aggregator_factory` exit — replacing the optimizer

*Contract:* `factory(ledger, verifier, audit, config, staleness_policy) -> aggregator`
with `ingest(card)` and `step() -> list[MergeReport]` (and optionally
`finalize()`), per `AggregatorProtocol`.

Every [policy field](policies.md) swaps **one decision** inside the shipped
seven-stage pipeline. Some optimizers need more: their own candidate pool,
their own admission rule, per-instance score rows, parent switches. For those,
`aggregator_factory=` replaces the optimizer wholesale — the sanctioned exit,
and the one the mechanism-heavy ports have always used.

## The single-head fact that makes this necessary

The engine's `selection` seam is honest about its limit: the ledger holds one
live `dev` branch, so a selection policy that names any starting point other
than the head is refused (`_check_selection` — "multi-head support is a
separate change"). Population search on a single-head ledger therefore lives
in the aggregator: **keep the pool in the optimizer, and make "selection" a
ledger commit that rewrites the head.**

## Implemented

| Aggregator | What it adds | Where |
|---|---|---|
| `Aggregator` | the shipped pipeline: dedupe, staleness, conflict, fusion, statistical acceptance, transactional commit, promotion | `agentdescent.aggregator` |
| `PopulationAggregator(Aggregator)` | an archive of every distinct committed head (with held-out score) + any standard `SelectionPolicy` picking the next parent, committed back to dev; `finalize()` lands the archive's best | `examples/_population.py` — how the candidate ports' `Policies(selection=…)` declarations actually run |
| `ParetoAggregator` | GEPA's pool with per-instance score rows and Algorithm-2 frontier sampling | `examples/gepa/` |
| `DGMArchiveAggregator` | DGM's keep-all archive with `sigmoid(perf) × 1/(1+children)` parent selection | `examples/dgm/` |
| `MetaSearchAggregator` | ADAS's keep-all archive over agent designs | `examples/adas/` |

## The one trap

The factory path bypasses the default-aggregator construction, so `Policies`
fields that the default path would wire (`conflict`, `fusion`, `acceptance`,
`promotion`) **do not reach a factory-built aggregator through the bundle**.
Pass them into your aggregator's constructor instead, and strip them from the
bundle — carried in both places, one copy is silently ignored, which is the
exact failure `require_supported` exists to prevent on the other path.
`examples/_population.py::population_factory` shows the pattern.

Prefer a policy field when one decision is enough; reach for the factory when
your mechanism needs state the pipeline does not keep.
