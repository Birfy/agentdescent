# RSI metrics — acceptance, cost and category frequency

A run store answers what happened; a product needs the aggregate: acceptance
rate, cost per accepted change, which categories appear most often, and a stable
text export for Prometheus.

```python
from agentdescent import alerts, metrics_from_log, to_prometheus

metrics = metrics_from_log(records)    # any iterable of decision records
print(metrics.acceptance_rate, metrics.cost_per_accepted)
print(to_prometheus(metrics))
for line in alerts(metrics, min_acceptance=0.2, max_cost_per_accepted=50):
    print("ALERT:", line)
```

- `metrics_from_records` accepts dicts or objects with `accepted` (bool),
  `category` (str) and `cost` (a mapping/object with `calls`,
  `prompt_tokens`, `completion_tokens`, `wallclock`).
- Missing fields count as zero. A malformed present field — a non-bool
  `accepted`, a negative/non-integer call count, a non-finite wall clock, a
  cost that is neither a mapping nor an object with cost fields — raises a
  `ValueError` naming the record index. These numbers drive alerts, so they are
  not silently coerced.
- `acceptance_rate`, `calls`, token totals, wall clock, `cost_per_accepted`
  (calls per accepted decision, `None` when nothing was accepted) and a category
  frequency histogram (all decisions, not only refusals).
- `to_prometheus` is deterministic. Every family has one `HELP`/`TYPE` pair;
  the per-category counter is its own family
  (`agentdescent_rsi_decisions_by_category_total`) so Prometheus/OpenMetrics
  parsers do not see two types for one sample name. Counter names use the
  `_total` suffix and label values escape backslash, quote and newline.
- Counters are snapshots over the iterable passed in, so they are only
  monotonic when the underlying log is append-only.
- `alerts` is a small, pure threshold list (minimum acceptance rate, maximum
  cost per accepted change). When calls were spent but nothing was accepted,
  `max_cost_per_accepted` still fires rather than hiding behind `None`.

All metrics are derived from decision records; no model call and no second
source of truth.
