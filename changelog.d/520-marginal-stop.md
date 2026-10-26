Added
`MarginalGainStop`, a stateful `stop_when` policy: it watches the recent
reward/call slope and stops after `patience` measurements whose per-call gain
is below `min_gain_per_call`. The per-call bar decides on its own under the
defaults, so a run that improves too slowly to pay for its calls now stops;
`min_gain` is an optional absolute-gain floor combined by OR. The `reason`
names the measured gain and call delta, a round with no new calls never counts
as low, and `reset()` makes the object reusable across runs. This is an
economic stop for long self-improvement runs, not a safety verdict.
