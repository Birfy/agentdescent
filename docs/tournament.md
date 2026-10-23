# Algorithm tournament — which method works for *this* skill?

A registry of algorithms is useful only if a user can answer "which one works
for my artifact?" without reading each paper. `agentdescent tournament` runs the
same spec once per requested algorithm and seed, under the same evolve budget,
and reports the measured result.

```bash
agentdescent tournament spec.json \
  --algorithm default \
  --algorithm rrsi \
  --algorithm archive_selection \
  --seed 0 --seed 1 --rounds 4
```

`default` is the spec's own recipe, including any `algorithm` the spec already
names. A named arm replaces `spec.algorithm` (and therefore its params) for that
arm only, via a deep copy; the file on disk is untouched. Every arm is an
ordinary `EvolveSpec -> compose() -> evolve()` run.

The winner is the best **algorithm aggregated over seeds** -- mean final reward,
ties to fewer mean calls -- not the luckiest single run. A tie across algorithms
is reported as a tie (`winner: null`, `tied: [...]`, exit code 1), not as a
measured win. Per-arm min/max are in the report so a wide spread is visible.

```python
from agentdescent import run_tournament
from agentdescent.evolvespec import load_spec

report = run_tournament(load_spec("spec.json"),
                        ["default", "rrsi"], seeds=(0, 1), rounds=4)
print(report.winner, report.reason)
for arm in report.entries:
    print(arm.algorithm, arm.seed, arm.final_reward, arm.outcomes)
```

## What it is, and is not

- It **is** an apples-to-apples comparison of the same spec under different
  methods: same data, same score, same agent/reflect model, same rounds or
  wall-clock per arm.
- It **is not** a paper-number reproduction. The `examples/` ports carry their
  own datasets and controls; this runs the method on *your* spec. The
  per-algorithm `fidelity` field says which is which.
- The winner is only as meaningful as the number of seeds and the size of the
  held-out set. One seed is a demo, not a result; `agentdescent plan` already
  warns when the gate is too small.

`--dry-run` composes every arm and prints the `plan` call estimate without
running anything; quote that total before starting an N x M tournament.

Temporary ledgers live under `--repo-root` (the system tempdir by default) and
are removed afterwards; `--keep` leaves them for diagnosis.
