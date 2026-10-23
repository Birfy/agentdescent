# Named algorithms

*Generated from `agentdescent.algorithms`; do not edit by hand.*

These are the methods a spec (or `agentdescent init`) can ask for by
name. An algorithm contributes **defaults** — policies, `evolve()`
knobs, an `agg_config` — that the same spec can override field by field.
The default recipe for a `kind` is used when no algorithm is named, and
`plan` always echoes what was resolved.

```bash
agentdescent algorithms list [--kind K] [--json]
agentdescent algorithms show NAME
agentdescent init PATH --algorithm NAME --param KEY=VALUE
```

## `archive_selection`

Parent sampling from an archive of every accepted candidate, tempered by novelty (DGM's rule by default) instead of always extending the head.

- **Kinds:** `text`, `skill_dir`, `agent_dir`, `agent_code`, `plugin`
- **Fidelity:** `preset`
- **Cost:** none
- **Source:** Darwin Gödel Machine (choose_selfimproves), ADAS archive, SICA best-agent
- **Spec:** `"algorithm": "archive_selection"`

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `sampling` | `str` | `sigmoid_novelty` | sigmoid_novelty: Darwin Gödel Machine's rule, sigmoid(10*(score-0.5)) / (1+times chosen); novelty: softmax over score divided by (1+times chosen); performance: softmax over score; best: always the top scorer (SICA); uniform: the ablation |
| `temperature` | `float` | `1.0` | softmax temperature for performance / novelty; must be > 0 |
| `seed` | `int` | `0` | seed of the parent draw |

## `gepa`

GEPA's genetic reflective loop: keep a Pareto pool of candidates scored per instance and mutate a win-frequency-selected parent instead of the single best average.

- **Kinds:** `text`
- **Fidelity:** `mechanism_microport`
- **Cost:** every admitted candidate is scored over the full held-out set once (the Pareto row); no extra reflection calls
- **Source:** GEPA: Reflective Prompt Evolution (Agrawal et al., 2025); arXiv:2507.19457
- **Details:** [algo-gepa.md](algo-gepa.md)
- **Spec:** `"algorithm": "gepa"`

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `seed` | `int` | `0` | seed for the Pareto parent draw |
| `eval_concurrency` | `int` | `8` | per-instance scorings run in parallel when a candidate enters the pool |

## `rrsi`

Regularized recursive self-improvement: an annealed edit budget, a leakage screen before measurement, and a historical-best acceptance floor.

- **Kinds:** `text`
- **Fidelity:** `mechanism_microport`
- **Cost:** none (the reflection model proposes as it would anyway; the screen is deterministic and runs before any held-out sweep)
- **Source:** Google RRSI (regularized harness selection); docs/algo-rrsi.md
- **Details:** [algo-rrsi.md](algo-rrsi.md)
- **Spec:** `"algorithm": "rrsi"`

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `beta0` | `float` | `0.0` | large-gain cost rule: tolerance |
| `beta1` | `float` | `0.0` | large-gain cost rule: slope |
| `w_s` | `float` | `1.0` | within-band score weight |
| `w_c` | `float` | `1.0` | within-band cost weight |
| `w_n` | `float` | `0.0` | within-band novelty weight |
| `noise_band` | `float` | `0.0` | historical-best floor tolerance (score >= S* - noise_band); the paper estimates it by re-evaluating the unchanged base |
| `b_min` | `int` | `1` | smallest edit budget (independent edits per candidate) |
| `b_max` | `int` | `5` | largest edit budget, at the start of the anneal |
| `T` | `int` | `20` | rounds the edit budget anneals over |
| `stall_window` | `int` | `5` | rounds without movement of S* that count as a stall |
| `prune_window` | `int` | `5` | rounds of component yield a component must show gain over to stay |
| `max_history_chars` | `int` | `1500` | how much per-edit history fits in the reflection prompt |
| `protected` | `list[str]` | `[]` | strings that must not leak into the artifact: task ids, gold answers, names from the evolve set (never the final holdout). A candidate containing one is rejected before it is measured |
| `base_score` | `float` | `0.0` | the base artifact's evolve-set score, to seed the historical-best floor S* (0 keeps the floor at the first measurement) |
