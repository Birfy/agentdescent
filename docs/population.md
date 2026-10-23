# Population engine

`PopulationAggregator` keeps an archive of committed candidates and asks a
`SelectionPolicy` which of them the next batch of workers should start from. A
declared selection policy (anything but `SingleHead`) installs it through
`Policies(selection=...)`. `population_islands` widens that into a
*persistent* population search: the archive plus a set of head-islands that
survive across steps, so a tree-shaped policy scores *clades*.

## Why islands

The archive already records `parent` for every candidate. But with the old
per-step behaviour, the forks (`head/<slot>`) were created each step from
`dev` and discarded at `finalize` — a candidate's lineage could not grow, so a
tree-shaped policy had at most one generation to score. A **persistent island**
keeps its fork across steps and records its lineage (`_branch_parent`), so
every new candidate admitted from island *i* is a child of that island's last
tip. That is the structure HGM's metaproductivity scores: a clade is a lineage
that accumulates.

## The dials

| Knob | What it does |
| --- | --- |
| `Policies(selection=CladeThompson())` | The population layer and its expansion rule. `CladeThompson` draws `theta_a ~ Beta(tau·(1+S_a), tau·(1+F_a))` over clade evidence and expands the argmax — HGM's answer to "which parent has the best *descendants*". |
| `population_islands=N` | `N` persistent head-islands on top of `dev`. `0` (the default) keeps the old per-step-fork behaviour exactly. A value above `0` without a declared selection policy is refused (the "honour or refuse" rule). |
| checkpointing | The archive, the islands (`forks`), the per-island lineage (`branch_parent`) and the selection counter all checkpoint/restore, so a resumed run keeps the population tree it had built. |

## Reading the lineages

A persistent-island run returns a read-only `EvolutionResult.lineages`: one row
per live lineage (`dev` plus each island) with its `candidates`, the archive's
`best_version` / `best_score` on that branch, the `selected` count and its
lineage `tip`. It is read from the archive the gate already scored, so it costs
no extra evaluation, and it outlives `finalize` (islands are reaped from the
ledger at the end, but the report still names them). `None` on a single-head
run. The official `state` / `rendered` remain the primary head; this is a
report, not a second way to commit.

```python
result = evolve(..., population_islands=4)
for line in result.lineages or []:
    print(line["branch"], line["best_score"], line["candidates"])
```

## Honest measurement

`bench/population_engine.py` A/Bs three arms (single-head / CladeThompson /
islands+CladeThompson) on a deterministic capability domain at equal budget.
On a **smooth monotone** landscape, single-head's greedy ascent is competitive
or better — population exploration spends budget it does not need there. The
population engine is for **rugged/exploratory** landscapes with many local
optima, where a lineage that keeps exploring past a dead end is what finds the
better basin. The bench is the instrument, not a claim; run it on your
landscape before choosing the arm.

## Verification

The mechanism is pinned at the aggregator level: islands are created and
persist across steps, a worker mutation on an island becomes a child of that
island's recorded tip, checkpoint/restore carries the islands and lineages,
and `island_count=0` keeps the old behaviour with no forks left behind.
