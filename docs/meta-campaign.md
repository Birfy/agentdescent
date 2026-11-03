# Meta-campaign — continuous meta-learning as a bounded loop

`agentdescent meta-campaign` runs repeated `continuous_meta_run` sessions over
one consolidation archive, each in a fresh ledger repo, under the existing
`RSIController` stop policy:

```text
session: evolve a slot -> validate on unseen problems -> archive an adopted rule
stop:    target / cost / patience / max_sessions
memory:  the archive is the only state shared between sessions
```

It does not reimplement meta evolution. It supplies the durable loop, the
per-session repo isolation and the archive continuity.

## Quickstart

```bash
agentdescent meta-campaign init --out meta.json --demo
agentdescent meta-campaign plan --spec meta.json
agentdescent meta-campaign run  --spec meta.json --json --out report.json
```

The demo uses `agentdescent.meta_demo`, a deterministic offline domain, so it
runs with no model or dataset. `run` exits 0 only when the campaign completed
without an errored session; a rejected rule is a normal measured session.

## Spec

```json
{
  "archive": "~/.agentdescent/archive",
  "repo_dir": "meta-repos",
  "slot": "task_sampler",
  "spec": "mypkg.meta:slot_spec",
  "tasks_for": "mypkg.meta:tasks_for",
  "run": "mypkg.meta:run",
  "reward": "mypkg.meta:reward",
  "inner_propose": "mypkg.meta:inner_propose",
  "inner_strategy": "mypkg.meta:inner_strategy",
  "propose": "mypkg.meta:propose",
  "problems": "mypkg.meta:problems",
  "val_problems": "mypkg.meta:val_problems",
  "campaign": {"max_sessions": 2, "patience": 5},
  "inner": {"rounds": 5, "n_workers": 1},
  "evolve": {"rounds": 1, "n_workers": 1},
  "seeds": [0, 1, 2, 3],
  "val_seeds": [10, 11]
}
```

- `spec`, `inner_strategy`, `problems`, `val_problems` are zero-argument
  builders (or ready objects).
- `tasks_for`, `run`, `reward`, `inner_propose`, `propose` are callables.
- `meta_reward` and `cost_fn` are optional callable refs.
- `campaign` is `MetaCampaignConfig`; `inner` is the inner `evolve()` settings;
  `evolve` is the outer meta `evolve()` settings.
- `seeds` / `val_seeds` must be disjoint; the validation set is fixed and never
  seen by the search.

`plan` resolves refs and validates configs without calling builders or creating
the archive. On MCP, refs are gated by `AGENTDESCENT_SURFACE_ALLOW` (empty
refuses) and archive/repo/report writes by `AGENTDESCENT_SURFACE_WRITE_DIRS`.
When `repo_dir` is omitted, `meta_campaign_run` creates a system temporary
directory for the per-session ledgers; that fallback is outside the write
allowlist by design, so pass `repo_dir` when you need the ledgers confined.
