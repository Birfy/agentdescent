# Self-improvement loop — let the evidence decide what to improve next

*Modules:* [`agentdescent.self_improve`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/self_improve.py),
[`agentdescent.evidence`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/evidence.py),
[`agentdescent.admit`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/admit.py)
· *Issue:* #465

The framework's rungs each know one thing: `portfolio` knows which decision
slot's evolution is paying for itself, `harness` knows which port wins a domain,
`meta_run` evolves one slot and validates it, `exam` / `audit` / `chain_audit`
know whether a rule can be trusted. What none of them does alone is **decide the
next move**. `self_improve` is that loop.

```python
from agentdescent import SelfImproveConfig, self_improve

cfg = SelfImproveConfig(
    slots=["selection", "task_sampler"],
    specs={"selection": priority_selection(), "task_sampler": task_sampler_spec},
    domains={"gsm": (train_problems, val_problems)},
    ticks=5, top_k=2, config=CONFIG)
report = self_improve(store, cfg, propose=offline_propose)
for tick in report.ticks:
    print(tick.targets, tick.verdicts, tick.budget_after)
```

## One tick

1. **Read evidence.** `evidence.snapshot(store)` composes the portfolio's
   per-slot return-per-cost and exam stamps, the harness leaderboard, and the
   archive's health counters into one read-only view. `evidence.rank_slots`
   returns the slots whose evolution is paying (return-per-cost above the
   floor, enough samples, no demotions), plus an **exploration share**: up to
   `explore` of the `k` targets go to slots with fewer than `min_samples`
   rules (including never-archived ones), so an unseen slot is tried at least
   once. The fallback order is **rotated by tick**, so a cold or tied archive
   moves through the slots instead of always picking the same prefix.
2. **Evolve (compounding).** Each target slot is evolved on each domain with
   `meta_run` (train on `problems`, validate on disjoint `val_problems`), and
   each target **starts from the archive's best rule for that slot** (passed as
   `initial_state`, like `continuous_meta_run`), so tick 2 refines tick 1's
   adoption instead of re-evolving the fresh seed. An `adopt` is written with
   `archive_rule`, tagged with its domain in the problem descriptor.
3. **Trust-gate + quarantine.** `admit.admit_record` composes the adoption's
   rings (its own `exam` stamp plus any audit / tournament / red-team / chain
   status the caller supplies) into one **fail-closed** verdict. A `demote` or
   `flag` is **stamped on the record** with `store.mark_exam(...)`, so the
   archive's own demotion/exclusion paths see it, not just this report. The
   gate only *composes* statuses; it never re-runs a mechanism.
4. **Move budget.** `portfolio_report(layers_from_archive(store))` returns the
   next tick's per-slot budget, so the budget follows the measured return.

With the **default** gate a freshly archived record has `exam=None` and no
other ring, so it passes *provisionally*; those ticks list the key under
`SelfImproveTick.unexamined` rather than counting as reviewed. Supply rings (or
run `exam_archive`) to make the gate bite.

Ticks repeat a bounded `ticks` times; the report is the per-tick decision trail
(targets, verdicts, gains, the budget move). The evolution step is a plain
per-slot `meta_run`, so the loop is deterministic and dependency-free; a joint
or population variant can be substituted without changing the loop's shape.

## Isolation

`val_problems`/`val_seeds` must be isolated from the training set (as
`meta_run` requires): overlapping problem names, shared `Problem` objects, and
overlapping seed sets all raise. The loop never measures a verdict on the
problems that shaped the value.
