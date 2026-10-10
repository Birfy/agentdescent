# Budget allocation

> **The token budget is a ceiling; this page is about how to spend it.** A
> `max_tokens` cap is a brake — when spend reaches it, the run stops. Everything
> on this page answers a different question: given that you have a budget, *which
> candidate is worth spending the expensive part of it on*?

## 1. The problem with a clock

`BudgetGovernor` (see [`evolution`](evolution.md) / [`async`](async.md)) degrades
the optional spend of a run — fusion tournaments, the self-verify rollout — as
the budget runs out, against two fixed floors:

| floor | what it shuts off | default |
|---|---|---|
| `soft_floor` | the fusion tournament (a held-out sweep used only to *rank* survivors) | 75% spent |
| `hard_floor` | the self-verify rollout (the before/after delta) | 90% spent |

That is fair, predictable, and exactly what a budget-aware allocator should
*not* do: it treats every candidate alike. When `spent / max_tokens` crosses the
floor, the *next* candidate stops getting the expensive part — whatever it is.
A candidate that arrived late in the run but would have been the best change of
the whole run is skipped because of the clock, not because of its value.

The EvoAlloc line of work measures the cost of this fixed allocation: when
evaluation is the scarce resource, spending it by time-since-start rather than
by expected value wastes it on candidates that cannot move the search while
starving ones that can. `evolve(allocator=...)` replaces the clock with a
**per-candidate** decision learned from the run's own merge history.

## 2. `ValueBudgetAllocator` — learn which candidates are worth it

```python
from agentdescent import ValueBudgetAllocator

alloc = ValueBudgetAllocator()
evolve(tasks, reward, agent=agent,
       max_tokens=500_000,           # the allocator decides how this is spent
       allocator=alloc)
```

Two halves, one object:

**Value-directed allocation.** Once the budget is tight (`remaining_fraction`
below the allocator's `budget_start`, default 0.5), the self-verify rollout is
spent only on candidates the value model expects to commit. `LearningValueModel`
is a tiny online logistic model over the features the merge path already
measures — the proposing rollout's local `before_after_delta`, its
group-relative `advantage`, the diff's size, the merge's P(improve), and the
candidate's distance from `stable`. It is fit online against commit outcomes
("did it commit"), so it needs no labelled data: the engine knows the label for
free. A logit clip and an L2 term keep one extreme candidate from blowing the
weights.

**Counterfactual exploration.** A candidate the model judged unworthy is
skipped — except with probability `epsilon`, when it is evaluated anyway. The
outcome is then *ground truth the model was wrong about*, and it learns from it.
Without this term the allocator only ever sees the candidates it already
believed in, and a biased value model is never corrected: a workload where
"low before/after delta" once meant "won't commit" would be skipped forever even
after it stopped meaning that. `epsilon` anneals from `epsilon_start` (0.3) down
to `epsilon_end` (0.05) over the first `epsilon_anneal` (40) observations, so a
run explores hardest when the model knows nothing.

## 3. How it is wired

The decision and the feedback are two separate points on the merge path:

- **The worker asks** — before spending the self-verify rollout, the budget
  governor calls the allocator's `decide(ctx)` with an
  [`AllocatorContext`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/allocator.py)
  built from the diff's size and the rollout's advantage. High-value candidates
  keep the spend; low-value ones lose it; counterfactual ones get it anyway.
- **The aggregator reports** — after each merge, the outcome
  (`observe(ctx, committed)`) is fed back, so the value model learns which
  features predict a commit.

`allocator=None` (the default) is the old behaviour byte for byte. The clock
thresholds remain the *ceiling* the allocator may never exceed: it can skip a
candidate the clock would have deep-evaluated, never spend on one the clock
refused.

## 4. What this does not do

- **It does not raise the budget.** `max_tokens` is still the cap; the allocator
  only decides how to spend what exists. Requires `max_tokens` to be set.
- **It does not replace the acceptance gate.** Every candidate that gets a
  self-verify rollout still goes through the same Beta-posterior acceptance and
  the audit gate. The allocator changes *how much finding-out is spent*, never
  *what is kept*.
- **It does not touch the oracle budget.** `oracle_budget` / `AuditScheduler`
  remain the mechanism for spending ground-truth checks; the allocator is about
  the *optional* self-verify spend.
