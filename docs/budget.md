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
is a tiny online logistic model over the **pre-spend features the worker
actually has at decision time** — the proposing rollout's group-relative
`advantage`, the diff's `size`, and the artifact's `blast_radius`. It is fit
online against commit outcomes ("did it commit"), so it needs no labelled data:
the engine knows the label for free.

**Decision and training see the same vector.** The post-outcome signals the
merge computes later — the before/after `delta` (the self-verify rollout's own
product), `P(improve)`, the distance from `stable` — are kept on the context
for the audit trail but **never enter the model's feature vector**. Training on
post-outcome features while deciding on pre-spend ones would fit a model that
explains the labels without teaching a usable pre-spend predictor; restricting
both sides to the pre-spend columns keeps the learning loop sound. A logit clip
and an L2 term keep one extreme candidate from blowing the weights.

**Every commit is attributed to a worker's actual pre-spend vector.** Which
card an outcome attaches to depends on how the candidate was chosen:

- **Fused** (the candidate is the union of several proposals): the commit label
  attaches to the **first contributor's** pre-spend vector — the card the
  tournament put forward, the same attribution the acceptance gate uses when it
  folds `before_after_delta` back into the posterior. A fusion's union size is
  a post-merge artefact no worker ever decided on.
- **Not fused** (the tournament chose a single proposal): the label attaches to
  the **winning proposal's** card, found by matching the selected diff — never
  an arbitrary first entrant, which may be a losing proposal. Labeling a losing
  card as committed would teach the model the wrong proposal succeeded and
  supply no observation for the actual winner.

Either way the vector is what a worker decided on (advantage, the card's own
diff size, the artifact's blast radius) — never a post-merge size.

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
- **The aggregator reports** — every measured decision is reported exactly
  once (`observe(ctx, committed)`): a commit, an oracle rejection, an
  acceptance rejection and a CAS conflict all produce an observation. Oracle
  and acceptance rejections are **negative labels** — without them the model
  would learn only from commits and never know which evaluated proposals fail.

`allocator=None` (the default) is the old behaviour byte for byte. The clock
thresholds remain the *ceiling* the allocator may never exceed: it can skip a
candidate the clock would have deep-evaluated, never spend on one the clock
refused.

## 4. The A/B — is it actually worth it?

The mechanism's own premise is *an analogy without an A/B is decoration*. The
equal-budget experiment is [`bench/ab_allocator.py`](https://github.com/Birfy/agentdescent/blob/main/bench/ab_allocator.py):

```bash
python -m bench.ab_allocator --dataset hotpotqa --budget-rollouts 96 \
    --seeds 0,1,2 --provider claude --model GLM-5.2 \
    --max-tokens 500_000 --yes
```

Two arms on identical workloads, budgets and seeds, one argument different:

| arm | what runs |
|---|---|
| `off` | `allocator=None` — the clock degrades optional spend |
| `on` | `allocator=ValueBudgetAllocator()` — learned per-candidate spend |

The table reports **test quality, measured tokens and whether the allocator
fired**. The fired check is the first thing to read: an allocator that agreed
with the clock everywhere (`fired == 0` with `observed > 0`) is a control
against a control, and its token column is about nothing. `--max-tokens` must
be set — below the budget the allocator stays inert, and the two arms are then
the same run.

If the `on` arm does not win here (equal-or-better quality at fewer tokens, or
better quality at the same tokens), delete it and record the negative result on
this page. That is the outcome the premise asks for explicitly.

## 5. What this does not do

- **It does not raise the budget.** `max_tokens` is still the cap; the allocator
  only decides how to spend what exists. Requires `max_tokens` to be set.
- **It does not replace the acceptance gate.** Every candidate that gets a
  self-verify rollout still goes through the same Beta-posterior acceptance and
  the audit gate. The allocator changes *how much finding-out is spent*, never
  *what is kept*.
- **It does not touch the oracle budget.** `oracle_budget` / `AuditScheduler`
  remain the mechanism for spending ground-truth checks; the allocator is about
  the *optional* self-verify spend.
