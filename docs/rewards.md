# Rewards — the scorers everyone writes

*Module:* [`agentdescent.rewards`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/rewards.py)
· *API:* [`exact_match`, `contains`, `last_number`, `numeric_close`](api.md#ready-made-scorers)

A reward is `(task, output) -> float` in `[0, 1]`, and writing one is easy —
which is why almost everyone writes the same three and gets the same details
wrong: thousands separators, a trailing period, a model that answers in a
sentence, a gold column that is a whole worked solution rather than a number.

```python
from agentdescent.rewards import last_number

evolve(tasks, last_number(), agent=agent)
```

They are a **convenience, not a contract** — bring your own for anything else.
Each is a factory: call it to get the scorer.

| scorer | 1.0 when | use for |
|---|---|---|
| `exact_match()` | output equals gold | labels, classes, short answers |
| `contains()` | gold appears anywhere in output | models that answer in a sentence |
| `last_number()` | the **last** number in the output matches gold | arithmetic word problems |
| `numeric_close(tolerance=0.01)` | `last_number` within a relative tolerance | rounded or derived figures |

All of them read the expected answer from `task.meta["gold"]` — the same place
the [reflector](evolution.md) looks. `gold_key=` points elsewhere.

## `normalise` is on for a reason

`exact_match` and `contains` casefold, collapse whitespace and strip surrounding
punctuation by default. Without it, a model that ends its answer with a period
scores zero — which looks like a reasoning failure and is not one, and the run
then spends every round asking the reflector to fix an answer that was already
right.

## `last_number` reads the *gold* the same way

This is the detail that silently ruins runs. A dataset's answer column is often
the whole worked solution ending in the figure — GSM8K's ends `#### 72`. Parsing
that as a bare number fails, every item scores 0, and it reads as a hopeless
model rather than a scorer mismatch.

Taking the last number from **both** sides handles `"72"`, `"#### 72"` and
`"The answer is 72."` alike. Thousands separators and a leading `$` / `£` / `€`
are handled too.

If the gold contains no number at all, `last_number` raises rather than scoring 0
forever — a scorer that can never match is a configuration error, not a result.

!!! warning "`contains()` is the easiest to fool"
    A gold of `"2"` is inside `"12"`. For numeric answers prefer `last_number()`.

## The contract the engine enforces

```python
reward(task, output) -> float in [0, 1]
```

Return something outside that range and the engine raises
`RewardContractError` immediately rather than letting the run continue. The
reason is specific: the engine treats `>= solved_threshold` (0.999) as solved, so
a scorer on the wrong scale — accuracy out of 100, say — makes **every task look
already solved**. Nothing is ever learned, no proposal is ever requested, and the
reported `final_reward` looks large and healthy. Normalise before returning.

## Graded scorers need `solved_threshold`

A ROUGE score or an LLM judge rarely reaches 0.999. With the default threshold
every rollout is treated as a failure, the reflector is asked to "fix" an answer
that scored 0.95, and the run reports `below-threshold` as if the reflector were
the problem:

```python
evolve(tasks, my_rouge_scorer, agent=agent,
       solved_threshold=0.8,                              # the engine
       task_sampler=DifficultyWeighted(pass_threshold=0.8))   # and the sampler
```

Both, or the two disagree about what a pass is. See
[sampling](sampling.md#pass_threshold-mirrors-the-engine).

## Writing your own

```python
def rubric_score(task, output):
    hits = sum(1 for k in task.meta["must_mention"] if k in output.lower())
    return hits / len(task.meta["must_mention"])       # already in [0, 1]

evolve(tasks, rubric_score, agent=agent, solved_threshold=0.9)
```

Anything callable with `(task, output)` works — including one that calls a model.
Remember that the reward runs on the hot path: every held-out measurement and
every candidate the [aggregator](aggregator.md) ranks goes through it, so an
LLM-judge reward multiplies the cost of the whole run. Bound it with
`cheap_eval_tasks=` and see the [cost model](directory-evolution.md#cost-the-first-order-design-constraint).
