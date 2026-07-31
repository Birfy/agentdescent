# Dataset → evolved skill, in one call

> **`evolve_skill` is a thin wrapper over [`evolve`](evolution.md)** — same
> engine, same result object. Everything it does, you can do by hand; it just
> stops you writing the same page of boilerplate every time.

Evolving a skill needs three things that are genuinely yours:

1. **your data**,
2. **how to score an answer**,
3. **which model**.

Everything else — wrapping rows as `Task`s, the lambda that puts the skill in
front of the question, the same last-number regex, a dozen knobs — is the same
every time.

```python
from agentdescent import evolve_skill
from agentdescent.agents import openai_compatible
from agentdescent.dataloader import hf_rows

rows = hf_rows("hotpotqa/hotpot_qa", "validation", config="distractor", limit=40)

result = evolve_skill(rows, model=openai_compatible(model="deepseek-v4-flash"),
                      prompt="question", gold="answer", score="exact")

print(result.rendered)        # the skill it learned
print(result.final_reward)    # held-out reward
print(result.outcomes())      # why it went that way
```

That is **11 lines against 21** for the same program written by hand, and more
importantly it removes the decisions a first-time user has no basis to make.

## What that call actually does — measured

The block above, run as written on 40 real HotpotQA items with
`deepseek-v4-flash` (12 held out):

| | held-out exact match |
|---|---|
| starting instruction (`"You are a helpful assistant."`) | 2/12 = **0.167** |
| after evolution | 7/12 = **0.583** |

Four rounds, stopped by `patience`; 338 model calls, ~25 min wall-clock. The skill
it wrote:

> *"Respond with only the requested answer, omitting any extra explanation or
> restatement."*

Which is exactly the failure it was looking at — asked for a short span, the model
was answering with a paragraph:

```
gold='Arena of Khazan'  got='In *Tunnels and Trolls*, an adventure is called a **tunnel** - a playf...'
```

`result.outcomes()` was `{'committed': 1, 'below-threshold': 3}`: one proposal
cleared the gate and three were rejected for not beating it — the gate doing its
job, not a stuck run.

## The arguments

| | |
|---|---|
| `data` | rows (dicts) from anywhere, or ready-made `Task`s |
| `model` | any [completion](agents.md) — API model, CLI agent, your own function |
| `prompt=`, `gold=` | which columns hold the question and the expected answer |
| `score=` | a name from `SCORERS`, or your own `(task, output) -> float` |
| `instruction=` | the starting skill; what the run learns replaces it |
| `template=` | where the skill meets the question — `"{skill}\n\n{prompt}"` by default |
| `reflect_with=` | the model that proposes improvements (a cheap one is a fine trade) |
| anything else | forwarded to [`evolve`](evolution.md) and overrides the defaults |

## The scorers

`agentdescent.rewards` covers the common cases, and gets the details right that
are easy to get wrong:

| `score=` | matches when | notes |
|---|---|---|
| `"last_number"` | the **last** number in the output equals the gold number | the default for arithmetic — models show their working, so the answer is the last number |
| `"exact"` | output equals the gold | casefolds, collapses whitespace, strips trailing punctuation |
| `"contains"` | the gold appears anywhere in the output | forgiving, and the easiest to fool: gold `"2"` is inside `"12"` |
| `"numeric_close"` | last number within a relative tolerance | for rounded answers |

!!! warning "A dataset's answer column is often not the answer"
    GSM8K's `answer` is the whole worked solution ending in `#### 72`. Parsing
    that as a bare number fails — and it fails **silently**: every item scores 0,
    which reads as a hopeless model rather than a scorer mismatch. Measured on
    real GSM8K, this was the difference between a reported **0/7** and the true
    **7/7**.

    `last_number` therefore reads the gold the same way it reads the output, so
    `"72"`, `"#### 72"` and `"The answer is 72."` all work. A gold with no number
    in it at all now raises instead of scoring zero.

## What it chooses for you

All defaults, all overridable — pass any of them and yours wins:

* `n_workers = max_concurrency = min(8, train tasks)`
* `rounds = 8`, `patience = 3`, `target_reward = 0.98`
* `held_out_frac = 0.3`

Early stopping is on so a small dataset does not buy eight rounds of nothing.

## When to drop to `evolve()`

The moment you want something this does not express — a different artifact shape
([`strategy=`](evolution.md#2-the-evolution-rule-strategy)), a custom optimizer
([`aggregator_factory=`](aggregator.md)), a multi-step agent as `run=`. You can
also get there gradually: every one of those is just a keyword argument here,
because they pass straight through.

```python
result = evolve_skill(rows, model=model, prompt="question", gold="answer",
                      asynchronous=True, max_seconds=600,      # barrier-free
                      reflect_with=cheap_model)                # cheaper reflection
```

## Building the pieces yourself

The two layers underneath are public, and useful on their own:

```python
from agentdescent.evolution import tasks_from
from agentdescent.rewards import last_number

tasks = tasks_from(rows, prompt="question", gold="answer", difficulty="level")
reward = last_number()
```

`tasks_from` numbers the rows, puts the gold in `meta` (where
[the reflector reads it](evolution.md#bring-an-agent-you-already-have)), and maps
any extra columns you name into `meta` too.
