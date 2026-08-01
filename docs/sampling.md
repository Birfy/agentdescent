# Task sampling — which rollout to spend

*Module:* [`agentdescent.sampling`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/sampling.py)
· *API:* [`TaskSampler`, `RoundRobin`, `DifficultyWeighted`](api.md#task-sampling)

[Parallelism](parallelism.md) decides which *shard* a worker owns. The sampler
decides which task inside that shard it rolls out **next** — and since a rollout
is the expensive unit of the whole system, that choice is where most of the
budget is either spent or wasted.

```python
from agentdescent import DifficultyWeighted, RoundRobin

evolve(tasks, reward, agent=agent, task_sampler=RoundRobin())            # default
evolve(tasks, reward, agent=agent, task_sampler=DifficultyWeighted())
```

## `RoundRobin` — the deterministic default

Cycles through the shard in order. Reproducible, dependency-free, and it spends
rollouts uniformly — including on tasks the agent already solves, which produce
no proposal and teach nothing.

That is fine while most tasks still fail. It stops being fine the moment the
artifact is good: most rollouts then land on solved tasks and the run keeps
paying for them.

## `DifficultyWeighted` — UCB over learning signal

Weight each task by how much signal it still carries:

```
weight = 4 * p * (1 - p)        # p = observed pass rate
```

The weight peaks at `p = 0.5` and vanishes at both extremes. A task that always
passes yields no gradient; neither does one that never passes whatever the
artifact says. This is the GRPO zero-advantage argument, applied to task
selection.

An untried task keeps the optimistic prior `p = 0.5` — exactly where the weight
is maximal — so exploration happens without needing a large exploration constant.

### `c` defaults low, and the number is measured

Share of rollouts that landed on an informative task (higher is better), on a
40-task workload where only 6 tasks carry signal:

| `c` | 1.4 | 0.7 | 0.4 | **0.2** | 0.1 | round-robin |
|---|---|---|---|---|---|---|
| clean | 15.5% | 19.3% | 21.0% | **23.4%** | 27.4% | 14.5% |
| 15% noise | — | — | 11.5% | **16.3%** | 18.1% | 7.3% |

The textbook `1.4` is *worse than round-robin's neighbourhood* here, because most
exploration already comes from the optimistic prior: a large `c` swamps the signal
term (capped at 1.0) and the sampler never stops re-trying tasks it has already
shown to be uninformative.

`0.2` is the default — 1.6–2.2× better than round-robin in both regimes, while
keeping twice the exploration of the empirical optimum, which matters when your
reward is noisier than the workloads measured here. Lower it to focus harder,
raise it to explore more.

!!! warning "That is a targeting measurement, not an accuracy claim"
    Landing more rollouts on failing tasks does **not** automatically produce a
    better artifact. On real
    [ACE / FiNER-139 runs](algo-ace.md#empirical-results-finer-139-with-deepseek)
    the difficulty-weighted sampler reached a lesson sooner (round 0 versus round
    2) but did not score better — and two runs of the *same* round-robin
    configuration differed by 4.8 points, so at that sample size neither sampler
    is distinguishable from the other.

    Concentrating on the hardest tasks can also yield lessons that fit those tasks
    and generalise worse. Treat this sampler as **worth trying and worth measuring
    on your own data**, not as a free win.

### `pass_threshold` mirrors the engine

A rollout counts as a pass when its reward reaches this, defaulting to the
engine's [`SOLVED`](evolution.md) (0.999) rather than repeating the literal.

!!! warning "Lower it for a graded scorer — in both places"
    A ROUGE score or an LLM judge rarely reaches 0.999, so with the default every
    task looks unsolved forever: the pass rate `p` stays near 0, the signal weight
    stays near 0, and the sampler degenerates. Pass
    `evolve(solved_threshold=0.8)` **and**
    `DifficultyWeighted(pass_threshold=0.8)`, or the two disagree about what a
    pass is.

## Writing your own

```python
class Hardest:
    name = "hardest-first"

    def __init__(self):
        self.seen = {}

    def pick(self, keys, round_index):        # never mutate `keys`
        return min(keys, key=lambda k: self.seen.get(k, 0.0))

    def record(self, task_id, score):         # 0..1
        self.seen[task_id] = score

evolve(tasks, reward, agent=agent, task_sampler=Hardest())
```

Two methods, no base class. `pick` returns one task id from the shard it is
given; `record` reports what that rollout scored.

!!! note "One sampler, many threads"
    With `max_concurrency > 1` the workers share a single sampler instance and
    call both methods concurrently. `DifficultyWeighted` guards its statistics
    with a lock; a custom sampler that keeps mutable state must do the same.
