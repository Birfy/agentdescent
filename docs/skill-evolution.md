# Example: skill evolution (real API, all modules)

The one complete, end-to-end example — a real dataset, a real LLM, and every
module wired together. It evolves a **skill playbook** (accumulated lessons) on a
**BIG-Bench-Hard** task, and it uses:

| Module | Used for |
|---|---|
| [`concordia.agents`](agents.md) | `claude(...)` / `openai_compatible(...)` → a `Completion` (provider layer) |
| [`concordia.evolution`](evolution.md) | `LLMAgent` + `evolve()` + `AppendRules` strategy (the engine + rule) |
| [`concordia.parallel`](parallelism.md) | `DataParallel` — the parallelism method |
| governance | `blast_radius=0.2` → the L2 skill layer |

Source:
[`examples/skill_evolution.py`](https://github.com/Birfy/concordia/blob/main/examples/skill_evolution.py).

```python
from concordia.agents import claude
from concordia.evolution import evolve, LLMAgent, AppendRules
from concordia.parallel import DataParallel

result = evolve(
    tasks, reward,                                   # your dataset + scorer
    agent=LLMAgent(claude(model="claude-haiku-4-5")),
    strategy=AppendRules(),                          # deduped, fused lessons
    parallel=DataParallel(),                         # parallelism method
    blast_radius=0.2,                                # L2 (a local skill)
    rounds=6, n_workers=4,
)
print(result.rendered, result.final_reward)
```

## Choosing a task that can actually improve

The default task is **`salient_translation_error_detection`** for a specific
reason. A capable model is at *ceiling* on most benchmarks (nothing to learn) or
at *floor* on the hardest (a text lesson can't help), and even a mid-range
*category* like MMLU-Pro law is a grab-bag of unrelated subtopics, so a lesson
learned on one problem doesn't transfer to the held-out.

The task that works is a **single-skill** one where every problem is the same
kind — here, "which of 6 error types does this translation have" — and the model
is **mid-range** (haiku 4.5 ≈ 0.75). Then one learned lesson transfers to every
held-out problem *and* there's headroom to improve.

## A real run

`--provider claude --model claude-haiku-4-5`:

```
round 0  reward=0.750  items=1  +1/-0
round 1  reward=0.750  items=2  +1/-0
round 2  reward=0.792  items=3  +1/-0    ← a lesson transferred and lifted held-out
round 3  reward=0.792  items=3  +0/-1    ← the gate rejects a lesson that didn't help
```

**0.750 → 0.792** held-out on a real benchmark, with the aggregator visibly
committing helpful lessons and rejecting a useless one. The gains are modest — a
text lesson only moves a strong model where its errors are *process* problems,
not knowledge gaps — but the machinery is exactly the same as the deterministic
examples that converge fully (router merge-vs-fork, etc.); there the agent can
*apply* every learned rule, so the curve is steep.

## Run it

The dataset is fetched through the [`concordia.dataloader`](dataloader.md) data
layer (cached, dependency-free), so the first `--dry-run` downloads it and every
later run is offline.

```bash
python -m examples.skill_evolution --dry-run                 # dataset + estimate, no API
python -m examples.skill_evolution --model claude-haiku-4-5   # real run (Anthropic)
python -m examples.skill_evolution --task hyperbaton --rounds 8
```

Any model — the provider layer is pluggable:

```bash
# GLM or any OpenAI-compatible endpoint (key + base URL read from your env):
export OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4
export OPENAI_API_KEY=...        # set in your own shell; never in code
python -m examples.skill_evolution --provider glm --model glm-4.6
```

!!! note "Robust to API failures"
    If the model backend fails mid-run (rate limit, credit exhaustion), the
    engine stops gracefully and returns the partial result — progress isn't lost.

!!! warning "Cost"
    A real run makes many calls (rollouts + held-out scoring + the aggregator's
    cheap-eval subsets). Defaults are small; `--dry-run` prints an estimate.
    Identical `(skill, task)` evaluations are memoized within a run.
