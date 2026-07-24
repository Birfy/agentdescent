# Example: skill evolution (real dataset + LLM)

One concrete use of the [general engine](evolution.md): evolve a **skill** — a
lesson playbook — on a real, hard **BIG-Bench-Hard** task, driven by a real
**Claude** agent. BBH tasks are deliberately hard for LLMs and scored by exact
match / graded overlap, so there is genuine headroom for a learned skill to
raise the score.

Source:
[`examples/skill_evolution.py`](https://github.com/Birfy/concordia/blob/main/examples/skill_evolution.py).

```python
from concordia.agents import claude
from concordia.evolution import evolve, LLMAgent, AppendRules

result = evolve(
    tasks, reward,                                  # your dataset + scorer
    agent=LLMAgent(claude(model="claude-haiku-4-5")),
    strategy=AppendRules(),                          # deduped, fused lessons
    blast_radius=0.2,                               # L2 (a local skill)
    artifact_id="skill", rounds=4, n_workers=2,
)
print(result.rendered, result.final_reward)
```

Each round, parallel workers run problems through the current skill via Claude
and, on a failure, ask Claude to propose a lesson. The aggregator dedupes, fuses
complementary lessons from parallel workers, and **commits a lesson only if it
improves held-out score** — so unhelpful lessons are rejected automatically.

## Run it

```bash
# inspect the dataset + a cost estimate, no API calls:
python -m examples.skill_evolution --dry-run

# the real thing (needs ANTHROPIC_API_KEY or `ant auth login`):
python -m examples.skill_evolution --task logical_deduction_seven_objects --rounds 5
python -m examples.skill_evolution --task word_sorting --model claude-haiku-4-5
```

`--dry-run` output:

```
Dataset : BIG-Bench-Hard / word_sorting
Loaded  : 250 examples; using 22 (12 train / 10 held-out)
Scoring : graded token overlap
Plan    : model=claude-opus-4-8, rounds=4, workers=2
Budget  : up to ~232 Claude calls (cached repeats are free; use --model claude-haiku-4-5 to cut cost)
```

!!! warning "Cost"
    A real-LLM run makes many calls (rollouts + held-out scoring + the
    aggregator's cheap-eval subsets). Defaults are small; the script prints an
    estimate and asks before spending. Identical `(skill, task)` evaluations are
    memoized within a run.

## Customizing

The three plug-ins — the **agent** (`solve`/`propose`, or any `run`/`propose`),
the **reward**, and the **`Strategy`** (evolution logic) — are documented on the
[engine page](evolution.md#what-you-provide). The no-API mechanics (any-agent
protocol, custom strategies, merge-over-fork, harmful-lesson rejection) are
exercised deterministically in the test suite
([`test_evolution.py`](https://github.com/Birfy/concordia/blob/main/tests/test_evolution.py),
[`test_strategy.py`](https://github.com/Birfy/concordia/blob/main/tests/test_strategy.py)).
