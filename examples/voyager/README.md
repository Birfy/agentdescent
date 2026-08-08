# Voyager — Embodied skill-library agent

Port onto the AgentDescent engine, as a declarative
[`MethodPolicy`](../_method_policy.py).

| | |
|---|---|
| Kind | Embodied skill-library agent |
| Fidelity class | `environment_analogue` |
| Paper | "Voyager: An Open-Ended Embodied Agent with Large Language Models", Wang et al., 2023 ([arXiv:2305.16291](https://arxiv.org/abs/2305.16291)) |
| Upstream code | [MineDojo/Voyager@55e45a88](https://github.com/MineDojo/Voyager/tree/55e45a880755d0c8c66ca7fb5fe7962ac8974f89) |
| Domain (compact) | deterministic crafting world (12 recipe goals, disjoint splits) |
| Definition plug-ins | `SkillLibrary` strategy; `Policies(task_sampler=DifficultyWeighted)`; `self_verify` critic; environment error feedback |

## Run

```bash
python -m examples.voyager.voyager_skill_library --dry-run
python -m examples.voyager.voyager_skill_library --provider openai --model glm-5.2
```

`--dry-run` prints the configuration and returns with **zero network access
and no API key**.

## What is in here

- [`voyager_skill_library.py`](
  voyager_skill_library.py) — the `MethodPolicy` definition and runnable entry point
- Mechanism notes, upstream trace, and every recorded boundary: [`docs/algo-voyager.md`](../../docs/algo-voyager.md)
- Offline tests: [`tests/test_candidate_methods.py`](../../tests/test_candidate_methods.py)

All MethodPolicy ports share one declarative contract (`examples/_method_policy.py`)
and one runner (`examples/_method_runner.py`); scheduling, budgets, and merging
never appear in a definition. The full serial/sync/async matrix is
`python -m bench.candidate_methods`.
