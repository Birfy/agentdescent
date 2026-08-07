# SkillWeaver — Web agent API synthesis

Candidate-method port onto the AgentDescent engine, as a declarative
[`MethodPolicy`](../_method_policy.py).

| | |
|---|---|
| Kind | Web agent API synthesis |
| Fidelity class | `environment_analogue` |
| Paper | "SkillWeaver: Web Agents can Self-Improve by Discovering and Honing Skills", Zheng et al., 2025 ([arXiv:2504.07079](https://arxiv.org/abs/2504.07079)) |
| Upstream code | [OSU-NLP-Group/SkillWeaver@f2a63d65](https://github.com/OSU-NLP-Group/SkillWeaver/tree/f2a63d65d0f6ff46ac30e817cede8797f8f25b97) |
| Domain (compact) | deterministic settings web service (12 form tasks, disjoint splits) |
| Definition plug-ins | `SkillLibrary` strategy; `Policies(task_sampler=DifficultyWeighted)`; `self_verify` reward model; site execution feedback |

## Run

```bash
python -m examples.skillweaver.skillweaver_web_apis --dry-run
python -m examples.skillweaver.skillweaver_web_apis --provider openai --model glm-5.2
```

`--dry-run` prints the configuration and returns with **zero network access
and no API key**.

## What is in here

- [`skillweaver_web_apis.py`](
  skillweaver_web_apis.py) — the `MethodPolicy` definition and runnable entry point
- Mechanism notes, upstream trace, and every recorded boundary: [`docs/algo-skillweaver.md`](../../docs/algo-skillweaver.md)
- Offline tests: [`tests/test_candidate_methods.py`](../../tests/test_candidate_methods.py)

All candidate ports share one declarative contract (`examples/_method_policy.py`)
and one runner (`examples/_method_runner.py`); scheduling, budgets, and merging
never appear in a definition. The full serial/sync/async matrix is
`python -m bench.candidate_methods`.
