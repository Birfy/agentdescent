# Reflexion — Verbal reinforcement / episodic memory

Port onto the AgentDescent engine, as a declarative
[`MethodPolicy`](../_method_policy.py).

| | |
|---|---|
| Kind | Verbal reinforcement / episodic memory |
| Fidelity class | `mechanism_microport` |
| Paper | "Reflexion: Language Agents with Verbal Reinforcement Learning", Shinn et al., 2023 ([arXiv:2303.11366](https://arxiv.org/abs/2303.11366)) |
| Upstream code | [noahshinn/reflexion@218cf0ef](https://github.com/noahshinn/reflexion/tree/218cf0ef1df84b05ce379dd4a8e47f17766733a0) |
| Domain (compact) | deterministic integer-cents arithmetic (12 tasks, disjoint splits) |
| Definition plug-ins | `WindowedMemory` strategy (bounded append-only); reflective merge |

## Run

```bash
python -m examples.reflexion.reflexion_episodic_memory --dry-run
python -m examples.reflexion.reflexion_episodic_memory --provider openai --model glm-5.2
```

`--dry-run` prints the configuration and returns with **zero network access
and no API key**.

## What is in here

- [`reflexion_episodic_memory.py`](
  reflexion_episodic_memory.py) — the `MethodPolicy` definition and runnable entry point
- Mechanism notes, upstream trace, and every recorded boundary: [`docs/algo-reflexion.md`](../../docs/algo-reflexion.md)
- Offline tests: [`tests/test_candidate_methods.py`](../../tests/test_candidate_methods.py)

All MethodPolicy ports share one declarative contract (`examples/_method_policy.py`)
and one runner (`examples/_method_runner.py`); scheduling, budgets, and merging
never appear in a definition. The full serial/sync/async matrix is
`python -m bench.candidate_methods`.
