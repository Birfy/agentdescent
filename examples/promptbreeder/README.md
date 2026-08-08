# PromptBreeder — Prompt self-evolution (genetic)

Port onto the AgentDescent engine, as a declarative
[`MethodPolicy`](../_method_policy.py).

| | |
|---|---|
| Kind | Prompt self-evolution (genetic) |
| Fidelity class | `mechanism_microport` |
| Paper | "Promptbreeder: Self-Referential Self-Improvement Via Prompt Evolution", Fernando et al., 2023 ([arXiv:2309.16797](https://arxiv.org/abs/2309.16797)) |
| Upstream code | paper only — no official released code |
| Domain (compact) | deterministic integer-cents arithmetic (12 tasks, disjoint splits) |
| Definition plug-ins | `FieldSlots` genome (task-prompt + mutation-prompt keys); `Policies(selection=BinaryTournament)`; reflective merge |

## Run

```bash
python -m examples.promptbreeder.promptbreeder_genetic_prompts --dry-run
python -m examples.promptbreeder.promptbreeder_genetic_prompts --provider openai --model glm-5.2
```

`--dry-run` prints the configuration and returns with **zero network access
and no API key**.

## What is in here

- [`promptbreeder_genetic_prompts.py`](
  promptbreeder_genetic_prompts.py) — the `MethodPolicy` definition and runnable entry point
- Mechanism notes, upstream trace, and every recorded boundary: [`docs/algo-promptbreeder.md`](../../docs/algo-promptbreeder.md)
- Offline tests: [`tests/test_candidate_methods.py`](../../tests/test_candidate_methods.py)

All MethodPolicy ports share one declarative contract (`examples/_method_policy.py`)
and one runner (`examples/_method_runner.py`); scheduling, budgets, and merging
never appear in a definition. The full serial/sync/async matrix is
`python -m bench.candidate_methods`.
