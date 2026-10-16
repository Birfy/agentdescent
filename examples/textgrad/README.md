# TextGrad — Textual-gradient prompt optimization

Port onto the AgentDescent engine, as a declarative
[`MethodPolicy`](../_method_policy.py).

| | |
|---|---|
| Kind | Reflection / refinement (backpropagated natural-language critique) |
| Fidelity class | `mechanism_microport` |
| Paper | "Optimizing generative AI by backpropagating language model feedback", Yuksekgonul et al., *Nature* 2025 ([doi](https://www.nature.com/articles/s41586-025-08661-4)); [arXiv:2406.07496](https://arxiv.org/abs/2406.07496) |
| Upstream code | [zou-group/textgrad@75e912e2](https://github.com/zou-group/textgrad/tree/75e912e210864b61999781778cdf756d4468120f) |
| Domain | GSM8K (64/64/64 windows) |
| Definition plug-ins | `TextVariable` (content-addressed single slot); two-call backward → TGD-step proposal; reflective merge |

## Run

```bash
python -m examples.textgrad.textgrad_prompt_optimization --dry-run
python -m examples.textgrad.textgrad_prompt_optimization --provider openai --model glm-5.2
```

`--dry-run` prints the configuration and returns with **zero network access
and no API key** (the `textgrad` package is never imported, and no model is
called).

## What is in here

- [`textgrad_prompt_optimization.py`](textgrad_prompt_optimization.py) — the
  `MethodPolicy` definition and runnable entry point
- Mechanism notes, upstream trace, and every recorded boundary:
  [`docs/algo-textgrad.md`](../../docs/algo-textgrad.md)
- Offline tests: [`tests/test_textgrad_example.py`](../../tests/test_textgrad_example.py)
