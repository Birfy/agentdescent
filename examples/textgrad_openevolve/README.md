# TextGrad and OpenEvolve

This example bundle contains two small GLM-5.2 algorithm demonstrations and a
serial/parallel/asynchronous comparison built directly on those implementations.

## Contents

- `textgrad_prompt_optimization.py`: one-variable textual gradient descent on
  BBH word sorting.
- `openevolve_program_search.py`: LLM program evolution with an AST gate and
  Bubblewrap evaluator.
- `parallel_async_benchmark.py`: serial, stage-barrier parallel, and
  barrier-free pipeline comparisons for both algorithms.
- `_openevolve_runner.py`: isolated candidate evaluator used inside Bubblewrap.
- `openevolve_best_program.py`: best program from the recorded small run.
- `results/`: machine-readable live results and the corresponding reports.

## Dry Runs

Dry runs make no API calls and need no key:

```bash
python -m examples.textgrad_openevolve.textgrad_prompt_optimization --dry-run
python -m examples.textgrad_openevolve.openevolve_program_search --dry-run
python -m examples.textgrad_openevolve.parallel_async_benchmark --dry-run
```

## Recorded Results

- [`results/textgrad-openevolve-small-experiments.md`](results/textgrad-openevolve-small-experiments.md)
- [`results/textgrad-openevolve-parallel-async.md`](results/textgrad-openevolve-parallel-async.md)
- [`results/textgrad-small-result.json`](results/textgrad-small-result.json)
- [`results/openevolve-small-result.json`](results/openevolve-small-result.json)
- [`results/algorithm-parallel-async-result.json`](results/algorithm-parallel-async-result.json)

The JSON files contain model outputs and usage metrics, but no API key. The
experiments read `OPENAI_API_KEY` and `OPENAI_BASE_URL` from the environment.
