# Candidate-method experiments

This directory contains compact, mechanism-level experiments for the eleven
issue #74 candidates left after TextGrad and OpenEvolve:

- PromptBreeder, AFlow, Reflexion, Self-Refine
- Voyager, SkillWeaver
- Absolute Zero, R-Zero, Agent0
- SICA, Godel Agent

The methods use compact deterministic domains with strict evaluators instead of
an LLM judge. Every comparison runs through AgentDescent itself:

- `serial`: `evolve(max_concurrency=1)`
- `sync_parallel`: `evolve(max_concurrency=workers)`
- `async_pipeline`: `async_evolve(...)`

Each mode reserves the same number of candidates and method-specific proposal
calls. Framework gate calls can differ because async has completion-order merge
sweeps; those calls are measured and reported instead of being hidden. The
implementation is split by mechanism:

- [`inference_methods.py`](inference_methods.py): PromptBreeder, AFlow,
  Reflexion, and Self-Refine
- [`environment_methods.py`](environment_methods.py): Voyager and SkillWeaver
- [`self_play_methods.py`](self_play_methods.py): Absolute Zero, R-Zero, and
  Agent0
- [`self_edit_methods.py`](self_edit_methods.py): SICA and Godel Agent
- [`framework.py`](framework.py): the shared `evolve` / `async_evolve` adapter
- [`runtime.py`](runtime.py): provider call timing only; it contains no scheduler

Port author: `cyanneko`.

These are not all faithful paper reproductions. `ALGORITHMS` in
`benchmark.py` labels each one:

- `mechanism_microport`: paper/released control flow on a compact API domain.
- `environment_analogue`: the loop is real but Minecraft or WebArena is
  replaced by a deterministic local environment.
- `inference_analogue`: proposer/solver/reward flow is preserved, while verbal
  memory replaces unavailable distributed RL weight updates.
- `self_edit_analogue`: real AST-gated source replacement over a small policy.

Preview the candidate and proposal-call budget without a key or network access:

```bash
python -m bench.candidate_methods --dry-run
```

Run one low-cost live matrix:

```bash
python -m bench.candidate_methods \
  --provider openai --model glm-5.2 --repeats 1 \
  --workers 2 --candidates 2 --yes
```

Run the paired three-seed matrix:

```bash
python -m bench.candidate_methods \
  --provider openai --model glm-5.2 --repeats 3 \
  --workers 2 --candidates 2 \
  --output examples/candidate_methods/results/candidate-methods-framework-raw.json \
  --yes
python -m bench.candidate_methods_merge \
  --inputs examples/candidate_methods/results/candidate-methods-framework-raw.json \
  --expected-seeds 0 100 200 \
  --output examples/candidate_methods/results/candidate-methods-framework-final.json
python -m bench.candidate_methods_report
```

The runner reads `OPENAI_API_KEY` and `OPENAI_BASE_URL` from the environment,
rotates mode order, writes every paid observation atomically, supports `--resume`,
and stores no raw model response or generated source in the result JSON. The
merge step validates runtime provenance, source hashes, budgets, failures, and
seed coverage before publishing a compact final JSON without per-call event
traces. Only direct `evolve` and `async_evolve` observations enter the report.
