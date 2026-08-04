# Contributing to AgentDescent

Thanks for your interest in improving AgentDescent. This is a research reference
implementation; contributions that keep it **correct, faithful, and testable**
are very welcome.

## Development setup

```bash
git clone https://github.com/Birfy/agentdescent
cd agentdescent
pip install -e ".[dev]"     # core engine has zero runtime deps; [dev] adds pytest
```

Python ≥ 3.9 is required.

## Shortest verification path

From a fresh checkout, the shortest environment and behavior check is:

```bash
git clone https://github.com/Birfy/agentdescent
cd agentdescent
pip install -e ".[dev]"
pytest -q
python -m examples.run_demo
```

Then temporarily change `rounds = 40` to `rounds = 8` in
`examples/run_demo.py`, run the demo again, observe the shorter curve, and
restore the line. That proves you are executing the checkout you are editing.

## Running the tests

The suite is **offline and deterministic** (no network, no model API):

```bash
pytest -q
```

CI runs the same suite on Python 3.9 / 3.11 / 3.12 for every push and PR. Please
add or update tests for any behavior change — the reference domain
(`agentdescent/domains/router.py`) and the example stubs let you test the full
parallel/async loop without an API key.

## Trying it end to end

```bash
python -m examples.run_demo                       # synchronous DP, no API
python -m examples.efficiency                     # parallel scaling + async tail-hiding
python -m examples.skill_evolution --dry-run      # the flagship LLM example, no API
```

Every faithful algorithm port has a zero-network `--dry-run` mode for inspecting
its CLI configuration without loading a dataset or calling a model.

## Building the docs

```bash
pip install -e ".[docs]"
mkdocs serve            # live preview at http://127.0.0.1:8000
mkdocs build --strict   # must pass with no warnings (CI enforces this)
```

## Guidelines

- **Keep the analogy honest.** The design mirrors the deep-learning training
  stack, but *gradients add and diffs do not* — new merge/acceptance logic must
  go through the aggregator's conflict-resolution + statistical-acceptance path,
  not silent averaging.
- **Faithful ports stay faithful.** The algorithm examples follow each source
  repo's *released code* (not just the paper). If code and paper disagree, follow
  the code and note it. Start from the [porting checklist](docs/porting-checklist.md)
  and `examples/_TEMPLATE.py`.
- **No hidden network in tests.** Anything requiring an API or heavy infra
  (SWE-bench Docker, gated datasets) must be gated behind a flag and documented.
- Match the style and comment density of the surrounding code.

## Pull requests

1. Branch from `main`.
2. `pytest -q` and `mkdocs build --strict` both pass.
3. Update `CHANGELOG.md` under **[Unreleased]**.
4. Open the PR with a clear description of *what* changed and *why*.
