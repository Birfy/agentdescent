# Install and first run

## Install

```bash
pip install agentdescent
```

The core engine has **zero required dependencies** and needs only Python ≥ 3.9.
That gives you the whole library: [`evolve()`](evolution.md), the
[aggregator](aggregator.md), the [agent layer](agents.md), the
[data layer](dataloader.md), [directory evolution](directory-evolution.md).

| extra | adds | for |
|---|---|---|
| `pip install -e ".[dev]"` | pytest | running the test suite |
| `pip install -e ".[docs]"` | MkDocs Material | building this site |
| `pip install anthropic` | the Claude SDK | [`claude(...)`](agents.md) |
| `pip install openhands-ai` | OpenHands SDK (Python ≥ 3.12) | [`openhands(...)`](backends.md) |

Nothing else is needed for an OpenAI-compatible endpoint — GLM, DeepSeek, a local
vLLM server — because [`openai_compatible`](agents.md) speaks HTTP directly.

## The examples need a checkout

They are research artifacts kept **outside** the installed package (they would
otherwise squat the top-level `examples` name), so every `python -m examples.…`
command needs a clone:

```bash
git clone https://github.com/Birfy/agentdescent && cd agentdescent
pip install -e ".[dev]"
```

## First run — no API key

```bash
python -m examples.run_demo
```

Runs the merge-based loop and a fork baseline on the same budget over the
[reference domain](orchestrator.md), then prints the learning curve and the
comparison. This is the framework's central claim, reproducible in seconds:

```
round  dev_acc   stable  commit  fused  stale  confl  oracle
    0    0.828    0.000       1      1      0      0       0
    3    1.000    0.000       1      0      0      1       0     ← a contradiction dropped
    8    1.000    1.000       0      0      0      0       0     ← stable branch catches up

AgentDescent (merge) held-out accuracy : 1.000
Fork/archive best-fork accuracy        : 0.379
merge advantage                        : +0.621
```

Two more that need nothing:

```bash
python -m examples.skill_dir_evolution    # evolve a skill DIRECTORY a real agent reads
python -m examples.efficiency             # parallel scaling + async tail-hiding
```

The complete list — every demo, every algorithm port, and what each one prints —
is in [run everything](usage.md#1-run-the-demos).

## First real run — with a model

Point the provider layer at whatever you have. Credentials are read from the
environment at call time and never pass through code:

```bash
export OPENAI_BASE_URL=https://api.deepseek.com     # or GLM, vLLM, OpenAI itself
export OPENAI_API_KEY=sk-...
```

```python
from agentdescent import evolve_skill, openai_compatible
from agentdescent.dataloader import hf_rows

rows = hf_rows("openai/gsm8k", config="main", split="train", limit=64)

result = evolve_skill(rows, model=openai_compatible(model="deepseek-v4-flash"),
                      prompt="question", gold="answer", score="last_number")

print(result.rendered)        # the skill it learned
print(result.final_reward)    # held-out reward
```

For Claude, `pip install anthropic` and use `claude(model="claude-haiku-4-5")`
instead — same call everywhere else. Full walkthrough:
[quickstart](quickstart-skill.md).

!!! tip "Inspect a faithful port with `--dry-run`"
    Every algorithm port (all eighteen) prints its configuration with no dataset
    download, model call, or API key. Other examples may still load or download data:

    ```bash
    python -m examples.ace.ace_context_evolution --dry-run
    python -m examples.gepa.gepa_prompt_evolution --dry-run
    ```

## Running the tests

The suite is offline and deterministic — no network, no model API:

```bash
pytest -q
```

CI runs it on Python 3.9 / 3.11 / 3.12 for every push and PR.

## Building the docs

```bash
pip install -e ".[docs]"
mkdocs serve                    # live preview at http://127.0.0.1:8000
mkdocs build --strict           # must pass with no warnings (CI enforces this)
python -m tools.gen_api_docs    # regenerate the API reference after a signature change
```

## Where to go next

| you want | go to |
|---|---|
| the shortest path from a dataset to a result | [Quickstart — a skill](quickstart-skill.md) |
| to evolve a folder a real agent reads | [Quickstart — a directory](quickstart-directory.md) |
| to understand why it is built this way | [Concepts](concepts.md) |
| every knob on the loop | [The `evolve` method](evolution.md) |
| a specific module | [Module map](modules.md) |
| a signature | [API reference](api.md) |
