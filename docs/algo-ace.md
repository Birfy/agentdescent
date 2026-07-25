# ACE — Agentic Context Engineering

> **Skill / context self-evolution.** Evolve a *playbook of lessons* (the model's
> context), not the weights. Runs through [`evolve()`](evolution.md) with a custom
> `Strategy`. Example: [`examples/ace_context_evolution.py`](https://github.com/Birfy/concordia/blob/main/examples/ace_context_evolution.py).

| | |
|---|---|
| **Paper** | *Agentic Context Engineering* — Zhang et al., 2025 ([arXiv:2510.04618](https://arxiv.org/abs/2510.04618)) |
| **Repo** | [`ace-agent/ace`](https://github.com/ace-agent/ace) |
| **Dataset** | **FiNER-139** (financial XBRL tagging) — `nlpaueb/finer-139` |
| **Layer** | L2 skill (`blast_radius=0.2`) |

## The algorithm

ACE evolves a **context playbook** through three roles, which map one-to-one onto
`evolve()`:

| ACE role | Concordia piece | Job |
|---|---|---|
| **Generator** | `LLMAgent.solve` (the `run`) | solve a task using the playbook |
| **Reflector** | `LLMAgent.propose` (ACE template) | distil ONE *delta bullet* from a failure |
| **Curator** | the **aggregator** | deterministic, non-LLM merge (dedup + statistical acceptance) |

Two invariants are preserved by the custom `ACEPlaybook` strategy:

* **Incremental delta updates.** `to_diff` only ever *appends* a new
  content-addressed bullet — never a monolithic rewrite — so ACE's **context
  collapse** (the model compressing an accumulated context into a lossy summary)
  cannot happen.
* **Grow-and-refine de-dup.** A near-duplicate bullet (lexical-Jaccard proxy for
  ACE's embedding de-dup) is dropped at insert time.

ACE's per-bullet **helpful / harmful** counters become the aggregator's per-diff
**Beta-posterior acceptance**: a bullet is committed only when it raises held-out
reward, and rejected otherwise.

## How it plugs into `evolve()`

```python
evolve(tasks, reward, agent=ace_agent(completion),
       strategy=ACEPlaybook(), blast_radius=0.2, artifact_id="ace_playbook")
```

* `strategy=ACEPlaybook()` — the itemised delta representation + grow-and-refine.
* `agent=` — Generator (`solve`) + Reflector (`propose`).
* the default aggregator **is** the Curator (dedup + Beta gate).

## Dataset

FiNER-139 framed as XBRL-tag classification of a highlighted numeric span,
restricted to the `--top-k` most frequent concepts so a learned lesson transfers.
ACE's full setup also runs AppWorld (a heavy simulator) — out of scope here and
documented in the module docstring.

## Run it

```bash
python -m examples.ace_context_evolution --dry-run
python -m examples.ace_context_evolution --model claude-haiku-4-5
python -m examples.ace_context_evolution --top-k 10 --rounds 6
```

Offline tests: `tests/test_ace_example.py`.
