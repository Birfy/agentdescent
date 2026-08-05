# DGM — Darwin Gödel Machine

> **Harness self-evolution.** A coding agent that *edits its own codebase*,
> keeping every variant in an open-ended archive. Runs through
> [`evolve()`](evolution.md) with a custom `Strategy` + `aggregator_factory` at
> **L1** governance. Example:
> [`examples/dgm/dgm_self_improve.py`](https://github.com/Birfy/agentdescent/blob/main/examples/dgm/dgm_self_improve.py).

| | |
|---|---|
| **Paper** | *Darwin Gödel Machine* — Zhang, Hu, Lu, Lange, Clune, 2025 ([arXiv:2505.22954](https://arxiv.org/abs/2505.22954)) |
| **Repo** | [`jennyzzt/dgm`](https://github.com/jennyzzt/dgm) |
| **Dataset** | **SWE-bench Verified** (real instance ids) |
| **Layer** | L1 harness (`blast_radius=0.6`) |

## The algorithm

Faithful to `DGM_outer.py`:

* **Keep-all archive** of agents — stepping stones are retained, not just the best.
* **Parent selection** = `p_i ∝ sigmoid(10·(score−0.5)) · 1/(1+children_i)` —
  favour high performers, discount already-explored parents (open-endedness).
  Ported exactly as `dgm_parent_weights` and unit-tested.
* **Self-modification**: a parent inspects its own eval logs, diagnoses a
  weakness, and adds "the next feature" to its harness → a child.
* **Staged empirical validation**: small=10 → medium=50 if score > 0.4 → big=140.

## How it plugs into `evolve()`

* `strategy=HarnessStrategy()` — a proposed capability → a `Diff` on the harness's
  capability set.
* `run` — the surrogate objective; `reward` — resolved / not.
* `propose` — inspect the failed instance, add the most-needed capability.
* `aggregator_factory` → `DGMArchiveAggregator` — the keep-all archive with staged
  eval and the sigmoid×novelty parent selection; it **commits the sampled parent
  as the dev head**, so `evolve()` mutates it open-endedly next round (not the
  greedy best).

## Honesty boundary

DGM's real objective runs each candidate patch inside the **SWE-bench Docker
harness** (per-task containers, real test suites, arbitrary code execution) —
out of scope for a dependency-free example. The objective here is a **transparent
surrogate** (each real SWE instance has a latent required-capability set an agent
must cover), so the DGM *algorithm* runs and is tested offline while the *scores*
are simulated, not SWE-bench results. Pass a real `evaluate_fn` to `run_dgm` to
plug in the actual Docker harness.

## Plug-ins implemented

In [`examples/dgm/dgm_self_improve.py`](https://github.com/Birfy/agentdescent/blob/main/examples/dgm/dgm_self_improve.py):

| Plug-in | `evolve()` slot | What it does |
|---|---|---|
| **`HarnessStrategy`** | `strategy=` | a proposed capability becomes a `Diff` on the harness capability set |
| **`DGMArchiveAggregator`** | `aggregator_factory=` | keep-all archive + staged eval (10→50→140) + sigmoid×novelty parent selection; sets the sampled parent as the dev head |
| **`dgm_parent_weights` / `choose_selfimproves`** | (selection) | the exact DGM rule `p_i ∝ sigmoid(10·(score−0.5)) · 1/(1+children_i)` |
| `propose` + `make_surrogate_evaluator` | `propose=` / objective | add the most-needed capability; the transparent surrogate objective (swap in a real Docker harness via `evaluate_fn`) |

## Measured — surrogate objective

`--generations 4 --provider openai --model deepseek-v4-flash`:

| | resolve rate |
|---|---|
| val, before → after | 0.000 → **0.300** |
| test (held out, never seen by selection) | 0.200 |

Archive: 4 agents (keep-all). The best lineage reached generation 3 with
capabilities `context-retrieval`, `dependency-resolver`, `diff-minimization`,
`regression-test-runner`.

!!! warning "This is the surrogate, not SWE-bench"
    The objective is a capability-cover stand-in; real DGM evaluates on SWE-bench
    Verified inside Docker, which this example does not run. What is faithful is
    the **archive, the selection rule and the staged escalation** — the numbers
    above measure those mechanics, not coding ability.

## Run it

```bash
python -m examples.dgm.dgm_self_improve                      # runs offline (surrogate)
python -m examples.dgm.dgm_self_improve --generations 12 --archive keep_all
python -m examples.dgm.dgm_self_improve --model claude-haiku-4-5   # LLM proposes modifications
```

Offline tests: `tests/test_dgm_example.py`.
