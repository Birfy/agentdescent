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
| `DGMParentSelection` | selection ([seam](selection.md)) | `sigmoid(10·(s−0.5)) × 1/(1+children)` parent sampling; default of `DGMArchiveAggregator` |
| **`HarnessStrategy`** | `strategy=` | a proposed capability becomes a `Diff` on the harness capability set |
| **`DGMArchiveAggregator`** | `aggregator_factory=` | keep-all archive + staged eval (10→50→140) + sigmoid×novelty parent selection; sets the sampled parent as the dev head |
| **`dgm_parent_weights` / `choose_selfimproves`** | (selection) | the exact DGM rule `p_i ∝ sigmoid(10·(score−0.5)) · 1/(1+children_i)` |
| `propose` + `make_surrogate_evaluator` | `propose=` / objective | add the most-needed capability; the transparent surrogate objective (swap in a real Docker harness via `evaluate_fn`) |

## Measured — the real objective

`--objective real` evolves the agent's own Python source and scores it by running
pytest. 64 vendored bugs in `examples/dgm/tasks/` (32 train / 32 held-out),
16 rollouts, `--staleness full`, `deepseek-v4-flash`, one seed:

| | resolve rate |
|---|---|
| seed agent, held out | **0.844** (27 of 32) |
| best archived child | **0.906** (29 of 32) |
| archive | 3 agents: 0.906, 0.906, 0.844 |
| the agent's own `solve.py` | 18 lines → **79 lines** |

**What it wrote for itself.** The seed agent sends `lib.py` and the pytest output
to a model and writes the reply straight back. Its own diagnosis produced three
changes, all of them aimed at failures that actually occurred:

```python
def _strip_code_fences(reply):     # replies arrived wrapped in ``` and broke the import
    if text.startswith("```"): ...
    match = re.search(r"```[a-zA-Z0-9_+-]*\s*\n(.*?)```", text, re.DOTALL)

def _is_success(out):              # "passed" alone is not success
    return " passed" in out and " failed" not in out and " error" not in out

test_source = (task_path / "test_lib.py").read_text()   # read the tests too
```

The fence stripper is the one that mattered: measured before the run, six of the
sixteen tasks the seed failed came back as `1 error` rather than `1 failed` --
the reply had been written into `lib.py` complete with its Markdown fence, so the
module would not import. Reading the test file is not a fix for anything that
failed; it is the agent giving itself an input the seed never had.

!!! warning "This is not SWE-bench"
    64 hand-written bugs are not 500 repository issues, and a number here cannot
    be compared with the paper. What this reproduces is the *shape*: real source,
    real execution, real pass/fail, and self-edits that can leave the agent worse
    -- or unable to run at all.

    The [surrogate objective](#honesty-boundary) remains the default, and its
    limitation is worth stating precisely: it is **monotone**. Adding a
    capability can never un-resolve a task, so a self-modification can never
    regress -- which removes the reason to keep an archive. Open-ended search
    retains stepping stones because a worse intermediate can lead somewhere
    better, and under a monotone objective there are no worse intermediates.
    Under the real objective there are: an earlier run archived children at
    **0.875 and 0.500**, the worse one kept, which is the behaviour `keep-all`
    exists for and which the surrogate cannot produce.

!!! danger "Two numbers on this page were wrong before they were right"
    An earlier version of this run reported **0.000 → 0.844**. The final figure
    was correct and the baseline was invented: the archive seeded itself inside
    `step()`, which only runs once a card reaches the merger, so a run whose
    self-modifications all failed never measured its own seed and printed
    `DGMContext.seed_score`'s 0.0 default. A wrong denominator under a right
    numerator reads as a large success. Seeding now happens when the archive is
    built, and a test pins it with an actor that proposes nothing at all.

    Before that, a self-edit that added a **retry loop** scored 0.875 → 0.500 and
    looked like a regression. It was not: the harness pre-fetched a single model
    reply and served it to every call, so the second and third attempts received
    the first answer again. "One model call per task" was a property of the
    harness, and the first improvement the agent ever proposed was the one thing
    that made impossible. The bridge is now a real request/reply channel with a
    call budget.

## Run it

```bash
python -m examples.dgm.dgm_self_improve                      # offline (surrogate)
python -m examples.dgm.dgm_self_improve --generations 12 --archive keep_all
python -m examples.dgm.dgm_self_improve --model claude-haiku-4-5   # LLM proposes modifications

# the real objective, as measured above
python -m examples.dgm.dgm_self_improve --yes --seed 0 --objective real \
    --selfimprove-size 2 --generations 9999 --budget-rollouts 16 \
    --staleness full --async --async-ratio 3 --max-seconds 2700 \
    --eval-concurrency 16 --model deepseek-v4-flash \
    --write-agent /tmp/evolved-agent
```

`--write-agent` is not optional if you want to see what the run produced: the
ledger is a scratch git repo that `evolve` reaps on exit, and for a self-editing
agent that source *is* the result.

Offline tests: `tests/test_dgm_example.py`, `tests/test_dgm_real_objective.py`.
