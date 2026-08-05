# Run everything, and extend it

[Install and first run](install.md) covers getting set up. This page is the
complete list of what you can run, the configuration reference, and how to plug
your own domain into the loop.

!!! tip "Looking for something specific?"
    [Module map](modules.md) indexes every module and its page ·
    [API reference](api.md) has every signature ·
    [The `evolve` method](evolution.md) is the entry point's own page.

## 1. Run the demos

!!! note "The demos need a checkout, not just `pip install`"
    The `examples/` directory ships with the **repository**, not the wheel, so
    `python -m examples.…` requires `git clone`. `pip install agentdescent` gives
    you the library only.


### RQ1 — merge vs fork (synchronous DP)

```bash
python -m examples.run_demo
```

Runs the merge-based `AgentDescent` loop and a DGM-style fork baseline on the same
budget, then prints the learning curve and the comparison:

```
round  dev_acc   stable  commit  fused  stale  confl  oracle
    0    0.828    0.000       1      1      0      0       0
    3    1.000    0.000       1      0      0      1       0     ← a contradiction dropped
    8    1.000    1.000       0      0      0      0       0     ← stable branch catches up

AgentDescent (merge) held-out accuracy : 1.000
Fork/archive best-fork accuracy     : 0.379
merge advantage                     : +0.621
```

### Async stage orchestration (FlashEvolve-style)

```bash
python -m examples.run_async
```

Compares the three staleness policies and sweeps the `async_ratio` lag budget.

### RQ2 — staleness tolerance sweep

```bash
python -m examples.rq2_staleness
```

### Evolving a directory (a skill folder a real agent reads)

```bash
python -m examples.skill_dir_evolution
```

Materialises a candidate skill directory into a throwaway workspace at
`.claude/skills/csv-total/` and lets a real (sub)process agent read it. Runs
offline in seconds; `--agent claude-code` swaps in the real CLI agent, and
`--install-to <dir>` writes the evolved skill back. See
[evolving a directory](directory-evolution.md).

### Self-evolution algorithm ports (real datasets)

Faithful ports of the latest skill- and harness-self-evolution algorithms — ACE,
GEPA, EvoSkill, SkillOpt, ADAS, DGM (see
[the catalog](self-evolution-examples.md)). Each loads a real benchmark through
the [`agentdescent.dataloader`](dataloader.md) data layer on a real run.
`--dry-run` prints the requested dataset and runtime configuration, then returns
before loading data or models: it needs no network and no API key.

```bash
python -m examples.ace.ace_context_evolution --dry-run     # ACE   / FiNER-139
python -m examples.gepa.gepa_prompt_evolution --dry-run     # GEPA  / HotpotQA
python -m examples.dgm.dgm_self_improve                    # DGM   / SWE-bench Verified (offline surrogate)
```

### Tests

```bash
pytest                          # the whole suite, no external services
pytest -q tests/test_async.py   # just the async runtime
```

---

## 2. Programmatic use

### The entry point — `evolve()`

Covered in full elsewhere, and not repeated here:

* **[Quickstart](quickstart-skill.md)** — a dataset to an evolved skill in one
  call, with the measured result of running it.
* **[The `evolve` method](evolution.md)** — the entry point underneath, with every
  parameter and what it plugs into.
* **[Connecting agents & LLMs](agents.md)** — any `prompt -> text` is a backend,
  including tool-using CLI agents.
* **[Measured results](results.md)** — every empirical claim with its setup.

### What a run cost

`EvolutionResult` carries the artifact *and* the bill. Every field below defaults
to zero and is read back with `.get`, so a result saved before they existed still
loads.

```python
r = evolve(tasks, reward, agent=agent)

r.cost_summary()          # one line: rollouts, wall-clock, model calls, ratios
r.time_to_quality(0.9)    # seconds to the first round at >= 0.9, or None
r.cost_to_quality(0.9)    # rollouts to that same round, or None
r.stale_rate()            # discarded / considered -- the ratio needs both
r.duplicate_rate()        # evaluation cache hit rate
```

| what | field |
|---|---|
| time | `wallclock`, and `RoundInfo.elapsed_s` per round |
| work | `rollouts`, `rollout_seconds` (a **sum** across workers, so it exceeds `wallclock` when they overlap), `eval_seconds` |
| model | `usage.calls` / `.seconds` / `.failures` |
| staleness | `stale_considered`, `stale_discarded` |
| recovery | `redispatched`, `duplicates_dropped` |
| recomputation | `cache_hits`, `cache_misses` |
| sandboxes | `sandbox_wait_s`, `sandbox_setup_s`, `sandboxes_created` / `_reused` / `_failures` |

Two things worth knowing before reading these numbers:

* **`usage.calls` counts actor invocations** — one `run` or `propose` — not
  provider requests. A `cli_agent` rollout is one call here and many requests to
  the model.
* **Token counts need a shared `Usage`.** `run` is `(rendered, task) -> str`, so
  an opaque actor cannot report tokens. Pass the same object to both and they
  accumulate together:

```python
from agentdescent import Usage
u = Usage()
r = evolve(tasks, reward, agent=LLMAgent(claude(usage=u)), usage=u)
print(r.usage.total_tokens, r.usage.estimated_cost(3.0, 15.0))
```

The sandbox fields are zero on the default path: one throwaway workspace per
rollout, nothing to queue for and no image to warm. They exist so that when a
pool does, "8 workers only bought 2x" can be attributed rather than guessed at.

### The reference stack — `AgentDescent` / `AsyncAgentDescent`

A **separate** runtime used by the RQ1/RQ2 and efficiency experiments on the
built-in synthetic router domain. It has the `TaskScheduler` / `EvidenceBuffer` /
duration-estimator machinery that `evolve()` does not — see the
[two-stack note](architecture.md#4-the-two-runtimes). Reach for it to reproduce
those experiments, not to evolve your own artifact.

```python
import tempfile
from agentdescent.domains.router import make_task_universe
from agentdescent import AgentDescent

universe = make_task_universe(seed=7)
with tempfile.TemporaryDirectory() as repo:
    system = AgentDescent(repo, universe, n_workers=6, noise=0.15, seed=1)
    history = system.run(rounds=40)
    print(system.final_accuracy())      # held-out accuracy on the dev branch
```

```python
import tempfile
from agentdescent import AsyncAgentDescent, AsyncConfig
from agentdescent.domains.router import make_task_universe
from agentdescent import get_policy

universe = make_task_universe(seed=7)
cfg = AsyncConfig(n_workers=6, async_ratio=4, target_accuracy=0.98, max_seconds=15.0)
with tempfile.TemporaryDirectory() as repo:
    system = AsyncAgentDescent(repo, universe, config=cfg,
                            staleness_policy=get_policy("reflective"))
    stats = system.run()
    print(stats.final_dev_accuracy, stats.commits, stats.discarded_stale)
```

---

## 3. Configuration reference

### `AggregatorConfig` ([aggregator](aggregator.md))

| Field | Default | Meaning |
|---|---|---|
| `batch_trigger` | 4 | `B`: fire a bucket once it holds this many cards |
| `max_wait_rounds` | 3 | `T_max`: fire a cold bucket after this many sweeps |
| `base_delta` | 0.5 | acceptance risk; threshold is `1 − δ`, annealed by version |
| `alpha_head` | 5 | staleness tolerance α for hot artifacts |
| `alpha_tail` | 1 | staleness tolerance α for cold artifacts |
| `trust_region_ops` | 6 | max edits per diff (trust region) |
| `trust_region_chars` | 32 000 | max characters in **one** op's value — for a [`FileTree`](directory-evolution.md) that is a per-file cap, and `TreeSpec.max_file_bytes` must stay under it |
| `anneal_half_life` | 64 | versions over which the acceptance risk decays |
| `promote_after_k` | 3 | dev→stable survival rounds (EMA) |

### `AsyncConfig` ([async](async.md))

| Field | Default | Meaning |
|---|---|---|
| `n_workers` | 6 | worker threads |
| `async_ratio` | 3 | ROLL Flash lag budget: max head-drift before a worker refreshes |
| `noise` | 0.15 | per-op probability that a *noisy* worker proposes a wrong label (every third worker is noisy; the rest are clean) |
| `target_accuracy` | 0.98 | stop early when the dev branch reaches this |
| `max_seconds` | 20.0 | wall-clock safety bound |
| `aggregator_interval` | 0.002 | sleep between aggregator sweeps |
| `worker_pause` | 0.001 | sleep between worker rollouts |
| `oracle_budget` | 400 | oracle calls the AuditScheduler may spend |
| `stall_patience` | 150 | no-commit sweeps before a backpressure sync |

`AsyncStats.error` is `None` on a clean run and carries the backend failure that ended it otherwise — check it, since a run whose workers all died otherwise returns normal-looking zeros.

### Staleness policy ([staleness](staleness.md))

```python
from agentdescent import get_policy
get_policy("full")        # accept stale diffs directly
get_policy("guarded")     # version-gated (default)
get_policy("reflective")  # always rebase + re-verify
```

---

## 4. Plug in your own domain

!!! note "Most callers never need this"
    A [strategy](strategies.md) over a flat `{key: value}` state covers nearly
    everything — including a whole [directory](directory-evolution.md). Implement
    [`Evolvable`](data-model.md) yourself only when your artifact genuinely is not
    that shape; [`domains/router.py`](orchestrator.md) is the worked example.

The whole framework is domain-agnostic: **what evolves is decided by
registration, not hard-coded.** To evolve something new, provide four things.

### 4.1 An `Evolvable`

Implement the protocol from `agentdescent/evolvable.py`
([reference: `RouterSkill`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/domains/router.py)):

```python
from agentdescent import Contract, Diff, EvidenceCard

class MyArtifact:
    def __init__(self, id, state, version=1, blast_radius=0.2):
        self.id = id
        self.version = version
        self.blast_radius = blast_radius            # → auto L0/L1/L2 layering
        self.contract = Contract(major=1)
        self.state = state

    def diff(self, other) -> Diff: ...              # difference to another instance
    def apply(self, diff: Diff) -> "MyArtifact":    # return a NEW instance, version+1
        ...
    def cheap_eval(self, evidence: EvidenceCard) -> float:
        # score on the tasks the evidence card carries (used by rebase re-verify)
        ...
    def full_eval(self, task_set) -> dict:          # ground-truth metrics
        ...
```

!!! important "`apply` must be pure"
    `apply` returns a **new** instance with `version + 1`; it never mutates
    `self`. The aggregator relies on this to test candidates without side
    effects.

### 4.2 Serialize / deserialize for the Ledger

The Ledger stores artifacts as JSON blobs in git, so give it two functions:

```python
def serialize_mine(a) -> dict:            # → JSON-friendly dict
    return {"state": a.state, "blast_radius": a.blast_radius}

def deserialize_mine(artifact_id, version, state) -> MyArtifact:
    return MyArtifact(artifact_id, state["state"], version, state["blast_radius"])

ledger = Ledger(repo_path, serialize_mine, deserialize_mine)
ledger.register(MyArtifact("my-art", initial_state))
```

### 4.3 An eval function for the verifier

Ground-truth scorer over a held-out task set:

```python
def my_eval(artifact, tasks) -> float:    # accuracy / reward in [0, 1]
    ...

verifier = ThreeLayerVerifier(eval_fn=my_eval, held_out=held_out_tasks)
```

### 4.4 A worker that proposes diffs

Workers turn observed failures into a `Diff` + `EvidenceCard`. The reference
`Worker` (agentdescent/worker.py) is a deterministic corrector; in a real system
this is where an LLM reflects on a trajectory and proposes an edit. Emit a card
with the `base_version` you read, the `touched` artifacts, and a local
`before_after_delta`, then `aggregator.ingest(card)`.

Once these four pieces exist, everything else — the Ledger, the aggregator, the
schedulers, the governance layers, both runtimes, and all three parallel
paradigms — works unchanged.

---

## 5. Building the documentation site

These docs render as a website via [MkDocs](https://www.mkdocs.org/):

```bash
pip install -e ".[docs]"
mkdocs serve      # live preview at http://127.0.0.1:8000
mkdocs build      # static HTML into ./site
```
