# Install, run, extend

How to install, run the demos, tune the knobs, and — most importantly — plug in
your own `Evolvable` domain.

---

## 1. Install

```bash
git clone https://github.com/Birfy/agentdescent
cd agentdescent
pip install -e ".[dev]"
```

No external services or model APIs are needed. Requires Python ≥ 3.9.

---

## 2. Run the demos

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

### Self-evolution algorithm ports (real datasets)

Faithful ports of the latest skill- and harness-self-evolution algorithms — ACE,
GEPA, EvoSkill, SkillOpt, ADAS, DGM (see
[the catalog](self-evolution-examples.md)). Each loads a real benchmark through
the [`agentdescent.dataloader`](dataloader.md) data layer and runs offline with
`--dry-run`:

```bash
python -m examples.ace_context_evolution --dry-run     # ACE   / FiNER-139
python -m examples.gepa_prompt_evolution --dry-run     # GEPA  / HotpotQA
python -m examples.dgm_self_improve                    # DGM   / SWE-bench Verified (offline surrogate)
```

### Tests

```bash
pytest            #  tests
pytest -q tests/test_async.py   # just the async runtime
```

---

## 3. Programmatic use

### The entry point — `evolve()`

Covered in full elsewhere, and not repeated here:

* **[Quickstart](quickstart-skill.md)** — a dataset to an evolved skill in one
  call, with the measured result of running it.
* **[The `evolve` method](evolution.md)** — the entry point underneath, with every
  parameter and what it plugs into.
* **[Connecting agents & LLMs](agents.md)** — any `prompt -> text` is a backend,
  including tool-using CLI agents.
* **[Measured results](results.md)** — every empirical claim with its setup.

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

## 4. Configuration reference

### `AggregatorConfig` (agentdescent/aggregator.py)

| Field | Default | Meaning |
|---|---|---|
| `batch_trigger` | 4 | `B`: fire a bucket once it holds this many cards |
| `max_wait_rounds` | 3 | `T_max`: fire a cold bucket after this many sweeps |
| `base_delta` | 0.5 | acceptance risk; threshold is `1 − δ`, annealed by version |
| `alpha_head` | 5 | staleness tolerance α for hot artifacts |
| `alpha_tail` | 1 | staleness tolerance α for cold artifacts |
| `trust_region_ops` | 6 | max edits per diff (trust region) |
| `promote_after_k` | 3 | dev→stable survival rounds (EMA) |

### `AsyncConfig` (agentdescent/async_runtime.py)

| Field | Default | Meaning |
|---|---|---|
| `n_workers` | 6 | worker threads |
| `async_ratio` | 3 | ROLL Flash lag budget: max head-drift before a worker refreshes |
| `noise` | 0.15 | fraction of workers that inject contradictory proposals |
| `target_accuracy` | 0.98 | stop early when the dev branch reaches this |
| `max_seconds` | 20.0 | wall-clock safety bound |
| `aggregator_interval` | 0.002 | sleep between aggregator sweeps |
| `worker_pause` | 0.001 | sleep between worker rollouts |
| `oracle_budget` | 400 | oracle calls the AuditScheduler may spend |
| `stall_patience` | 150 | no-commit sweeps before a backpressure sync |

`AsyncStats.error` is `None` on a clean run and carries the backend failure that ended it otherwise — check it, since a run whose workers all died otherwise returns normal-looking zeros.

### Staleness policy

```python
from agentdescent import get_policy
get_policy("full")        # accept stale diffs directly
get_policy("guarded")     # version-gated (default)
get_policy("reflective")  # always rebase + re-verify
```

---

## 5. Plug in your own domain

The whole framework is domain-agnostic: **what evolves is decided by
registration, not hard-coded.** To evolve something new, provide four things.

### 5.1 An `Evolvable`

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

### 5.2 Serialize / deserialize for the Ledger

The Ledger stores artifacts as JSON blobs in git, so give it two functions:

```python
def serialize_mine(a) -> dict:            # → JSON-friendly dict
    return {"state": a.state, "blast_radius": a.blast_radius}

def deserialize_mine(artifact_id, version, state) -> MyArtifact:
    return MyArtifact(artifact_id, state["state"], version, state["blast_radius"])

ledger = Ledger(repo_path, serialize_mine, deserialize_mine)
ledger.register(MyArtifact("my-art", initial_state))
```

### 5.3 An eval function for the verifier

Ground-truth scorer over a held-out task set:

```python
def my_eval(artifact, tasks) -> float:    # accuracy / reward in [0, 1]
    ...

verifier = ThreeLayerVerifier(eval_fn=my_eval, held_out=held_out_tasks)
```

### 5.4 A worker that proposes diffs

Workers turn observed failures into a `Diff` + `EvidenceCard`. The reference
`Worker` (agentdescent/worker.py) is a deterministic corrector; in a real system
this is where an LLM reflects on a trajectory and proposes an edit. Emit a card
with the `base_version` you read, the `touched` artifacts, and a local
`before_after_delta`, then `aggregator.ingest(card)`.

Once these four pieces exist, everything else — the Ledger, the aggregator, the
schedulers, the governance layers, both runtimes, and all three parallel
paradigms — works unchanged.

---

## 6. Building the documentation site

These docs render as a website via [MkDocs](https://www.mkdocs.org/):

```bash
pip install -e ".[docs]"
mkdocs serve      # live preview at http://127.0.0.1:8000
mkdocs build      # static HTML into ./site
```
