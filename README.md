# AgentDescent

> **Gradient descent — but the parameters are agents.** A parallel, asynchronous
> framework for self-evolving agents (skills, prompts, harnesses) where **diffs
> are the gradients** and **the aggregator is the optimizer**.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22348027.svg)](https://doi.org/10.5281/zenodo.22348027)
[![PyPI](https://img.shields.io/pypi/v/agentdescent)](https://pypi.org/project/agentdescent/)
[![tests](https://github.com/Birfy/agentdescent/actions/workflows/tests.yml/badge.svg)](https://github.com/Birfy/agentdescent/actions/workflows/tests.yml)
[![docs](https://img.shields.io/badge/docs-mkdocs--material-1f6feb)](https://birfy.github.io/agentdescent/)
[![python](https://img.shields.io/badge/python-%E2%89%A53.9-1f6feb)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-3fb950)](https://github.com/Birfy/agentdescent/blob/main/LICENSE)

*N* workers propose edits to a shared artifact **in parallel**; a barrier-free
aggregator merges them into one version-controlled library. Serial
self-improvement is bounded at one accepted change per iteration, and merging
concurrent edits is the attempt to lift that bound.

The place the analogy has to break is the whole design: **gradients add, diffs do
not.** So aggregation is not averaging but conflict resolution, fusion,
statistical acceptance and a transactional commit.

## What it measures

Every row names the setting that produced it, and
[the full results page](https://birfy.github.io/agentdescent/results/) also
reports the runs where there was nothing left to learn.

| | measured | setting |
|---|---|---|
| An artifact held in **one key**, where a keyed union can never fuse | keyed union fuses **0 of 48** merges · reflective merge **42 of 48** | BBH `dyck_languages`, `GLM-5.2`, *N*=4, 4 seeds |
| What that costs, at a pinned rollout budget | **40% fewer model calls** (95% CI 27–54%, same direction on every seed) | the same runs |
| Wall-clock against a *faithful* serial control | median **6.8×** (three seeds, 3.1–9.5×) | GEPA on HotpotQA, *N*=4, 16 rollouts pinned |
| The one-call path, end to end | held-out exact match **0.167 → 0.583** | 40 HotpotQA items, 12 held out |

Quality is claimed only where the design can support it. The
[paper](https://doi.org/10.5281/zenodo.22348027) reports intervals rather than
*p*-values where a comparison cannot reach significance, and says so.

## How it works

![AgentDescent architecture: N workers roll out against ledger snapshots and emit
diffs with evidence cards into a buffer; a single aggregator thread runs the
five-stage merge pipeline and commits to a git-backed ledger; four pluggable
seams sit under the components they select; the L0 governance layer gates the
audit step.](docs/assets/architecture.png)

**Solid, top:** the fixed data path. *N* workers roll out tasks against ledger
snapshots and emit diffs with evidence cards; one aggregator thread runs the
five-stage merge and commits winners; the green edge is the only feedback path,
bounded by the lag budget.

**Dashed, bottom:** the seams, each selected by one keyword argument of
`evolve()`. **Red:** the governance layer, deliberately *not* a seam — a
self-modifying system must not be able to replace its own evaluator.

Stages 1–5 are one optimizer step over a discrete space. Whether that step has
anything to do at all is decided upstream, by the **key space** your strategy
writes — edits on disjoint keys fuse, edits on the same key conflict. That is
why the table above starts with a one-key artifact.

The figure is the paper's, rendered from its TikZ source by
[`tools/gen_architecture_figure.py`](tools/gen_architecture_figure.py) so it
cannot drift from what the paper shows.

## Install and run something in 30 seconds

```bash
pip install agentdescent
```

The core engine has **zero required dependencies** and needs only Python ≥ 3.9.
The examples are research artifacts kept outside the installed package — they
would otherwise squat the top-level `examples` name — so **clone the repo** to
run them:

```bash
git clone https://github.com/Birfy/agentdescent && cd agentdescent
pip install -e ".[dev]"
python -m examples.run_demo      # no API key, no network
```

```
round  dev_acc   stable  commit  fused  stale  confl  oracle
    0    0.604    0.000       1      1      0      0       0
    1    0.707    0.707       1      1      0      0       0
    2    1.000    1.000       1      1      2      0       0
    3    1.000    1.000       0      0      0      0       0
```

Three rounds commit, then the gate stops accepting because there is nothing left
to improve — `commit`, `fused`, `stale` and `confl` are the aggregator's own
counters, and every run prints them.

## Quickstart — a dataset to an evolved skill

One entry point, `evolve()`, and three building blocks that turn a dataset into
its arguments. The decisions that are actually yours — your data, how to score
it, which model — are the ones you still make.

```python
from agentdescent import SingleSlot, evolve, openai_compatible, reflector, scorer, tasks_from
from agentdescent.dataloader import hf_rows

rows = hf_rows("hotpotqa/hotpot_qa", "validation", config="distractor", limit=40)
model = openai_compatible(model="deepseek-v4-flash")

tasks = tasks_from(rows, prompt="question", gold="answer")     # rows -> Task objects
run = lambda skill, task: model(f"{skill}\n\n{task.prompt}")   # the skill meets the question

result = evolve(tasks, scorer("exact"), run=run, propose=reflector(model),
                strategy=SingleSlot(initial_value="You are a helpful assistant."),
                rounds=8, n_workers=8, max_concurrency=8, held_out_frac=0.3,
                patience=3, target_reward=0.98)

print(result.rendered)        # the skill it learned
print(result.final_reward)    # held-out reward
print(result.outcomes())      # why it went that way
```

That run is the last row of the table above. It learned *"Respond with only the
requested answer, omitting any extra explanation or restatement."*

<details>
<summary>The same thing without a dataset. Runnable as-is — no API key, no dependencies.</summary>

```python
from agentdescent import Task, evolve

tasks = [Task(id=f"t{i}", prompt=f"item {i}") for i in range(12)]

def reward(task, output):                  # must return [0, 1]
    return 1.0 if "2026" in output else 0.0

def run(rendered, task):                   # your solver
    return "answer" + (" 2026" if "year" in rendered else "")

def propose(rendered, task, output, reward):   # what to add on a failure
    return "always state the year"

result = evolve(tasks, reward, run=run, propose=propose,
                rounds=6, n_workers=3, max_concurrency=3)
print(result.rendered, result.final_reward, result.error)
```

Swap in a real model or agent by passing `agent=` instead of `run`/`propose`:

```python
from agentdescent import LLMAgent, claude, openai_compatible, claude_code

evolve(tasks, reward, agent=LLMAgent(claude(model="claude-haiku-4-5")))
evolve(tasks, reward, agent=LLMAgent(openai_compatible(model="deepseek-v4-flash")))
evolve(tasks, reward, agent=LLMAgent(claude_code()))     # Claude Code CLI
# ...or run barrier-free: evolve(..., asynchronous=True, async_ratio=3)
```

</details>

## What you can replace

Two seams carry the algorithm, and both are `typing.Protocol`s — nothing to
inherit from, and the contracts are re-derived from the engine's own call sites
by a test, so the published interface cannot drift from what runs.

**The strategy — what evolves.** Three methods over a flat `{key: value}` state.
Its key space is what decides whether concurrent proposals can fuse at all.

| strategy | the artifact is | concurrent proposals |
|---|---|---|
| `AppendRules` | a deduped list of lessons, keyed by content | almost always **fuse** |
| `KeyedRules(categories)` | one entry per named category | contradict within, fuse across |
| `FileTree(files)` | a directory, one key per path | contradict per file |
| `SingleSlot` | one value — a prompt, an instruction | always contradict |

**Eight policy slots — the rules of evolution.** Each is one field of a
`Policies` bundle, each defaults to the shipped rule, and filling one leaves the
other seven alone. A driver *refuses* a field it cannot honour rather than
ignoring it — a policy that installs but never runs is the one failure a caller
cannot detect from a completed run.

| | | | |
|---|---|---|---|
| `selection` | `task_sampler` | `proposal` | `staleness` |
| `conflict` | `fusion` | `acceptance` | `promotion` |

Mechanisms that need state the pipeline does not keep — an archive, per-instance
score rows, an island pool — take the `aggregator_factory` exit instead.
[How to fill a slot →](https://birfy.github.io/agentdescent/policy-guide/)

## Nineteen algorithm ports

Published self-evolution algorithms run as plug-ins rather than forks, under
serial, synchronous or barrier-free scheduling without touching the engine —
which is what makes the scheduler a controlled variable instead of a property of
each paper's own loop.

**Benchmark-faithful (8):** ACE · GEPA · EvoSkill · SkillOpt · ADAS · DGM ·
OpenEvolve · ERA
**Microports and analogues (11):** PromptBreeder · AFlow · Self-Refine ·
Reflexion · SICA · Gödel Agent · Voyager · SkillWeaver · Absolute Zero · R-Zero ·
Agent0

Fidelity is declared per port, follows each project's *released code* where it
diverges from its paper, and analogues are not to be cited as benchmark
reproductions. Every port has a `--dry-run` mode that needs no API key.
[All nineteen, with their measured results →](https://birfy.github.io/agentdescent/self-evolution-examples/)

## Documentation

Full docs render at **[birfy.github.io/agentdescent](https://birfy.github.io/agentdescent/)**.

| Start here | |
|---|---|
| [Quickstart — dataset to skill](https://birfy.github.io/agentdescent/quickstart-skill/) | One call: your data, how to score it, which model |
| [Measured results](https://birfy.github.io/agentdescent/results/) | Every empirical claim with the setup that produced it |
| [Architecture](https://birfy.github.io/agentdescent/architecture/) | Components, data flow, the two runtimes |
| [Concepts](https://birfy.github.io/agentdescent/concepts/) | The training↔RSI analogy, staleness, governance |

| Going further | |
|---|---|
| [Evolving anything](https://birfy.github.io/agentdescent/evolution/) · [Strategies](https://birfy.github.io/agentdescent/strategies/) | Evolve any artifact by writing its `Strategy` |
| [Choosing policies](https://birfy.github.io/agentdescent/policies/) · [Using the slots](https://birfy.github.io/agentdescent/policy-guide/) | The decision plane, and how to write for it |
| [Agents & LLMs](https://birfy.github.io/agentdescent/agents/) · [Datasets](https://birfy.github.io/agentdescent/dataloader/) | The provider layer and the data layer |
| [Parallelism](https://birfy.github.io/agentdescent/parallelism/) · [Execution](https://birfy.github.io/agentdescent/execution/) · [Sandboxes](https://birfy.github.io/agentdescent/sandboxes/) | Where rollouts run, and how they are isolated |
| [Efficiency](https://birfy.github.io/agentdescent/efficiency/) · [Runtime matrix](https://birfy.github.io/agentdescent/matrix-overview/) | Measured scaling and the scheduler comparison |

## Citing

The paper — *AgentDescent: Asynchronous Parallel Self-Evolution of LLM Agents by
Merging Conflicting Edits* — gives the design, a closed-form condition for when
merging can fire at all, and a live-model evaluation with every generating script
named.

```bibtex
@article{chen2026agentdescent,
  title  = {{AgentDescent}: Asynchronous Parallel Self-Evolution of {LLM} Agents
            by Merging Conflicting Edits},
  author = {Chen, Danyang},
  year   = {2026},
  doi    = {10.5281/zenodo.22348027}
}
```

LaTeX source and the built PDF live on the
[`paper`](https://github.com/Birfy/agentdescent/tree/paper/paper) branch, not
here: the write-up and the code change on different clocks. Per-result raw data
is in [`bench/results/`](bench/results/) for most results; the
merge-versus-selection sweeps are re-measurable from the command the paper names
rather than recomputable, because their per-run data was not retained — the paper
says so too.

## Scope

A **research reference implementation**, not a production system. The novelty is
a narrow engineering synthesis — concurrent, staleness-bounded,
conflict-resolved diff-level merge over a git-backed versioned ledger — and its
throughput premise is a testable hypothesis rather than community consensus.
Contributions welcome: [CONTRIBUTING.md](CONTRIBUTING.md). The suite is offline
and deterministic, so `pytest -q` needs no API key.
