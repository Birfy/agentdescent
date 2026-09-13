# EvoX Genesis — persistent recursive worlds

Port of the released implementation onto the AgentDescent engine, as pluggable
policies rather than a fork of the loop.

| | |
|---|---|
| Kind | Software world (a directory that grows) |
| Governance layer | L2 (`blast_radius=0.2`) — the agents cannot reach the evaluator |
| Paper | "Persistent Recursive Worlds Enable Autonomous Software Evolution", Beichen Huang et al., 2026 ([arXiv:2608.10450](https://arxiv.org/abs/2608.10450)) |
| Upstream code | https://github.com/EMI-Group/genesis @ v0.12.6 (Elixir, AGPL-3.0) |
| Domains | Four compact formation runs — **not** upstream's 123.4 h C-compiler benchmark |
| `evolve()` plug-ins | `strategy` + `Policies(proposal=, acceptance=, conflict=)` |
| Fidelity | `mechanism_microport` |

## Run

```bash
python -m examples.genesis.genesis_recursive_worlds                       # offline, no API key
python -m examples.genesis.genesis_recursive_worlds --domain stackvm      # the deeper world
python -m examples.genesis.genesis_recursive_worlds --domain jqx          # a program, not a package
python -m examples.genesis.genesis_recursive_worlds --domain md           # scored by tests, no oracle
python -m examples.genesis.genesis_recursive_worlds --episodes 96 --workers 8
python -m examples.genesis.genesis_recursive_worlds --keyed-union         # control: no three-way merge
python -m examples.genesis.genesis_recursive_worlds --engine-gate         # control: the Beta gate
python -m examples.genesis.genesis_recursive_worlds --model claude-haiku-4-5   # real agents
```

`--dry-run` prints the configuration and returns with **zero network access and
no API key**. The default run needs no key either: the manager and executor are
rule-based offline actors, which exercise the mechanism and say nothing about a
model's ability to write code.

## The mechanism, as four components

The paper's model is `w = (v, p)` — an accepted version and a repository path —
with two operations that differ in exactly one way: recursive delegation moves
the path, an accepted event moves the version. Each piece is one file here.

| file | what it is | upstream |
|---|---|---|
| [`_world.py`](_world.py) | `LocalWorld(v, p)`, the `CONTEXT.md` chain **and its routing table**, the episode archive | paper §3.1, `core/context_node.ex` |
| [`_delegation.py`](_delegation.py) | `ProposalPolicy`: one rollout is a whole episode tree at one version | paper §3.2, `agent/subagent_processing.ex` |
| [`_spatial.py`](_spatial.py) | `Strategy`: an agent writes only inside its own subtree | `agents/manager.ex:179` |
| [`_octopus.py`](_octopus.py) | `ConflictPolicy`: three-way merge, so two agents in one file both survive | `Git.merge_octopus/2` |
| [`_judge.py`](_judge.py) | `AcceptancePolicy`: the parent's rule — *partial progress is accepted* | `agents/manager.ex:58` |
| [`_suite.py`](_suite.py) | the machinery a domain needs and does not own: the harness, both loaders (`Suite` scores against a reference, `TestSuite` against a frozen test suite with no reference in the loop), the runner, the parent's integration check, the LLM actors | — |
| [`_domain.py`](_domain.py) | **minilang** — an integer expression language, 2 nodes deep, 4 files | — |
| [`_stackvm.py`](_stackvm.py) | **stackvm** — a stack machine and its assembler, 4 nodes deep, 10 files | — |
| [`_jqx.py`](_jqx.py) | **jqx** — a JSON query tool whose command-line entry point is frozen, 4 nodes, 9 files | — |
| [`_md.py`](_md.py) | **md** — Lennard-Jones molecular dynamics under a frozen driver, 6 nodes, 14 files, scored by a frozen test suite and audited by tests that are not in the repository | — |

Nothing in `agentdescent/` changed.

## What is in here

- [`genesis_recursive_worlds.py`](genesis_recursive_worlds.py) — the runnable port
- Port notes, upstream trace, every recorded deviation and the measured results:
  [`docs/algo-genesis.md`](../../docs/algo-genesis.md)
- Offline tests: [`tests/test_genesis_example.py`](../../tests/test_genesis_example.py)

The shared command-line contract (`--provider/--model/--seed/--async/--async-ratio/--max-seconds/--dry-run/--yes/--serial`)
lives in [`examples/_common.py`](../_common.py) and is enforced by
[`tests/test_example_entrypoints.py`](../../tests/test_example_entrypoints.py).
