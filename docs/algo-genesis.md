---
description: EvoX Genesis (persistent recursive worlds) as pluggable AgentDescent policies - w=(v,p), recursive delegation, the octopus merge and the parent's verdict, measured on a compact formation run.
---

# Genesis — Persistent Recursive Worlds

> **A software world that persists while its agents do not.** An agent is
> situated by an accepted version *and* a repository path; delegation moves the
> path, only acceptance moves the version. Runs through
> [`evolve()`](evolution.md) as one `Strategy` and three `Policies` fields, with
> **no engine change**. Example:
> [`examples/genesis/genesis_recursive_worlds.py`](https://github.com/Birfy/agentdescent/blob/main/examples/genesis/genesis_recursive_worlds.py).

| | |
|---|---|
| **Paper** | *Persistent Recursive Worlds Enable Autonomous Software Evolution* — Beichen Huang, Zhenyu Liang, Bowen Zheng, Ran Cheng, 2026 ([arXiv:2608.10450](https://arxiv.org/abs/2608.10450)) |
| **Upstream code** | [`EMI-Group/genesis`](https://github.com/EMI-Group/genesis) @ v0.12.6 (Elixir, AGPL-3.0) |
| **Example** | [`examples/genesis/genesis_recursive_worlds.py`](https://github.com/Birfy/agentdescent/blob/main/examples/genesis/genesis_recursive_worlds.py) |
| **Domain** | A compact formation run: a language toolchain grown from an implementation-empty repository against a frozen staged suite |
| **Layer** | L2 (`blast_radius=0.2`) — the evaluator is outside the artifact |
| **Fidelity** | `mechanism_microport` — [what the classes mean](port-fidelity.md) |

## The mechanism

The paper's model is four objects and two operations.

A **local software world** is a pair

    w = (v, p)

of an accepted version and a repository-relative path. `v` fixes the whole
inheritable project; `p` says where an agent starts and what it is answerable
for. A finite-lived agent produces a candidate `Δᵢ = Aᵢ((v,p), gᵢ)` and then
ceases to exist — its conversation is not the identity of any later agent.

The two operations differ in exactly one way:

| operation | version | path |
|---|---|---|
| recursive delegation `(v,p) ⇝ (v,q)` | **unchanged** | moves to `q` |
| accepted event `(v,p) → (v′,p′)` | **advances** | stays, or is explicitly mapped |

Managers decompose and judge; leaf executors write files. A parent merges what
its children return with `git merge --octopus` and resolves a real conflict
itself. Only what the parent accepts is offered to the version history; a
refused change leaves nothing behind except, where the refusal was recorded in
`CONTEXT.md`, its reason.

Three rules keep that recursion honest, and each is upstream's:

* **A node is a directory.** Delegating to a file is refused — a path is a node
  when it can hold children.
* **An agent writes only inside its own path**, and a change it needs *outside*
  it is **reported upward**, not made: the ancestor with authority there handles
  it, under its own name. Upstream says this in as many words — *"report the need
  back up to your parent agent, which will handle it"* (`agents/executor.ex:64`),
  and the manager prompt explains it is the spatial contract, not an exception to
  it.
* **A sibling touching the same file is a merge, not a fault.** One child may sit
  inside another's subtree, so both may legally write one path; the parent
  three-way merges them and only a real overlap goes back as re-planning, which
  is what `git merge --octopus` plus the conflict-file list does upstream.

`CONTEXT.md` is not documentation in this system, and the port treats it the same
way. It is part of `v`, so a later agent inherits it; the chain from the root
down to `p` is what an entering agent is *given*; and its **routing table is
where a manager may delegate**. A node its parent does not route to is a node
later agents cannot find, so a manager that opens one writes the entry at its own
level — which is a write to the accepted version like any other, gated like any
other.

## How it plugs into `evolve()`

Every piece is a seam the engine already had. Nothing in `agentdescent/` changed.

| Plug-in | `evolve()` slot | What it does |
|---|---|---|
| [`SpatialContract`](https://github.com/Birfy/agentdescent/blob/main/examples/genesis/_spatial.py) | `strategy=` | A directory, one key per path. An edit carries the path of the agent that made it, and an edit outside that subtree is dropped and **counted**. |
| [`RecursiveDelegation`](https://github.com/Birfy/agentdescent/blob/main/examples/genesis/_delegation.py) | `Policies(proposal=)` | One rollout is a whole episode tree at **one** version: manager → children → leaf executors → the parent's verdict → the merged edit set. |
| [`OctopusConflict`](https://github.com/Birfy/agentdescent/blob/main/examples/genesis/_octopus.py) | `Policies(conflict=)` | Three-way merges the contested values, so two agents editing two functions of one file both survive. Real overlaps fall through to the shipped rule. |
| [`ParentJudge`](https://github.com/Birfy/agentdescent/blob/main/examples/genesis/_judge.py) | `Policies(acceptance=)` | Upstream's monotone rule: *partial progress is accepted*; a regression is refused; a tie commits and sends more work to that subtree. |
| [`suite_review`](https://github.com/Birfy/agentdescent/blob/main/examples/genesis/_domain.py) | `RecursiveDelegation(review=)` | The parent's own test run on **one child's** contribution, inside the episode — the tests and integration evidence of paper §3.3, which the acceptance gate cannot see because it only ever sees what the whole episode returned. |
| [`LocalWorld` / `WorldLog`](https://github.com/Birfy/agentdescent/blob/main/examples/genesis/_world.py) | — | `(v,p)` itself; the `CONTEXT.md` chain an entering agent is given; `routing()`, the table that decides where a manager may delegate; and the archive that outlives the agents. |

`--keyed-union` and `--engine-gate` turn the third and fourth rows back into the
engine's own defaults, which is how the rows below were measured.

## Measured results — compact formation domain

Offline rule-based actors, `--episodes 96`, three seeds, one machine. The
repository starts at **0.000** with five `CONTEXT.md` files and no implementation;
held-out reward is the frozen staged suite (12 of 30 cases held out).

| arm | held-out reward | accepted events | agent episodes | observed depth | wall-clock |
|---|---|---|---|---|---|
| `--serial` (upstream semantics, 1 worker) | **1.000** ×3 | 6 | 18 | 2 | 4.6 s |
| 4 workers | **1.000** ×3 | 6 | 55 | 2 | 2.7 s |
| 8 workers | **1.000** ×3 | **4** | 98 | 2 | 2.7 s |

Two things in that table are the port's own findings rather than restatements of
the paper.

**The three-way merge buys accepted events, not quality.** At eight workers the
run reaches the same 1.000 in **four** accepted events instead of six, on every
seed, and the counter says why: `merged=1`. Two agents fixed two different node
kinds in one file in one round. Under `--keyed-union` — the engine's rule, where
"same key" means "same file" — the same three seeds take **6** commits and
`merged` is not available at all. That is the README's headline comparison at a
finer granularity: a keyed union cannot fuse an intra-file edit, and a textual
three-way can.

**The engine's statistical gate cannot get a formation run started.** Same budget,
same everything, `--engine-gate`:

| seed | held-out reward | accepted | refused |
|---|---|---|---|
| 0 | 0.333 | 3 | 9 |
| 1 | 0.750 | 4 | 8 |
| 2 | **0.000** | 1 | 11 |

The reason is structural rather than statistical. The first files of an empty
repository are structure — a package marker, an entry point — and structure moves
no test case. A gate that requires a measured improvement refuses them, so the
run never reaches the steps that *would* have improved. Upstream's rule accepts
them and keeps going, which is what "partial progress is accepted" is for. This is
the mirror image of ACE's departure on [the fidelity page](port-fidelity.md): there
the engine's gate emptied an artifact whose claim was accumulation; here it stops
one from being built at all.

!!! warning "What these numbers are not"
    Upstream's formation run is **123.4 hours, US$44.38, 248,989 lines and one
    sample**; its continuation and MESA-redevelopment runs are one sample each.
    None of that is reproduced here and none of it is claimed. The domain is a
    stand-in sized to finish offline, the actors are rule-based, and the
    wall-clock column is dominated by child processes rather than model calls —
    so the speedup is a property of this domain, not a result about Genesis.
    What the rows support is that the **mechanism** runs, and how it differs from
    the engine's defaults when it does.

## What a real-model run found that the offline arm could not

The offline actors are rule-based, so they never mistake a file for a node and
never write outside their subtree. A run against `deepseek-v4-flash` through an
Anthropic-shaped endpoint did both within forty episodes, and turned up two
defects that no offline test could have reached.

**A node is a directory, and nothing said so.** A manager delegated to
`src/frontend/lexer.py` as though it were a node. The child was situated there,
wrote `lexer.py/CONTEXT.md` and `lexer.py/__init__.py` *inside* it, and every
stage accepted it: the key space is a flat dict, so `a/b.py` and `a/b.py/c.py`
coexist happily until something tries to put them on a filesystem. The run died
in `EvolutionResult.write_to` with `FileExistsError` after 328 model calls — the
whole thing lost at the one point where it was being saved. Two rules now stop
it, and they are separate on purpose because they need opposite fixes:
`RecursiveDelegation` refuses a delegation whose target is an existing file
(`mistaken_nodes`), and `SpatialContract` refuses any edit that would make the
key space stop being a tree (`shape_violations`).

**The manager prompt never taught what a node is.** With the rule in place the
refusals were counted rather than fatal — 32 of them in one run — which is the
counter doing its job and the prompt not doing its own. It now states that a node
is a directory, lists what the node's routing table actually routes to, and says
that naming a new child adds it to that table.

**Two more mechanisms were missing, and the model found them too.** With the node
rule in place, one run logged 70 refused children and 17 dropped edits: executors
kept producing changes outside their own path — `src/frontend` wanting to write
`src/__init__.py`. Upstream that is not an overstep, it is a **report**: *"report
the need back up to your parent agent, which will handle it"*
(`agents/executor.ex:64`), and the manager prompt says in as many words that this
*is* the spatial contract rather than an exception to it. The port had no such
channel, so every one of those needs died as a refusal. It has one now — the
ancestor with authority over the path makes the change, under its own name — and
the same run goes to **0 refusals and 0 dropped edits, with 53 of 99 requests
handled**. The other was the sibling case: a child situated inside another's
subtree may legally write the same file, and rejecting on that collision threw
away a whole contribution for touching a file a sibling also touched. That is now
a three-way merge, like every other concurrent edit here.

This is the general point about the offline arm, stated in one place: it
exercises the mechanism and it cannot exercise the mechanism's *failure* paths,
because a rule-based actor does not fail that way. Every counter on the world line
exists because a model does.

**Where the model arm stands, without dressing it up.** Across four runs of
`deepseek-v4-flash` at 40–60 episodes, held-out reward stayed at **0.000**. The
mechanism is visibly running — observed depth 3, real refusals, real merges, real
upward requests — and the toolchain still does not come out working. The current
failure mode is the actor, not the harness: executors ask for files at paths the
project does not use (`lexer.py` at the repository root rather than under `src/`),
and since the root *does* have authority there, the request is granted and the
tree grows a second, wrong copy. A parent that "handles" a request by applying it
is the weakest reading of upstream's rule; a parent that re-delegates it to the
node that owns it is the next thing to try. Reported here rather than tuned away,
because a port page that showed only the offline 1.000 would be describing a run
nobody made.

## Honesty boundary

`--offline` — the default — proposes with rule-based actors that reveal
pre-written module implementations one step at a time, the same device as
[DGM](algo-dgm.md)'s surrogate objective and the [molecule search](porous-molecules.md)'s
offline operators. It exercises the port's mechanism and says **nothing** about a
model's ability to write software. `--model` supplies manager and executor agents
that are asked instead of computed.

Upstream's own task-level acceptance is a **human action**: `merge_and_report/4`
never merges, it only creates a `genesis/agent_<hex>` branch, and the merge is a
button on the dashboard (`EvoGit.Review.merge_branch/2,3`). The judging this port
automates is the one that runs *inside* an episode — a parent on its child — plus
the task-level gate, which is the aggregator. A reader who takes "123 hours
autonomous" to mean "no human in the accept path" is reading upstream wrong, and
so would a reader of this page without this paragraph.

## Recorded deviations

* **The benchmark is not reproduced.** See the warning above. Fidelity class is
  `mechanism_microport` for that reason and no other — the mechanism maps cleanly.
* **Tensor parallelism is refused rather than used.** TP is the engine's own
  "each worker owns a disjoint section, conflict-free by construction", which is
  word for word the spatial contract. It cannot be used here: its ownership map is
  built once before round 0 from a declared key space, and a key outside it
  belongs to no section — so **every newly created file is a `section-violation`**
  (`evolution.py:2739`; `FileTree.keys` says so itself). Formation is nothing but
  creating files. `SpatialContract` therefore declares no key space, which makes
  `evolve()` *refuse* TP with a message naming the reason instead of silently
  discarding the run's entire output, and enforces the contract in `to_diff`,
  where creation is free.
* **The world is L2, not L1.** The audit gate runs *before* acceptance and, above
  `FAST_MAX`, vetoes every candidate that does not strictly improve — which
  contradicts "partial progress is accepted" outright, and at L1 this domain
  commits nothing at all. L2 is honest here because the agents cannot reach the
  evaluator: the suite, its expectations and the harness that runs them live
  outside the artifact, and `spec/**` is refused to every proposal and restored
  pristine before scoring. Governance is deliberately not a seam, so this is
  recorded as a boundary rather than routed around.
* **"Request more work" is reconstructed.** `AcceptDecision` is a boolean whose
  `category` is a closed vocabulary — the aggregator turns it into a
  `MergeOutcome`, so an invented name raises inside the merge. A parent's *not
  yet* therefore commits the partial step and leaves a request in the `WorldLog`
  that the next round's manager re-delegates from, rather than becoming a third
  engine outcome.
* **Observed depth is 2 here, 4–8 upstream**, because the domain's decomposition
  is three nodes deep. The number reported is always the depth *observed*, never
  the depth configured.
* **Multi-repository work (`foreign_repos`), the Tauri desktop shell, the Phoenix
  dashboard and peak-hour scheduling are out of scope.**
* **`CONTEXT.md` is inherited and routed from, but only partly maintained.**
  The read half is faithful: the chain is assembled root-first exactly as
  `ContextNode.build_context/2` does, and the routing table is load-bearing —
  the offline manager's decomposition is *parsed from it*, not hardcoded, so
  editing the table changes where the run delegates (there is a test that does
  exactly that). The write half is not: upstream every `:read_write` agent keeps
  its node current — intent, API surface, known issues — and the read-only roles
  (`Investigator`, `ContextExtractor`) exist to do nothing else. Here an agent
  writes `CONTEXT.md` on its own in exactly two cases, the routing entry for a
  node it opened and the refusal note for a child it rejected. An LLM executor
  may write more; nothing requires it to.
* **Skills are inherited but never extracted.** `LocalWorld.skills()` collects
  `.agents/skills/` along the node chain and puts the *names* in the brief, which
  is what `hierarchical_skill_names/2` does upstream and what the paper means by
  listing "reusable skills" among what an accepted version carries (§3.1). What
  is missing is the other end: upstream's `SkillExtractor` distils a completed
  contribution into a new skill, and nothing here does. The domain ships one
  human-written skill so the inheritance path is live rather than decorative.
* **Only two roles are ported.** The released code has ten agent modules; the
  paper's appendix §1.3 says the model has two — manager and leaf executor — and
  that "codebase lead / investigator / task scheduler are implementation labels,
  not additional roles". Porting ten would be porting an implementation.

## Run it

```bash
python -m examples.genesis.genesis_recursive_worlds                            # offline, no API key
python -m examples.genesis.genesis_recursive_worlds --episodes 96 --workers 8
python -m examples.genesis.genesis_recursive_worlds --episodes 96 --serial     # upstream semantics
python -m examples.genesis.genesis_recursive_worlds --keyed-union              # control: engine merge
python -m examples.genesis.genesis_recursive_worlds --engine-gate              # control: engine gate
python -m examples.genesis.genesis_recursive_worlds --model claude-haiku-4-5   # real agents
python -m examples.genesis.genesis_recursive_worlds --write-repo /tmp/world    # keep what it grew
```

`--dry-run` prints the plan with zero network access and no API key.

Offline tests:
[`tests/test_genesis_example.py`](https://github.com/Birfy/agentdescent/blob/main/tests/test_genesis_example.py).
