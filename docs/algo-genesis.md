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

### With a real model

`deepseek-v4-flash` through an Anthropic-shaped endpoint, `--episodes 60
--workers 4 --no-thinking`, three seeds. The repository starts implementation-empty
and the model writes every line of what comes out.

| seed | held-out | accepted events | agent episodes | observed depth | model calls | wall-clock |
|---|---|---|---|---|---|---|
| 0 | **1.000** | 5 | 34 | 3 | 284 | 135 s |
| 1 | **1.000** | 5 | — | 3 | 306 | 173 s |
| 2 | **1.000** | 5 | — | 2 | 242 | 100 s |

Seed 0's repository was re-scored independently from what `--write-repo` wrote:
**30/30 cases**. What it grew is `src/__init__.py` plus
`frontend/{lexer,parser}.py` and `backend/evaluator.py`, and the `__init__.py`
imports each stage lazily because the specification says to — so the agents read
the contract rather than guessing it.

Three defects of this port stood between a model and that result, each invisible
until the one before it was gone; they are the section below, and none of them was
the model.

### With the offline actors

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
    None of that is reproduced here and none of it is claimed.

    The **model** rows are three seeds, one model, one endpoint, one small domain.
    They support "this organization can take an implementation-empty repository to
    a working, spec-conformant one, and here is what it cost" — not any comparison
    with upstream's scale, and not a claim about models in general.

    The **offline** rows run rule-based actors, and their wall-clock is dominated
    by child processes rather than model calls, so the speedup column is a property
    of this domain. What they support is that the mechanism runs and how it differs
    from the engine's defaults when it does — including the failure paths a
    rule-based actor can never reach, which is why the section below exists.

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

**"Relative to what?" cost four files and a whole run.** The run above put
`lexer.py`, `parser.py`, `evaluator.py` and `__init__.py` at the **top of the
repository** while the suite stayed at 0.000, and the first reading of that was
"the model asks for paths the project does not use". It was not. The root
`__init__.py` it wrote says *"Frontend package: tokenizer and parser"* and
`from . import lexer`: the agent situated at `src/frontend` meant
`src/frontend/__init__.py` and wrote the path **relative to its own node**. The
protocol said "relative path" without saying relative to what, this port read it
as repository-relative, the files fell outside the agent's subtree, became
upward requests, and the root — which does have authority everywhere — granted
them at the top level. Every layer behaved exactly as designed.

Three defects of this port's own stood between a model and a working toolchain,
and the second and third were only visible once the first was gone.

**"Relative to what?"** Two fixes, because one alone is not enough. The protocol now names the agent's own
path in every example it shows and is rendered per episode rather than once, so
"relative" has one meaning. And `resolve_edit_path` resolves a path against the
node before anything else looks at it, ordered so that a genuine cross-node
request is never mangled into a nested copy of its own path: already inside the
node, or an existing file of the node's, means node-relative; a path whose first
segment is an existing top-level entry is repository-relative and therefore a
request about somebody else's node. Resolutions are **counted**
(`node_relative_paths`), because the alternative to counting is a file appearing
somewhere nobody asked for it. After it, every file landed where it belongs.

**The agents could not see the contract they were judged against.** The language
specification lives at `spec/CONTEXT.md` — a sibling of `src/`, so on nobody's
`CONTEXT.md` chain — and `situate()` handed an agent the chain, its own file
listing, and nothing else. So every agent inferred the whole language from one
failing input, and built a coherent toolchain with `('NUMBER', '1')` tokens and a
`parse(tokens)` signature against a specification that says `("num", 1)` and
`parse(source)`. Five correct files, every one to the wrong contract.

That was a straight misreading of the paper: *"An agent may inspect the complete
project represented by v, but it begins from p"* (§3.1). The chain is where an
agent **begins**, not a wall around what it may read — upstream it would open the
spec with a read tool, and an agent here has no tools, so the human-supplied
contract travels in the brief or it is invisible. `situate(contracts=)` now shows
it in full wherever the agent stands. The next run's lexer emitted
`[('num', 1), ('op', '+'), ('num', 2)]`.

**A manager that delegates was never asked to finish.** With the contract visible
and every path correct, the run produced five correct modules and **no
`src/__init__.py`** — the public surface the specification names did not exist, so
every case still scored zero, and the `src` node's own `CONTEXT.md` had told its
manager *"Own `src/__init__.py`"*. It delegated every time instead.

Upstream's Architect works in three phases — *architecture & design →
implementation delegation → **review & accountability*** — and is "ACCOUNTABLE for
all code in its node path" (`agents/architect.ex:23`). This port stopped after the
second. A manager now gets one turn at its own node **after** its children return,
seeing the tree as they left it, restricted to files directly at the node because a
manager free to rewrite its children's work would make the decomposition
decorative (`--no-accountability` turns it back into a pure router). `situate()`
also splits "files AT this node (yours to write)" from "files below it (each
child's own)", because an agent is bad at noticing an absence inside a long list
and the absence is the actionable part.

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
