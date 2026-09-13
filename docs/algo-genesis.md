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
| **Domains** | Two compact formation runs, `--domain`: **minilang** (an integer expression language, 2 nodes deep, 4 files) and **stackvm** (a stack machine and its assembler, 4 nodes deep, 10 files), each grown from an implementation-empty repository against a frozen staged suite |
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
| [`Suite.review`](https://github.com/Birfy/agentdescent/blob/main/examples/genesis/_suite.py) | `RecursiveDelegation(review=)` | The parent's own test run on **one child's** contribution, inside the episode — the tests and integration evidence of paper §3.3, which the acceptance gate cannot see because it only ever sees what the whole episode returned. |
| [`LocalWorld` / `WorldLog`](https://github.com/Birfy/agentdescent/blob/main/examples/genesis/_world.py) | — | `(v,p)` itself; the `CONTEXT.md` chain an entering agent is given; `routing()`, the table that decides where a manager may delegate; and the archive that outlives the agents. |

`--keyed-union` and `--engine-gate` turn the third and fourth rows back into the
engine's own defaults, which is how the rows below were measured.

## Four domains, and what each one answers

`--domain minilang` is the original: a lexer, a parser and an evaluator under
`src/frontend` and `src/backend`, two nodes deep, four files for the agents to
write. It exercises every mechanism, and its recursion bottoms out at depth 2 —
which is fine for a merge comparison and thin for a system whose whole claim is
recursive organisation.

`--domain stackvm` grows a stack machine and its assembler: labels the assembler
resolves to instruction indices, a dispatch table that depends on three sibling
modules, and a loop in the suite. `src/vm/ops` is a node whose **parent is itself a
child**, so an episode reaches depth 3 on the way to a leaf, and `arith.py` holds
one independently-fillable function per opcode — so two agents working from two
different failing programs edit two different parts of one file, which is the case
a keyed union cannot fuse and a three-way merge can.

`--domain jqx` grows a JSON query language whose **command-line entry point is
frozen beside the specification**. The agents never write `jqx.py`; they write the
library it imports. A finished run is therefore a program you can run, not a
package nobody can invoke — and the difference showed up immediately as a class of
bug the other two domains cannot have, because a library that satisfies every case
can still be unusable from a shell.

`--domain md` grows Lennard-Jones molecular dynamics in reduced units — geometry,
two pair potentials, velocity Verlet, thermodynamic observables, a pair-distance
histogram — under a frozen driver that writes XYZ trajectories and an ASCII `g(r)`.
It answers the thing the first three cannot, and it is a change of kind rather than
of size: **there is no oracle in its scoring path**.

### Two ways to score a formation run, and only one of them is honest about it

The first three domains are scored by [`Suite`](https://github.com/Birfy/agentdescent/blob/main/examples/genesis/_suite.py):
a human writes a reference implementation, the loader runs it to compute the
expected answer for every case, and a candidate is scored by matching. It is cheap,
it is exact, and it is circular — you cannot ask a system to grow software you had
to write first. It also shapes what can be asked: an expected-output line cannot
carry a tolerance, an invariant, or "these two ways of computing this must agree".

`md` uses `TestSuite` instead. A human writes a specification and a **test suite**,
which is what a human actually writes. One task is one test function, a task's
prompt is that test's own source — prelude included, so the agent sees what
`CONFIG` is — and the reward is whether it passes. Nothing is compared against a
reference. This is also what upstream does: Genesis validates against c-testsuite,
LLVM and Csmith, which are assertions, with no reference compiler anywhere in the
loop.

What that buys is assertions an oracle cannot make. Many of `md`'s tests are
**invariants**: the forces sum to zero, every force component matches a central
difference of the energy, energy and momentum survive a trajectory, reversing the
velocities retraces it. One of them asserts that a far-too-large timestep does
*not* conserve energy, so an implementation that fakes conservation fails. Euler
with a single force evaluation is stable, plausible, and caught — `tests/test_genesis_example.py`
breaks the integrator that way and checks the suite notices, because a sampled
position from a bounded trajectory often agrees to six decimal places.

It also buys a trap, and a real run walked into it. **A field that returns zero
everywhere satisfies every invariant above**: zero sums to zero, zero is the
gradient of a constant, nothing moves so nothing drifts, and reversing nothing
retraces it. A run shipped exactly that — `d -= box * (d / box + 0.5) // 1`, where
the `// 1` binds after the multiplication, computing `floor(d + L/2)` instead of
`L * floor(d/L + 0.5)` and pushing every pair in a periodic box past the cutoff —
and 42 of the then 44 tests passed. Two noticed: the harmonic gradient, because
harmonic has no cutoff, and the negative control, whose whole docstring is *the
conservation tests must be able to fail, or they assert nothing*.

And a suite is only as good as **how it is run**. The finished md run reported
`audit reward 1.000`, and its repository contains this:

```python
# src/observe/__init__.py
def rdf(positions, box, bins, rmax):
    from .rdf import histogram      # importing the submodule rebinds
    ...                             # src.observe.rdf from this function to
                                    # the module: it destroys itself
```

The first call in a process works and every call after it raises
`TypeError: 'module' object is not callable`. Scored one test per process — which is
what one-task-per-test requires — it is perfect. Run the way a person runs a suite, in
one interpreter, five tests fail. Upstream has no such blind spot: its manager runs
`mix test` and its executor is told to "run ALL tests". So `TestSuite.suite_failures`
now runs the whole suite in one process, the run reports it beside the per-task
number, and `--complete-task` uses it as its precondition rather than the per-task
score. `SUITE_ONLY_BASELINES` keeps that exact code as a baseline: 65/65 per test,
60/65 in one process, and a test asserts both halves.

The lesson is not "add more invariants". A suite of invariants needs **value
anchors**, and this one was missing the ones that matter: not a single driven test
pinned a nonzero number in a periodic box. It does now — a pair in mid-box and a
pair across the seam against the closed form, a large box against no box, and an
explicit "this field is not identically zero".

And the mechanism, rather than my having noticed: `TestSuite.baselines` holds
**deliberately wrong implementations**, and a test requires every one of them to fail
the suite — to at least two tests each, because the zero field died to exactly one
and that one was the negative control. Six for md: the zero field, the real
precedence bug, `round` for `floor`, an unshifted Lennard-Jones, Euler dressed as
Verlet, and a centre-of-mass-corrected temperature. Writing it found two further
holes in the same sitting: the `round` geometry — *the one thing the specification
explicitly forbids* — passed the entire suite, and the temperature requirement hung
on a single test. It is mutation testing in its small form, and note what it does not
need: no reference in the **scoring** path. The reference sits beside the domain for
jobs that are not scoring, and suite QA is one of them. Coverage would have caught
none of it; the zero field executes every line of every test.

A reference implementation still exists next to the domain, for two jobs that are
not scoring: driving the offline rule-based actor, and letting a test prove the
suite is passable at all. Shipping a specification nobody has ever seen satisfied
is its own kind of dishonesty.

### The held-out tail is not a validation split

`evolve()` holds out the tail of the task list and reports its reward as
`final_reward`. For the oracle domains that is a generalisation estimate and it
means something: the cases are sampled inputs over one grammar, and passing inputs
you were not shown is the claim.

For a test suite it is a category error, and an expensive one. Every task is a
distinct **requirement**, so holding back 40% of them means refusing to tell the
system four tenths of what it has to do and then grading it on them. Worse, the
search never sees those requirements fail, so nothing is ever proposed for them.
That is exactly how the `jqx` run stalled at 0.923 with `stop reason: rounds`: the
case list was ordered by filter, two builtins appeared only in the held-out tail,
and no rollout could ever fail on them. A test guards the property now, for every
domain, and `TestSuite` interleaves its tasks across files so a positional split
cannot quietly hide a file.

Dropping the split entirely is also wrong, though, and for a reason specific to
this port: the executor is handed **the source of the test it is failing**, so it
could satisfy one assertion at a time without the physics underneath holding
together. So `md` splits the two jobs the single knob was doing:

* every **driven** test is part of the specification and every one of them drives
  the search — `HELD_OUT_FRAC` is *computed* so that `evolve()`'s positional cut
  lands exactly on the boundary;
* the tail is an **audit set** of thirteen further tests that are **not in the
  repository at all** — `frozen` stops a file being written, not read, and an audit
  test in the tree is one the executor can read and write code against. They are
  injected into the scratch copy only while a held-out task is being scored: a
  different box shape, a brute-force search over periodic images, a harmonic
  oscillator's period, the structure layer checked against the geometry layer.

`audit reward` in the run header is therefore the honest headline number — the
score on tests nothing that wrote the code has ever seen.

### What the human supplies, and what the run has to invent

A formation domain starts from fifteen files and not one line of implementation.
That number deserves a challenge, and it got one. Sorted by who the author should
be:

| what | why a human writes it |
|---|---|
| `spec/CONTEXT.md` | the goal, in prose and formulas. Unavoidable: something has to say what to build. |
| `tests/**` | the scoring. Upstream's equivalents are c-testsuite, LLVM and Csmith — also human-supplied, also frozen. |
| `md.py` / `jqx.py` | a frozen entry point, so the result is a program rather than a package nobody can invoke. A choice, and a defensible one. |
| `CONTEXT.md` **per node, with routing tables** | …this one is not defensible. |

Six of md's fifteen files are `CONTEXT.md` records that name the nodes and what
each is for: `src/potentials/pair/ -> one module per pair interaction`. That is the
decomposition, and the paper's first phase is *architecture and design* — so a tree
that ships with it has had its architecture handed to it. The machinery to grow one
was always there and always measured (`routes_opened`: a manager may name a node
its table does not reach, and the table records it afterwards); there was simply
never a domain that started without one.

`--cold-start` takes it away. What is left is the goal, the contract, the suite and
the frozen driver — eight files for md — with the root's routing table emptied and
the skills removed, because a hint about how to lay out Python packages is a hint
about the shape of the answer. Nothing frozen is touched, so the scoring is
identical; what changes is that the run has to write its own `CONTEXT.md` chain as
it goes.

Two defects were hiding behind the scaffolding, both of them invisible while every
domain shipped a table at the root:

* `routing()` falls back to the sub-directories that exist when a node has no
  table. Cold-started, the root's only sub-directories are `spec/` and `tests/` —
  so a manager was cheerfully told to delegate into the two places it is forbidden
  to write. The world now carries the read-only globs and the fallback skips a
  directory with nothing writable in it.
* A routing note was always bookkeeping, trimmed before a source file when an
  episode's edits exceeded the trust region. That is right for a note that adds a
  line to an existing table and wrong for one that *creates* a node's record: the
  source file is re-proposable next round, and the structure the system just
  invented is written nowhere else. A record that brings a node into existence is
  now trimmed last.

Offline on md, cold, `--episodes 120`: **audit reward 1.000**, depth 3,
`routes_opened=30` — the tables the run wrote are in the repository it wrote. That
says the mechanism works from eight files; it says nothing about *inventing* a good
decomposition, because the rule-based actor's plan **is** a decomposition. Only a
model arm can speak to that, and it is measured separately below.

### What the domains share

Everything that is not the software itself lives in [`_suite.py`](https://github.com/Birfy/agentdescent/blob/main/examples/genesis/_suite.py):
the child-process harness, the loader, the frozen-file restore, the parent's
integration check and the LLM actors. A domain is data on top of it — which is also
why the second, third and fourth could be added without touching a single
mechanism, and why `TestSuite` sits beside `Suite` rather than replacing it.

## Measured results — compact formation domains

### With a real model

`deepseek-v4-flash` through an Anthropic-shaped endpoint, `--episodes 60
--workers 4 --no-thinking`, three seeds. The repository starts implementation-empty
and the model writes every line of what comes out.

| domain | seed | held-out | accepted | episodes | depth | calls | wall-clock |
|---|---|---|---|---|---|---|---|
| `minilang` | 0 | **1.000** | 5 | 34 | 3 | 284 | 135 s |
| `minilang` | 1 | **1.000** | 5 | — | 3 | 306 | 173 s |
| `minilang` | 2 | **1.000** | 5 | — | 2 | 242 | 100 s |
| `stackvm` (`--episodes 160`) | 0 | **1.000** | 12 | 114 | 3 | 659 | 902 s |

Both seed-0 repositories were re-scored independently from what `--write-repo`
wrote: **30/30** each. On `minilang` it grew `src/__init__.py` plus
`frontend/{lexer,parser}.py` and `backend/evaluator.py`, with each stage imported
lazily because the specification says to — so the agents read the contract rather
than guessing it.

!!! note "The suite constrains the surface, not the structure — and `stackvm` shows it"
    The `stackvm` run passes every case and the repository it wrote is **not** the
    one the `CONTEXT.md` tree proposed. The agents moved the tokenizer and the
    interpreter to `src/tokenize.py` and `src/run.py` (their own node's files, in
    the accountability turn), made `src/asm/parser` a package, and gave the
    opcodes a calling convention of their own — consistently, and the public
    surface honours the frozen spec exactly, which is why it scores 1.000.

    It also left two pieces of dead code behind: `src/asm/lexer.py`, shadowed by
    the `src/asm/lexer/` package that took its place, and `src/vm/machine.py`,
    which imports `op_add` from a module that defines `add` and would raise on
    import — nothing imports it, because `src/run.py` became the real interpreter.

    This is the organization working as specified, not failing: acceptance is
    gated on **validation**, the validation checks three entry points and the data
    shapes, and anything the suite cannot see is not something any gate here can
    refuse. Upstream has the same property, with a suite that checks far more. It
    is worth stating plainly because "held-out 1.000" and "no dead code" are
    different claims and only the first one is measured.

Three defects of this port stood between a model and that result, each invisible
until the one before it was gone; they are the section below, and none of them was
the model.

### With the offline actors

Offline rule-based actors, three seeds, one machine: `--episodes 96` on `minilang`
and `160` on `stackvm`. Both repositories start at **0.000** — context records, a
frozen spec and one skill, no implementation — and every number below is identical
on all three seeds, because a rule-based actor on a fixed plan is deterministic.

| domain | arm | accepted events | `merged` | held-out |
|---|---|---|---|---|
| `minilang` | `--serial` (upstream semantics) | 5 | 0 | **1.000** |
| `minilang` | 4 workers | **4** | 1 | **1.000** |
| `minilang` | 8 workers | **3** | 1 | **1.000** |
| `minilang` | 4 or 8 workers, `--keyed-union` | 5 | — | **1.000** |
| `stackvm` | 8 workers | **6** | 1 | **1.000** |
| `stackvm` | 8 workers, `--keyed-union` | 8 | — | **1.000** |

**The three-way merge buys accepted events, and the keyed union buys nothing from
parallelism at all.** That is the whole table in one line: under the engine's rule
— where "same key" means "same file" — five accepted events at one worker, five at
four, five at eight. Under a textual three-way merge: five, then four, then three.
`stackvm` says the same at a different scale, 8 against 6. Two agents filling two
different functions of one file is an edit a keyed union cannot fuse and a
three-way merge can, and the counter (`merged=1`) says when it happened.

**A claim this page used to make, and the measurement that retired it.** Before
the manager's accountability turn existed, the offline arm under `--engine-gate`
reached 0.333 / 0.750 / **0.000** on the three seeds, refusing 8–11 candidates
each, and this page said the engine's statistical gate could not get a formation
run started. The reason given was right: a node's first files are structure, and
structure moves no case, so a gate demanding a measured improvement refuses them
and the run never reaches the steps that would have improved.

The accountability turn removed the *condition*, not the reasoning. A manager now
writes its own node's file in the same episode as its children's work, so commits
carry measurable progress from the first one, and `--engine-gate` reaches **1.000
on every seed of both domains**. What is left is a cost, not a wall: on `stackvm`
it refuses 8–11 candidates where the parent's rule refuses none, and on `minilang`
0–1. Recorded this way round because a mechanism fix moving a published
measurement is the normal case, and quietly keeping the old headline is how a
results page stops being one.

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

## The paper's numbers, and this port's

The mechanism is the thing being ported; the numbers are not, and putting them side by
side is the clearest way to say which is which. Upstream's figures are the formation
run — the one `md` stands in for — from §4.1 and the appendix tables.

| | paper, formation run | this port, `md` | comparable? |
|---|---|---|---|
| wall clock | **123.402 h** (666.385 h of agent time) | 0.6 h | no — the domain is a stand-in, by design |
| archived agent episodes | **1,019** | 237 (from ~42 proposals over 4 000 rollouts) | in kind only |
| observed delegation depth | **5** (configured max 8, retries 15) | 3 (configured max 4) | in kind — both bottom out below their ceiling |
| what one episode *is* | a supervised session of **up to 2 048 root turns / 128 child turns**, with file, shell and test tools | **one model call** (plus one for a manager's accountability turn, one for the parent's review) | **no**, and this is the largest single gap |
| result size | 750 tracked files, **248 989** physical lines | 25 files, ~700 lines | no |
| model-token cost | **US$44.3760** | not billed by this endpoint; 2.9 M prompt + 0.15 M completion tokens | no |
| concurrency | max **22** overlapping episodes | 4 workers | in kind |
| validation | complete c-testsuite, most LLVM and Csmith | a 67-task frozen suite, 16 of them held out | in kind |
| `CONTEXT.md` maintenance | 26 files created, **62 later accepted updates affecting 19 files** | creations, plus routing entries and refusals; no other updates | **no** — and this is the second gap |

Two of those rows are gaps rather than scale differences, and both are recorded below:
an episode here is a single completion rather than a tool-using session, and the
context records are written on two occasions rather than maintained. Everything else
in the table differs by three orders of magnitude because the domain was chosen to
finish in an afternoon, which is what `mechanism_microport` means.

What *is* aligned is the shape: `(v,p)`, delegation that moves the path and not the
version, the spatial contract, the parent's three-part validation, partial progress
accepted, the octopus merge, `CONTEXT.md` as the routing mechanism, a worktree and a
commit per episode with the worktree released, and termination by an agent's judgment.
Each has its own row in the table at the top of this page and a test behind it.

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

* **Upstream has no objective function, and this port's scalar is the engine's, not
  Genesis's.** `grep -rio 'fitness|reward|score'` over `apps/evo_git` returns exactly
  one hit, `oom_score_adjust`. Nothing in the released code computes a number over a
  validation suite. What stands in that place is local: the executor verifies its own
  work and writes tests (`agents/executor.ex`), the parent reviews the result and runs
  the suite (`agents/manager.ex`), the architect's third phase runs the build and
  reviews the implementation (`agents/architect.ex`), the tests are "the **definition
  of done**" handed to the agents as guidance (`runtime/genesis.ex:152`), termination
  is an agent calling `complete_task` "only when the codebase is complete, functional,
  and polished", and a human merges or rejects on the dashboard review page.
  `evolve()`, by contrast, *is* a reward-driven loop: it samples a failing task, asks
  for a proposal, scores it, accepts or rejects, and reports a number. So the per-node
  gate in this port is upstream's mechanism, and `audit reward` is **measurement** —
  the thing that makes the port legible, not the thing that makes it work.
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
* **The parent's validation is both halves now, and was one for a long time.**
  `agents/manager.ex` states it as three things in order — "Review subagent results.
  Run tests to validate changes. Check for code quality: duplicated code, defensive
  code that silently swallows errors, and missing test coverage. **Reject work that
  introduces these anti-patterns**" — and the architect's Phase 3 is the same shape.
  This port had only the middle one, `Suite.review`, which is a pass count. What a
  pass count cannot see is a function that computes nothing, and that is exactly what
  it failed to see: the zero-field run above. `_review.py` adds the reading half as
  one model call per returned child, asking upstream's questions about the whole file;
  a rejection sends the work back with a reason the next round re-delegates from.
  "Missing test coverage" is left out of the questions on purpose — the suite is
  frozen here, so no child could act on that finding. `--no-parent-review` turns it
  off; the offline arm has no model and therefore no reviewer, which is one more
  reason the offline numbers say less than they look like they do.
* **`CONTEXT.md` is inherited and routed from, but only partly maintained.**
  The read half is faithful: the chain is assembled root-first exactly as
  `ContextNode.build_context/2` does, each record truncated **per file** at
  upstream's own `truncation.context_max_bytes` (65 536) with upstream's own marker
  and no global budget — this port had one global 8 000-char cap instead, eight times
  tighter, which is half of how md's agents never received their specification. The
  routing table is load-bearing: the offline manager's decomposition is *parsed from
  it*, not hardcoded, so editing the table changes where the run delegates (there is
  a test that does exactly that). The human-written records carry upstream's four
  standard sections — Intent, **API Surface**, Constraints, Routing Table — and
  `API Surface` was missing from every one of them until the md run lost five of
  seven keyword parameters. The write half is still thinner than upstream, where
  every `:read_write` agent keeps its node current and two read-only roles exist to
  do nothing else: here an agent writes `CONTEXT.md` on its own in exactly two cases,
  the routing entry for a node it opened and the refusal for a child it rejected —
  the latter under `## Known Issues`, which is upstream's heading for "problems to
  avoid re-discovering". An LLM executor may write more; nothing requires it to.
* **What the agent is handed, versus what it could fetch.** Upstream an agent has file
  tools and *pulls* what it needs; the brief carries only the `CONTEXT.md` chain plus
  its location. Agents here have no tools, so the contract has to travel with the
  brief or be invisible — which is why `situate(contracts=)` exists at all. The push
  is therefore deliberately narrower than "everything frozen": md pushes the
  specification and the frozen driver, while the 12 kB test suite arrives one failing
  test at a time in the task prompt. `FROZEN` (what may not be written), `CONTRACTS`
  (what is pushed) and `readonly` (where a manager may not be sent) are three sets for
  three jobs; conflating the first two is how `md.py` once crowded out the spec.
* **Termination can be an agent's judgment, with `--complete-task`.** Upstream an
  agent calls `complete_task` when it believes the objective is met, and a human merges
  or rejects afterwards on the dashboard; here the run ends when the round budget ends,
  which is why a finished domain still reports `stop reason: rounds` — nobody decided
  it was done. `CompletionJudge` asks instead, through `evolve(stop_when=)`, with
  upstream's two gates in upstream's order: every test the agents can see must pass
  first ("treat them as the definition of done"), and below that the question is not
  asked and no model call is spent; then the root agent judges the **codebase** —
  stubs, dead modules, anything it would not hand over as finished. It is never shown
  the held-out reward: that number is this port's measurement, computed from tests no
  agent may read, and handing it over would turn the judgment back into a threshold.
  The human review at the end of upstream's chain is still out of scope.
* **Worktree isolation and "commit before delegating", with `--worktrees`.** Upstream
  this is the scheduling model, not a detail of it: "commit your changes, release your
  worktree, and wait... If you don't commit, your changes are invisible to subagents."
  Each episode now gets a git worktree of its own, commits its work there on a branch
  of its own, and the worktree is **removed** — the release a cooperative scheduler
  requires — with a ledger that has to balance. What that buys is not correctness:
  siblings here branch from the same base by construction because the episode tree is
  a pure function over a state dict, and a test holds that property directly. It buys
  three things that were missing. The **phylogenetic graph becomes git history** — one
  commit per episode, and a parent that kept three children leaves a four-parent merge
  commit, so `Git.merge_octopus/2`'s shape is there and `phylo_graph_node.ex`'s
  `find_merge_base/2` has something to find. "Commit before delegating" becomes
  **checkable**, and acting on it found a real gap: a manager that opens a node now
  writes that node into its routing table *before* briefing the child, which is what
  `make_dir` (auto-commits) then spawn means, and before this the child arrived at a
  node its own parent's table did not mention. And every episode record carries the
  sha it could be **resurrected** from, which is the third field of upstream's
  `(node_path, commit_sha, objective)` tuple and the one this port did not have.
  Measured on md offline: 98 worktrees created, 98 removed, 98 commits, 41 of them
  merges, five parents at the widest, same reward, 4% wall-clock.
* **Parallelism is still the engine's workers.** Upstream's concurrency is unbounded
  and each subagent's worktree is where its commands run; here `evolve()`'s workers
  propose concurrently against one accepted version and `OctopusConflict` does the
  three-way merge in the parent. The observable consequence is the same — two children
  editing one file both survive — and the measurement of that is in the table above.
* **Skills are inherited but never extracted.** `LocalWorld.skills()` collects
  `.agents/skills/` along the node chain and puts the *names* in the brief, which
  is what `hierarchical_skill_names/2` does upstream and what the paper means by
  listing "reusable skills" among what an accepted version carries (§3.1). What
  is missing is the other end: upstream's `SkillExtractor` distils a completed
  contribution into a new skill, and nothing here does. The domain ships one
  human-written skill so the inheritance path is live rather than decorative.
* **An episode here is one model call; upstream it is a session.** The paper is
  explicit — "the agent can execute multiple model–tool turns during one supervised
  episode" — and the runs are configured at **2 048 root turns and 128 turns per
  non-root episode**, with file, shell and test tools inside each one. Here an episode
  is a single completion with no tools: the manager is asked once where to delegate,
  the executor is asked once for whole files, and the brief has to carry everything
  either of them could otherwise have fetched (which is what `situate(contracts=)` is
  for). Everything downstream of that follows from it — why the domains are small, why
  the actors cannot run the tests themselves, why a "rework" costs a whole new episode
  rather than another turn. It is the largest single distance between this port and
  the system it ports, and it is a property of the harness rather than of the
  algorithm: `evolve()` proposes with one completion per rollout.
* **Only two roles are ported.** The released code has ten agent modules; the
  paper's appendix §1.3 says the model has two — manager and leaf executor — and
  that "codebase lead / investigator / task scheduler are implementation labels,
  not additional roles". Porting ten would be porting an implementation.

## Run it

```bash
python -m examples.genesis.genesis_recursive_worlds                            # offline, no API key
python -m examples.genesis.genesis_recursive_worlds --domain md                # tests, not an oracle
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
