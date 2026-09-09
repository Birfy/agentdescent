# Evolving a directory — a skill, an agent, or its code

Everything else in AgentDescent evolves *text*: an instruction, a playbook, a
prompt. This page is about evolving a **directory** — a skill folder with a
`SKILL.md` and a `references/` beside it, a folder of subagent definitions, or a
small codebase that *is* the agent — where each rollout is performed by a **real
agent** that reads those files off disk with its own tools.

```python
from agentdescent import FileTree, evolve, load_tree, scorer, tree_reflector, tree_runner
from agentdescent.agents import claude_code, openai_compatible
from agentdescent.governance import SKILL_BLAST_RADIUS

path = "~/.claude/skills/pdf-audit"
tree = load_tree(path)                                   # the directory as artifact state
strategy = FileTree(tree, max_files_per_diff=2)          # file paths are the keys
run = tree_runner(claude_code(extra_args=["--permission-mode", "acceptEdits"]),
                  layout="claude_skill", name="pdf-audit",
                  overlay=strategy.frozen_files(tree))

result = evolve(tasks, scorer("contains"), run=run, strategy=strategy,
                propose=tree_reflector(openai_compatible(model="deepseek-v4-flash"),
                                       strategy=strategy),
                artifact_id="pdf-audit", blast_radius=SKILL_BLAST_RADIUS,
                self_verify=False, cheap_eval_tasks=4,
                rounds=6, n_workers=4, max_concurrency=4, held_out_frac=0.3)

print(result.final_reward, result.outcomes())
result.write_to(path)     # opt in; backs up first
```

The design record — why it is built this way, and the three problems found while
reviewing it — is [in the design document](design-directory-evolution.md).

## Why this works without touching the optimizer

An artifact's state is a flat `{key: value}` dict and the engine never interprets
the keys. It only asks two questions: do two diffs touch the same key
(`diffs_contradict`), and can non-overlapping ones be unioned (`fuse_diffs`).

So make the keys **relative file paths** and the whole aggregation stack acquires
file-level semantics for nothing:

| Two workers… | …produce | …and the aggregator |
|---|---|---|
| edit different files | complementary diffs | fuses them into one candidate |
| edit the same file | a contradiction | keeps whichever scores better held-out |
| propose the identical file | duplicate diffs | collapses them |

This is the same machinery ACE, GEPA and EvoSkill run on. Nothing in
`aggregator.py`, `ledger.py`, `async_evolve.py`, `parallel.py` or `governance.py`
knows that a directory is involved.

## The four moving parts

```
~/.claude/skills/pdf-audit/            your directory
   │  load_tree()                      → {"SKILL.md": "...", "references/rules.md": "..."}
   ▼
FileTree strategy                      keys are paths; frozen paths are read-only
   ▼
evolve()                               unchanged
   │  every rollout: tree_runner()
   ▼
/tmp/agentdescent-ws-xxxx/             one throwaway workspace per rollout
   .claude/skills/pdf-audit/SKILL.md   ← the layout decides where it lands
   <the task's fixtures>
   │  claude_code().in_workspace(ws)(prompt)
   ▼
answer → reward → reflection → {"path": "...", "content": "..."} → Diff
   ▼
result.write_to("~/.claude/skills/pdf-audit")
```

### 1. `filetree` — directory ↔ state

```python
from agentdescent import TreeSpec, load_tree, materialize, canonical, parse_tree

state = load_tree("~/.claude/skills/pdf-audit")          # {relpath: text}
materialize(state, "/tmp/ws", prefix=".claude/skills/pdf-audit")
```

`TreeSpec` says what belongs to the artifact (`include` / `exclude` globs) and
how big it may get. Two behaviours are deliberate:

* **A file that matches `include` but cannot be represented — binary, or over
  `max_file_bytes` — raises.** Skipping it silently would mean the evolved tree
  is not the tree you pointed at, and `write_to` would later write back a
  directory missing files. Put it in `exclude` to say "not part of the artifact"
  on purpose.
* **`max_file_bytes` defaults to 28 000, below the aggregator's
  `trust_region_chars` (32 000).** A larger file can be loaded but never changed:
  every diff touching it is rejected as `oversized`. `TreeSpec.validate_against`
  catches the mismatch before the run rather than five rounds in.

`canonical` / `parse_tree` are the lossless serialisation pair.
`Strategy.render` is both what `run` receives *and* the evaluation-cache key, so
it has to be exact — two different trees must never share a cached score. It is
JSON rather than a prettier `--- file: x ---` format precisely because a file's
own content must not be able to forge its container's delimiter.

### 2. `FileTree` — the strategy

```python
from agentdescent import FileTree

strategy = FileTree(load_tree("./my-skill"),
                    editable=["**"],          # globs the loop may write
                    frozen=["tests/**"],      # globs it may only read
                    max_files_per_diff=2)
```

`frozen` is L0 expressed in paths. `governance.py` freezes whole *artifacts* by
id, which cannot say "this skill may evolve, but not its test suite" — and
without that, the shortest path to a high score is to weaken the thing measuring
it.

Proposals use a small protocol, because one string can no longer mean "the new
artifact":

```
<EDITS>
{"rationale": "the skill never said what to do when a table spans pages",
 "edits": [{"path": "SKILL.md", "content": "...the complete new file..."}]}
</EDITS>
```

Whole-file replacement, not a unified diff: a model-written patch that does not
apply fails *silently* (wrong context lines, drifted offsets), whereas a whole
file either parses or does not. `parse_edits` also accepts a fenced or bare JSON
object, drops hostile paths, and returns `{}` — never raises — when a reflector
ignores the protocol, so that shows up as a counted non-proposal rather than a
crashed run. `{"path": "old.md", "delete": true}` deletes a file.

`tree_reflector(complete, strategy=...)` builds the reflection prompt. It does
**not** put the whole tree in it: `render()` is the full serialisation because
the cache needs it lossless, but a reflection only needs to see the files it
might change, so it inlines the `context_files` within `max_context_chars` and
lists the rest by path and size.

### 3. `runners` — giving a real agent the candidate

```python
from agentdescent import tree_runner, code_runner, LAYOUTS

run = tree_runner(claude_code(), layout="claude_skill", name="pdf-audit")
```

| `layout` | where the tree is written in the workspace |
|---|---|
| `claude_skill` | `.claude/skills/<name>/` |
| `skill_library` | `.claude/skills/` (a directory *of* skills) |
| `claude_agent` | `.claude/agents/` |
| `dsh_skill` | `.dsh/skills/<name>/` (DeepSeek Harness project skill) |
| `agents_skill` | `.agents/skills/<name>/` (the Agent Skills standard; dsh and Codex both read it) |
| `root` | the workspace itself |

Any literal prefix works too. Per rollout the runner makes a fresh workspace,
materialises the candidate, overlays the pristine frozen files, stages the task's
fixtures (`task.meta["fixtures"]`, or a `fixtures=` callable), runs the agent
bound to that directory, and deletes the workspace.

!!! warning "One workspace per call, always"
    `evolve(max_concurrency=N)` runs workers in threads and
    `EvolvingArtifact.score` opens its own pool of `eval_concurrency` threads. A
    shared directory would let two candidates overwrite each other and produce
    scores that look perfectly ordinary and are simply wrong.

`tree_runner` requires a [`WorkspaceAgent`](agents.md) (`claude_code()`,
`codex()`, `cli_agent([...])`, `openhands()`). A plain API completion cannot read
files, so it raises rather than quietly scoring the model's prior knowledge.

### 4. `write_to` — installing the result

```python
plan = result.write_to("~/.claude/skills/pdf-audit", dry_run=True)
# {"written": [...], "extra": ["notes.md"], "deleted": [], "backup": []}
```

This is the only call in the package that writes into a directory you care about,
so it is conservative: `backup=True` (default) copies the directory to
`<path>.bak-N` first; files present in the target but not in the artifact are
reported as `extra` and **left alone** unless `prune=True`; `dry_run=True` returns
the plan without touching anything. The run itself never writes to your directory.

## The three workloads

One call, three settings. What differs is governance and what guards the tree:

| workload | `blast_radius=` | runner / reward | what makes it different |
|---|---|---|---|
| a skill folder | `SKILL_BLAST_RADIUS` (`0.2`, L2) | `tree_runner(layout="claude_skill")`, `scorer(…)` | merges on held-out reward alone |
| an agent folder | `HARNESS_BLAST_RADIUS` (`0.6`, L1) | `tree_runner(layout="claude_agent")`, `scorer(…)` | an agent definition is a harness: every merge additionally passes the oracle |
| agent code | `HARNESS_BLAST_RADIUS` + test gate | `code_runner(entrypoint, test_cmd=…)`, `gated_reward(scorer(…))` | the tree is **executed**; a frozen test suite guards it |

Both constants live in `agentdescent.governance`, next to the L1/L2 boundary
they are chosen against.

### Evolving agent code

```python
from agentdescent import code_runner, gated_reward
from agentdescent.governance import HARNESS_BLAST_RADIUS

tree = load_tree("./my-agent")
strategy = FileTree(tree, frozen=["tests/**", "conftest.py"], max_files_per_diff=2)
run = code_runner(["python", "main.py"], name="my-agent",     # + task.prompt as the last argv
                  test_cmd=["python", "-m", "pytest", "-q"],
                  overlay=strategy.frozen_files(tree))

result = evolve(tasks, gated_reward(scorer("contains")), run=run, strategy=strategy,
                propose=tree_reflector(openai_compatible(model="deepseek-v4-flash"),
                                       strategy=strategy, context_files=("**/*.py",)),
                artifact_id="my-agent", blast_radius=HARNESS_BLAST_RADIUS,
                self_verify=False, cheap_eval_tasks=4)
```

Each rollout materialises the candidate, **overwrites every frozen path with its
pristine content**, runs `setup_cmd` then `test_cmd`, and only then executes the
entrypoint. A failing gate is not an exception: the output becomes
`TEST_FAILURE_MARKER` plus the captured output; `gated_reward` scores that 0
and the reflector gets something concrete to fix.

!!! danger "Isolation, not a sandbox"
    Candidate code runs in a throwaway workspace, with a trimmed environment
    (`HOME` and `TMPDIR` point inside the workspace, so it cannot read
    `~/.claude` or `~/.aws`), under a hard timeout, in its own process group so
    a timeout takes its children with it. It still runs **as your user, with
    your network**. Use a container for anything you would not run by hand.

    `sandbox.py` manages the *lifetime* of a workspace -- how many exist, who
    owns one, who cleans up when an owner dies. Isolation strength is a separate
    axis, and it is a property of the **provider**. For a real boundary, use the
    container provider below.

Both halves of `frozen` are needed and they do different jobs:

* the **proposal filter** stops the reflector from editing the test suite;
* the **overlay** stops the *candidate* from rewriting it at run time
  (a `conftest.py`, a monkeypatched assertion, an early `exit(0)`).

## Where a candidate runs

A candidate that is code needs somewhere to run, and that is its own subject:
lifetime (who owns a workspace, when it comes back, what happens if its owner
dies) and isolation (what the candidate can reach while it runs) are independent
questions with different answers. Both are on the [Sandboxes](sandboxes.md) page:

* [leases rather than deletion by age](sandboxes.md#lifetime-leases-not-deletion-by-age),
  and one ceiling shared by rollouts and the gate;
* [one ceiling across processes](sandboxes.md#one-ceiling-across-processes), read
  from the lease directory rather than from a server;
* [three isolation levels](sandboxes.md#isolation-strength-three-levels) — a
  trimmed environment is **not** a boundary; `ContainerProvider` is.

```python
from agentdescent.sandbox import SandboxPool
from agentdescent.sandbox_container import ContainerProvider

pool = SandboxPool(ContainerProvider("python:3.11-slim"), max_sandboxes=8)
run = code_runner(["python", "main.py"], test_cmd=["pytest", "-q"],
                  sandbox_pool=pool)
```

## Cost — the first-order design constraint

One rollout is one real agent invocation. Count them per round:

| source | calls per round | knob |
|---|---|---|
| worker rollouts | `n_workers` | — |
| self-verify re-run | `n_workers` | `self_verify=False` **(pass it)** |
| conflict resolution + fusion tournament ranking | `candidates × cheap set` | `cheap_eval_tasks=4` **(pass it)** |
| Beta acceptance test (winner: base + candidate) | `1 × |held_out|` | — (this is the commit gate) |
| L1 oracle gate | **0** | see below |

Two counter-intuitive points, both verified against the engine:

1. **The oracle gate is free.** `oracle_eval` and `eval_counts` call the same
   `eval_fn` over the same held-out set, and the evaluation cache is keyed on
   `(render(), task.id)` — so L1's forced audit spends `oracle_budget` counter,
   not agent calls. Evolving an agent directory is not more expensive than
   evolving a skill.
2. **The plain-engine defaults are the expensive ones.** With
   `cheap_eval_tasks=None` the cheap layer is pinned to the whole held-out set,
   which makes *ranking* the dominant cost when `eval_fn` runs an agent; and
   `self_verify=True` adds a second rollout per proposal. Every directory
   example in these pages flips both, because getting them wrong costs money
   rather than a warning.

Bound a rollout with the agent's own timeout (`cli_agent(timeout=...)`, which
really kills the process), not with `round_timeout` — that only stops *waiting*.

## Evaluation noise: the cache turns one sample into ground truth

`_EvalCache` evaluates each `(tree, task)` pair exactly once, ever. For a
deterministic scorer that is free speed. For a real agent it means **a single
sample is treated as the truth**, so the Beta posterior underestimates variance
and can accept noise as improvement. Mitigations, cheapest first: `temperature=0`
where the provider supports it; a majority vote of *k* inside your `reward`; and
a bigger held-out set — the engine warns below 4 tasks, but 4 is nowhere near
enough for a stochastic agent.

## Parallelism

`DataParallel` (the default) shards tasks and any editable path can be created.

`TensorParallel` gives each worker a disjoint set of *files*, which fits "one
worker owns `SKILL.md`, another owns `references/`" — but **no worker can create
a file under TP**. The section map is built once from `strategy.keys()` before the
first round, and an undeclared path maps to `None`, which never equals a worker's
section, so every creation is counted as `section-violation`. Declare the paths
up front with `FileTree(planned_paths=[...])`, or stay on DataParallel.

Fusion is at whole-file granularity, so two *complementary* edits to the same
`SKILL.md` are still a contradiction and only one survives. That is an argument
for the thing skill authors should do anyway: keep `SKILL.md` small and put the
detail in `references/*.md`, one concern per file.

## Known limitations

* **Whole-file replacement** costs tokens on large files; `max_file_bytes` and
  splitting the skill are the answer.
* **No renames** — a rename is a delete plus a create, which the protocol can
  express but which shows up as two ops.
* **Binary files are out of scope.** The state is `Dict[str, str]`.
* **Every ledger commit stores the whole tree** as one JSON blob
  (`artifacts/<id>.json`); git delta-compresses it, but `max_total_bytes` exists
  for a reason.
* **Only EvoSkill has been migrated.** ADAS and DGM still evolve their own
  representations; moving them here would let their candidates actually execute,
  but that needs an evaluation harness rather than these modules. See §5 of the
  [design document](design-directory-evolution.md).

## The acceptance test: EvoSkill

The abstraction is only worth having if it can express what the repo already
does. [EvoSkill](algo-evoskill.md)'s artifact was `{skill name: SKILL.md body}` —
a skill directory that never touched disk — so it is the natural test, and it now
runs on `SkillLibraryTree`, a `FileTree` subclass:

```
{"defense-lookup": "..."}   →   {"skills/defense-lookup/SKILL.md": "..."}
```

Two things it proved, both worth knowing before you migrate something yourself:

1. **A port can keep its own prompt format.** `render()` has to be the lossless
   serialisation because it is the cache key — but `run` is where an artifact
   becomes a prompt (the framework never injects one for you), so EvoSkill parses
   the tree and re-renders it in its own `### skill: <name>` format. The
   retriever path's prompt is byte-identical to before the move, which a test
   asserts; the port measures the same thing it did.
2. **A port can keep its own proposal protocol.** `SkillLibraryTree` overrides
   `to_diff` to accept the repo's `name :: body` instead of `<EDITS>` JSON. What
   is faithful about EvoSkill is the two-role Proposer/Generator induction, not
   the separator — and switching protocols would have changed the Generator's
   prompt, which is exactly the kind of silent behaviour change a "pure
   refactor" should not make.

The payoff is on the tool-using path: with `--backend claude-code`, the skills are
now materialised into the agent's workspace and the prompt points at them, so the
agent opens the one skill it needs. Before, every skill in the library rode along
in every prompt.

## Running a real agent non-interactively

`claude -p` needs its tool permissions settled up front or it will stall waiting
for a prompt that no one can answer:

```python
claude_code(extra_args=["--permission-mode", "acceptEdits"], timeout=300)
```

And check that the skill is actually being *used*: run one round against an empty
skill directory as a control. If the score does not move, what you are measuring
is the model's prior knowledge, not your skill.
