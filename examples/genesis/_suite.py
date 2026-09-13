"""The machinery a formation domain needs, with the domain itself left out.

A formation run needs four things that have nothing to do with *which* software is
being grown: a frozen validation suite, a way to run a candidate repository in a
child process, the parent's integration check, and a pair of LLM actors. All four
are here, and the domains -- :mod:`examples.genesis._domain` (minilang),
:mod:`examples.genesis._stackvm`, :mod:`examples.genesis._jqx`,
:mod:`examples.genesis._md` -- are data on top of them.

The suite comes in two shapes, and the difference matters more than it looks.
:class:`Suite` computes each case's expected answer by running a **reference
implementation**: cheap to write, and circular -- you cannot ask a system to grow
software you had to write first. :class:`TestSuite` scores against a **frozen test
suite** instead: one task per test function, reward is whether it passes, and no
reference appears anywhere in the scoring path. That is what upstream does
(c-testsuite, LLVM, Csmith) and what a human actually writes.

Split out when the second domain arrived, for the reason the first one should have
been: the two differ in the software they grow and in nothing else, and a copy of
the harness per domain is two places for the frozen-file restore to drift.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from itertools import zip_longest
from typing import (Callable, Dict, List, Mapping, Optional, Sequence,
                    Tuple)

from agentdescent.evolution import Task
from agentdescent.filetree import (canonical, match_any, materialize,
                                   parse_tree)

from ._delegation import Brief, Delegation, Edit
from ._spatial import SITUATED_EDIT_PROTOCOL, parse_situated_edits
from ._world import CONTEXT_FILE, ROUTING_HEADING, SKILLS_DIR, normalise

__all__ = ["CRASHED", "PYTHON_MODULE_SKILL", "Suite", "TestSuite", "llm_executor",
           "llm_manager", "reward", "reward_test", "run_cases", "run_test"]

#: One human-written skill, shared by the formation domains because the advice
#: is about writing a Python module rather than about either language. Shipped so
#: the skill-inheritance path (`LocalWorld.skills`) is live rather than
#: decorative; nothing here extracts a new one -- see docs/algo-genesis.md.
PYTHON_MODULE_SKILL = '''# Writing a module in this project

A file is complete or it is not written. Never leave a partial module: the
validation suite imports what is there, and a half-written module fails the
stages after it as well as its own.

- Keep one concern per file, and name it after that concern.
- A package directory needs an `__init__.py`, even an empty one.
- Import a sibling module relatively (`from .lexer import tokenize`).
- Do not import a stage you do not need: `src/__init__.py` imports lazily on
  purpose, and a module-level import undoes that.
'''


#: What a case scores when the workspace could not be run at all -- a syntax
#: error in a generated module, a timeout, a missing package marker. Distinct
#: from a wrong answer only in the transcript; both score zero, and conflating
#: them in the *reward* would be the bug, not here.
CRASHED = "ERROR:workspace"

#: Executed inside the materialised workspace, in a child process. A candidate is
#: agent-written code: it can loop, raise or exit, and none of those may take the
#: run with it. The stage -> attribute map arrives as JSON so the harness itself
#: is domain-independent.
_HARNESS = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
cases = json.loads(sys.argv[2])
stages = json.loads(sys.argv[3])
digits = json.loads(sys.argv[4])
import src


def canonical(value):
    # Rounded before repr, for a domain whose answers are floating point: two
    # correct implementations of the same formula differ in the last bits by
    # summation order alone, and comparing full repr would score a correct
    # candidate wrong. `None` leaves the value untouched.
    if digits is None:
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        rounded = round(value, digits)
        return 0.0 if rounded == 0 else rounded
    if isinstance(value, list):
        return [canonical(v) for v in value]
    if isinstance(value, tuple):
        return tuple(canonical(v) for v in value)
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in value.items()}
    return value


out = []
for kind, source in cases:
    try:
        out.append(repr(canonical(getattr(src, stages[kind])(source))))
    except Exception as exc:            # a stage that is not built yet
        out.append("ERROR:" + type(exc).__name__)
print(json.dumps(out), end="")
"""


def run_cases(state: Mapping[str, str], cases: Sequence[Sequence[str]],
              stages: Mapping[str, str], *, digits: Optional[int] = None,
              timeout: float = 60.0) -> List[str]:
    """Materialise ``state`` and evaluate every case in one child process.

    ``digits`` rounds every float in a result before it is compared. A domain
    whose answers are exact -- a token list, a parse tree -- leaves it ``None``;
    one whose answers are floating point needs it, because two correct
    implementations of one formula differ in the last bits by summation order.
    """
    if not cases:
        return []
    workspace = tempfile.mkdtemp(prefix="genesis-run-")
    try:
        materialize(state, workspace)
        proc = subprocess.run(
            [sys.executable, "-c", _HARNESS, workspace, json.dumps(list(cases)),
             json.dumps(dict(stages)), json.dumps(digits)],
            capture_output=True, text=True, timeout=timeout, cwd=workspace)
        if proc.returncode != 0:
            return [CRASHED] * len(cases)
        return list(json.loads(proc.stdout))
    except (OSError, ValueError, subprocess.SubprocessError):
        return [CRASHED] * len(cases)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def reward(task: Task, output: str) -> float:
    """Exact match against the frozen suite's expectation."""
    return 1.0 if output == task.meta.get("gold") else 0.0


@dataclass(frozen=True)
class Suite:
    """One formation domain's data: what is given, what is grown, how it is scored.

    ``stages`` maps a case kind to the attribute of ``src`` that answers it, which
    is what makes the suite *staged*: a case exercising only the first stage scores
    as soon as that stage lands, so the run has a gradient before the toolchain is
    complete. Without staging a formation run has literally zero signal until the
    last file arrives, and no acceptance rule can help.
    """

    name: str
    #: Human-supplied and never the agents': context records, constraints, the
    #: specification, any skill. This is the repository a run starts from.
    given: Mapping[str, str]
    #: Globs refused to every proposal and restored pristine before scoring.
    frozen: Sequence[str]
    #: ``{case kind: attribute of src}``.
    stages: Mapping[str, str]
    #: ``(kind, source)`` pairs. Interleaved by stage so the held-out tail
    #: ``evolve()`` cuts is a mix rather than one stage's worth.
    cases: Sequence[Sequence[str]]
    #: The finished repository: the oracle the expectations come from, and what
    #: the offline actors reveal.
    reference: Mapping[str, str]
    #: Decimal places every float in a result is rounded to before comparison.
    #: ``None`` for a domain whose answers are exact.
    digits: Optional[int] = None

    # -- the repository -----------------------------------------------------

    def initial_files(self) -> Dict[str, str]:
        return dict(self.given)

    def reference_tree(self) -> Dict[str, str]:
        return dict(self.given, **dict(self.reference))

    # -- the suite ----------------------------------------------------------

    def build_tasks(self, limit: Optional[int] = None) -> List[Task]:
        """The suite, with every expected answer computed from the reference.

        Computed rather than typed out: a hand-written expectation drifts from the
        implementation it is supposed to pin, and a suite that disagrees with its
        own oracle scores a correct candidate wrong. This is the loader, so it is
        also the boundary ``--dry-run`` must not cross.
        """
        cases = list(self.cases)[:limit] if limit else list(self.cases)
        gold = run_cases(self.reference_tree(), cases, self.stages,
                         digits=self.digits)
        tasks: List[Task] = []
        for i, ((kind, source), expected) in enumerate(zip(cases, gold)):
            if expected.startswith("ERROR:"):
                raise RuntimeError(
                    f"the {self.name} reference failed case {kind} {source!r} "
                    f"({expected}); the suite would score a correct candidate wrong")
            tasks.append(Task(id=f"{kind}{i:02d}", prompt=source,
                              meta={"kind": kind, "gold": expected}))
        return tasks

    def make_runner(self) -> Callable[[str, Task], str]:
        """``run(rendered, task)`` -- one case against one candidate repository."""

        def run(rendered: str, task: Task) -> str:
            try:
                state = dict(parse_tree(rendered))
            except Exception:  # noqa: BLE001 - an empty artifact, before anything
                return CRASHED
            # The frozen files are restored from the human's copy rather than
            # taken from the candidate: a proposal can never write them, and a
            # resumed ledger or a hand-built aggregator could still put state in
            # front of this. Scoring against the agents' own copy of the suite is
            # the one failure mode that cannot be detected from a completed run.
            for path, content in self.given.items():
                if match_any(path, self.frozen):
                    state[path] = content
            return run_cases(state, [(task.meta["kind"], task.prompt)],
                             self.stages, digits=self.digits)[0]

        return run

    # -- the parent's integration evidence ---------------------------------

    def review(self, tasks: Sequence[Task], *, sample: int = 10):
        """A :data:`~examples.genesis._delegation.Review`: the suite on a child's work.

        The paper's parent decides "using the available tests, constraints and
        integration evidence" (3.3), and that is a different decision from the
        acceptance gate's: it happens **inside** the episode, on one child's
        contribution, before anything is offered to the version history.

        Refuses a regression and says nothing otherwise -- structural work that
        moves no case is what upstream's "partial progress is accepted" is about,
        so a parent that demanded a gain would refuse the first file of every node.

        ``sample`` bounds the cost: this runs per child per episode, so it is a
        subset of the suite rather than the whole of it -- integration evidence a
        parent can afford, not the gate's measurement.
        """
        chosen = list(tasks)[:sample]
        cases = [(t.meta["kind"], t.prompt) for t in chosen]
        gold = [t.meta["gold"] for t in chosen]
        cached: Dict[int, int] = {}

        def score(state: Mapping[str, str]) -> int:
            got = run_cases(state, cases, self.stages, digits=self.digits)
            return sum(1 for out, want in zip(got, gold) if out == want)

        def review(parent, returned):
            key = id(parent.state)
            if key not in cached:
                cached[key] = score(parent.state)
            candidate = dict(parent.state)
            for edit in returned:
                if edit.content is None:
                    candidate.pop(edit.path, None)
                else:
                    candidate[edit.path] = edit.content
            after = score(candidate)
            if after < cached[key]:
                return ("rejected", f"integration check regressed "
                                    f"{cached[key]}/{len(cases)} -> {after}/{len(cases)}")
            return None

        return review


# ---------------------------------------------------------------------------
# The LLM actors: the same two seams, asked rather than computed
# ---------------------------------------------------------------------------

_MANAGER_PROMPT = """You are a manager agent in a recursive software world. You are \
situated at the repository path `{path}` and you own everything under it.

{context}

This node's routing table says its children are: {routes}

OBJECTIVE
{objective}

You do not write code. Decide whether to delegate to more specific paths inside \
your own subtree, or to handle this yourself.

Reply with ONE JSON object and nothing else:
{{"delegations": [{{"path": "<a node inside {path}>", "objective": "<one sentence>"}}]}}

Rules:
- You are ACCOUNTABLE for all code under `{path}`, and delegating does not \
discharge that. The files AT `{path}` itself are nobody else's to write: after \
your children return you get one more turn to write them.
- A node is a **directory**, never a file. `src/frontend` is a node; \
`src/frontend/lexer.py` is a file that belongs to the agent situated at \
`src/frontend`, and delegating to it is refused.
- Prefer a child this node already routes to. Naming a new one is allowed and \
adds it to this node's routing table -- do that only when the work genuinely \
belongs to a new part of the tree.
- An empty list means you will handle it at your own path, writing the files \
that belong to `{path}` itself.
- Never name a path outside your subtree -- it belongs to another agent."""

_EXECUTOR_PROMPT = """You are an executor agent in a recursive software world. You are \
situated at the repository path `{path}` and you may write ONLY files under it.

{context}

OBJECTIVE
{objective}

{failure}

{protocol}"""

#: How an oracle-scored domain shows the executor what went wrong: one input, the
#: output it produced, and the score that got.
ORACLE_FAILURE = """A validation case failed:
  input    {prompt}
  produced {output}
  score    {reward:.2f}"""

#: How a test-scored domain shows it: the failing test's own source. The
#: specification is executable here, so the evidence *is* the specification --
#: including the tolerance, which an expected-output line cannot carry.
TEST_FAILURE = """A test in the frozen suite is failing. Here it is, exactly as it \
runs:

{prompt}

and it reports: {output}

`tests/` is read-only. The implementation has to meet the test, never the other \
way round -- and the tests that already pass have to keep passing."""


def llm_manager(complete) -> Callable[[Brief], Sequence[Delegation]]:
    """Ask a model where to situate children. A bad reply means "handle it here"."""

    def manager(brief: Brief) -> Sequence[Delegation]:
        routes = brief.world.routing(brief.state)
        reply = _ask(complete, _MANAGER_PROMPT.format(
            path=brief.world.path or "./", context=brief.context,
            routes=", ".join(f"`{r}/`" for r in routes) or "(none yet)",
            objective=brief.objective))
        try:
            data = json.loads(_first_object(reply) or "{}")
            items = data.get("delegations") or []
        except Exception:  # noqa: BLE001 - malformed model output, not a bug
            return ()
        out = []
        for item in items:
            if isinstance(item, dict) and item.get("path"):
                out.append(Delegation(normalise(str(item["path"])),
                                      str(item.get("objective", brief.objective))))
        return out

    return manager


def llm_executor(complete, *, editable: Sequence[str] = ("**",),
                 frozen: Sequence[str] = (),
                 failure: str = ORACLE_FAILURE) -> Callable[[Brief], Sequence[Edit]]:
    """Ask a model for situated edits. Unparseable replies cost their episode.

    The protocol is rendered **per episode** rather than once, because it names the
    agent's own path in every example it shows. A protocol that says "relative
    path" without saying relative to what is read both ways, and the wrong reading
    puts a node's files at the top of the repository.

    ``failure`` is how the evidence is worded: :data:`ORACLE_FAILURE` for a domain
    scored against a reference, :data:`TEST_FAILURE` for one scored by a test suite,
    where "input / produced" describes nothing the agent can act on.
    """

    def executor(brief: Brief) -> Sequence[Edit]:
        owner = brief.world.path or "."
        protocol = SITUATED_EDIT_PROTOCOL.format(
            owner=owner, editable=", ".join(editable) or "(none)",
            frozen=", ".join(frozen) or "(none)")
        reply = _ask(complete, _EXECUTOR_PROMPT.format(
            path=brief.world.path or "./", context=brief.context,
            objective=brief.objective,
            failure=failure.format(prompt=getattr(brief.task, "prompt", ""),
                                   output=(brief.output or "")[:400],
                                   reward=brief.reward),
            protocol=protocol))
        return [Edit(owner=brief.world.path or "", path=edit["path"],
                     content=edit["content"])
                for edit in parse_situated_edits(reply)]

    return executor


def cold_start(files: Mapping[str, str]) -> Dict[str, str]:
    """The same repository with the **decomposition taken out**.

    A formation domain ships a `CONTEXT.md` per node, each with a routing table, and
    that is a fair reading of upstream -- the records are part of the accepted
    version and a human writes the first ones. It is also, for every node below the
    root, the system's own first job done for it: the paper's phase 1 is
    *architecture and design*, and a tree that already says `src/potentials/pair/ ->
    one module per interaction` has had its architecture handed to it.

    So this strips every node record except the root's, and the root's routing table
    with them, leaving only what a human cannot avoid supplying: the goal, the
    contract, the suite that scores it, and any frozen entry script. The skills go
    too -- a hint about how to lay out Python packages is a hint about the shape of
    the answer.

    Nothing frozen is touched, so scoring is identical; what changes is that the run
    has to write its own `CONTEXT.md` chain as it goes, which `routing()` and
    `_routing_note` already support (that is what `routes_opened` counts).
    """
    kept: Dict[str, str] = {}
    for path, body in files.items():
        if path == CONTEXT_FILE:
            kept[path] = _without_routing(body)
        elif path.endswith(CONTEXT_FILE) and not path.startswith("spec/"):
            continue                                   # a node record: the run's job
        elif f"/{SKILLS_DIR}/" in f"/{path}" or path.startswith(f"{SKILLS_DIR}/"):
            continue                                   # a hint about the shape
        else:
            kept[path] = body
    return kept


def _without_routing(body: str) -> str:
    """The root record with its routing table removed, and a note saying why."""
    lines, out, dropping = body.splitlines(), [], False
    for line in lines:
        if line.strip().lower().startswith(ROUTING_HEADING.lower()):
            dropping = True
            out.append(ROUTING_HEADING)
            out.append("- (empty: open the nodes this needs, and record them here)")
            out.append("")
            continue
        if dropping:
            if line.startswith("## "):
                dropping = False
            else:
                continue
        out.append(line)
    return "\n".join(out) + "\n"


def plan_delegations(brief: Brief, owes: Callable[[Mapping[str, str], str], bool],
                     nodes: Sequence[str]) -> List[Delegation]:
    """Where the rule-based manager sends work, cold start included.

    The routing table first, which is what that table is for upstream: the
    decomposition is read from `CONTEXT.md` rather than hardcoded, and only the
    choice of *which* routed child to work on is a surrogate. But a cold-started
    world has no tables at all, so a manager that can only delegate to what it
    already routes to would have nothing to do forever. Upstream a manager may name
    a new node inside its own subtree and the table records it afterwards, so this
    one does the same: if nothing routed owes work, open the direct children its
    plan still owes work to.

    That makes the offline cold-start arm a test of the *mechanism* -- tables get
    written, the recursion reaches depth -- and nothing at all about inventing a
    decomposition, because the plan is the decomposition. Only a model arm can say
    anything about that.
    """
    here = normalise(brief.world.path)
    routed = [node for node in brief.world.routing(brief.state)
              if owes(brief.state, node)]
    if not routed:
        prefix = f"{here}/" if here else ""
        depth = len(prefix.split("/")) if here else 1
        routed = sorted({node for node in nodes
                         if node.startswith(prefix) and node != here
                         and owes(brief.state, node)},
                        key=len)
        # direct children only: the recursion is what reaches the rest
        routed = [node for node in routed
                  if len(normalise(node).split("/")) == depth]
    return [Delegation(node, f"clear the outstanding work under {node}/")
            for node in routed]


def preflight(complete) -> None:
    """One call before the run, so a dead backend costs a second instead of an hour.

    ``evolve()`` swallows an exception raised inside a proposal policy, and it is
    right to: one model call that times out should cost its episode and nothing
    more. But the same tolerance turns a *wrong endpoint* into a run that looks like
    a failure of the mechanism. A base URL with ``/v1/messages`` already on it -- the
    SDK appends its own -- 404s every call, the actors return no edits, every
    proposal is an empty accepted step, and thirteen minutes later the run reports
    ``reward 0.000  depth=0  accepted=400``. The only evidence was ``2400 failed``
    in the usage line, and nothing in the header said the model had never answered.

    So ask it one question first, and let that failure be loud.
    """
    try:
        complete("Reply with the single word OK.")
    except Exception as exc:  # noqa: BLE001 - anything at all means do not start
        raise SystemExit(
            f"model backend unreachable: {type(exc).__name__}: {exc}\n"
            "check --provider/--model and the endpoint -- ANTHROPIC_BASE_URL is the "
            "BASE url, and the SDK appends /v1/messages to it") from exc


def _ask(complete, prompt: str) -> str:
    try:
        return complete(prompt) or ""
    except Exception:  # noqa: BLE001 - a dead call costs this episode, not the run
        return ""


def _first_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            esc = (c == "\\") and not esc
            if c == '"' and not esc:
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


# ---------------------------------------------------------------------------
# Scoring by a frozen test suite instead of by an oracle
# ---------------------------------------------------------------------------

#: One test, in a child process. Executed rather than imported as a module, so a
#: test file needs no package plumbing and a candidate that cannot even be
#: imported fails the test it was asked about instead of taking the run down.
_TEST_HARNESS = r"""
import sys
workspace, path, func = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, workspace)
try:
    namespace = {"__name__": "_genesis_test", "__file__": path}
    with open(path, encoding="utf-8") as handle:
        exec(compile(handle.read(), path, "exec"), namespace)
    namespace[func]()
except BaseException as exc:                 # a failed assertion is a result
    detail = " ".join(str(exc).split())[:240]
    print("FAIL:" + type(exc).__name__ + (": " + detail if detail else ""), end="")
else:
    print("PASS", end="")
"""

#: What a test scores when it could not be run at all.
TEST_CRASHED = "FAIL:workspace"


def reward_test(task: Task, output: str) -> float:
    """One frozen test, passed or not. No expected value anywhere."""
    return 1.0 if output == "PASS" else 0.0


def run_test(state: Mapping[str, str], path: str, func: str,
             *, timeout: float = 60.0) -> str:
    """Materialise ``state`` and run one test function in a child process."""
    workspace = tempfile.mkdtemp(prefix="genesis-test-")
    try:
        materialize(state, workspace)
        proc = subprocess.run(
            [sys.executable, "-c", _TEST_HARNESS, workspace,
             os.path.join(workspace, *path.split("/")), func],
            capture_output=True, text=True, timeout=timeout, cwd=workspace)
        out = proc.stdout.strip()
        return out if out.startswith(("PASS", "FAIL")) else TEST_CRASHED
    except subprocess.TimeoutExpired:
        return "FAIL:Timeout"
    except (OSError, ValueError, subprocess.SubprocessError):
        return TEST_CRASHED
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def _discover_tests(files: Mapping[str, str]) -> List[Tuple[str, str, str]]:
    """``(file, function, prompt)`` for every top-level ``test_*`` in ``files``.

    The prompt is the test's own source **with its file's prelude above it** --
    imports, module constants, helper functions -- because a test body that reads
    ``forces(CONFIG, box=BOX)`` says nothing on its own about what ``CONFIG`` is.
    Everything in the file that is not itself a test goes in, in file order.
    """
    import ast

    def segment(body: str, node) -> str:
        return ast.get_source_segment(body, node) or ""

    found: List[Tuple[str, str, str]] = []
    for path in sorted(files):
        if not path.endswith(".py"):
            continue
        body = files[path]
        nodes = ast.parse(body, path).body
        tests = [n for n in nodes
                 if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]
        prelude = "\n\n".join(s for s in (segment(body, n) for n in nodes
                                           if n not in tests) if s)
        for node in tests:
            source = segment(body, node) or node.name
            found.append((path, node.name,
                          f"# {path}\n{prelude}\n\n\n{source}" if prelude else source))
    return found


@dataclass(frozen=True)
class TestSuite:
    """A formation domain scored by a **frozen test suite**, with no oracle.

    :class:`Suite` needs a reference implementation: it runs it to compute the
    expected answer for every case, and a candidate is scored by matching it. That
    is circular for the thing this port is for -- you cannot ask a system to grow
    software you had to write first -- and it is not what upstream does. Genesis
    validates against **c-testsuite, LLVM and Csmith**: assertions, not expected
    outputs, and no reference compiler anywhere in the loop.

    So here a task is one **test function**, its prompt is that test's own source,
    and its reward is whether it passes. The human writes a specification and a
    test suite, which is what a human actually writes; what the tests assert --
    tolerances, invariants, equivalences -- is theirs to choose, and nothing has to
    agree bit for bit with an implementation that already exists.

    A reference implementation may still exist beside the domain, for two jobs that
    are not scoring: driving the offline rule-based actor, and letting a test prove
    the suite is passable at all. Shipping a suite nobody has ever seen pass is its
    own kind of dishonesty.

    **The split is not a train/validation split**, and trying to make it one is a
    mistake this port made once already. ``evolve()`` holds out the tail of the task
    list and reports its reward. For :class:`Suite`, whose cases are sampled inputs
    over one grammar, that is a generalisation estimate and it means something. Here
    every task is a distinct **requirement**, so holding 40% of them back means
    refusing to tell the system four tenths of what it has to do and then grading it
    on them -- and the search never even sees those requirements fail, so nothing is
    ever proposed for them. That is exactly how the jqx run stalled at 0.923.

    The tail is an **audit set** instead: :attr:`audit` holds test files that are not
    in the repository at all, so no agent can read them, injected only when a held-out
    task runs. The driven suite is the whole specification; the audit set is the
    honest headline number, because the executor is shown the source of the test it
    is failing and could otherwise write to that one assertion. :meth:`held_out_frac`
    returns the fraction that makes ``evolve()``'s positional split land exactly on
    the boundary.
    """

    name: str
    #: Human-supplied and never the agents': context records, the specification,
    #: the test suite, any entry script.
    given: Mapping[str, str]
    #: Globs refused to every proposal and restored pristine before scoring. The
    #: tests belong here -- a system that can edit its own tests has no tests.
    frozen: Sequence[str]
    #: Where the test functions live.
    tests: Sequence[str] = ("tests/**",)
    #: ``path -> source`` for tests the agents never see. Deliberately *not* part of
    #: :meth:`initial_files`: frozen stops a file being written, not read, and an
    #: audit test in the repository is an audit test the executor can read.
    audit: Mapping[str, str] = field(default_factory=dict)
    #: ``name -> file overrides``: implementations that are **deliberately wrong**,
    #: applied on top of the reference. A suite is only as good as what it rejects,
    #: and nothing else in this design checks that. See :meth:`kill_report`.
    baselines: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    #: Seconds one test may take.
    timeout: float = 60.0

    def initial_files(self) -> Dict[str, str]:
        return dict(self.given)

    def kill_report(self, reference: Mapping[str, str], *,
                    stop_after: int = 0) -> Dict[str, List[str]]:
        """``baseline -> the task ids that fail on it``. Empty list means survived.

        This is the mechanism the md domain was missing, and the cost of missing it
        was a run that reported 1.000 with no force field in it. A suite validated
        only against a *correct* implementation is not validated: it has been shown
        to accept the right answer, and nothing has been shown about what it
        refuses. A field that returns zero everywhere satisfies every invariant --
        zero sums to zero, zero is the gradient of a constant, nothing moves so
        nothing drifts -- so a suite of invariants is exactly the shape that looks
        thorough and rejects nothing.

        Mutation testing is the standard answer and this is its small form: keep the
        wrong implementations, score each, and require every one of them to die.
        Note what this does *not* need: a reference in the **scoring** path. The
        reference exists beside the domain for jobs that are not scoring -- driving
        the offline actor, proving the suite passable -- and suite QA is a third one.

        Coverage, for the record, would not have caught it. The zero field executes
        every line of every test.

        This is the **harness**'s honesty check, not the algorithm's. Upstream has no
        objective function at all -- ``grep -rio 'fitness|reward|score'`` over
        ``apps/evo_git`` returns one hit, ``oom_score_adjust`` -- and what stands in
        its place is a parent that reviews the diff and runs the tests
        (``agents/manager.ex``). A reviewer reading the change is what rejects a
        registry that pushes every pair past the cutoff; a scalar cannot. Both belong
        in the port, in different places: this one keeps the *measurement* honest.

        ``stop_after`` stops scoring a baseline once that many tests have killed it,
        which is all a pass/fail guard needs and is much cheaper than the full grid.
        """
        run, tasks = self.make_runner(), self.build_tasks()
        report: Dict[str, List[str]] = {}
        for name, overrides in self.baselines.items():
            rendered = canonical(dict(reference, **overrides))
            killers: List[str] = []
            for task in tasks:
                if reward_test(task, run(rendered, task)) == 0.0:
                    killers.append(task.id)
                    if stop_after and len(killers) >= stop_after:
                        break
            report[name] = killers
        return report

    def held_out_frac(self) -> float:
        """The ``held_out_frac`` that puts exactly the audit tasks in the tail.

        ``evolve()`` cuts at ``round(n * (1 - frac))`` and refuses 0.0, so a domain
        with no audit set falls back to holding out one task.
        """
        total = len(self.build_tasks())
        audited = sum(1 for t in self.build_tasks() if t.meta.get("audit"))
        return max(audited, 1) / total

    def discover(self) -> List[Tuple[str, str, str]]:
        """``(file, function, source)`` for every ``test_*`` in the driven suite.

        Parsed rather than imported, so discovery works against a repository with
        no implementation in it -- which is every repository this port starts from.
        """
        return _discover_tests({path: body for path, body in self.given.items()
                                if match_any(path, self.tests)})

    def discover_audit(self) -> List[Tuple[str, str, str]]:
        """The same, over the audit files the repository does not contain."""
        return _discover_tests(self.audit)

    def build_tasks(self, limit: Optional[int] = None) -> List[Task]:
        """One task per test. This is the loader, so ``--dry-run`` stops before it.

        The tasks are **interleaved across files**, one from each in turn, because
        ``evolve()`` splits train from held-out by position: grouped by file, the
        last file or two would sit entirely in the held-out tail, no rollout could
        ever fail on them, and nothing would ever be proposed for the layer they
        test. That failure cost a whole jqx run to find.
        """
        found = self.discover()
        if not found:
            raise RuntimeError(f"the {self.name} suite declares no tests under "
                               f"{', '.join(self.tests)}")
        by_file: Dict[str, List[Tuple[str, str, str]]] = {}
        for entry in found:
            by_file.setdefault(entry[0], []).append(entry)
        ordered: List[Tuple[str, str, str]] = []
        for row in zip_longest(*by_file.values()):
            ordered.extend(entry for entry in row if entry is not None)
        tasks = [self._task(entry) for entry in (ordered[:limit] if limit else ordered)]
        # Last, and in file order: `evolve()` splits by position, so this is what
        # puts exactly the audit set in the held-out tail.
        tasks += [self._task(entry, audit=True) for entry in self.discover_audit()]
        return tasks

    def _task(self, entry: Tuple[str, str, str], *, audit: bool = False) -> Task:
        path, func, source = entry
        stem = path.rsplit("/", 1)[-1].removeprefix("test_").removesuffix(".py")
        return Task(id=f"{'audit:' if audit else ''}{stem}::{func}", prompt=source,
                    meta={"file": path, "func": func, "kind": stem, "audit": audit})

    def make_runner(self) -> Callable[[str, Task], str]:
        """``run(rendered, task)`` -- one test against one candidate repository."""

        def run(rendered: str, task: Task) -> str:
            try:
                state = dict(parse_tree(rendered))
            except Exception:  # noqa: BLE001 - an empty artifact, before anything
                return TEST_CRASHED
            for path, content in self.given.items():
                if match_any(path, self.frozen):
                    state[path] = content
            # Into the scratch copy the subprocess sees, never into the artifact:
            # the audit tests exist only for the length of one evaluation.
            state.update(self.audit)
            return run_test(state, task.meta["file"], task.meta["func"],
                            timeout=self.timeout)

        return run

    def review(self, tasks: Sequence[Task], *, sample: int = 8):
        """The parent's integration evidence: how many of these tests still pass."""
        chosen = [t for t in tasks if not t.meta.get("audit")][:sample]
        cached: Dict[int, int] = {}

        def passes(state: Mapping[str, str]) -> int:
            return sum(1 for t in chosen
                       if run_test(state, t.meta["file"], t.meta["func"],
                                   timeout=self.timeout) == "PASS")

        def review(parent, returned):
            key = id(parent.state)
            if key not in cached:
                cached[key] = passes(parent.state)
            candidate = dict(parent.state)
            for edit in returned:
                if edit.content is None:
                    candidate.pop(edit.path, None)
                else:
                    candidate[edit.path] = edit.content
            after = passes(candidate)
            if after < cached[key]:
                return ("rejected", f"integration check regressed "
                                    f"{cached[key]}/{len(chosen)} -> "
                                    f"{after}/{len(chosen)} tests")
            return None

        return review
