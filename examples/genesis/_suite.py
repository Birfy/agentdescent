"""The machinery a formation domain needs, with the domain itself left out.

A formation run needs four things that have nothing to do with *which* software is
being grown: a frozen validation suite whose expectations are computed from a
reference rather than typed out, a way to run a candidate repository in a child
process, the parent's integration check, and a pair of LLM actors. All four are
here, parameterised by :class:`Suite`; the domains -- :mod:`examples.genesis._domain`
(minilang) and :mod:`examples.genesis._stackvm` -- are data on top of it.

Split out when the second domain arrived, for the reason the first one should have
been: the two differ in the software they grow and in nothing else, and a copy of
the harness per domain is two places for the frozen-file restore to drift.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence

from agentdescent.evolution import Task
from agentdescent.filetree import match_any, materialize, parse_tree

from ._delegation import Brief, Delegation, Edit
from ._spatial import SITUATED_EDIT_PROTOCOL, parse_situated_edits
from ._world import normalise

__all__ = ["CRASHED", "PYTHON_MODULE_SKILL", "Suite", "llm_executor",
           "llm_manager", "reward", "run_cases"]

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

A validation case failed:
  input    {prompt}
  produced {output}
  score    {reward:.2f}

{protocol}"""


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
                 frozen: Sequence[str] = ()) -> Callable[[Brief], Sequence[Edit]]:
    """Ask a model for situated edits. Unparseable replies cost their episode.

    The protocol is rendered **per episode** rather than once, because it names the
    agent's own path in every example it shows. A protocol that says "relative
    path" without saying relative to what is read both ways, and the wrong reading
    puts a node's files at the top of the repository.
    """

    def executor(brief: Brief) -> Sequence[Edit]:
        owner = brief.world.path or "."
        protocol = SITUATED_EDIT_PROTOCOL.format(
            owner=owner, editable=", ".join(editable) or "(none)",
            frozen=", ".join(frozen) or "(none)")
        reply = _ask(complete, _EXECUTOR_PROMPT.format(
            path=brief.world.path or "./", context=brief.context,
            objective=brief.objective,
            prompt=getattr(brief.task, "prompt", ""),
            output=(brief.output or "")[:400], reward=brief.reward,
            protocol=protocol))
        return [Edit(owner=brief.world.path or "", path=edit["path"],
                     content=edit["content"])
                for edit in parse_situated_edits(reply)]

    return executor


def _ask(complete, prompt: str) -> str:
    try:
        return complete(prompt) or ""
    except Exception:  # noqa: BLE001 - a dead backend costs this episode, not the run
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
