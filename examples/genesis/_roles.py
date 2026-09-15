"""The Manager and the ContextExtractor as sessions, on :mod:`._session`.

Two of upstream's root-capable roles, each of which this port ran as a single
completion returning JSON.

**Manager** (``agents/manager.ex``) is the orchestrator: its moduledoc opens with "The
Manager does NOT implement features directly" and lists five jobs -- analyse the
objective, plan the work, delegate to subagents, **validate results and handle
conflicts**, report completion. It is ``:read_write`` and ``delegation_level :high``,
and in Mode B it is the *second* root agent: the Architect designs and commits, the
Manager starts from that commit and fills the tree in.

Two of those five jobs are model calls in this port, and both get a session here:

* *plan and delegate* -- :class:`ManagerSession`. The completion decided where to
  delegate from the routing table and the record alone. A session can **look**: the
  routing table is there so a parent need not investigate (upstream's own words: "you
  don't need to investigate the subtree first"), but *need not* is not *cannot*, and
  the Manager is the role upstream gives read tools to precisely so it can check.
* *validate results* -- :class:`ReviewSession`. The completion was handed a rendered
  diff, truncated at 12 000 characters. A session reads **the files the child actually
  wrote**, in place, with Glob and Grep. That difference is the whole point of the
  role: a reviewer judging a truncated rendering of work is not reviewing the work.

**ContextExtractor** (``agents/context_extractor.ex``) is Mode A's root agent, and the
reason Mode A exists: a repository with code and no ``CONTEXT.md`` cannot be worked on
by recursive delegation at all, because the routing table *is* the map. It is ``:read``
-- it may not change the code -- but it writes ``CONTEXT.md``, which is why upstream's
read-only tool set keeps ``context_write``/``context_edit``. :class:`ExtractSession`
is that: read tools plus permission to write exactly one record.

Its record is richer than an architect's, and upstream says why -- the sections it
lists include Design Decisions, **Notes for Agents** ("this file is generated, don't
split it"), Dependencies, Test Strategy and Status, on the rule that a section earns
its place if it "would save an agent from re-investigating or re-discovering
something". An architect inventing a directory has nothing to record there yet; an
extractor reading one does.

Both keep this port's standing difference: **the recursion is in the driver.** One
session does one node, and the phase or the delegation policy drives the next.
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional, Sequence, Tuple

from ._session import (READ_ONLY_TOOLS, SCRATCH_DIR, AgentSession)
from ._world import CONTEXT_FILE, normalise

__all__ = ["EXTRACT_BRIEF", "MANAGER_BRIEF", "REVIEW_BRIEF",
           "ExtractSession", "ManagerSession", "ReviewSession"]


_PLAN = f"{SCRATCH_DIR}/plan.md"
_VERDICT = f"{SCRATCH_DIR}/verdict.md"

#: One delegation per line, `path -> objective`. A line and not JSON because the
#: failure this whole file exists to remove is "the structured reply did not parse":
#: a plan of five children where the fourth line is malformed should delegate four
#: children, not none.
_PLAN_LINE = re.compile(
    r"""^\s*[-*]?\s*`?\s*(?P<path>[A-Za-z0-9._\-/]+?)\s*/?`?\s*
        (?:[-=]+>|→|:)\s*(?P<objective>.+?)\s*$""", re.VERBOSE)

MANAGER_BRIEF = """You are a manager agent in a recursive software world, situated at \
the repository path `{path}`. You own everything under it. You do not write code.

{context}

This node's routing table says its children are: {routes}

THE OBJECTIVE
{objective}

Decide whether to delegate to more specific paths inside your own subtree, or to \
handle this here. You have read tools -- the routing table means you do not *have* to \
investigate the subtree, but you may, and a delegation aimed at the wrong child costs \
a whole episode.

Rules:
- You are ACCOUNTABLE for all code under `{path}`, and delegating does not discharge \
that. The files AT `{path}` itself are nobody else's to write: after your children \
return you get one more turn to write them.
- A node is a **directory**, never a file. `src/frontend` is a node; \
`src/frontend/lexer.py` is a file belonging to the agent situated at `src/frontend`, \
and delegating to it is refused.
- Delegate only inside `{path}`. A path outside your subtree is refused and counted.

--- HOW TO DELIVER THIS ---

Write your plan to `{plan}`, one delegation per line, and stop:

    <path> -> <one sentence saying what that child is to do>

An empty file means you are handling this node yourself. Do not reply with the plan; \
what you wrote is read from the working tree, and a reply is not read at all. Change \
no other file."""

REVIEW_BRIEF = """You are the parent agent at `{path}`, reviewing what a child \
returned before you accept it. You are ACCOUNTABLE for all code under your node.

{context}

THE OBJECTIVE
{objective}

The child situated at `{child}` has written its work into this working tree. **Read \
the files it wrote.** They are on disk, and you have Read, Glob and Grep -- do not \
judge from the summary below, which is only a list of what changed:

{changed}

Reject only for something that is wrong: code that cannot run, an import that does not \
resolve, a contract the record forbids, a file written where a child had no business \
writing. Do not reject for style, for work that is merely incomplete -- partial \
progress is accepted here -- or for something you would rather have been done \
differently.

--- HOW TO DELIVER THIS ---

Write your verdict to `{verdict}` and stop. The first line is exactly `ACCEPT` or \
`REJECT`; if you reject, the lines after it say why, concretely enough that the child \
can act on it. Change no other file."""

EXTRACT_BRIEF = """You are a context extractor: an expert software architect reading \
an existing codebase and writing the map others navigate it by. You are situated at \
`{path}`. **You may not change the code** -- you read it and you describe it.

{context}

THE OBJECTIVE THIS REPOSITORY SERVES
{objective}

Read what is actually in `{path}` and write its `CONTEXT.md`. You are describing what \
is there, not what should be: if the code went somewhere no design anticipated, the \
code is what is true.

Sections, in this order. The first four are required; include the rest when they would \
save a later agent from re-investigating something, and leave them out when they would \
not:

- `## Intent` -- what this directory is for.
- `## API Surface` -- the files this node owns and what each exposes.
- `## Constraints` -- rules for code here. Omit what the parent already says.
- `## Routing Table` -- one line per child directory that exists, written \
`- ./{path_prefix}<name>/ (<N> files) -> <what it handles>`.
- `## Design Decisions` -- why something is the way it is, where that is not obvious.
- `## Known Issues` -- gotchas, subtle bugs, tricky behaviour. Where two things \
collide and one of them has to go, say so and say which.
- `## Notes for Agents` -- what would waste an agent's time. "This file is generated, \
do not split it" is the shape.
- `## Dependencies` -- what is needed beyond the package manager.
- `## Test Strategy` -- how this directory is tested, and what is not covered.

Only directories that **exist** go in the routing table: you are finding children, not \
inventing them.

--- HOW TO DELIVER THIS ---

Write the record to `{record}` and stop. Create no other file, change no code, and do \
not reply with the record -- what you wrote is read from the working tree."""


class ManagerSession:
    """``manager(brief) -> [Delegation]``, as a session that may read the subtree."""

    def __init__(self, delegation_cls, **kwargs):
        self._delegation = delegation_cls
        kwargs.setdefault("tools", READ_ONLY_TOOLS + ("Write",))
        self.session = AgentSession(**kwargs)

    def __call__(self, brief) -> Sequence[object]:
        path = normalise(brief.world.path)
        routes = brief.world.routing(brief.state)
        prompt = MANAGER_BRIEF.format(
            path=path or "./", context=brief.context, plan=_PLAN,
            routes=", ".join(f"`{r}/`" for r in routes) or "(none yet)",
            objective=brief.objective)
        got = self.session.run(brief.state, prompt, read=[_PLAN])
        return [self._delegation(p, o) for p, o in _plan(got.get(_PLAN, ""))]

    def summary(self) -> str:
        return self.session.summary()


class ReviewSession:
    """``review(parent, returned) -> ("rejected", why) | None``, reading real files.

    The rejection shape is the one :func:`examples.genesis._review.chain_reviews`
    already composes, so this drops in beside the deterministic checks.
    """

    def __init__(self, *, contracts: Sequence[str] = (), **kwargs):
        self._contracts = tuple(contracts)
        kwargs.setdefault("tools", READ_ONLY_TOOLS + ("Write",))
        self.session = AgentSession(**kwargs)
        #: How many children were read, and how many were sent back.
        self.reviewed = 0
        self.rejected = 0
        self.unparsed = 0

    def __call__(self, parent, returned) -> Optional[Tuple[str, str]]:
        work = [e for e in returned
                if e.kind == "work" and e.content is not None]
        if not work:
            return None                      # bookkeeping only: nothing to read
        candidate = dict(parent.state)
        for edit in work:
            candidate[edit.path] = edit.content
        child = _child_of(work)
        prompt = REVIEW_BRIEF.format(
            path=normalise(parent.world.path) or "./", child=child,
            context=parent.world.situate(parent.state, contracts=self._contracts),
            objective=parent.objective, verdict=_VERDICT,
            changed="\n".join(f"- {e.path}" for e in work[:60]))
        self.reviewed += 1
        got = self.session.run(candidate, prompt, read=[_VERDICT])
        verdict, reason = _verdict(got.get(_VERDICT, ""))
        if verdict != "reject":
            # An unparseable verdict is not a rejection: the child did the work, and
            # a reviewer that cannot speak is not evidence against it.
            self.unparsed += int(verdict is None)
            return None
        self.rejected += 1
        return ("rejected", f"parent review: {reason or 'unspecified'}")

    def summary(self) -> str:
        return (f"{self.session.summary()} read={self.reviewed} "
                f"rejected={self.rejected} unparsed={self.unparsed}")


class ExtractSession:
    """``extract(state, path, objective) -> record | None`` for one existing node."""

    def __init__(self, **kwargs):
        # `:read` upstream, plus the one thing a read-only agent still writes: its
        # own `CONTEXT.md` (`agent/tools.ex` keeps `context_write` in the read-only
        # set for exactly this). Write is granted; the brief scopes it to one file
        # and the caller reads back only that file.
        kwargs.setdefault("tools", READ_ONLY_TOOLS + ("Write",))
        self.session = AgentSession(**kwargs)

    def __call__(self, state, path: str, objective: str, *,
                 context: str = "") -> Optional[str]:
        path = normalise(path)
        record = f"{path}/{CONTEXT_FILE}" if path else CONTEXT_FILE
        prompt = EXTRACT_BRIEF.format(
            path=path or "./", path_prefix=f"{path}/" if path else "",
            objective=objective, context=context, record=record)
        got = self.session.run(state, prompt, read=[record])
        body = (got.get(record) or "").strip()
        return (body + "\n") if body else None

    def summary(self) -> str:
        return self.session.summary()


# -- readers --------------------------------------------------------------------

def _plan(body: str) -> List[Tuple[str, str]]:
    """``(path, objective)`` per usable line; a malformed line costs only itself."""
    out: List[Tuple[str, str]] = []
    seen = set()
    for line in (body or "").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = _PLAN_LINE.match(line)
        if not match:
            continue
        path = normalise(match.group("path"))
        if path and path not in seen:
            seen.add(path)
            out.append((path, match.group("objective").strip()))
    return out


def _verdict(body: str) -> Tuple[Optional[str], str]:
    """``("accept"|"reject"|None, why)``. The first non-empty line decides."""
    lines = [l.strip() for l in (body or "").splitlines()]
    lines = [l for l in lines if l]
    if not lines:
        return None, ""
    head = lines[0].strip("`*# ").upper()
    reason = " ".join(lines[1:]).strip()
    if head.startswith("REJECT"):
        return "reject", reason
    if head.startswith("ACCEPT"):
        return "accept", reason
    return None, reason


def _child_of(work) -> str:
    """The deepest directory every edit in `work` sits under."""
    parts: Optional[List[str]] = None
    for edit in work:
        here = normalise(edit.path).split("/")[:-1]
        parts = here if parts is None else [
            a for a, b in zip(parts, here) if a == b]
    return "/".join(parts or []) or "./"
