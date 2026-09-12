"""Recursive delegation as a :class:`~agentdescent.policies.ProposalPolicy`.

``(v, p) ⇝ (v, q)`` is the operation the engine has no counterpart for, and the
reason it fits *inside* a proposal rather than beside it is the paper's own
distinction: delegation **does not advance the accepted version**. A parent hands
work to a child at a deeper path, the child returns a candidate, the parent
accepts or refuses it -- and the project history has not moved. Only the
accepted event ``(v, p) → (v′, p)`` moves it, and that is the aggregator's
commit.

So one rollout here is one whole episode tree: a root manager decomposes, children
are situated deeper, leaf executors write files, each parent judges what came back,
and the surviving edits are octopus-merged into the single proposal the engine
consumes. Every version in that tree is the same ``v``; nothing in the middle
touches the ledger. That is the shape upstream runs, and it is expressible with no
engine change because ``ProposalPolicy`` is free to do whatever it likes between
being handed a failure and returning a proposal.

What the engine *does* see is unchanged: one proposal per rollout. The tree is
recorded in the :class:`~examples.genesis._world.WorldLog` instead, which is where
the observed depth and the per-episode verdicts come from -- upstream's archive,
minus the parts that only matter to a desktop app.

The two roles are callables the caller supplies, because that is the seam where a
domain plugs in:

    manager(brief)  -> [Delegation(path, objective), ...]   empty = do it here
    executor(brief) -> [Edit(owner, path, content), ...]

An offline rule-based pair and an LLM-backed pair are then the same mechanism run
with different actors, which is what makes the offline mode a test of *this file*
rather than a different algorithm.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from agentdescent.filetree import parse_tree

from ._world import (CONTEXT_FILE, EpisodeRecord, LocalWorld, WorldLog,
                     normalise, owns, routing_entry)

__all__ = ["Brief", "Delegation", "Edit", "RecursiveDelegation", "render_edits"]


@dataclass(frozen=True)
class Delegation:
    """``(v, p) ⇝ (v, q)``: where a child is situated, and what it is for."""

    path: str
    objective: str


@dataclass(frozen=True)
class Edit:
    """One file write by the agent situated at ``owner``. ``content=None`` deletes.

    ``kind`` separates the work from the bookkeeping. Maintaining ``CONTEXT.md``
    is an obligation upstream puts on every agent, but it must never cost the
    change it is describing: the trust region is small, and an edit set trimmed
    to fit should lose its routing note before it loses a source file.
    """

    owner: str
    path: str
    content: Optional[str]
    kind: str = "work"          # "work" | "context"


@dataclass(frozen=True)
class Brief:
    """Everything a finite-lived agent is given when it enters its world.

    Deliberately not "the whole repository": ``context`` is the ``CONTEXT.md``
    chain from the root down to this node plus a listing of the files this node
    is responsible for, which is what the path coordinate is *for*. ``state`` is
    the full tree, because an agent may read anything under ``v`` even though it
    may only write under ``p`` -- that asymmetry is the paper's, not a shortcut.
    """

    world: LocalWorld
    objective: str
    context: str
    state: Mapping[str, str]
    task: Any
    output: str
    reward: float
    depth: int


Manager = Callable[[Brief], Sequence[Delegation]]
Executor = Callable[[Brief], Sequence[Edit]]
Review = Callable[[Brief, Sequence[Edit]], Optional[str]]


def render_edits(edits: Sequence[Edit], rationale: str) -> str:
    """Serialise an edit set into the situated proposal protocol."""
    return "<EDITS>" + json.dumps({
        "rationale": rationale,
        "edits": [
            {"owner": e.owner, "path": e.path, **(
                {"delete": True} if e.content is None else {"content": e.content})}
            for e in edits
        ],
    }) + "</EDITS>"


@dataclass
class RecursiveDelegation:
    """One rollout = one recursive episode tree rooted at ``(v, root_path)``.

    ``max_depth`` bounds the recursion the way upstream's controller limits do.
    ``max_edits`` bounds what comes back: the strategy drops a proposal that
    exceeds its own per-diff cap *entirely*, so a tree that returns more edits
    than the trust region allows would lose all of them rather than the excess.
    Truncating here, and counting it, is the difference between a bounded
    proposal and a silently empty round.
    """

    manager: Manager
    executor: Executor
    log: WorldLog
    max_depth: int = 2
    max_edits: int = 4
    root_path: str = ""
    #: Optional real judging of a child's returned work. ``None`` runs the scope
    #: check alone -- the cheap half of the upstream rule, with the test half
    #: living in the acceptance gate where the held-out suite actually is.
    review: Optional[Review] = None
    truncated: int = 0
    #: Child nodes opened at a path their parent did not yet route to, and
    #: therefore written into the parent's routing table.
    routes_opened: int = 0
    #: Delegations refused because the target was a file, not a node.
    mistaken_nodes: int = 0

    # -- the ProposalPolicy protocol ---------------------------------------

    def propose(self, ctx) -> Sequence[str]:
        state = _state_of(ctx.rendered)
        root = LocalWorld(version=int(ctx.base_version or 0), path=self.root_path)

        # A parent that asked for more work gets it before anything else is
        # chosen: this is the third verdict, arriving one round later because
        # `AcceptDecision` cannot carry it (see examples/genesis/_judge.py).
        pending = self.log.take_rework()
        if pending is not None and owns(root.path, pending[0]):
            world, objective = root.delegate(pending[0]), f"rework: {pending[1]}"
            edits, _ = self._episode(world, objective, state, ctx, depth=1)
            rationale = f"rework at {pending[0] or './'}"
        else:
            edits, _ = self._episode(root, self._objective(ctx), state, ctx, depth=0)
            rationale = f"episode at {root.path or './'}"

        edits = self._bound(edits)
        if not edits:
            return []
        return [render_edits(edits, rationale)]

    # -- the recursion -----------------------------------------------------

    def _episode(self, world: LocalWorld, objective: str, state: Mapping[str, str],
                 ctx, *, depth: int) -> "tuple[List[Edit], EpisodeRecord]":
        """One finite-lived agent. Returns what it proposes, and its own record.

        The record travels back with the work because the verdict on an episode
        is its **parent's**, not its own: an agent that returns something it was
        pleased with and a parent that refuses it are one episode, not two.
        """
        brief = Brief(world=world, objective=objective,
                      context=world.situate(state), state=state, task=ctx.task,
                      output=ctx.output, reward=ctx.reward, depth=depth)

        delegations: Sequence[Delegation] = ()
        if depth < self.max_depth:
            delegations = [d for d in self.manager(brief) or ()
                           if self._is_node(world, d, state)]

        if not delegations:
            # A leaf executor: it writes files, it does not delegate. Its edits
            # are NOT filtered here -- the scope check belongs to the parent that
            # asked for the work (and, at the root, to the strategy), and doing
            # it twice would hide every refusal the parent is supposed to make.
            edits = [Edit(owner=world.path, path=e.path, content=e.content)
                     for e in self.executor(brief) or ()]
            record = self.log.record(EpisodeRecord(
                agent_id=self.log.next_id("executor"), role="executor",
                path=world.path, depth=depth, version=world.version,
                objective=objective, n_edits=len(edits)))
            return edits, record

        record = self.log.record(EpisodeRecord(
            agent_id=self.log.next_id("manager"), role="manager", path=world.path,
            depth=depth, version=world.version, objective=objective))

        merged: List[Edit] = []
        claimed: Dict[str, str] = {}          # path -> the child that took it
        routed = world.routing(state)
        opened: List[Delegation] = []         # children this node had not routed to
        for delegation in delegations:
            child = world.delegate(delegation.path)
            returned, child_record = self._episode(child, delegation.objective, state,
                                                   ctx, depth=depth + 1)
            child_record.verdict, child_record.reason = self._judge(
                child, returned, brief, claimed)
            if child_record.verdict == "rejected":
                # The rejected code does not survive; the *reason* can, as a
                # record at the child's own node. Paper, appendix 1.4: a saved
                # failure reason is a separate accepted event, and the code it
                # came from is not.
                note = self._failure_note(child, state, child_record.reason)
                if note is not None:
                    merged.append(note)
                continue
            if child_record.verdict == "rework":
                # Not a refusal: the parent wants another attempt, so the node is
                # queued rather than annotated. Writing a note for every "try
                # again" would turn CONTEXT.md into a transcript, which is the
                # one thing upstream says it must not become.
                self.log.request_rework(child.path, child_record.reason)
                continue
            for edit in returned:
                claimed[edit.path] = child.path
            merged.extend(returned)
            if normalise(delegation.path) not in routed:
                opened.append(delegation)

        # A node whose parent does not route to it is a node later agents cannot
        # find. Upstream the manager that opens one writes the entry at its own
        # level; here the same write, from the same agent, on its own CONTEXT.md.
        note = self._routing_note(world, state, opened)
        if note is not None:
            merged.append(note)

        record.n_edits = len(merged)
        return merged, record

    def _is_node(self, world: LocalWorld, delegation: Delegation,
                 state: Mapping[str, str]) -> bool:
        """May a child be situated here at all?

        Inside the parent's subtree, not the parent itself, and **not an existing
        file**. A node is a directory: upstream a path is a node when it holds at
        least one tracked file, and delegating to `lexer.py` as though it were one
        makes every write land *inside* a file. A real run did exactly that.
        """
        path = normalise(delegation.path)
        if not owns(world.path, path) or path == world.path:
            return False
        if path in state:
            self.mistaken_nodes += 1
            return False
        return True

    def _judge(self, child: LocalWorld, returned: Sequence[Edit], parent: Brief,
               claimed: Mapping[str, str]) -> "tuple[str, str]":
        """The responsible parent's verdict on one child's returned work.

        Scope first, because it is the check that does not need the suite: an
        edit outside the child's subtree is not the child's to make, and a
        sibling that has already claimed a path means two agents were given
        overlapping responsibility -- a decomposition fault, which upstream sends
        back rather than merging.
        """
        if not returned:
            return "rework", "the child returned no change"

        outside = [e.path for e in returned if not owns(child.path, e.path)]
        if outside:
            return "rejected", f"edit outside its subtree: {outside[0]}"
        collision = next((e.path for e in returned if e.path in claimed), None)
        if collision is not None:
            return "rejected", (f"{collision} was already written by a sibling at "
                                f"{claimed[collision]}; the decomposition overlaps")
        if self.review is not None:
            complaint = self.review(parent, returned)
            if complaint:
                return "rework", complaint
        return "accepted", ""

    def _routing_note(self, world: LocalWorld, state: Mapping[str, str],
                      opened: Sequence[Delegation]) -> Optional[Edit]:
        """Add every newly opened child to this node's routing table, once.

        One edit for all of them rather than one each: the trust region counts
        files, and a node's table is one file.
        """
        if not opened:
            return None
        key = world.context_key()
        body = state.get(key, f"# {world.path or './'}\n")
        changed = False
        for delegation in opened:
            updated = routing_entry(body, delegation.path,
                                    delegation.objective.strip() or "delegated work")
            if updated is not None:
                body, changed = updated, True
        if not changed:
            return None
        self.routes_opened += len(opened)
        return Edit(owner=world.path, path=key, content=body, kind="context")

    def _failure_note(self, child: LocalWorld, state: Mapping[str, str],
                      reason: str) -> Optional[Edit]:
        """Write the refusal into the child's ``CONTEXT.md``, if it is new.

        Appending unboundedly would make the note the artifact; upstream's own
        guidance is that ``CONTEXT.md`` records current state and is pruned when
        it grows. So one line, and only when it is not already there.
        """
        if not child.path:
            return None
        key = f"{child.path}/{CONTEXT_FILE}"
        line = f"- refused: {reason}"
        body = state.get(key, f"# {child.path}\n")
        if line in body:
            return None
        return Edit(owner=child.path, path=key, kind="context",
                    content=body.rstrip("\n") + "\n" + line + "\n")

    # -- bounds ------------------------------------------------------------

    def _bound(self, edits: Sequence[Edit]) -> List[Edit]:
        """Last write per path wins, then the trust region, counted.

        Work before bookkeeping, so a trimmed edit set loses a routing note
        before it loses a source file -- see :class:`Edit`.
        """
        by_path: Dict[str, Edit] = {}
        for edit in edits:
            by_path[edit.path] = edit
        work = sorted((p for p, e in by_path.items() if e.kind == "work"))
        context = sorted((p for p, e in by_path.items() if e.kind != "work"))
        ordered = [by_path[p] for p in work + context]
        if len(ordered) > self.max_edits:
            self.truncated += len(ordered) - self.max_edits
            ordered = ordered[:self.max_edits]
        return ordered

    def _objective(self, ctx) -> str:
        return (f"validation case {getattr(ctx.task, 'id', '?')} scored "
                f"{ctx.reward:.2f}: {getattr(ctx.task, 'prompt', '')}")

    def stats(self) -> str:
        return (f"delegation: {self.log.summary()} truncated_edits={self.truncated} "
                f"routes_opened={self.routes_opened} "
                f"mistaken_nodes={self.mistaken_nodes}")


def _state_of(rendered: str) -> Dict[str, str]:
    """The tree behind a rendered artifact. ``{}`` when it has not been built yet."""
    try:
        return dict(parse_tree(rendered))
    except Exception:  # noqa: BLE001 - an empty or not-yet-a-tree render
        return {}
