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
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from agentdescent.filetree import parse_tree

from ._octopus import three_way

from ._world import (CONTEXT_FILE, KNOWN_ISSUES, EpisodeRecord, LocalWorld,
                     WorldLog, directly_at, normalise, owns, resolve_edit_path,
                     routing_entry, under_heading)

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

    With one exception, and ``--cold-start`` is what found it. A note that *creates*
    a node's record is ``"record"``, and it is trimmed **last** -- because then it is
    not bookkeeping at all. It is the only place the decomposition the system just
    invented is written down, and nothing else in the accepted version carries it;
    the source file it would be dropped for is re-proposable next round, the
    structure is not. While every domain shipped a ``CONTEXT.md`` per node this
    could not show: the note only ever added a line to a table that already existed.
    """

    owner: str
    path: str
    content: Optional[str]
    kind: str = "work"          # "record" | "work" | "context"


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

#: What the responsible parent does with a returned contribution beyond checking
#: its scope. Returns ``None`` to accept, or ``(verdict, reason)`` with a verdict
#: of ``"rejected"`` or ``"rework"``. Two outcomes rather than a complaint string,
#: because the paper's parent has three: *accepted, rejected, or requires further
#: work* (3.3), and collapsing the last two loses the one that comes back.
Review = Callable[[Brief, Sequence[Edit]], Optional[Tuple[str, str]]]


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
    #: Human-supplied, read-only files every agent is shown in full, wherever it
    #: stands -- the specification, the constraints, the validation contract. The
    #: paper's agent may inspect the whole project; one with no read tool can only
    #: inspect what the brief carries.
    contracts: Sequence[str] = ()
    #: Globs that are read-only. Routing must not send a manager into a directory
    #: where nothing is writable; separate from :attr:`contracts`, which is what
    #: gets *pushed into the brief*. Upstream an agent pulls files with tools, so
    #: "what may not be written" and "what is handed over unasked" are two sets.
    #: Defaults to ``contracts``.
    readonly: Sequence[str] = ()
    #: Upstream's review-and-accountability phase: after its children return, a
    #: manager gets one turn at its own node. Off makes a manager a pure router,
    #: which is what this port was and why a node's own file never appeared.
    accountability: bool = True
    #: The parent's judgement beyond scope. Upstream a parent decides "using the
    #: available tests, constraints and integration evidence" (paper 3.3), which
    #: is a *test* run on the child's work before it is offered to the version
    #: history at all -- not the same thing as the acceptance gate, which sees
    #: only what the whole episode returned. ``None`` runs the scope check alone.
    review: Optional[Review] = None
    truncated: int = 0
    #: Child nodes opened at a path their parent did not yet route to, and
    #: therefore written into the parent's routing table.
    routes_opened: int = 0
    #: Delegations refused because the target was a file, not a node.
    mistaken_nodes: int = 0
    #: Sibling edits to one path reconciled by three-way merge, and not.
    sibling_merges: int = 0
    sibling_conflicts: int = 0
    #: Files a manager wrote at its own node in the accountability phase, and
    #: edits it offered there that belonged to a child and were declined.
    accountability_edits: int = 0
    accountability_declined: int = 0
    #: Edits whose path was written relative to the agent's own node rather than
    #: to the repository. Counted because the alternative to counting is a file
    #: appearing somewhere nobody asked for it.
    resolved_relative: int = 0
    #: Changes an agent needed outside its own path: raised, handled by an
    #: ancestor, and left unmet at the top of the chain.
    requests_raised: int = 0
    adopted_requests: int = 0
    unmet_requests: int = 0

    # -- the ProposalPolicy protocol ---------------------------------------

    def propose(self, ctx) -> Sequence[str]:
        state = _state_of(ctx.rendered)
        root = LocalWorld(version=int(ctx.base_version or 0), path=self.root_path,
                          readonly=tuple(self.readonly or self.contracts))

        # A parent that asked for more work gets it before anything else is
        # chosen: this is the third verdict, arriving one round later because
        # `AcceptDecision` cannot carry it (see examples/genesis/_judge.py).
        pending = self.log.take_rework()
        if pending is not None and owns(root.path, pending[0]):
            world, objective = root.delegate(pending[0]), f"rework: {pending[1]}"
            edits, unmet, _ = self._episode(world, objective, state, ctx, depth=1)
            rationale = f"rework at {pending[0] or './'}"
        else:
            edits, unmet, _ = self._episode(root, self._objective(ctx), state, ctx,
                                            depth=0)
            rationale = f"episode at {root.path or './'}"
        # A need that reached the top of this episode's chain and still fell
        # outside its root's authority. Counted rather than dropped quietly: it
        # means the decomposition put work where nobody could do it.
        self.unmet_requests += len(unmet)

        edits = self._bound(edits)
        if not edits:
            return []
        return [render_edits(edits, rationale)]

    # -- the recursion -----------------------------------------------------

    def _episode(self, world: LocalWorld, objective: str, state: Mapping[str, str],
                 ctx, *, depth: int) -> "tuple[List[Edit], List[Edit], EpisodeRecord]":
        """One finite-lived agent: what it proposes, what it asks for, its record.

        The record travels back with the work because the verdict on an episode
        is its **parent's**, not its own: an agent that returns something it was
        pleased with and a parent that refuses it are one episode, not two.
        """
        brief = Brief(world=world, objective=objective,
                      context=world.situate(state, contracts=self.contracts),
                      state=state, task=ctx.task, output=ctx.output,
                      reward=ctx.reward, depth=depth)

        delegations: Sequence[Delegation] = ()
        if depth < self.max_depth:
            delegations = [d for d in self.manager(brief) or ()
                           if self._is_node(world, d, state)]

        if not delegations:
            # A leaf executor: it writes files, it does not delegate. What it
            # produces outside its own path is not a violation -- upstream an
            # agent that needs a change it has no authority for "reports the need
            # back up to the parent agent, which will handle it"
            # (`agents/executor.ex:64`). So the split here is edits / requests,
            # and the parent decides what to do with the second list.
            produced = []
            for raw in self.executor(brief) or ():
                # "relative path" means two things to an agent standing in a
                # subtree, and reading it the wrong way sends real work to the
                # repository root. Resolve it against the node before anything
                # else looks at it, and count the ones that needed it.
                path, relative = resolve_edit_path(state, world.path, raw.path)
                self.resolved_relative += int(relative)
                produced.append(Edit(owner=world.path, path=path,
                                     content=raw.content, kind=raw.kind))
            edits = [e for e in produced if owns(world.path, e.path)]
            requests = [e for e in produced if not owns(world.path, e.path)]
            self.requests_raised += len(requests)
            record = self.log.record(EpisodeRecord(
                agent_id=self.log.next_id("executor"), role="executor",
                path=world.path, depth=depth, version=world.version,
                objective=objective, n_edits=len(edits)))
            return edits, requests, record

        record = self.log.record(EpisodeRecord(
            agent_id=self.log.next_id("manager"), role="manager", path=world.path,
            depth=depth, version=world.version, objective=objective))

        held: Dict[str, Edit] = {}            # path -> what the parent holds so far
        owner_of: Dict[str, str] = {}         # path -> the child that wrote it
        notes: List[Edit] = []
        pending: List[Edit] = []              # needs this node cannot meet either
        routed = world.routing(state)
        opened: List[Delegation] = []         # children this node had not routed to
        for delegation in delegations:
            child = world.delegate(delegation.path)
            returned, asked, child_record = self._episode(
                child, delegation.objective, state, ctx, depth=depth + 1)
            child_record.verdict, child_record.reason = self._judge(
                child, returned, asked, brief)
            if child_record.verdict != "rejected" and asked:
                # "Report the need back up to the parent, which will handle it."
                # This node handles what falls inside its own authority and passes
                # the rest further up; at the root, authority is the whole world,
                # so nothing is left unmet.
                mine = [replace(e, owner=world.path) for e in asked
                        if owns(world.path, e.path)]
                if mine:
                    self.adopted_requests += len(mine)
                    self._fold(held, owner_of, mine, world, state)
                pending += [e for e in asked if not owns(world.path, e.path)]
            if child_record.verdict == "rejected":
                # The rejected code does not survive; the *reason* can, as a
                # record at the child's own node. Paper, appendix 1.4: a saved
                # failure reason is a separate accepted event, and the code it
                # came from is not.
                note = self._failure_note(child, state, child_record.reason)
                if note is not None:
                    notes.append(note)
                continue
            if child_record.verdict == "rework":
                # Not a refusal: the parent wants another attempt, so the node is
                # queued rather than annotated. Writing a note for every "try
                # again" would turn CONTEXT.md into a transcript, which is the
                # one thing upstream says it must not become.
                self.log.request_rework(child.path, child_record.reason)
                continue
            conflicts = self._fold(held, owner_of, returned, child, state)
            if conflicts:
                # Upstream's octopus merge returns the conflicting file list to
                # the parent, which resolves, aborts, or re-plans
                # (`agent/subagent_processing.ex:527`). Re-planning is the one a
                # parent here can take: the child's other edits stand, and the
                # overlapping one goes back to it.
                child_record.verdict = "rework"
                child_record.reason = (
                    f"{conflicts[0]} was also written by a sibling at "
                    f"{owner_of.get(conflicts[0], '?')}, and the two edits overlap")
                self.log.request_rework(child.path, child_record.reason)
            if normalise(delegation.path) not in routed:
                opened.append(delegation)

        # Upstream's third phase. An Architect works "architecture & design ->
        # implementation delegation -> **review & accountability**" and is
        # "ACCOUNTABLE for all code in its node path" (`agents/architect.ex:23`):
        # delegating does not discharge that, and the files that belong to the
        # node itself are nobody else's to write. Without this a manager
        # delegated forever and its own node's file was never written -- measured:
        # five correct modules and no `src/__init__.py`, so the public surface the
        # specification names did not exist and every case scored zero.
        if self.accountability:
            own = self._accountability_pass(world, objective, state, held, ctx, depth)
            if own:
                self._fold(held, owner_of, own, world, state)

        merged: List[Edit] = list(held.values()) + notes

        # A node whose parent does not route to it is a node later agents cannot
        # find. Upstream the manager that opens one writes the entry at its own
        # level; here the same write, from the same agent, on its own CONTEXT.md.
        routing = self._routing_note(world, state, opened)
        if routing is not None:
            merged.append(routing)

        record.n_edits = len(merged)
        return merged, pending, record

    def _accountability_pass(self, world: LocalWorld, objective: str,
                             state: Mapping[str, str], held: Mapping[str, Edit],
                             ctx, depth: int) -> List[Edit]:
        """One turn for the manager at its own node, after its children return.

        It sees the tree **as its children just left it**, because what the node
        still needs depends on what came back. Restricted to files directly at the
        node: a manager is accountable for its whole subtree but a child's files
        are the child's to write, and a manager free to rewrite them would make
        the decomposition decorative.
        """
        amended = dict(state)
        for edit in held.values():
            if edit.content is None:
                amended.pop(edit.path, None)
            else:
                amended[edit.path] = edit.content
        brief = Brief(world=world, objective=f"review and accountability: {objective}",
                      context=world.situate(amended, contracts=self.contracts),
                      state=amended, task=ctx.task, output=ctx.output,
                      reward=ctx.reward, depth=depth)
        out: List[Edit] = []
        for raw in self.executor(brief) or ():
            path, relative = resolve_edit_path(amended, world.path, raw.path)
            self.resolved_relative += int(relative)
            if directly_at(world.path, path):
                out.append(Edit(owner=world.path, path=path, content=raw.content,
                                kind=raw.kind))
            else:
                self.accountability_declined += 1
        self.accountability_edits += len(out)
        return out

    def _fold(self, held: Dict[str, Edit], owner_of: Dict[str, str],
              returned: Sequence[Edit], child: LocalWorld,
              state: Mapping[str, str]) -> List[str]:
        """Octopus-merge one child's edits into what the parent already holds.

        Two children can legally write the same path when one is situated inside
        the other's subtree, and upstream does not treat that as a fault: the
        parent runs ``git merge --octopus`` over what came back and only a real
        overlap becomes a conflict. Rejecting on a bare path collision -- which is
        what this used to do -- discards a whole contribution for touching a file
        a sibling also touched, which is exactly the merge the system is for.
        """
        conflicts: List[str] = []
        for edit in returned:
            mine = held.get(edit.path)
            if mine is None:
                held[edit.path] = edit
                owner_of[edit.path] = child.path
                continue
            if mine.content == edit.content:
                continue                      # the two agreed; nothing to merge
            if mine.content is None or edit.content is None:
                conflicts.append(edit.path)   # a delete against an edit
                self.sibling_conflicts += 1
                continue
            fused = three_way(state.get(edit.path, ""), mine.content, edit.content)
            if fused is None:
                conflicts.append(edit.path)
                self.sibling_conflicts += 1
                continue
            held[edit.path] = replace(mine, content=fused)
            self.sibling_merges += 1
        return conflicts

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

    def _judge(self, child: LocalWorld, returned: Sequence[Edit],
               asked: Sequence[Edit], parent: Brief) -> "tuple[str, str]":
        """The responsible parent's verdict on one child's returned work.

        Two things are deliberately *not* judged here. A change the child wanted
        outside its own path is a **request**, not an overstep -- upstream an
        agent reports such a need upward and the ancestor with authority handles
        it. And an overlap with a sibling is a **merge**, handled in
        :meth:`_fold`. What is left is the question the paper's parent actually
        answers: is this contribution good, on the tests, constraints and
        integration evidence available (3.3)?
        """
        if not returned:
            if asked:
                return "rework", (f"the child did no work at its own path; it asked "
                                  f"for {len(asked)} change(s) elsewhere, starting "
                                  f"with {asked[0].path}")
            return "rework", "the child returned no change"
        if self.review is not None:
            verdict = self.review(parent, returned)
            if verdict is not None:
                return verdict
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
        # Creating the record outranks the work; adding a line to one does not.
        return Edit(owner=world.path, path=key, content=body,
                    kind="context" if key in state else "record")

    def _failure_note(self, child: LocalWorld, state: Mapping[str, str],
                      reason: str) -> Optional[Edit]:
        """Write the refusal into the child's ``CONTEXT.md``, if it is new.

        Under ``## Known Issues``, which is upstream's own heading for exactly this:
        "findings worth preserving belong in CONTEXT.md... `## Known Issues`
        (problems to avoid re-discovering)" (``agents/manager.ex``). A refusal is the
        cheapest such finding there is -- the next agent at this node would otherwise
        rediscover it by being refused again.

        Appending unboundedly would make the note the artifact; upstream's guidance is
        that ``CONTEXT.md`` records current state and is pruned when it grows. So one
        line, and only when it is not already there.
        """
        if not child.path:
            return None
        key = f"{child.path}/{CONTEXT_FILE}"
        line = f"- refused: {reason}"
        body = state.get(key, f"# {child.path}\n")
        if line in body:
            return None
        return Edit(owner=child.path, path=key, kind="context",
                    content=under_heading(body, KNOWN_ISSUES, line))

    # -- bounds ------------------------------------------------------------

    def _bound(self, edits: Sequence[Edit]) -> List[Edit]:
        """Last write per path wins, then the trust region, counted.

        Work before bookkeeping, so a trimmed edit set loses a routing note before
        it loses a source file -- and a record that brings a node into existence
        before either, because nothing else in the accepted version says the node
        exists. See :class:`Edit`.
        """
        by_path: Dict[str, Edit] = {}
        for edit in edits:
            by_path[edit.path] = edit
        order = {"record": 0, "work": 1}
        ordered = [by_path[p] for p in
                   sorted(by_path, key=lambda p: (order.get(by_path[p].kind, 2), p))]
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
                f"mistaken_nodes={self.mistaken_nodes} "
                f"sibling_merges={self.sibling_merges} "
                f"sibling_conflicts={self.sibling_conflicts} "
                f"requests={self.requests_raised}/"
                f"{self.adopted_requests}/{self.unmet_requests} "
                f"node_relative_paths={self.resolved_relative} "
                f"accountability={self.accountability_edits}/"
                f"{self.accountability_declined}")


def _state_of(rendered: str) -> Dict[str, str]:
    """The tree behind a rendered artifact. ``{}`` when it has not been built yet."""
    try:
        return dict(parse_tree(rendered))
    except Exception:  # noqa: BLE001 - an empty or not-yet-a-tree render
        return {}
