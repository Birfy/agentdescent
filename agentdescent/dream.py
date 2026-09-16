"""Dream-RSI: a finished discovery run, replayed as a simulator of itself.

`meta.py` evolves one decision slot of `evolve()`, and states the cost plainly:
*"One outer rollout is therefore an entire inner search."* A selection rule is
judged by running a whole search under it, so a meta-round costs a meta-round's
worth of model calls, and the measured run asked for 8 rounds and got 2.

Dream-RSI (Zheng et al., 2026, `arxiv/dream-rsi`) removes that cost for the
decisions an exploration policy actually makes. Its observation: a completed
discovery run *already recorded* the outcome of every attempt it made, arranged
as a tree. An alternative policy that walks the same tree -- taking a different
subset of branches, in a different order, in different parallel groupings,
stopping at a different point -- can be scored by **revealing outcomes that are
already on disk**. No agent runs, no evaluator runs. One expensive online
rollout buys thousands of off-policy evaluations::

    pool = SimulatorPool()
    spec = exploration_policy()                      # the policy, as gated source
    result = dream_rsi(
        continue_fn,                                 # the discovery agent: node -> Attempt
        spec=spec, model=model, pool=pool,
        objective=ReplayObjective.scaled(max_nodes=40, n_workers=4),
        rounds=4, n_workers=4)
    policy = spec.compile(result.rendered)            # the redeployable policy

The loop is the paper's three stages, and each is an object here:

1. **Online explore** -- :func:`explore` runs the current policy against a real
   discovery agent for at most ``K1`` decision rounds and records what happened
   as a :class:`DiscoveryTree`.
2. **Construct replay simulator** -- the tree joins a :class:`SimulatorPool`,
   which is the history ``H_t = (T_1, ..., T_t)``.
3. **Dreaming** -- every world in the pool becomes a
   :class:`~agentdescent.meta.Problem`, and :func:`~agentdescent.meta.meta_evolve`
   runs over them exactly as it runs over live inner searches. The artifact is
   the policy's source, the reward is the replay objective of Equation 1, and
   the reflector reads replay *trajectories* rather than a live curve.

Then the selected policy is redeployed online and the pool grows. That is the
recursive part: the worlds the policy is dreamt in are the worlds its own
earlier deployments built.

**Why this fits here without a new engine.** The paper's exploration policy maps
a discovery tree to a batch of nodes to continue. That is
:class:`~agentdescent.selection.SelectionPolicy` -- ``select(ctx, n)`` over
``ctx.candidates``, with ``n`` the worker count -- so the artifact is the
``selection`` slot, the gate is :func:`~agentdescent.meta.compile_policy_source`,
and the outer loop is :func:`~agentdescent.meta.meta_evolve` unchanged. What is
new is only the *problem*: a replay instead of a run.

**What is faithful, and what is not**, stated once here and again in
``docs/algo-dream-rsi.md``: no code has been released for Dream-RSI, so this
follows the paper's equations rather than a repository, and where the main text
and Appendix B.2 disagree it follows the appendix and says so. The results in
the paper are not reproduced here -- the shipped example is an offline synthetic
world, and nothing in this module has been measured against a published number.
"""

from __future__ import annotations

import concurrent.futures
import json
import math
import statistics
import threading
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence,
    Tuple,
)

from .agents import Completion
from .evolution import Task
from .meta import (MetaOutcome, MetaReward, Problem, SlotSpec, SourceSlot,
                   meta_evolve, policy_source)
from .selection import Candidate, SelectionContext, SelectionPolicy

__all__ = [
    "Attempt",
    "Continuation",
    "DiscoveryNode",
    "DiscoveryTree",
    "DreamResult",
    "DreamRound",
    "PARALLEL_REFINE_SEED",
    "Replay",
    "ReplayObjective",
    "ReplayRound",
    "ReplayWorld",
    "SimulatorPool",
    "dream_rsi",
    "explore",
    "exploration_policy",
    "mean_replay_value",
    "replay_problem",
    "replay_reflector",
    "replay_value",
]

#: The artifact id every world's nodes carry. Selection policies never read it;
#: it exists because :class:`~agentdescent.selection.Candidate` requires one.
ARTIFACT_ID = "discovery-node"


# ---------------------------------------------------------------------------
# The discovery tree
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Attempt:
    """What one ``CONTINUE(v)`` produced: a score, or a failure.

    ``score`` is ``None`` for an attempt that produced no number at all -- a
    program that would not run, an evaluator that raised. That is deliberately
    not ``0.0`` and deliberately not ``-inf``: the paper's replay maximises the
    best score *attained*, and a policy has to be able to tell "this direction
    was measured and is bad" from "this direction broke and may be repairable",
    which Appendix B.2 makes a whole section of ("a local implementation failure
    does not by itself prove that its parent direction is poor"). ``-inf`` also
    does not survive JSON, and a world has to.

    ``detail`` is whatever the replaying policy should see about the attempt --
    ``fail_class`` and ``error`` in the paper's ``Observation``, a change
    summary, a diagnostic. String values reach the policy through
    ``Candidate.state``; numeric ones through ``Candidate.per_task``.
    """

    score: Optional[float] = None
    valid: bool = True
    detail: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DiscoveryNode:
    """One recorded attempt: where it started, what it scored, what it said."""

    index: int
    #: ``None`` only for the root, which is the initial workspace rather than an
    #: attempt.
    parent: Optional[int]
    score: Optional[float] = None
    valid: bool = True
    detail: Mapping[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        return {"index": self.index, "parent": self.parent, "score": self.score,
                "valid": self.valid, "detail": dict(self.detail)}


class DiscoveryTree:
    """A rooted tree of attempts -- one online rollout, recorded.

    Node 0 is the root: the initial workspace, which has a score (the baseline)
    and no parent. Every other node is one ``CONTINUE(v)``: the discovery agent
    resumed ``v``'s workspace, produced a candidate, and the evaluator scored
    it.

    **Shape.** Under the paper's online rule a non-root node is continued at
    most once -- continuing a leaf advances that branch to its new child, and
    the child is the leaf from then on -- so only the root has several children,
    one per branch. :attr:`paper_shaped` reports whether a tree obeys that, and
    a tree that does not (a flat-PUCT tree, whose interior nodes are re-expanded)
    is still usable: the children of an interior node are simply never reachable
    during replay, because ``A(T) = {root} u leaves`` cannot name their parent.
    :attr:`unreachable` counts them, so a world that silently throws away half
    its record says so instead.
    """

    def __init__(self, nodes: Optional[Sequence[DiscoveryNode]] = None,
                 *, name: str = "world") -> None:
        self.name = name
        self.nodes: List[DiscoveryNode] = list(nodes or [])
        #: How many decision rounds produced this tree. Set by :func:`explore`;
        #: zero on a tree assembled by hand, and carried through JSON because
        #: the parallelism term of Equation 1 is meaningless without it.
        self.decision_rounds = 0
        self._children: Dict[int, List[int]] = {}
        for node in self.nodes:
            if node.parent is not None:
                self._children.setdefault(node.parent, []).append(node.index)
        self._lock = threading.Lock()

    # -- construction -------------------------------------------------------

    @classmethod
    def rooted(cls, score: Optional[float] = 0.0, *, name: str = "world",
               detail: Optional[Mapping[str, Any]] = None) -> "DiscoveryTree":
        """A tree holding only the root, at the baseline ``score``."""
        return cls([DiscoveryNode(0, None, score, True, dict(detail or {}))], name=name)

    def add(self, parent: int, attempt: Attempt) -> DiscoveryNode:
        """Append one recorded attempt under ``parent`` and return its node."""
        with self._lock:
            if not 0 <= parent < len(self.nodes):
                raise ValueError(f"no node {parent} to continue from")
            node = DiscoveryNode(len(self.nodes), parent, attempt.score,
                                 attempt.valid, dict(attempt.detail))
            self.nodes.append(node)
            self._children.setdefault(parent, []).append(node.index)
            return node

    # -- structure ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.nodes)

    def children(self, index: int) -> Tuple[int, ...]:
        """The recorded children of ``index``, in the order they were created."""
        return tuple(self._children.get(index, ()))

    def depth(self, index: int) -> int:
        depth, cursor = 0, self.nodes[index]
        while cursor.parent is not None:
            cursor = self.nodes[cursor.parent]
            depth += 1
        return depth

    def branch(self, index: int) -> int:
        """Which root child heads ``index``'s branch; ``-1`` for the root itself."""
        cursor = self.nodes[index]
        if cursor.parent is None:
            return -1
        while cursor.parent not in (None, 0):
            cursor = self.nodes[cursor.parent]        # type: ignore[index]
        return cursor.index

    @property
    def paper_shaped(self) -> bool:
        """True when only the root has more than one recorded child."""
        return all(len(kids) <= 1 for index, kids in self._children.items() if index != 0)

    @property
    def unreachable(self) -> int:
        """Recorded nodes no replay can reach: the 2nd+ child of a non-root node.

        Zero for a tree :func:`explore` built. Positive means the tree came from
        a search that re-expands interior nodes, and that much of the record is
        invisible to the paper's ``A(T) = {root} u leaves``.
        """
        total = 0
        for index, kids in self._children.items():
            if index != 0 and len(kids) > 1:
                total += self._subtree_size(kids[1:])
        return total

    def _subtree_size(self, roots: Iterable[int]) -> int:
        stack, seen = list(roots), 0
        while stack:
            index = stack.pop()
            seen += 1
            stack.extend(self._children.get(index, ()))
        return seen

    def best(self) -> Optional[float]:
        """The best score anywhere in the record -- the ceiling for any replay."""
        scores = [n.score for n in self.nodes if n.score is not None]
        return max(scores) if scores else None

    # -- persistence --------------------------------------------------------

    def to_json(self) -> str:
        return json.dumps({"name": self.name, "decision_rounds": self.decision_rounds,
                           "nodes": [n.to_payload() for n in self.nodes]},
                          separators=(",", ":"), default=str)

    @classmethod
    def from_json(cls, text: str) -> "DiscoveryTree":
        payload = json.loads(text)
        nodes = [DiscoveryNode(int(n["index"]),
                               None if n.get("parent") is None else int(n["parent"]),
                               None if n.get("score") is None else float(n["score"]),
                               bool(n.get("valid", True)),
                               dict(n.get("detail", {})))
                 for n in payload.get("nodes", [])]
        tree = cls(nodes, name=str(payload.get("name", "world")))
        tree.decision_rounds = int(payload.get("decision_rounds", 0))
        return tree


# ---------------------------------------------------------------------------
# The decision interface, shared by both phases
# ---------------------------------------------------------------------------


def _candidates(tree: DiscoveryTree, revealed: Sequence[int],
                eligible: Mapping[int, int]) -> Tuple[Candidate, ...]:
    """The revealed subtree, as the policy sees it.

    Every revealed node is handed over, not only the legal ones: Appendix B.2
    gives the policy both ``observed()`` (the whole revealed prefix) and
    ``legal_actions()``, and a policy that may only see its legal moves cannot
    do the thing that prompt spends two sections demanding -- reconstruct a
    branch's *trajectory* and tell a repairable failure from a dead direction.

    The mapping onto :class:`~agentdescent.selection.Candidate` is fixed here so
    an evolved policy can rely on it:

    ============== =========================================================
    ``version``    the node index
    ``parent``     its parent's index, ``None`` for the root
    ``score``      the recorded score; ``None`` when the attempt produced none
    ``selected``   how many children of this node are already revealed
    ``per_task``   ``score``, ``delta_vs_parent``, ``delta_vs_baseline``,
                   ``depth``, ``branch``, and any numeric ``detail`` entry
    ``state``      ``legal`` (``"1"``/``"0"``), ``valid``, and any string
                   ``detail`` entry -- ``fail_class`` and ``error`` included
    ============== =========================================================

    Nothing about *unrevealed* nodes appears anywhere in it. That is the
    paper's prefix-only constraint, and it is enforced by construction rather
    than by asking the policy to behave: the count of recorded children still
    waiting behind a node is information about the future, so it is not exposed
    even though the simulator obviously knows it.
    """
    rows: List[Candidate] = []
    baseline = tree.nodes[0].score
    shown = set(revealed)
    for index in revealed:
        node = tree.nodes[index]
        parent_score = (tree.nodes[node.parent].score
                        if node.parent is not None else None)
        numbers: Dict[str, float] = {"depth": float(tree.depth(index)),
                                     "branch": float(tree.branch(index))}
        if node.score is not None:
            numbers["score"] = float(node.score)
            if parent_score is not None:
                numbers["delta_vs_parent"] = float(node.score - parent_score)
            if baseline is not None:
                numbers["delta_vs_baseline"] = float(node.score - baseline)
        text: Dict[str, str] = {"legal": "1" if index in eligible else "0",
                                "valid": "1" if node.valid else "0"}
        for key, value in node.detail.items():
            if isinstance(value, bool):
                text[str(key)] = "1" if value else "0"
            elif isinstance(value, (int, float)) and math.isfinite(float(value)):
                numbers[str(key)] = float(value)
            else:
                text[str(key)] = str(value)
        rows.append(Candidate(
            artifact_id=ARTIFACT_ID, version=index, state=text, score=node.score,
            per_task=numbers,
            selected=sum(1 for kid in tree.children(index) if kid in shown),
            parent=node.parent))
    return tuple(rows)


def _context(rows: Sequence[Candidate], eligible: Mapping[int, int], round_index: int,
             n_workers: int) -> SelectionContext:
    """``head`` is the best-scoring *legal* node, root as the fallback.

    ``SelectionContext.head`` is documented as "what the engine would have used
    with no policy at all", so it has to be an answer a policy can return
    unchanged and still be doing something sensible. For a discovery tree that
    is "refine the best thing you have", which is also what the stock
    :class:`~agentdescent.selection.SingleHead` computes -- so a plain
    ``[ctx.head] * n`` is greedy depth-first refinement here rather than a crash.
    """
    legal = [c for c in rows if c.version in eligible]
    scored = [c for c in legal if c.score is not None]
    head = (max(scored, key=lambda c: (c.score, -c.version)) if scored
            else (legal[0] if legal else rows[0]))
    return SelectionContext(head=head, candidates=rows, round=round_index,
                            n_workers=n_workers)


def _eligible(tree: DiscoveryTree, revealed: Sequence[int],
              n_workers: int) -> Dict[int, int]:
    """``A(T) = {root} u {leaves}``, with the multiplicity each node may carry.

    The set is the paper's, and it is the *same formula in both phases* -- which
    is the one structural claim Dream-RSI makes about its own design: *"Both
    phases use the same tree-based decision interface. Their difference lies in
    how a continuation produces its next observation."* So this function is
    called by :func:`explore` and by :meth:`ReplayWorld.replay` alike, and a
    policy cannot tell which one it is running in by looking at its legal set.

    The multiplicity is ``1`` for every leaf -- continuing one twice in a round
    would give it two children, which is the shape no replay could reproduce --
    and ``n_workers`` for the root, which is the appendix's "a batch may contain
    several roots". Nothing about the *record* enters it: a leaf whose branch
    ended is legal here and reveals nothing when it is played, which is how a
    replay charges a policy for a worker it would have wasted online.
    """
    shown = set(revealed)
    caps: Dict[int, int] = {}
    for index in revealed:
        if index == 0:
            caps[index] = n_workers
        elif not any(kid in shown for kid in tree.children(index)):
            caps[index] = 1
    return caps


def _normalise_batch(batch: Any, caps: Mapping[int, int],
                     n_workers: int) -> Tuple[List[int], int, int]:
    """``(picks, illegal, over_budget)`` -- the three hard constraints, applied.

    Appendix B.2: *"A selected batch must be legal, have no duplicate ids, and
    contain at most ``question.max_parallelism`` cells."* Here ``caps`` carries
    the multiplicity each node may appear with rather than a flat "no
    duplicates", which is the one place this follows the appendix over the main
    text. The main text's ``C subset-of A(T)`` lets the root into a batch once,
    so opening ``W`` parallel branches takes ``W`` decision rounds -- yet the
    paper's own seed policy is described as one that *"launches multiple
    independent exploration workspaces in parallel"*, and the appendix says a
    batch "may contain several roots". Those are reconcilable only if the root
    is several cells, one per unopened branch, which is what ``caps`` makes it.
    Every non-root node still has a cap of one, so the two readings agree on
    every batch that does not touch the root.

    A pick the simulator refuses is *counted*, never raised on: an exploration
    policy is a model's rewrite, and the reward is what tells it the batch was
    wasteful. The counts reach the reflector through the replay's ``detail``.
    """
    picks: List[int] = []
    used: Dict[int, int] = {}
    illegal = 0
    for item in (batch or ()):
        index = getattr(item, "version", item)
        if not isinstance(index, int) or isinstance(index, bool) or index not in caps:
            illegal += 1
            continue
        if used.get(index, 0) >= caps[index]:
            illegal += 1
            continue
        used[index] = used.get(index, 0) + 1
        picks.append(index)
    over = max(0, len(picks) - n_workers)
    return picks[:n_workers], illegal, over


def _attempt(continue_fn: "Continuation", tree: "DiscoveryTree",
             parent: "DiscoveryNode", index: int) -> "Attempt":
    """One ``CONTINUE(v)``, with a raising agent turned into a failed attempt.

    See the note on :func:`explore`: a discovery agent that throws is the
    ordinary case for a provider under load, and a tree is already the right
    place to record an attempt that produced nothing.
    """
    try:
        return continue_fn(tree, parent, index)
    except Exception as error:      # noqa: BLE001 - recorded as the node it is
        return Attempt(None, False, {"fail_class": "agent-error",
                                     "error": f"{type(error).__name__}: {error}"})


def _fresh(policy: Any) -> Any:
    """A policy instance with no memory of another episode.

    Appendix B.2's policies keep per-episode state and start each one with
    ``question.reset()``; ours are ordinary objects with ``__init__``, and the
    gate guarantees the class takes no constructor arguments. So a new instance
    is the reset, and every replay and every online rollout gets one. Without
    it a policy that remembers "I already opened four branches" would carry that
    into the next world, and :func:`~agentdescent.meta.meta_validate`'s paired
    comparison -- the same value on the same problem -- would stop being paired.
    """
    try:
        return type(policy)()
    except Exception:       # noqa: BLE001 - a policy we cannot rebuild is used as-is
        return policy


# ---------------------------------------------------------------------------
# Stage 1: online exploration
# ---------------------------------------------------------------------------


class Continuation(Protocol):
    """The discovery agent: resume a node's workspace, produce one scored child.

    It is handed the whole ``tree`` and not only ``parent``, because that is
    what the paper's discovery agent gets -- its exploration prompt opens with
    *"You must read every historical proposal before proposing or implementing
    a new solution"* and spends a section on learning from the failures in it.
    An agent that only saw its own parent would be a different method.

    ``attempt`` is the running count of attempts, handed over so a caller can
    derive a deterministic stream from it -- which is what makes an offline
    world reproducible, and what :func:`~agentdescent.meta.cached_completion`
    does for a live one.

    The tree is being appended to by the other workers in the same batch while
    this runs, so read it, do not hold it: everything already in it is final,
    and nothing about what the rest of the batch is doing is.
    """

    def __call__(self, tree: DiscoveryTree, parent: DiscoveryNode,
                 attempt: int) -> Attempt: ...


def explore(continue_fn: Continuation, policy: SelectionPolicy, *,
            n_workers: int = 4, max_rounds: int = 8, root_score: float = 0.0,
            root_detail: Optional[Mapping[str, Any]] = None,
            max_concurrency: Optional[int] = None, attempt_offset: int = 0,
            name: str = "world") -> DiscoveryTree:
    """Stage 1 -- run ``policy`` against a real discovery agent, and record it.

    At most ``max_rounds`` decision rounds (the paper's ``K1``). Each round the
    policy is shown the tree so far and returns a batch of at most ``n_workers``
    nodes to continue; the batch runs in parallel, the children are attached to
    their selected parents, and the next round sees them. An empty batch ends
    the rollout, which is the policy's own stopping decision.

    The returned tree is the world. It is the *only* thing stages 2 and 3 use --
    the agent, the evaluator and the workspaces are gone by then, which is the
    whole point.

    Parameters
    ----------
    continue_fn:
        The discovery agent, as a :class:`Continuation`.
    policy:
        The exploration policy. A fresh instance is taken (see :func:`_fresh`),
        so the caller's object is not left holding this rollout's state.
    n_workers:
        ``W`` -- how many continuations may run at once, and the batch cap.
    max_rounds:
        ``K1`` -- the decision-round budget. The rollout also ends early on an
        empty batch.
    root_score, root_detail:
        The baseline: what the initial workspace scores before any attempt, and
        whatever the policy should read about it.
    max_concurrency:
        Threads used for one batch; ``None`` means ``n_workers``. Set to ``1``
        for a serial control, which changes nothing about the decisions.
    attempt_offset:
        Where this rollout's attempt counter starts. The counter is the only
        thing a deterministic ``continue_fn`` has to vary on, so two rollouts
        that share an offset are the same rollout -- which is what
        :func:`dream_rsi` is avoiding when it passes a running total.
    name:
        The world's name, which becomes its task id during dreaming.

    Notes
    -----
    A ``continue_fn`` that raises **does not end the rollout**: the exception is
    recorded as a failed attempt, which is what a node whose program would not
    build already is, and the rollout carries on. Losing a whole online rollout
    -- the expensive half of the method -- to one flaky provider call would be
    the wrong trade. The cost is that a ``continue_fn`` which is simply broken
    produces a tree of failures rather than a traceback, so a caller that sees
    a world with no valid node should read the root's neighbours' ``detail``:
    the exception's type and message are in ``error``, under
    ``fail_class="agent-error"``.
    """
    if n_workers < 1:
        raise ValueError(f"n_workers must be at least 1, got {n_workers}")
    tree = DiscoveryTree.rooted(root_score, name=name, detail=root_detail)
    runner = _fresh(policy)
    revealed = [0]
    attempts = attempt_offset
    for round_index in range(max_rounds):
        caps = _eligible(tree, revealed, n_workers)
        rows = _candidates(tree, revealed, caps)
        ctx = _context(rows, caps, round_index, n_workers)
        picks, _, _ = _normalise_batch(runner.select(ctx, n_workers), caps, n_workers)
        if not picks:
            break
        parents = [tree.nodes[index] for index in picks]
        limit = max(1, min(len(picks), max_concurrency or n_workers))

        def run(pair: Tuple[int, DiscoveryNode]) -> Attempt:
            offset, parent = pair
            return _attempt(continue_fn, tree, parent, attempts + offset)

        work = list(enumerate(parents))
        if limit == 1:
            results = [run(pair) for pair in work]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=limit) as pool:
                results = list(pool.map(run, work))
        attempts += len(parents)
        for parent, attempt in zip(parents, results):
            revealed.append(tree.add(parent.index, attempt).index)
        tree.decision_rounds = round_index + 1
    return tree


# ---------------------------------------------------------------------------
# Stage 2: the replay simulator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayObjective:
    """Equation 1 -- quality, minus execution cost, plus a parallelism bonus.

    ``V = max score revealed  -  b1 * N  +  b2 * N / max(1, k)``

    with ``N`` the number of revealed non-root nodes and ``k`` the number of
    completed decision rounds. The third term is the mean batch size, so it
    rewards a policy that issues useful continuations *together* rather than one
    at a time.

    **The paper does not publish its coefficients**, so both default to zero --
    which makes ``V`` the best score attained and nothing else, the honest
    reading of "we did not say". :meth:`scaled` is the constructor to use when
    you want the trade-off: it derives ``b1`` and ``b2`` from the budget, so
    that spending the whole node budget costs ``cost_weight`` of a full unit of
    quality and running perfectly parallel earns ``parallel_weight`` of one.
    """

    beta1: float = 0.0
    beta2: float = 0.0

    def __post_init__(self) -> None:
        if self.beta1 < 0 or self.beta2 < 0:
            raise ValueError("beta1 and beta2 must be non-negative (Equation 1)")

    @classmethod
    def scaled(cls, *, max_nodes: int, n_workers: int, cost_weight: float = 0.25,
               parallel_weight: float = 0.10) -> "ReplayObjective":
        """Coefficients on the scale of a ``[0, 1]`` score, given the budget.

        ``b1 = cost_weight / max_nodes`` and ``b2 = parallel_weight /
        n_workers``: spending the whole node budget costs ``cost_weight`` of a
        unit of quality, and running at full parallelism earns
        ``parallel_weight`` of one.

        **The two weights are not equal, and that is the whole content of this
        constructor.** Together the cost and parallelism terms are
        ``N * (b2 / k - b1)``, which depends on the *round* count and not on
        ``N`` at the point where it vanishes: the net is zero at
        ``k = b2 / b1 = parallel_weight * max_nodes / (cost_weight *
        n_workers)``, whatever ``N`` is. Set the weights equal and that break-even
        lands on ``max_nodes / n_workers`` -- exactly the round budget a policy
        with ``max_nodes`` continuations and ``n_workers`` workers is expected to
        use -- so the objective becomes **blind to how many continuations were
        spent** for every policy that runs its rounds out.

        Measured on one world of the example domain, at ``max_nodes=24,
        n_workers=4``: the shipped seed reaches 0.818 on 24 continuations in 6
        rounds, and the example's evolved policy reaches the same 0.818 on
        **16** in the same 6 rounds. At ``0.25 / 0.25`` both score
        ``V = 0.8181`` -- a third of the discovery budget saved, worth exactly
        nothing, and the whole cost story of the paper gone. At the default
        ``0.25 / 0.10`` the break-even is ``0.4 * max_nodes / n_workers`` and
        the same pair scores 0.6681 against 0.7181, so an ordinary-length
        rollout pays per continuation and parallelism only breaks ties.

        The paper publishes no coefficients. These put cost ahead of
        parallelism, which is the direction its results run: it reports
        discovery-call reductions of 1.7x to 162x, and treats batching as what
        makes a given number of calls cheaper in wall clock rather than as an
        end. :func:`dream_rsi` passes ``max_nodes = n_workers * online_rounds``,
        the online budget, so the scale is the deployment being improved for.
        """
        if max_nodes < 1 or n_workers < 1:
            raise ValueError("max_nodes and n_workers must be at least 1")
        return cls(beta1=cost_weight / max_nodes, beta2=parallel_weight / n_workers)

    def value(self, quality: float, nodes: int, rounds: int) -> float:
        return (quality - self.beta1 * nodes
                + self.beta2 * (nodes / max(1, rounds)))

    def bounds(self, world: "ReplayWorld", *, n_workers: int,
               max_rounds: int) -> Tuple[float, float]:
        """The range ``V`` can take in ``world`` -- what :func:`replay_value` divides by.

        Both ends are the world's own, not a guess: a replay cannot score better
        than the best node the record holds, and cannot score worse than the
        root, which is always revealed. The cost term is worst at the full node
        budget and the parallelism bonus is best at ``n_workers``.
        """
        root = world.tree.nodes[0].score
        floor = 0.0 if root is None else float(root)
        best = world.tree.best()
        ceiling = floor if best is None else max(floor, float(best))
        budget = min(len(world.tree) - 1, n_workers * max(1, max_rounds))
        return (floor - self.beta1 * budget, ceiling + self.beta2 * n_workers)


@dataclass(frozen=True)
class ReplayRound:
    """One decision round of a replay: what was asked for, what was revealed."""

    index: int
    batch: Tuple[int, ...]
    revealed: Tuple[int, ...]
    illegal: int
    over_budget: int
    best: float


@dataclass(frozen=True)
class Replay:
    """One policy's whole trajectory through one recorded world.

    ``value`` is Equation 1 on ``quality``, ``nodes`` and ``decision_rounds``;
    the three are kept separately because the reflector is told the decomposition
    rather than a single number, which is what Appendix B.2's prompt does.
    """

    world: str
    rounds: Tuple[ReplayRound, ...]
    revealed: Tuple[int, ...]
    stop_reason: str
    quality: float
    nodes: int
    decision_rounds: int
    parallelism: float
    value: float
    illegal: int
    barren: int
    error: Optional[str] = None

    def curve(self) -> List[float]:
        """Best-so-far after each completed round -- the outcome's ``curve``."""
        return [r.best for r in self.rounds]

    def trajectory(self, *, max_rounds: int = 12) -> List[Dict[str, Any]]:
        """A compact per-round record for a prompt or a report."""
        return [{"round": r.index, "continued": list(r.batch),
                 "revealed": list(r.revealed), "best": round(r.best, 6),
                 "illegal": r.illegal}
                for r in self.rounds[:max_rounds]]

    def to_detail(self) -> Dict[str, Any]:
        return {"world": self.world, "quality": self.quality, "nodes": self.nodes,
                "decision_rounds": self.decision_rounds,
                "parallelism": self.parallelism, "value": self.value,
                "illegal": self.illegal, "barren": self.barren,
                "stop_reason": self.stop_reason, "error": self.error,
                "trajectory": self.trajectory()}


class ReplayWorld:
    """A recorded :class:`DiscoveryTree`, played back as a simulator.

    Replay starts from the root alone. At each round the policy sees the
    revealed subtree and which of its nodes are legal to continue, and returns a
    batch; the simulator reveals the next recorded child of each selected node.
    The transition is **deterministic** -- that is the whole difference from
    online exploration, where the same workspace can produce different children.

    Replay ends when the policy returns an empty batch, when a round reveals
    nothing (the paper's *"no recorded continuation remains available"*), or at
    ``max_rounds`` (``K2``). Every policy starts from the root again: a version
    does not inherit the subtree an earlier version revealed.

    **The legal set is the paper's**, ``A(T) = {root} u {leaves of the revealed
    subtree}``, and it is *not* narrowed to nodes that still have something
    recorded behind them. Picking a leaf whose branch ended in the record is a
    legal move that reveals nothing, and it costs the policy a slot -- which is
    the honest simulation of a policy that would, online, have spent a worker
    there. Those picks are counted as ``barren`` rather than ``illegal``.
    """

    def __init__(self, tree: DiscoveryTree, *, name: Optional[str] = None) -> None:
        self.tree = tree
        self.name = name or tree.name

    def __repr__(self) -> str:       # pragma: no cover - debugging convenience
        return f"ReplayWorld({self.name!r}, nodes={len(self.tree)})"

    def replay(self, policy: SelectionPolicy, objective: ReplayObjective, *,
               n_workers: int = 4, max_rounds: int = 16) -> Replay:
        """Walk ``policy`` through this world and score it with ``objective``."""
        runner = _fresh(policy)
        tree = self.tree
        revealed: List[int] = [0]
        cursor: Dict[int, int] = {}          # node -> how many children revealed
        rounds: List[ReplayRound] = []
        illegal_total = barren_total = 0
        best = tree.nodes[0].score
        best = 0.0 if best is None else float(best)
        stop_reason, error = "max-rounds", None
        for round_index in range(max_rounds):
            caps = _eligible(tree, revealed, n_workers)
            rows = _candidates(tree, revealed, caps)
            ctx = _context(rows, caps, round_index, n_workers)
            try:
                batch = runner.select(ctx, n_workers)
            except Exception as exc:         # noqa: BLE001 - the policy's failure, scored
                stop_reason, error = "error", f"{type(exc).__name__}: {exc}"
                break
            picks, illegal, over = _normalise_batch(batch, caps, n_workers)
            illegal_total += illegal + over
            if not picks:
                stop_reason = "empty-batch"
                break
            newly: List[int] = []
            for index in picks:
                taken = cursor.get(index, 0)
                children = tree.children(index)
                if taken >= len(children):
                    barren_total += 1
                    continue
                cursor[index] = taken + 1
                child = children[taken]
                revealed.append(child)
                newly.append(child)
                score = tree.nodes[child].score
                if score is not None:
                    best = max(best, float(score))
            if not newly:
                stop_reason = "exhausted"
                break
            rounds.append(ReplayRound(round_index, tuple(picks), tuple(newly),
                                      illegal + over, over, best))
        nodes = len(revealed) - 1
        completed = len(rounds)
        return Replay(
            world=self.name, rounds=tuple(rounds), revealed=tuple(revealed),
            stop_reason=stop_reason, quality=best, nodes=nodes,
            decision_rounds=completed,
            parallelism=nodes / max(1, completed),
            value=objective.value(best, nodes, completed),
            illegal=illegal_total, barren=barren_total, error=error)


class SimulatorPool:
    """``H_t = (T_1, ..., T_t)`` -- every world recorded so far.

    The pool is what makes the loop recursive: each online deployment appends
    the tree it built, and the next dreaming phase evaluates every policy
    version on *all* of them. A policy that overfits the world its predecessor
    happened to record is caught by the older worlds still being in the pool.
    """

    def __init__(self, worlds: Optional[Sequence[ReplayWorld]] = None) -> None:
        self.worlds: List[ReplayWorld] = list(worlds or [])

    def __len__(self) -> int:
        return len(self.worlds)

    def __iter__(self):
        return iter(self.worlds)

    def add(self, tree: DiscoveryTree, *, name: Optional[str] = None) -> ReplayWorld:
        """Append one recorded rollout; the name defaults to ``t<index>``.

        A name already in the pool is suffixed rather than accepted, because
        :meth:`problems` keys a mapping on it and
        :func:`~agentdescent.meta.meta_evolve` keys a task id on that -- so two
        worlds sharing a name is not a cosmetic clash, it is one of them
        silently not being dreamt in. It happens the moment a pool outlives a
        single :func:`dream_rsi` call, which is exactly what a pool is for.
        """
        wanted = name or f"t{len(self.worlds) + 1}"
        taken = {world.name for world in self.worlds}
        unique, suffix = wanted, 2
        while unique in taken:
            unique, suffix = f"{wanted}#{suffix}", suffix + 1
        world = ReplayWorld(tree, name=unique)
        self.worlds.append(world)
        return world

    def problems(self, objective: ReplayObjective, *, n_workers: int = 4,
                 max_rounds: int = 16) -> Dict[str, Problem]:
        """The pool as :func:`~agentdescent.meta.meta_evolve`'s ``problems``."""
        return {world.name: replay_problem(world, objective, n_workers=n_workers,
                                           max_rounds=max_rounds)
                for world in self.worlds}

    def to_json(self) -> str:
        return json.dumps([{"name": w.name, "tree": json.loads(w.tree.to_json())}
                           for w in self.worlds], separators=(",", ":"))

    @classmethod
    def from_json(cls, text: str) -> "SimulatorPool":
        worlds = []
        for row in json.loads(text):
            tree = DiscoveryTree.from_json(json.dumps(row["tree"]))
            worlds.append(ReplayWorld(tree, name=str(row.get("name", tree.name))))
        return cls(worlds)


# ---------------------------------------------------------------------------
# Stage 3: dreaming -- the replay as a meta-evolution problem
# ---------------------------------------------------------------------------


def replay_problem(world: ReplayWorld, objective: ReplayObjective, *,
                   n_workers: int = 4, max_rounds: int = 16) -> Problem:
    """One world as a :class:`~agentdescent.meta.Problem`.

    ``(compiled policy, seed) -> MetaOutcome``, with the outcome's ``curve``
    the best-so-far after each round (so the stock
    :func:`~agentdescent.meta.auc` still means what it always meant), ``final``
    the raw ``V`` of Equation 1, ``rollouts`` the ``N`` continuations the
    trajectory represents, and ``detail`` the whole replay trajectory for the
    reflector.

    **The seed is ignored, and that is a property rather than an oversight.**
    Replay is deterministic, so ``problem(value, 0)`` and ``problem(value, 7)``
    are the same run -- which is exactly the condition
    ``docs/meta-evolution.md`` states for a seed not to be a replicate. Pass
    ``seeds=[0]`` and spend the budget on more *worlds*.
    """
    lo, hi = objective.bounds(world, n_workers=n_workers, max_rounds=max_rounds)

    def problem(value: Any, seed: int) -> MetaOutcome:
        replayed = world.replay(value, objective, n_workers=n_workers,
                                max_rounds=max_rounds)
        detail = replayed.to_detail()
        detail.update({"value_lo": lo, "value_hi": hi, "seed": seed})
        return MetaOutcome(curve=replayed.curve(), final=replayed.value,
                           rollouts=replayed.nodes, detail=detail)

    return problem


def replay_value() -> MetaReward:
    """Equation 1, mapped into ``[0, 1]`` by the world's own attainable range.

    ``evolve()`` compares rewards *across tasks* -- the held-out gate averages
    them -- and raw ``V`` is not comparable across worlds: a world whose best
    recorded node scores 0.9 and one whose best scores 0.2 put the same policy
    at different heights for reasons that have nothing to do with the policy. So
    each world's ``V`` is divided by its own range, computed in
    :meth:`ReplayObjective.bounds` from the record itself: 0 is the root for the
    maximum price, 1 is the best node in the record for free at full
    parallelism. Neither end is reachable, which is why
    :func:`~agentdescent.meta.meta_parts` puts ``solved_threshold`` above one.

    This is a **deviation from the paper**, which averages raw ``V`` over
    worlds, and it is the one place the port changes an equation rather than
    reading it. The raw value is kept in the outcome's ``detail`` under
    ``value``, so a report can show both.
    """

    def reward(outcome: MetaOutcome) -> float:
        detail = outcome.detail or {}
        lo = float(detail.get("value_lo", 0.0))
        hi = float(detail.get("value_hi", 1.0))
        if not math.isfinite(lo) or not math.isfinite(hi) or hi - lo < 1e-12:
            return min(1.0, max(0.0, float(outcome.final)))
        return min(1.0, max(0.0, (float(outcome.final) - lo) / (hi - lo)))

    return reward


def mean_replay_value(policy: SelectionPolicy, pool: SimulatorPool,
                      objective: ReplayObjective, *, n_workers: int = 4,
                      max_rounds: int = 16) -> float:
    """``V^m`` -- the mean normalised replay value of one policy over the pool.

    The paper's selection rule reads this: after ``M`` revisions the next online
    policy is ``argmax_m V^m``, and because the candidate set contains the
    current policy the selected one is never worse *on the fixed history*. That
    is the guarantee :func:`dream_rsi` enforces explicitly rather than inferring
    from the fact that a gate ran.
    """
    if not len(pool):
        return 0.0
    score = replay_value()
    values = []
    for world in pool:
        problem = replay_problem(world, objective, n_workers=n_workers,
                                 max_rounds=max_rounds)
        values.append(score(problem(policy, 0)))
    return statistics.fmean(values)


# ---------------------------------------------------------------------------
# The exploration policy, as an evolvable slot
# ---------------------------------------------------------------------------


#: The paper's manually designed starting policy, transcribed: *"it launches
#: multiple independent exploration workspaces in parallel, with each workspace
#: maintaining its own local discovery trajectory and repeatedly refining its
#: current candidate based on the history accumulated within that workspace."*
#:
#: Round one has nothing but the root, so the batch is ``W`` root continuations
#: and ``W`` branches open at once. Every round after that refines each open
#: branch's frontier, one worker each, and opens a new branch only when a
#: workspace has run out of recorded continuations. It never prunes, never
#: reorders, and never stops early -- which is the point of a baseline: those
#: are exactly the decisions dreaming is supposed to discover.
PARALLEL_REFINE_SEED = '''class Policy:
    # Parallel refining (Dream-RSI's manually designed exploration policy):
    # open W workspaces from the root, then refine every open workspace's
    # frontier, one worker each, for as long as the budget lasts.
    def __init__(self):
        self.width = 0

    def select(self, ctx, n):
        legal = [c for c in ctx.candidates if c.state.get("legal") == "1"]
        if not legal:
            return []
        if self.width < 1:
            self.width = n                      # W, fixed for the episode
        roots = [c for c in legal if c.parent is None]
        frontier = sorted((c for c in legal if c.parent is not None),
                          key=lambda c: c.version)
        batch = frontier[:self.width]
        room = min(n, self.width) - len(batch)
        if roots and room > 0:
            batch = batch + [roots[0]] * room   # one cell per unopened branch
        return batch[:n]
'''


_NOTES = """This is a DISCOVERY-TREE EXPLORATION POLICY (Dream-RSI). `ctx.candidates` is the
revealed part of a recorded discovery tree, newest last; `select(ctx, n)` returns
the batch of nodes to continue next, and `n` is the worker count W.

Per candidate:
- `version` is the node index, `parent` the index it was continued from (None
  for the root, which is the initial workspace);
- `score` is the recorded score, or None when the attempt produced no number at
  all -- a failure, which may be repairable and is NOT the same as a low score;
- `state["legal"] == "1"` marks a node you may continue this round: the root,
  and every leaf of the revealed subtree. Continuing the root opens a new
  branch, and the root may appear in a batch several times, once per branch you
  want opened. Every other node may appear at most once.
- `state["valid"]`, and whatever the domain recorded -- `fail_class`, `error`;
- `per_task` carries the numbers: `score`, `delta_vs_parent`,
  `delta_vs_baseline`, `depth`, `branch`;
- `selected` is how many children of this node are already revealed.

Return a list of at most `n` candidates drawn from `ctx.candidates`. An empty
list STOPS the rollout, which is a real decision and sometimes the right one:
the objective charges for every continuation. A pick that is not legal, or that
exceeds a node's multiplicity, is dropped and counted against you.

You are scored by V = (best score revealed) - b1 * (continuations spent)
+ b2 * (continuations per decision round). So: reach a good node, do not pay for
attempts that will not help, and put independent continuations in the SAME batch
rather than in consecutive rounds.

You may only use what has been revealed. There is no way to see an unrevealed
node's score, and no fixed node index is worth hardcoding -- the policy is
scored on worlds it has never seen."""


def _smoke_exploration(policy: Any) -> None:
    """Drive the candidate through a small fixed world before it is ever used.

    The shipped ``selection`` smoke test cannot be reused: it requires
    ``select`` to return at least one candidate, and for an exploration policy
    the empty batch is a legal, meaningful move. What has to be checked instead
    is the contract of a *batch*: no more than ``n``, drawn from the candidates
    it was handed, no mutation of the list it was given, and termination.

    The world is two branches of two attempts with a failed node in one of them,
    which is the smallest shape that exercises the branches a policy has -- a
    root to open, a frontier to refine, a leaf whose branch ended, and a node
    whose ``score`` is ``None``.
    """
    tree = DiscoveryTree.rooted(0.10, name="smoke")
    a = tree.add(0, Attempt(0.40, True, {"fail_class": "ok"}))
    tree.add(a.index, Attempt(0.55, True, {"fail_class": "ok"}))
    b = tree.add(0, Attempt(None, False, {"fail_class": "compile_other",
                                          "error": "did not build"}))
    tree.add(b.index, Attempt(0.30, True, {"fail_class": "ok"}))
    world = ReplayWorld(tree)

    runner = _fresh(policy)
    revealed = [0]
    for round_index in range(6):
        caps = _eligible(tree, revealed, 2)
        rows = _candidates(tree, revealed, caps)
        frozen = list(rows)
        ctx = _context(rows, caps, round_index, 2)
        batch = list(runner.select(ctx, 2))
        if list(ctx.candidates) != frozen:
            raise ValueError("select() must not mutate the candidates it is given")
        if len(batch) > 2:
            raise ValueError(f"select() returned {len(batch)} picks for n=2")
        if any(c not in frozen for c in batch):
            raise ValueError("select() must return candidates from ctx.candidates")
        if not batch:
            break
        for cand in batch[:2]:
            if cand.version in caps and tree.children(cand.version):
                revealed.append(tree.children(cand.version)[0])
    # ...and once more end to end, so a policy that cannot terminate is refused
    # here rather than after it has burned an outer round.
    world.replay(policy, ReplayObjective(), n_workers=2, max_rounds=8)


def exploration_policy(seed: str = PARALLEL_REFINE_SEED) -> SourceSlot:
    """The Dream-RSI exploration policy, as an evolvable ``selection`` slot.

    :func:`~agentdescent.meta.policy_source` with two substitutions: the seed is
    :data:`PARALLEL_REFINE_SEED` rather than the engine's own default rule, and
    the smoke test is :func:`_smoke_exploration`, which permits the empty batch
    the shipped one forbids. Everything else -- the AST allowlist, the
    ``isinstance`` check against
    :class:`~agentdescent.selection.SelectionPolicy`, the restricted namespace --
    is :mod:`agentdescent.meta`'s, unchanged.

    The value is therefore compilable straight into ``Policies(selection=...)``
    as well as into a replay, which is what "redeploy the improved policy
    online" means here.
    """
    return policy_source("selection", seed, smoke=_smoke_exploration, notes=_NOTES)


def replay_reflector(complete: Completion, spec: SlotSpec, *,
                     max_history: int = 4,
                     max_outcome_chars: int = 2_600) -> Callable[..., Optional[str]]:
    """The policy-development agent: it reads *trajectories*, not just a score.

    :func:`~agentdescent.meta.slot_reflector` shows the model the value, the
    outcome JSON and one number. The paper's development agent is given more,
    and the difference is the part that can actually be acted on: it *"examines
    the replay trajectories and scores of the current policy, together with
    feedback from earlier revisions, to identify successful decisions and
    recurring failures."*

    So this prompt carries three things the stock one does not:

    * the **round-by-round trajectory** -- which nodes each batch continued,
      what that revealed, and where the best score stopped moving;
    * the **decomposition** of Equation 1 into quality, continuations spent and
      mean batch size, so a policy that lost on cost is not told merely that it
      lost;
    * the **last few revisions and what they scored**, which is the "feedback
      from earlier revisions" and the only thing stopping the model from
      proposing the same rewrite every round.

    The history is this reflector's own memory of the calls it has made, bounded
    to ``max_history`` entries, and it is keyed on nothing -- concurrent workers
    share one list, which is what makes it a record of the *run* rather than of
    a worker.
    """
    history: List[Tuple[float, str]] = []
    lock = threading.Lock()

    def propose(rendered: str, task: Task, output: str, reward: float) -> Optional[str]:
        try:
            outcome = MetaOutcome.from_json(output)
            detail = outcome.detail or {}
        except (ValueError, TypeError, json.JSONDecodeError):
            detail = {}
        with lock:
            earlier = list(history[-max_history:])
            history.append((reward, _digest(rendered)))
        lines = [
            "You are improving the EXPLORATION POLICY of a discovery search.",
            "",
            spec.describe(),
            "",
            f"Current policy:\n```python\n{rendered}\n```",
            "",
            f"Replay world: {task.prompt}",
        ]
        if detail:
            lines += [
                "",
                "How the current policy did there:",
                f"- best score reached: {detail.get('quality')}"
                f" (the best node anywhere in this world would be worth more)",
                f"- continuations spent (N): {detail.get('nodes')}",
                f"- decision rounds (k): {detail.get('decision_rounds')}",
                f"- mean batch size (N/k): {detail.get('parallelism')}",
                f"- illegal or over-budget picks: {detail.get('illegal')}",
                f"- picks on a branch with nothing recorded left: {detail.get('barren')}",
                f"- why it stopped: {detail.get('stop_reason')}",
                "",
                "Round by round (`continued` are node indices it asked to continue,",
                "`revealed` what that produced, `best` the best score so far):",
                json.dumps(detail.get("trajectory", []), separators=(",", ":"))[:max_outcome_chars],
            ]
        else:
            lines += ["", "Outcome JSON:", output[:max_outcome_chars]]
        if earlier:
            lines += ["", "Earlier revisions this run, oldest first "
                      "(reward, policy digest):",
                      ", ".join(f"{value:.3f}/{tag}" for value, tag in earlier)]
        lines += [
            "",
            f"Its reward on this world was {reward:.3f} (higher is better; 1.0 is "
            "the unreachable ceiling -- the best node in the record, revealed for "
            "free, at full parallelism).",
            "",
            "Propose ONE revised policy that would raise that reward on worlds "
            "like this one -- not on this world alone. Say in the leading comment "
            "which decision you changed and what evidence in the trajectory "
            "prompted it.",
        ]
        return complete("\n".join(lines))

    return propose


def _digest(text: str) -> str:
    return f"{abs(hash(text.strip())) % 0xFFFF:04x}"


# ---------------------------------------------------------------------------
# The outer loop
# ---------------------------------------------------------------------------


@dataclass
class DreamRound:
    """One outer iteration ``t``: explore, construct, dream, select.

    ``value_before`` and ``value_after`` are the mean normalised replay value of
    the deployed policy and of the selected one, both measured on the pool *as
    it stands after this round's tree was added* -- the paper's fixed ``H_t``.
    ``redeployed`` says whether the selection actually changed the policy.
    """

    index: int
    online_nodes: int
    online_best: Optional[float]
    online_rounds: int
    pool_size: int
    value_before: float
    value_after: float
    redeployed: bool
    rendered: str
    stop_reason: str = ""

    def to_payload(self) -> Dict[str, Any]:
        return {"round": self.index, "online_nodes": self.online_nodes,
                "online_best": self.online_best, "online_rounds": self.online_rounds,
                "pool_size": self.pool_size, "value_before": self.value_before,
                "value_after": self.value_after, "redeployed": self.redeployed,
                "stop_reason": self.stop_reason}


@dataclass
class DreamResult:
    """What :func:`dream_rsi` did, round by round.

    ``rendered`` is the last selected policy -- hand it to ``spec.compile`` to
    get the object ``Policies(selection=...)`` takes. ``pool`` is the history,
    and it is worth keeping: it is the expensive half, and
    :meth:`SimulatorPool.to_json` will carry it to the next process.
    """

    rendered: str
    rounds: List[DreamRound] = field(default_factory=list)
    pool: SimulatorPool = field(default_factory=SimulatorPool)

    @property
    def online_curve(self) -> List[Optional[float]]:
        """The best score each online rollout reached, in order."""
        return [r.online_best for r in self.rounds]

    def to_payload(self) -> Dict[str, Any]:
        return {"rounds": [r.to_payload() for r in self.rounds],
                "worlds": len(self.pool), "rendered": self.rendered}


def dream_rsi(
    continue_fn: Continuation,
    *,
    spec: Optional[SlotSpec] = None,
    seed_policy: str = PARALLEL_REFINE_SEED,
    propose: Optional[Callable[..., Optional[str]]] = None,
    model: Optional[Completion] = None,
    objective: Optional[ReplayObjective] = None,
    pool: Optional[SimulatorPool] = None,
    rounds: int = 3,
    n_workers: int = 4,
    online_rounds: int = 8,
    online_repeats: int = 4,
    min_worlds: int = 4,
    replay_rounds: int = 16,
    dream_rounds: int = 3,
    dream_workers: Optional[int] = None,
    max_concurrency: Optional[int] = None,
    root_score: float = 0.0,
    on_round: Optional[Callable[[DreamRound], None]] = None,
    **dream_kwargs: Any,
) -> DreamResult:
    """The recursive loop: explore online, dream offline, redeploy.

    Each outer iteration deploys the current policy against ``continue_fn``,
    appends the tree it built to the simulator pool, evolves the policy over the
    *whole* pool with :func:`~agentdescent.meta.meta_evolve`, and then makes the
    paper's explicit selection: the next policy is whichever of {current,
    evolved} has the higher mean replay value on the fixed history. Because the
    current policy is in that set, ``value_after >= value_before`` every round,
    by construction rather than by hope -- which is the one formal guarantee the
    paper states, and the only thing here that is cheap enough to be worth
    guaranteeing.

    Parameters
    ----------
    continue_fn:
        The discovery agent, ``(tree, parent, attempt) -> Attempt``. This is
        the expensive thing, and the reason the whole method exists.
    spec, seed_policy:
        The artifact. ``spec`` defaults to ``exploration_policy(seed_policy)``;
        pass one directly to change the gate or the seed's own description.
    propose, model:
        The policy-development agent. Pass ``propose`` directly, or ``model`` to
        get :func:`replay_reflector` over the spec. One of the two is required.
    objective:
        Equation 1's coefficients. ``None`` is
        ``ReplayObjective.scaled(max_nodes=n_workers * online_rounds,
        n_workers=n_workers)`` -- the trade-off on the scale of this budget,
        since the paper publishes no coefficients.
    pool:
        An existing :class:`SimulatorPool` to extend. This is how a second
        process continues the loop: the worlds are the expensive part and they
        serialise.
    rounds:
        Outer iterations ``t``. Each one costs ``online_repeats`` whole online
        rollouts.
    n_workers:
        ``W`` -- the batch cap, online and in replay.
    online_repeats, min_worlds:
        **This port's own choice, and not the paper's.** Dream-RSI deploys the
        policy once per iteration, so ``H_1`` is a single tree. ``evolve()``
        refuses fewer than four tasks, and more to the point, dreaming on one
        world while the gate holds out that same world is precisely the
        fit-to-the-training-landscape failure ``docs/meta-evolution.md``
        documents and ``meta_validate`` exists to catch. So an iteration
        deploys the policy ``online_repeats`` times -- genuinely different
        trees, because the online transition is stochastic -- and dreaming is
        skipped, with ``stop_reason="pool-too-small"`` recorded, until the pool
        holds ``min_worlds``. Set ``online_repeats=1`` to follow the paper
        exactly; dreaming then starts at the fourth iteration.
    online_rounds, replay_rounds:
        ``K1`` and ``K2``: the decision-round caps for an online rollout and for
        one replay.
    dream_rounds, dream_workers:
        The dreaming phase's own budget -- rounds of
        :func:`~agentdescent.meta.meta_evolve` per outer iteration, and how many
        policy revisions it proposes in parallel. Named separately from
        ``rounds`` and ``n_workers`` because the two levels are budgeted in
        different currencies: an outer round costs a real discovery rollout, a
        dreaming round costs one model call per worker and no execution at all.
        ``dream_workers`` defaults to ``n_workers`` capped at four.
    max_concurrency:
        Threads for one online batch; ``None`` means ``n_workers``.
    root_score:
        What the initial workspace scores before any attempt.
    on_round:
        Called with each :class:`DreamRound` as it completes -- a progress hook,
        since an outer round is long.
    **dream_kwargs:
        Passed to :func:`~agentdescent.meta.meta_evolve` for the dreaming phase
        (``rounds``, ``held_out_frac``, ``max_seconds`` ...). ``slot``, ``spec``,
        ``problems``, ``meta_reward`` and ``seeds`` are this function's.
    """
    if propose is None and model is None:
        raise ValueError("dream_rsi() needs propose= or model=")
    for taken in ("slot", "spec", "problems", "meta_reward", "seeds"):
        if taken in dream_kwargs:
            raise TypeError(f"dream_rsi() sets {taken}= itself")
    if rounds < 1:
        raise ValueError(f"rounds must be at least 1, got {rounds}")
    spec = spec if spec is not None else exploration_policy(seed_policy)
    objective = objective if objective is not None else ReplayObjective.scaled(
        max_nodes=max(1, n_workers * online_rounds), n_workers=n_workers)
    pool = pool if pool is not None else SimulatorPool()
    rendered = spec.render(spec.initial())
    result = DreamResult(rendered=rendered, pool=pool)
    meta_kwargs: Dict[str, Any] = {
        "rounds": dream_rounds,
        "n_workers": dream_workers if dream_workers is not None
        else min(4, max(2, n_workers)),
    }
    meta_kwargs.update(dream_kwargs)

    attempts = 0
    for index in range(rounds):
        # 1 -- online explore, with the policy currently deployed.
        policy = spec.compile(rendered)
        trees = []
        for repeat in range(max(1, online_repeats)):
            tree = explore(continue_fn, policy, n_workers=n_workers,
                           max_rounds=online_rounds, root_score=root_score,
                           max_concurrency=max_concurrency,
                           attempt_offset=attempts,
                           name=f"t{index + 1}.{repeat + 1}")
            attempts += max(0, len(tree) - 1)
            trees.append(tree)
            # 2 -- construct the replay simulator and add it to the pool.
            pool.add(tree, name=tree.name)
        before = mean_replay_value(policy, pool, objective, n_workers=n_workers,
                                   max_rounds=replay_rounds)
        best_tree = max(trees, key=lambda t: (t.best() is not None, t.best() or 0.0))
        if len(pool) < min_worlds:
            record = DreamRound(
                index=index + 1, online_nodes=sum(len(t) - 1 for t in trees),
                online_best=best_tree.best(),
                online_rounds=sum(t.decision_rounds for t in trees),
                pool_size=len(pool), value_before=before, value_after=before,
                redeployed=False, rendered=rendered, stop_reason="pool-too-small")
            result.rounds.append(record)
            if on_round is not None:
                on_round(record)
            continue
        problems = pool.problems(objective, n_workers=n_workers,
                                 max_rounds=replay_rounds)
        # 3 -- dream: evolve the policy over every world recorded so far. The
        # spec is rebuilt on the *current* policy, so round t+1 starts from
        # pi_t rather than from the seed, which is what "the updated policy is
        # then redeployed" requires and what a fresh `exploration_policy()`
        # would silently undo.
        round_spec = exploration_policy(rendered)
        reflector = propose or replay_reflector(model, round_spec)  # type: ignore[arg-type]
        evolved = meta_evolve(problems, slot="selection", spec=round_spec,
                              propose=reflector, meta_reward=replay_value(),
                              seeds=[0], **meta_kwargs)
        # 4 -- selection. `meta_evolve` already gated on held-out worlds and the
        # oracle, and this is still not that: it is the paper's argmax over the
        # evaluated versions *including the current one*, measured on the whole
        # fixed history rather than on the gate's slice of it. The two agree in
        # the ordinary case; when they disagree the guarantee is what holds.
        candidate = evolved.rendered
        after = before
        redeployed = False
        if candidate and candidate.strip() != rendered.strip():
            try:
                value = mean_replay_value(spec.compile(candidate), pool, objective,
                                          n_workers=n_workers,
                                          max_rounds=replay_rounds)
            except ValueError:
                value = float("-inf")     # a value that will not compile is no value
            if value > before:
                rendered, after, redeployed = candidate, value, True
        record = DreamRound(
            index=index + 1, online_nodes=sum(len(t) - 1 for t in trees),
            online_best=best_tree.best(),
            online_rounds=sum(t.decision_rounds for t in trees),
            pool_size=len(pool), value_before=before, value_after=after,
            redeployed=redeployed, rendered=rendered,
            stop_reason=evolved.stop_reason)
        result.rounds.append(record)
        result.rendered = rendered
        if on_round is not None:
            on_round(record)
    return result
