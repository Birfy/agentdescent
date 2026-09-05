"""Which candidate the next batch of workers starts from.

`policies.py` made conflict, fusion, acceptance and promotion replaceable, and
missed a decision: *where does the next round begin?* The engine has one `dev`
head and starts every worker there. The `TaskScheduler`'s UCB looks like the
missing piece and is not -- it chooses a **task**, not a candidate.

The consequence is visible in the ports. GEPA's Pareto frontier, EvoSkill's
top-K aggregate frontier, DGM's archive and ADAS's archive are each a
candidate-selection rule, and each is written out by hand inside its own example
because the engine has nowhere to put one. Three things follow:

**"We did not change the semantics" cannot be checked.** The parallelisation
matrix claims each port's published selection rule is untouched; while the rule
lives in the example, that claim rests on a human reading the file.

**Tree search cannot be expressed at all.** With one head there is no place for
beam search or MCTS to keep the frontier they are made of.

**Pareto is implemented twice**, and the difference between the two --
per-instance (GEPA) versus top-K aggregate (EvoSkill) -- is a fidelity detail
this repository has documented in prose. It should be an argument.

This module is that seam, and *only* the seam. `SingleHead` is the default and
reproduces today's behaviour exactly; `tests/test_selection.py` asserts it.

**Selection and merging are not alternatives.** One selected starting point still
has N/k workers under it proposing diffs that the aggregator merges back into it:

    SelectionPolicy picks k starting points
      └─ N/k workers under each, each proposing a diff
           └─ the aggregator merges them into that starting point

So the merge layer sits *under* any search strategy rather than competing with
one.

## What is deliberately not here yet

Multiple **live** heads. The ledger has one `dev` branch, staleness is defined as
``eta = max(head - base)``, and promotion compares `dev` against `stable` -- all
three assume "head" names one thing. A policy that returns several distinct
starting points is therefore *refused* by both drivers rather than silently
collapsed to the first, and every policy below is usable today in the shape that
returns one. Making the ledger hold concurrent branches, and redefining `eta`
when `head` is plural, is a separate change.

"Both drivers" is load-bearing and was not always true: `async_evolve` listed
`selection` among the bundle fields it honours and then never consulted it, so
the same `Policies(selection=...)` raised on the barrier path and finished with a
reward on the barrier-free one. See :class:`MultiHeadUnsupported` for why the
refusal needs a type of its own to survive the merger thread.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import (
    Dict, List, Mapping, Optional, Protocol, Sequence, Set, Tuple,
    runtime_checkable,
)

from .evolvable import ContractError

__all__ = [
    "Archive",
    "Beam",
    "Candidate",
    "FlatPuct",
    "MCTS",
    "MultiHeadUnsupported",
    "ParetoFrontier",
    "SelectionContext",
    "SelectionPolicy",
    "SingleHead",
    "pareto_front",
    "pareto_win_frequency",
    "sigmoid_novelty_weights",
]


class MultiHeadUnsupported(ContractError, NotImplementedError):
    """A policy named a starting point the ledger cannot hold yet.

    Both bases carry their weight. ``NotImplementedError`` is what a caller
    catches and what this raised before it had a name. ``ContractError`` is what
    routes it: both engines run their merge on a background thread and absorb
    ordinary exceptions there as backend failures, retrying past them -- so a
    plain ``NotImplementedError`` raised inside the barrier-free merger would be
    counted as a transient, retried until the sweep budget ran out, and reported
    as though a provider had flaked. `ContractError` is the channel this
    repository already uses for "the caller's own code makes this run
    meaningless": it is stashed and re-raised on the caller's thread. Same
    reasoning, same shape as `ProposalContractError` and `RewardContractError`.
    """


@dataclass(frozen=True)
class Candidate:
    """One starting point the next batch could be launched from.

    Read-only, and deliberately not an :class:`~agentdescent.evolvable.Evolvable`:
    a selection policy chooses *among* candidates and must not be able to mutate
    one. ``state`` is carried so a policy can measure distance or novelty without
    reaching back into the ledger.
    """

    artifact_id: str
    version: int
    state: Mapping[str, str] = field(default_factory=dict)
    #: Aggregate held-out score, when one has been measured. ``None`` means "not
    #: scored yet", which a policy must distinguish from "scored zero" -- ranking
    #: an unmeasured candidate as the worst is how a new branch never gets tried.
    score: Optional[float] = None
    #: Per-task scores, keyed by task id. Empty unless the caller measured them.
    #: This is what separates per-instance Pareto from aggregate ranking, and a
    #: policy that needs it and finds it empty must say so rather than guess.
    per_task: Mapping[str, float] = field(default_factory=dict)
    #: How many times this candidate has already been selected. Archive sampling
    #: uses it as the novelty term; without it a "novelty" score is just age.
    selected: int = 0
    #: Version this candidate was derived from, for tree-shaped policies.
    parent: Optional[int] = None
    #: A policy prior -- AlphaZero's ``P(s,a)``, the slot :class:`FlatPuct`
    #: otherwise fills with a uniform ``1/N``. ``None`` means "no opinion" and
    #: falls back to uniform, which is what an unrated candidate must get: zero
    #: would bar it from ever being explored on the strength of a missing
    #: number. Higher means "worth expanding", not "already good" -- the score
    #: says how good it is, and the two are different claims about a candidate
    #: that is slow today because it is a first draft of the right idea.
    prior: Optional[float] = None


@dataclass(frozen=True)
class SelectionContext:
    """What a :class:`SelectionPolicy` is allowed to look at.

    ``head`` is the current `dev` head -- what the engine would have used with no
    policy at all, and therefore the answer a policy returns when it has no
    reason to do anything else.
    """

    head: Candidate
    #: Every candidate the caller knows about, including ``head``. On the single
    #: head path this is ``(head,)``; a policy must work in that case and not
    #: assume an archive exists.
    candidates: Sequence[Candidate] = ()
    round: int = 0
    n_workers: int = 1


@runtime_checkable
class SelectionPolicy(Protocol):
    """Given the candidates, return the ``n`` starting points for the next batch.

    Returning the same candidate several times is meaningful and normal: it means
    "put this many workers on that point". Returning fewer than ``n`` is also
    allowed -- the engine assigns workers round-robin over whatever comes back.
    """

    def select(self, ctx: SelectionContext, n: int) -> Sequence[Candidate]: ...


class SingleHead:
    """Every worker starts from the current head. Today's behaviour, exactly.

    The default, and the reason this module can be added without a measurement
    changing. `Beam(1)` computes the same answer by a different route, and
    `tests/test_selection.py` asserts the two agree -- which is what makes the
    beam implementation trustworthy, since it has no separate ground truth."""

    def select(self, ctx: SelectionContext, n: int) -> Sequence[Candidate]:
        return [ctx.head] * n


class Beam(SingleHead):
    """Keep the ``k`` best-scoring candidates and spread the workers over them.

    Unscored candidates sort *first*, not last. A candidate with ``score=None``
    has not been measured, and ranking it as the worst is how a freshly created
    branch is never explored -- the classic way a beam collapses to a single line
    of descent.

    The walk over the beam is **continuous across calls**, offset by
    ``ctx.round``, not restarted at the best member each time. That is what makes
    ``k`` mean anything where a caller asks for one starting point at a time: the
    engine's ledger holds one live head, so the population layer asks for one per
    merge, and a beam that answered "the best" every time was `Beam(1)` under
    another name. Rotating expands each of the ``k`` in turn instead -- serial
    where textbook beam search is parallel, and the same frontier.

    At ``ctx.round == 0`` this is exactly the old per-call round-robin, which is
    what makes the change checkable: `tests/test_selection.py` pins that every
    policy's answer at round 0 is the answer it gave before.
    """

    def __init__(self, k: int = 1) -> None:
        if k < 1:
            raise ValueError(f"beam width must be at least 1, got {k}")
        self.k = k

    def select(self, ctx: SelectionContext, n: int) -> Sequence[Candidate]:
        if self.k == 1 and len(ctx.candidates) <= 1:
            return super().select(ctx, n)
        # Unscored first (`c.score is None` sorts as 1 under reverse), then by
        # score descending. See the class docstring: ranking an unmeasured
        # candidate as the worst is how a beam collapses to one line of descent.
        ranked = sorted(ctx.candidates,
                        key=lambda c: (c.score is None, c.score or 0.0),
                        reverse=True)
        beam = ranked[:self.k] or [ctx.head]
        return [beam[(ctx.round + i) % len(beam)] for i in range(n)]


def pareto_win_frequency(
        scores: Sequence[Sequence[float]]) -> Tuple[Set[int], Dict[int, int]]:
    """GEPA Algorithm 2, steps 1-4: the frontier and how often each member wins.

    ``scores[c][i]`` is candidate ``c``'s score on Pareto instance ``i``.

    1. per-instance best; 2. union of the candidates that tie it; 3. drop any
    candidate strictly dominated by another survivor; 4. count, for each
    survivor, how many instances it still ties the best on.

    This is **not** :func:`pareto_front`, and the difference is the point.
    `pareto_front` keeps everything nothing else dominates; this keeps only what
    ties the best on at least one instance, then prunes. A candidate that is
    merely un-dominated -- never the best anywhere -- is on the first and not on
    the second. Two published rules, two functions, so a run can say which it
    used.
    """
    if not scores or not scores[0]:
        return set(), {}
    n_cand, n_inst = len(scores), len(scores[0])
    # 1 + 2: candidates that tie the best score on at least one instance.
    best = [max(scores[c][i] for c in range(n_cand)) for i in range(n_inst)]
    pstar = [{c for c in range(n_cand) if scores[c][i] == best[i]}
             for i in range(n_inst)]
    kept: Set[int] = set().union(*pstar) if pstar else set()
    # 3: iteratively remove any candidate strictly dominated by another in the
    # surviving set -- not by any candidate, which would prune against rows that
    # are themselves already out.
    changed = True
    while changed:
        changed = False
        for b in list(kept):
            if any(a != b
                   and all(x >= y for x, y in zip(scores[a], scores[b]))
                   and any(x > y for x, y in zip(scores[a], scores[b]))
                   for a in kept):
                kept.discard(b)
                changed = True
                break
    # 4: frequency over the pruned per-instance frontier.
    freq: Dict[int, int] = {c: 0 for c in kept}
    for winners in pstar:
        for c in (winners & kept):
            freq[c] += 1
    return kept, freq


def sigmoid_novelty_weights(scores: Sequence[float],
                            children: Sequence[int]) -> List[float]:
    """DGM's ``score_child_prop``: ``sigmoid(10*(s-0.5)) * 1/(1+children)``.

    Favour high performers, discount parents already explored. Normalised, and
    an all-zero raw vector divides by 1 rather than by 0 -- the degenerate case
    an archive of untouched zero-scorers actually reaches.

    Here rather than in a port because it was in *two*: `examples/dgm` and
    `examples/adas` each carried a byte-identical copy, one selecting a parent
    and one choosing what to show the meta-agent. Two copies of a published
    formula is how the two quietly stop being the same formula.
    """
    raw = [(1.0 / (1.0 + math.exp(-10.0 * (s - 0.5)))) * (1.0 / (1.0 + c))
           for s, c in zip(scores, children)]
    total = sum(raw) or 1.0
    return [r / total for r in raw]


def pareto_front(candidates: Sequence[Candidate], *,
                 tasks: Sequence[str]) -> List[Candidate]:
    """Candidates no other candidate beats on every task and betters on one.

    Plain domination, on the task ids given. Candidates missing a task score for
    a task are treated as scoring zero on it -- explicit, because the alternative
    (skipping the task) lets a candidate dominate by being measured on less.
    """
    def scores(c: Candidate) -> List[float]:
        return [c.per_task.get(t, 0.0) for t in tasks]

    front: List[Candidate] = []
    for c in candidates:
        cs = scores(c)
        if not any(all(o >= v for o, v in zip(scores(other), cs))
                   and any(o > v for o, v in zip(scores(other), cs))
                   for other in candidates if other is not c):
            front.append(c)
    return front


class ParetoFrontier(SingleHead):
    """Three published frontier rules, as one class and one argument.

    ``win_frequency``
        **GEPA's Algorithm 2**, exactly: the per-instance winners, pruned of
        anything a survivor dominates, sampled in proportion to how many
        instances each survivor still wins. One draw per call, repeated for
        every worker, because Algorithm 2 selects *a* parent per iteration.
        Needs ``Candidate.per_task`` and an rng.
    ``per_instance``
        Plain Pareto: a candidate survives when nothing else dominates it, and
        the survivors are walked round-robin. Close to GEPA and not GEPA -- it
        keeps candidates that are best at nothing, and it does not weight the
        draw. This is the mode the class shipped with under GEPA's name; the
        name was wrong and `win_frequency` is what that name meant.
    ``topk_aggregate``
        EvoSkill's released code. Rank by the aggregate score and keep the top
        ``k``. Cheaper, and it discards the specialist the other two keep.

    Making this an argument is the point: a fidelity difference that lives in
    hand-written per-port implementations drifts, and a reader of a result
    cannot tell which rule produced it.
    """

    MODES = ("win_frequency", "per_instance", "topk_aggregate")

    def __init__(self, mode: str = "per_instance", k: int = 5,
                 seed: int = 0, rng: Optional["random.Random"] = None) -> None:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, not {mode!r}")
        self.mode = mode
        self.k = k
        self.seed = seed
        #: Passed in when the caller owns the stream. A port migrating off a
        #: hand-written rule has to keep drawing from *its* rng in *its* order,
        #: or a seeded run picks different parents and every measured number it
        #: published moves. A fresh `Random(seed)` otherwise.
        self.rng = rng

    def _draw(self, ctx: SelectionContext):
        import random

        if self.rng is not None:
            return self.rng
        return random.Random(self.seed * 1_000_003 + ctx.round)

    def _win_frequency(self, ctx: SelectionContext, n: int) -> Sequence[Candidate]:
        """Algorithm 2 step 5: one draw, weighted by win frequency.

        One `rng.choices(..., k=1)` for the whole batch, not `k=n`: Algorithm 2
        selects a parent per iteration and every worker mutates it. Drawing per
        worker would consume the stream differently and pick different parents
        on a seeded re-run of an already-published number.
        """
        candidates = list(ctx.candidates)
        tasks = sorted({t for c in candidates for t in c.per_task})
        if not tasks:
            raise ValueError(
                "ParetoFrontier(mode='win_frequency') is GEPA Algorithm 2 and "
                "needs Candidate.per_task scores; none were provided. Use "
                "mode='topk_aggregate' if aggregate ranking is what you want.")
        rows = [[c.per_task.get(t, 0.0) for t in tasks] for c in candidates]
        kept, freq = pareto_win_frequency(rows)
        pool = [c for c in sorted(kept) if freq[c] > 0]
        if not pool:
            # Degenerate -- every row identical, so nothing uniquely wins.
            # Best aggregate, which is where upstream lands too, and no draw:
            # spending a random number here would desynchronise the stream
            # against a run that had one more candidate.
            best = max(range(len(rows)), key=lambda c: sum(rows[c]))
            return [candidates[best]] * n
        chosen = self._draw(ctx).choices(pool, weights=[freq[c] for c in pool],
                                         k=1)[0]
        return [candidates[chosen]] * n

    def select(self, ctx: SelectionContext, n: int) -> Sequence[Candidate]:
        if len(ctx.candidates) <= 1:
            return super().select(ctx, n)
        if self.mode == "topk_aggregate":
            return Beam(self.k).select(ctx, n)
        if self.mode == "win_frequency":
            return self._win_frequency(ctx, n)
        tasks = sorted({t for c in ctx.candidates for t in c.per_task})
        if not tasks:
            # Refused rather than silently degraded to aggregate ranking: that
            # would be *EvoSkill's* rule reported under GEPA's name, which is the
            # exact confusion this class exists to remove.
            raise ValueError(
                "ParetoFrontier(mode='per_instance') needs Candidate.per_task "
                "scores and none were provided; use mode='topk_aggregate' if "
                "aggregate ranking is what you want")
        front = pareto_front(ctx.candidates, tasks=tasks) or [ctx.head]
        return [front[(ctx.round + i) % len(front)] for i in range(n)]


class Archive(SingleHead):
    """DGM's and ADAS's archive sampling: performance, tempered by novelty.

    ``sampling='performance'`` is a softmax over score alone. ``'novelty'``
    divides by ``1 + selected``, so a candidate that has already been the parent
    many times is chosen less -- which is what stops an archive from behaving
    like a greedy hill climb. ``'uniform'`` is the ablation, and having it here
    means the archive's contribution can be measured rather than assumed.

    ``'best'`` does not sample at all: it returns the highest scorer every time,
    which is SICA's rule -- ``get_best_agent_iteration`` takes ``idxmax()`` of
    the mean benchmark score and the next meta-improvement starts from exactly
    that agent. It is here rather than inside one example because it is a
    published archive rule, and because the difference from ``'performance'`` is
    easy to miss: a softmax over scores in ``[0, 1]`` at temperature 1 puts
    ``exp(1)/exp(0) = 2.7`` between the best and worst candidate, so a run
    configured as "performance" picks the worst entry roughly a quarter of the
    time where SICA would never pick it.

    ``'sigmoid_novelty'`` is **DGM's** ``choose_selfimproves`` exactly:
    :func:`sigmoid_novelty_weights`, i.e. ``sigmoid(10*(s-0.5)) / (1+selected)``.
    The shape is `'novelty'`'s and the numbers are not -- a sigmoid at gain 10
    is nearly a step at 0.5, where a temperature-1 softmax is nearly flat, so
    the two disagree most exactly where an archive spends its time. It is a
    separate mode rather than a tuning of `'novelty'` for that reason.

    Deterministic given ``seed``: an archive that samples differently on a
    re-run makes a seeded comparison meaningless. Pass ``rng`` instead when the
    caller owns the stream -- a port migrating off a hand-written rule has to
    keep drawing from *its* rng in *its* order, or every measured number it
    published moves.
    """

    MODES = ("performance", "novelty", "sigmoid_novelty", "uniform", "best")

    def __init__(self, sampling: str = "novelty", temperature: float = 1.0,
                 seed: int = 0, rng: Optional["random.Random"] = None) -> None:
        if sampling not in self.MODES:
            raise ValueError(f"sampling must be one of {self.MODES}, not {sampling!r}")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.sampling = sampling
        self.temperature = temperature
        self.seed = seed
        self.rng = rng

    def _weight(self, c: Candidate) -> float:
        if self.sampling == "uniform":
            return 1.0
        # An unscored candidate is given the mean, not zero: the archive's job is
        # to keep trying things that have not been tried.
        score = 0.5 if c.score is None else c.score
        weight = math.exp(score / self.temperature)
        return weight / (1 + c.selected) if self.sampling == "novelty" else weight

    def select(self, ctx: SelectionContext, n: int) -> Sequence[Candidate]:
        import random

        if len(ctx.candidates) <= 1:
            return super().select(ctx, n)
        if self.sampling == "best":
            # First maximum, as `idxmax` takes: ties go to the earlier entry, so
            # a later candidate has to actually beat the incumbent to displace it.
            best = max(ctx.candidates,
                       key=lambda c: (0.0 if c.score is None else c.score))
            return [best] * n
        pool = list(ctx.candidates)
        if self.sampling == "sigmoid_novelty":
            # An unscored candidate is a zero here, not the mean: DGM's archive
            # holds evaluated agents, and inventing 0.5 for one would place it
            # exactly on the sigmoid's steepest point.
            weights = sigmoid_novelty_weights(
                [0.0 if c.score is None else c.score for c in pool],
                [c.selected for c in pool])
        else:
            weights = [self._weight(c) for c in pool]
        # An int, not a tuple: tuple seeding is deprecated from 3.9 and would
        # eventually start raising in the middle of a long run.
        rng = self.rng or random.Random(self.seed * 1_000_003 + ctx.round)
        return rng.choices(pool, weights=weights, k=n)


class FlatPuct(SingleHead):
    """ERA's Flat UCB tree search -- the PUCT rule from `google-research/era`.

    Two things separate it from :class:`MCTS` below, and both matter.

    **Flat.** Every node in the tree is a selection candidate, always. There is
    no descent from the root through children, so a good node found at depth 1
    competes with one at depth 9 on equal terms; "flat" names that, not a depth
    limit.

    **Ranked, not scored.** The exploitation term is the candidate's *rank*
    among all nodes, normalised to ``[0, 1]``, rather than its held-out score.
    The formula is therefore invariant to the metric's units -- which is what
    lets one search run against RMSE, log-likelihood, or accuracy without
    retuning ``c_puct``. Upstream ``implementation/futs.py``::

        rank_score = rank / (len(nodes) - 1)     # 0.5 when there is one node
        puct = rank_score + c_puct * (1/N) * sqrt(total_visits) / (1 + visits)

    ``visits`` is :attr:`Candidate.selected`, and it must be the **subtree**
    count: upstream backpropagates every expansion up the ``parent`` chain, so a
    caller that increments only the node it handed out is running a different
    rule with the same name. The prior is uniform (``1/N``) because a program is
    not a game move -- there is no policy network to ask which sibling is
    promising, so AlphaZero's ``P(s, a)`` degenerates to the uniform one.

    A FUTS tree never holds an unscored node -- a node exists only after
    ``execute_fn`` has returned -- so ``score=None`` is ranked as ``-inf`` here
    rather than getting this module's usual unscored-first treatment. Ties keep
    the order in which ``ctx.candidates`` presents them, and the winner is the
    first maximal PUCT, both matching upstream's stable ``sorted`` and ``max``.

    Upstream is **serial**: one node is expanded per iteration, so ``n > 1`` has
    no published behaviour. Each pick here reserves a visit on itself and its
    ancestors before the next pick is computed, which is what the serial loop
    would have seen had those expansions already been dispatched -- and is
    upstream exactly at ``n == 1``.
    """

    def __init__(self, c_puct: float = 1.0, prior_exponent: float = 0.0) -> None:
        self.c_puct = c_puct
        #: How sharply :attr:`Candidate.prior` bends the exploration term.
        #: ``0.0`` ignores priors entirely and is upstream's uniform ``1/N``;
        #: ``2.0`` weights by the square, which is what separates a promising
        #: candidate from a dead-end one rather than merely ordering them.
        self.prior_exponent = prior_exponent

    def _priors(self, rows: Sequence[Candidate]) -> List[float]:
        """``P(s,a)``, normalised to sum to 1 so ``c_puct`` keeps its meaning.

        Upstream ERA has no prior: ``futs.py`` uses ``1 / len(nodes)`` for every
        node, so the exploration budget is spread evenly over a tree in which
        most nodes are dead ends. With ``prior_exponent == 0`` that is exactly
        what this returns, to the floating-point bit, and the port stays
        faithful by default.

        Above zero the priors are raised to that power and renormalised. The
        power is the point rather than a knob for its own sake: with a prior
        proportional to the rating, a candidate rated 8 against a mean of 5.5
        gets 1.45x the exploration term and a dead end rated 2 still gets 0.36x
        of it. Squared, those become 2.12x and 0.13x -- the difference between
        widening exploration and *aiming* it.

        A candidate with no prior is given the mean of those that have one, not
        zero: an unrated node must not be barred from selection by the absence
        of a number.
        """
        rated = [c.prior for c in rows
                 if c.prior is not None and math.isfinite(c.prior) and c.prior > 0]
        if self.prior_exponent <= 0.0 or not rated:
            return [1.0 / len(rows)] * len(rows)
        fallback = sum(rated) / len(rated)
        raw = [
            (c.prior if c.prior is not None and math.isfinite(c.prior) and c.prior > 0
             else fallback) ** self.prior_exponent
            for c in rows
        ]
        total = sum(raw)
        if not total or not math.isfinite(total):
            return [1.0 / len(rows)] * len(rows)
        return [value / total for value in raw]

    @staticmethod
    def _rank_scores(rows: Sequence[Candidate]) -> List[float]:
        if len(rows) == 1:
            return [0.5]
        scores = [-math.inf if c.score is None else c.score for c in rows]
        order = sorted(range(len(rows)), key=lambda i: scores[i])
        ranks = [0.0] * len(rows)
        for rank, index in enumerate(order):
            ranks[index] = rank / (len(rows) - 1)
        return ranks

    def select(self, ctx: SelectionContext, n: int) -> Sequence[Candidate]:
        rows = list(ctx.candidates)
        if len(rows) <= 1:
            return super().select(ctx, n)
        ranks = self._rank_scores(rows)
        visits = [c.selected for c in rows]
        by_version = {c.version: i for i, c in enumerate(rows)}
        priors = self._priors(rows)
        picked: List[Candidate] = []
        for _ in range(n):
            total = sum(visits)
            best, best_puct = 0, -math.inf
            for i, row in enumerate(rows):
                puct = ranks[i] + self.c_puct * priors[i] * math.sqrt(total) / (
                    1 + visits[i])
                if puct > best_puct:
                    best, best_puct = i, puct
            picked.append(rows[best])
            # Reserve the visit upstream would have backpropagated once this
            # expansion finished -- self, then every ancestor, guarded against a
            # parent chain that loops or points outside the pool.
            seen: Set[int] = set()
            node: Optional[int] = best
            while node is not None and node not in seen:
                seen.add(node)
                visits[node] += 1
                parent = rows[node].parent
                node = by_version.get(parent) if parent is not None else None
        return picked


class MCTS(SingleHead):
    """UCT over the candidate tree: one evolve step is one rollout.

    ``value`` is the candidate's held-out reward, ``visits`` is
    ``Candidate.selected``, and the backup runs up ``Candidate.parent``. An
    unvisited candidate has infinite UCT score and so is always tried first,
    which is the standard rule and the one that keeps a shallow tree from being
    ignored.
    """

    def __init__(self, exploration: float = 1.4) -> None:
        self.exploration = exploration

    def _uct(self, c: Candidate, total: int) -> float:
        if c.selected == 0:
            return math.inf
        exploit = 0.0 if c.score is None else c.score
        return exploit + self.exploration * math.sqrt(
            math.log(max(1, total)) / c.selected)

    def select(self, ctx: SelectionContext, n: int) -> Sequence[Candidate]:
        if len(ctx.candidates) <= 1:
            return super().select(ctx, n)
        total = sum(c.selected for c in ctx.candidates)
        ranked = sorted(ctx.candidates, key=lambda c: self._uct(c, total),
                        reverse=True)
        # `n` distinct arms when there are enough of them -- sending every worker
        # to the same arm would make a batch worth one rollout of information.
        chosen = ranked[:n] or [ctx.head]
        return [chosen[i % len(chosen)] for i in range(n)]
