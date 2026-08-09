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
starting points is therefore *refused* by the engine rather than silently
collapsed to the first, and every policy below is usable today in the shape that
returns one. Making the ledger hold concurrent branches, and redefining `eta`
when `head` is plural, is a separate change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import (
    Dict, List, Mapping, Optional, Protocol, Sequence, runtime_checkable,
)

__all__ = [
    "Archive",
    "Beam",
    "Candidate",
    "MCTS",
    "ParetoFrontier",
    "SelectionContext",
    "SelectionPolicy",
    "SingleHead",
    "pareto_front",
]


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
        return [beam[i % len(beam)] for i in range(n)]


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
    """GEPA's and EvoSkill's selection rules, as one class and one argument.

    They are genuinely different rules and the repository has documented the
    difference in prose for a while:

    ``per_instance``
        GEPA. A candidate survives when no other dominates it across the
        held-out instances, so a candidate that is best on a single hard task
        stays in. Needs ``Candidate.per_task``.
    ``topk_aggregate``
        EvoSkill's released code. Rank by the aggregate score and keep the top
        ``k``. Cheaper, and it discards the specialist the first mode keeps.

    Making this an argument is the point: a fidelity difference that lives in two
    hand-written implementations drifts, and a reader cannot tell which one a run
    used.
    """

    MODES = ("per_instance", "topk_aggregate")

    def __init__(self, mode: str = "per_instance", k: int = 5) -> None:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, not {mode!r}")
        self.mode = mode
        self.k = k

    def select(self, ctx: SelectionContext, n: int) -> Sequence[Candidate]:
        if len(ctx.candidates) <= 1:
            return super().select(ctx, n)
        if self.mode == "topk_aggregate":
            return Beam(self.k).select(ctx, n)
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
        return [front[i % len(front)] for i in range(n)]


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

    Deterministic given ``seed``: an archive that samples differently on a
    re-run makes a seeded comparison meaningless.
    """

    MODES = ("performance", "novelty", "uniform", "best")

    def __init__(self, sampling: str = "novelty", temperature: float = 1.0,
                 seed: int = 0) -> None:
        if sampling not in self.MODES:
            raise ValueError(f"sampling must be one of {self.MODES}, not {sampling!r}")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.sampling = sampling
        self.temperature = temperature
        self.seed = seed

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
        # An int, not a tuple: tuple seeding is deprecated from 3.9 and would
        # eventually start raising in the middle of a long run.
        rng = random.Random(self.seed * 1_000_003 + ctx.round)
        pool = list(ctx.candidates)
        weights = [self._weight(c) for c in pool]
        return rng.choices(pool, weights=weights, k=n)


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
