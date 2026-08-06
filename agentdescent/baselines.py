"""The control every efficiency number in this repository is missing.

Everything measured here so far is **throughput speedup**: how much faster the
same work finishes on N workers. That number cannot answer the question the
project is actually making a claim about, because population-based methods are
already parallel -- fork-and-select saturates N workers just as well, and its
speedup is also close to N. A speedup table therefore cannot tell *merging*
apart from *sampling and selecting*.

One quantity can: **held-out quality at equal rollout budget, merge-of-N against
best-of-N fork.** This module runs the three arms that produce it.

    serial          n_workers=1, B rollouts
    best_of_n_fork  N runs that never talk to each other, B/N rollouts each
    merge_of_n      evolve(n_workers=N), B rollouts

Four things this module refuses to let a caller get wrong, because each of them
has produced a published-looking number somewhere before:

**One workload, not three.** The tasks, reward, actors, strategy and every other
`evolve()` argument come from a single :class:`Workload`, so an arm cannot
quietly get a different sampler or a different held-out fraction. Only the
execution shape varies -- that is the whole design.

**Budget in rollouts *and* calls -- and they cannot both be equalised.** This is
the uncomfortable finding, and it is measured rather than assumed
(``tests/test_baselines.py``). Four forks that never talk to each other each start
from nothing, so nearly every rollout of theirs fails and asks for a proposal;
merge-of-4 shares what the others already learned, so more of its rollouts solve
their task outright and never call the proposer. Fix rollouts and the fork arm
spends over twice the model. Fix calls and the fork arm gets a *quarter* of the
rollouts. The ratio is not a nuisance parameter -- it is a real difference between
the arms, and no budget can make it go away.

So :func:`compare` takes the unit being held fixed and reports the other one
separately: drift in the fixed unit is a **broken comparison**, divergence in the
other is a **confound that has to be printed next to the result**. A merge arm
that wins at equal rollouts *while spending fewer calls* is a robust result. One
that wins at equal rollouts while spending more calls is not, and the table says
so instead of putting "equal budget" in the caption. Both numbers are reported as
measured, never as requested: the synchronous engine checks its budget at the
round barrier and overshoots by up to one round.

**Fork gets two numbers, not one.** The oracle fork (the best fork *on test*) is
an upper bound nobody can ship -- picking it requires the answer. The selected
fork (best on dev, reported on test) is what fork-and-select actually delivers.
Reporting only the first flatters fork; reporting only the second flatters merge.
`run_fork_baseline`'s docstring already drew this distinction for the synthetic
router domain; it belongs on the real path too.

**The domain has to be able to say no.** The router domain is additive by
construction -- each keyword is an independent row, so two workers' diffs almost
never conflict and merging necessarily wins. Demonstrating "diffs can be merged"
there is circular: the property being measured was put in by the domain. Use this
module on a workload where diffs can genuinely contradict each other.

A negative result is a result. If merge-of-N lands within the spread of
best-of-N fork at equal budget, that says parallel merging is a throughput
optimisation on this workload and not a new mechanism, and the claim that
"gradients add, diffs do not" implies more than engineering convenience has to be
written down as unsupported *on that workload, at that budget*.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple,
)

from .agents import Usage
from .evolution import EvolutionResult, FusionStats, Reward, Task, evolve

__all__ = [
    "ArmResult",
    "Budget",
    "Comparison",
    "ForkOutcome",
    "Workload",
    "best_of_n_fork",
    "compare",
    "merge_of_n",
    "serial",
    "to_markdown",
]


# ---------------------------------------------------------------------------
# What is held fixed
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Budget:
    """What every arm is allowed to spend.

    Both units, always. Matched on rollouts alone, an arm that asks for more
    proposals per rollout spends more model and the table still says "equal
    budget"; matched on calls alone, an arm with a cheaper proposer buys more
    rollouts. Neither is derivable from the other -- a rollout that solves its
    task never proposes at all.
    """

    rollouts: int
    #: ``None`` means "do not bound calls", which is the right setting when every
    #: arm shares one actor and the ratio is stable. Set it as soon as arms
    #: differ in how often they propose.
    calls: Optional[int] = None

    def split(self, ways: int) -> "Budget":
        """The share of this budget one of ``ways`` independent runs may spend."""
        if ways < 1:
            raise ValueError(f"cannot split a budget {ways} ways")
        return Budget(rollouts=self.rollouts // ways,
                      calls=None if self.calls is None else self.calls // ways)


@dataclass(frozen=True)
class Workload:
    """The half of the comparison that must not vary, in one object.

    Passing tasks and actors to three call sites is how two arms end up with
    different held-out fractions and the difference gets called merging.

    ``test_eval`` scores a finished run on a set **the gate never saw**.
    ``evolve()`` optimises against its own held-out split and stops when that
    split says so, so ``result.final_reward`` is a training score for the purpose
    of this comparison: fork's selection step in particular would be choosing on
    the same numbers it is then judged by. Hold a third split out of ``tasks``
    entirely and score it here.
    """

    tasks: Sequence[Task]
    reward: Reward
    test_eval: Callable[[EvolutionResult], float]
    #: Exactly as `evolve()` takes them -- ``agent=`` or ``run=``/``propose=``.
    agent: Optional[Any] = None
    run: Optional[Any] = None
    propose: Optional[Any] = None
    strategy: Optional[Any] = None
    #: Everything else `evolve()` accepts, applied identically to every arm.
    #: `rounds` belongs here and should be set far above what the budget allows:
    #: the budget is meant to be what ends the run, and a round cap that bites
    #: first silently reintroduces the budget-in-rounds mistake.
    evolve_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def _evolve(self, *, seed: int, n_workers: int, budget: Budget,
                usage: Usage) -> EvolutionResult:
        kwargs: Dict[str, Any] = dict(self.evolve_kwargs)
        for name, value in (("agent", self.agent), ("run", self.run),
                            ("propose", self.propose),
                            ("strategy", self.strategy)):
            if value is not None:
                kwargs[name] = value
        kwargs.setdefault("rounds", 10_000)
        return evolve(
            self.tasks, self.reward, seed=seed, n_workers=n_workers,
            max_concurrency=kwargs.pop("max_concurrency", n_workers),
            max_rollouts=budget.rollouts, max_calls=budget.calls,
            usage=usage, **kwargs)


def _scored(workload: Workload, result: EvolutionResult) -> Tuple[Optional[float], str]:
    """Score a finished run on the test split, surviving a backend failure.

    `evolve()` absorbs backend failures into `result.error` and still returns the
    artifact it evolved. `test_eval` runs *after* it, outside that protection, and
    is a plain sequence of model calls -- so one transient dropped connection
    there used to raise out of the arm, out of the thread pool, and discard every
    arm that had already finished.

    This is not a hypothetical: `agents.py` records the identical failure ("one
    `RemoteDisconnected` during an example's final held-out evaluation ...
    discarded the entire run") and it happened again here, in a sweep that had
    already paid for three arms. A missing score is a missing score; it is not a
    reason to throw away the ones that exist.
    """
    try:
        return workload.test_eval(result), ""
    except Exception as e:  # noqa: BLE001 - report it, keep every other arm
        return None, f"test_eval failed: {type(e).__name__}: {str(e)[:160]}"


# ---------------------------------------------------------------------------
# What an arm produced
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForkOutcome:
    """One member of a fork arm, kept so the selection step can be audited."""

    seed: int
    dev_reward: float
    #: ``None`` when the scoring pass failed. Never substituted with zero.
    test_reward: Optional[float]
    rollouts: int
    calls: int


@dataclass(frozen=True)
class ArmResult:
    """One arm, one seed, and the spend it actually incurred.

    ``rollouts`` and ``calls`` are measured, not requested. A budget is checked
    at the round barrier on the synchronous path, so an arm ends past its bound
    by up to one round -- and a wider arm's round is bigger. Comparing on the
    requested number hides exactly that.
    """

    arm: str
    seed: int
    #: Workers inside one run (merge), or independent runs (fork).
    width: int
    rollouts: int
    calls: int
    prompt_tokens: int
    completion_tokens: int
    #: Sum of every run in the arm. For fork this is what it cost, not what it
    #: would take: N forks are independent and run concurrently given N workers,
    #: which is why `wallclock_parallel` exists beside it.
    wallclock: float
    #: The longest single run in the arm -- fork's wall-clock if its members run
    #: at the same time. Equal to `wallclock` for the single-run arms.
    wallclock_parallel: float
    #: What each run's own gate saw. Not comparable across arms as a quality
    #: number: it is the split the run optimised against and stopped on.
    dev_reward: float
    #: Quality on the set no gate saw. This is the comparison. ``None`` when the
    #: scoring pass itself failed -- distinct from a low score, and never
    #: substituted with one.
    test_reward: Optional[float]
    #: Fork only. `test_reward` above is the *selected* fork -- best on dev,
    #: reported on test, which is what fork-and-select can ship. `test_oracle` is
    #: the best fork on test, an upper bound that requires the answer to pick.
    test_oracle: Optional[float] = None
    forks: Tuple[ForkOutcome, ...] = ()
    stop_reason: str = ""
    #: Non-empty when a run inside this arm ended on a failure. A quality number
    #: from a partial run is not a quality number.
    error: Optional[str] = None
    #: The fusion tournament's record, summed over every run in the arm. Carried
    #: here because the merge arm is the *only* place the fusion question can be
    #: answered -- serial has one proposal per step and fork never merges at all
    #: -- and because paying for a real run twice to answer two questions about
    #: the same mechanism would be absurd. Empty for the arms that cannot fuse,
    #: which is itself the right answer for them.
    fusion: Optional["FusionStats"] = None


def _tally(arm: str, seed: int, width: int, runs: Sequence[EvolutionResult],
           usages: Sequence[Usage], *, scoring_error: str = "",
           **extra: Any) -> ArmResult:
    # One arm may be several runs (fork), so the trials are concatenated before
    # being summarised -- summarising each run and averaging the rates would
    # weight a run that held two tournaments the same as one that held twenty.
    trials = [t for r in runs for t in r.fusion_trials]
    return ArmResult(
        fusion=FusionStats.of(trials) if trials else None,
        arm=arm, seed=seed, width=width,
        rollouts=sum(r.rollouts for r in runs),
        calls=sum(u.calls for u in usages),
        prompt_tokens=sum(u.prompt_tokens for u in usages),
        completion_tokens=sum(u.completion_tokens for u in usages),
        wallclock=sum(r.wallclock for r in runs),
        wallclock_parallel=max(r.wallclock for r in runs),
        stop_reason=",".join(sorted({r.stop_reason for r in runs})),
        error="; ".join([r.error for r in runs if r.error]
                        + ([scoring_error] if scoring_error else [])) or None,
        **extra)


# ---------------------------------------------------------------------------
# The three arms
# ---------------------------------------------------------------------------


def serial(workload: Workload, *, budget: Budget, seed: int = 0) -> ArmResult:
    """One worker, improving itself in sequence. The floor.

    Without it a comparison between two parallel arms cannot say whether either
    of them beat doing nothing clever at all.
    """
    usage = Usage()
    result = workload._evolve(seed=seed, n_workers=1, budget=budget, usage=usage)
    test, failure = _scored(workload, result)
    return _tally("serial", seed, 1, [result], [usage], scoring_error=failure,
                  dev_reward=result.final_reward, test_reward=test)


def merge_of_n(workload: Workload, n: int, *, budget: Budget,
               seed: int = 0) -> ArmResult:
    """N workers proposing into one artifact, merged every round. The claim."""
    if n < 1:
        raise ValueError(f"merge_of_n needs at least one worker, got {n}")
    usage = Usage()
    result = workload._evolve(seed=seed, n_workers=n, budget=budget, usage=usage)
    test, failure = _scored(workload, result)
    return _tally(f"merge-of-{n}", seed, n, [result], [usage],
                  scoring_error=failure, dev_reward=result.final_reward,
                  test_reward=test)


def best_of_n_fork(workload: Workload, n: int, *, budget: Budget,
                   seed: int = 0, concurrency: int = 1) -> ArmResult:
    """N runs that never see each other, each on its share of the budget.

    The forks differ only in seed. That is deliberate: any other difference
    (different data, different actors) would make this a comparison between two
    workloads rather than between merging and selecting.

    ``concurrency`` runs that many forks at once. It does not change any result:
    a fork's seed is fixed before it starts, its ledger is its own scratch repo,
    and by construction no fork can observe another -- that is what makes this
    the fork arm. It changes only how long the call takes, which is why
    wall-clock is reported both ways: ``wallclock`` sums the runs (what the work
    cost) and ``wallclock_parallel`` takes the longest (what N machines would
    take). Neither is affected by how many threads this process happened to use.

    Leave it at ``1`` when the backend is rate-limited; the arm is otherwise the
    slowest of the three by a factor of N.
    """
    if n < 1:
        raise ValueError(f"best_of_n_fork needs at least one fork, got {n}")
    if concurrency < 1:
        raise ValueError(f"concurrency must be at least 1, got {concurrency}")
    share = budget.split(n)

    def _one(i: int):
        # A fork's seed has to move, or N forks are one fork run N times and the
        # arm measures nothing. Spread them far enough apart that two forks
        # cannot collide with a neighbouring `seed` argument.
        fork_seed = seed * 1_000 + i
        usage = Usage()
        result = workload._evolve(seed=fork_seed, n_workers=1, budget=share,
                                  usage=usage)
        test, failure = _scored(workload, result)
        return result, usage, ForkOutcome(
            seed=fork_seed, dev_reward=result.final_reward, test_reward=test,
            rollouts=result.rollouts, calls=usage.calls), failure

    if concurrency > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(concurrency, n)) as pool:
            # `map` preserves input order, so the forks are collected in seed
            # order however they finish -- a selection that depended on
            # completion order would not be reproducible.
            done = list(pool.map(_one, range(n)))
    else:
        done = [_one(i) for i in range(n)]

    results: List[EvolutionResult] = [d[0] for d in done]
    usages: List[Usage] = [d[1] for d in done]
    outcomes: List[ForkOutcome] = [d[2] for d in done]
    failures = "; ".join(d[3] for d in done if d[3])

    # Selection, both ways. `max` over dev breaks ties on the first fork, which
    # is the pessimistic choice and the honest one: a real selector has no reason
    # to prefer a later tie.
    selected = max(outcomes, key=lambda f: f.dev_reward)
    # `None` sorts below every score rather than raising: a fork whose scoring
    # pass died is not the best fork on test, and it is not zero either.
    scored = [f for f in outcomes if f.test_reward is not None]
    oracle = max(scored, key=lambda f: f.test_reward) if scored else selected
    return _tally(f"fork-of-{n}", seed, n, results, usages,
                  scoring_error=failures,
                  dev_reward=selected.dev_reward,
                  test_reward=selected.test_reward,
                  test_oracle=oracle.test_reward,
                  forks=tuple(outcomes))


# ---------------------------------------------------------------------------
# Reading the arms against each other
# ---------------------------------------------------------------------------


@dataclass
class Comparison:
    """Several seeds of several arms, and whether they are comparable at all."""

    arms: Dict[str, List[ArmResult]]
    #: The unit the comparison holds fixed: ``"rollouts"`` or ``"calls"``.
    fixed: str = "rollouts"
    #: Arms whose spend in the **fixed** unit drifted from the median by more
    #: than the tolerance, as ``(arm, unit, spend, median)``. Non-empty means the
    #: comparison is broken -- these arms did not get the budget they were meant
    #: to, and no reading of the quality column is valid.
    unequal: List[Tuple[str, str, float, float]] = field(default_factory=list)
    #: The same, for the unit that was *not* fixed. Not a bug: the two units
    #: cannot both be equalised, because the arms genuinely differ in how often a
    #: rollout has to ask the model for anything. It is a confound, and it has to
    #: appear beside the result -- an arm that wins while also spending more here
    #: has not been shown to win.
    confounded: List[Tuple[str, str, float, float]] = field(default_factory=list)
    tolerance: float = 0.1

    def spread(self, arm: str) -> Optional[Tuple[float, float, float]]:
        """(min, median, max) test quality. Not a confidence interval.

        Three seeds cannot support one. This repository has published a point
        estimate that moved 4.8 points between two runs of one configuration, so
        the observed spread is the honest thing to show; a t-interval over three
        points is arithmetic dressed as statistics.
        """
        values = sorted(r.test_reward for r in self.arms[arm]
                        if r.test_reward is not None)
        if not values:
            return None
        return values[0], statistics.median(values), values[-1]

    def separates(self, a: str, b: str, *, min_seeds: int = 3) -> bool:
        """Whether ``a``'s seeds are all above ``b``'s, with no overlap.

        The weakest claim worth making from a handful of seeds, and the one this
        comparison exists to test. Overlapping spreads mean the experiment did
        not distinguish merging from selecting at this budget -- which is a
        finding, and has to be written as one.

        **False below ``min_seeds``, always.** With one seed per arm "no overlap"
        degenerates to ``a > b``, and this method would announce a separation
        from a single pair of numbers -- the exact overclaim the module exists to
        prevent, made by the thing meant to prevent it. It was written that way
        first and a one-seed pilot printed "above on every seed"; the guard is
        here because the mistake is that easy to make.
        """
        if min(self.scored(a), self.scored(b)) < min_seeds:
            return False
        sa, sb = self.spread(a), self.spread(b)
        return bool(sa and sb and sa[0] > sb[2])

    def underpowered(self, *arms: str, min_seeds: int = 3) -> bool:
        """Whether any named arm has too few seeds to support a comparison.

        Separate from :meth:`separates` so a caller can tell "we looked and found
        no difference" from "we could not have found one" -- they read the same
        in a table and mean opposite things about what to do next.
        """
        return any(self.scored(arm) < min_seeds for arm in arms)

    def scored(self, arm: str) -> int:
        """Seeds of ``arm`` that produced a test score at all.

        Not ``len(self.arms[arm])``: an arm whose scoring pass died has a row and
        no number, and counting it toward the seed requirement would let a
        two-seed comparison call itself three."""
        return sum(1 for r in self.arms.get(arm, ())
                   if r.test_reward is not None)


def compare(results: Sequence[ArmResult], *, fixed: str = "rollouts",
            tolerance: float = 0.1) -> Comparison:
    """Group arm results by arm and check what the comparison actually held fixed.

    ``fixed`` names the unit the budget was set in. Drift there goes to
    ``unequal`` and invalidates the comparison; divergence in the other unit goes
    to ``confounded``, which is a caveat rather than a defect -- see the module
    docstring for why it cannot be designed away.

    ``tolerance`` is a fraction of the median spend. The default allows the
    barrier overshoot -- an arm can end up to one round past its bound, and a
    wide arm's round is wide -- while still catching an arm that ran on twice
    the model.
    """
    if fixed not in ("rollouts", "calls"):
        raise ValueError(f"fixed must be 'rollouts' or 'calls', not {fixed!r}")
    by_arm: Dict[str, List[ArmResult]] = {}
    for r in results:
        by_arm.setdefault(r.arm, []).append(r)

    comparison = Comparison(arms=by_arm, fixed=fixed, tolerance=tolerance)
    for unit in ("rollouts", "calls"):
        spends = {arm: statistics.median([getattr(r, unit) for r in group])
                  for arm, group in by_arm.items()}
        median = statistics.median(list(spends.values()))
        if median <= 0:
            continue                      # nothing was spent in this unit at all
        bucket = comparison.unequal if unit == fixed else comparison.confounded
        for arm, spend in spends.items():
            if abs(spend - median) > tolerance * median:
                bucket.append((arm, unit, spend, median))
    return comparison


def to_markdown(comparison: Comparison) -> str:
    """A table whose caption cannot claim more than the numbers support."""
    lines = ["| arm | seeds | rollouts | calls | dev | test (min/med/max) | "
             "fork oracle |",
             "|---|---|---|---|---|---|---|"]
    for arm in sorted(comparison.arms):
        group = comparison.arms[arm]
        spread = comparison.spread(arm)
        oracles = [r.test_oracle for r in group if r.test_oracle is not None]
        oracle = f"{statistics.median(oracles):.3f}" if oracles else "—"
        # An arm whose scoring pass died has a row and no number. Printing a
        # blank is the point: it is not a low score, and the seed count beside it
        # says how many of its seeds are actually behind the interval.
        quality = ("—" if spread is None else
                   f"{spread[0]:.3f} / {spread[1]:.3f} / {spread[2]:.3f}")
        seeds = (f"{len(group)}" if comparison.scored(arm) == len(group)
                 else f"{comparison.scored(arm)}/{len(group)}")
        lines.append(
            f"| {arm} | {seeds} | "
            f"{statistics.median([r.rollouts for r in group]):.0f} | "
            f"{statistics.median([r.calls for r in group]):.0f} | "
            f"{statistics.median([r.dev_reward for r in group]):.3f} | "
            f"{quality} | {oracle} |")

    lines.append("")
    lines.append(f"Budget held fixed in **{comparison.fixed}**.")

    if comparison.unequal:
        lines.append("")
        lines.append("**Not an equal-budget comparison.** These arms missed the "
                     "budget they were given by more than "
                     f"{comparison.tolerance:.0%}, so the quality column is not "
                     "like-for-like:")
        for arm, unit, spend, median in comparison.unequal:
            lines.append(f"- `{arm}` spent {spend:.0f} {unit} against a median "
                         f"of {median:.0f}")

    if comparison.confounded:
        lines.append("")
        lines.append("**Confound.** Fixing one unit cannot fix the other: the "
                     "arms differ in how often a rollout has to call the model "
                     "at all. An arm that wins the quality column *while also* "
                     "spending more below has not been shown to win.")
        for arm, unit, spend, median in comparison.confounded:
            lines.append(f"- `{arm}` spent {spend:.0f} {unit} against a median "
                         f"of {median:.0f}")

    errored = [r for group in comparison.arms.values() for r in group if r.error]
    if errored:
        lines.append("")
        lines.append("Runs that ended on a failure, whose quality numbers are "
                     "from partial runs or missing entirely: "
                     + ", ".join(f"`{r.arm}` seed {r.seed}" for r in errored))

    fused = [r for group in comparison.arms.values() for r in group
             if r.fusion is not None and r.fusion.contested]
    if fused:
        lines.append("")
        lines.append("Fusion tournaments, which only the merge arm can hold — "
                     "serial has one proposal per step and fork never merges:")
        lines.append("")
        lines.append("| arm | contested | fused wins | mean gain | losses | "
                     "below baseline |")
        lines.append("|---|---|---|---|---|---|")
        for r in fused:
            f = r.fusion
            lines.append(
                f"| {r.arm} (seed {r.seed}) | {f.contested} | "
                f"{f.fused_wins} ({f.win_rate:.0%}) | {f.mean_gain:+.3f} | "
                f"{f.negative} (worst {f.worst_loss:+.3f}) | {f.below_baseline} |")

    if any(r.test_oracle is not None
           for group in comparison.arms.values() for r in group):
        lines.append("")
        lines.append("`fork oracle` is the best fork **on test** -- an upper "
                     "bound that requires the answer to select. The `test` "
                     "column for a fork arm is the fork chosen on dev, which is "
                     "what fork-and-select can actually ship.")
    return "\n".join(lines)
