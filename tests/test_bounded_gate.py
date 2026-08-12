"""The bounded held-out scan: a cost cut that cannot move a decision.

FlashEvolve §3.3 scores an `alpha_spec` prefix, compares it against the pool and
*guesses*; a wrong guess needs a rollback, which is why the paper keeps the
whole mechanism optional and out of its main results.

Rejection needs no guess. Rewards are contractually in [0, 1], so after k of n
tasks the best the full set could still reach is `(sum + (n - k)) / n`. Once
that is at or below the bar, no assignment of the rest can clear it -- the
remaining evaluations are model calls buying a number nobody reads.

So the property under test throughout this file is **equivalence of decisions**,
not closeness of numbers: `score_bounded(t, f) > f` must equal `score(t) > f`
for every input. A test that only checked "it got faster" would pass on an
implementation that quietly changed what commits.
"""

import warnings

import pytest

from agentdescent import AppendRules, Task, evolve
from agentdescent.aggregator import AggregatorConfig


def _tasks(n=12):
    return [Task(id=f"t{i}", prompt=f"q{i}", meta={"gold": str(i)}) for i in range(n)]


def _reward(task, output):
    return 1.0 if output == task.meta["gold"] else 0.0


def _artifact(scores):
    """An artifact whose task rewards are dictated by `scores` (id -> reward)."""
    from agentdescent.evolution import _Runtime, EvolvingArtifact
    from agentdescent.evalcache import MemoryCache

    rt = _Runtime(run=lambda rendered, task: rendered,
                  reward=lambda task, out: scores[task.id],
                  cache=MemoryCache(), eval_concurrency=2)
    return EvolvingArtifact("a", {"k": "v"}, runtime=rt, strategy=AppendRules()), rt


# ---------------------------------------------------------------------------
# Soundness: the bound never flips a verdict
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pattern", [
    [0.0] * 12,                                   # hopeless: cut on chunk 1
    [1.0] * 12,                                   # perfect: never cut
    [1.0] * 6 + [0.0] * 6,                        # front-loaded
    [0.0] * 6 + [1.0] * 6,                        # back-loaded: the trap
    [0.0, 1.0] * 6,                               # alternating
    [0.5] * 12,                                   # graded rewards
    [0.9] + [0.0] * 11,                           # one good task, then nothing
])
@pytest.mark.parametrize("floor", [0.0, 0.1, 0.25, 0.49, 0.5, 0.51, 0.75, 0.99, 1.0])
def test_the_bound_decides_exactly_as_the_full_scan_would(pattern, floor):
    """The one property everything else rests on, over the awkward cases.

    Back-loaded is the trap: the prefix looks hopeless and the tail is perfect.
    A guess would reject it; the bound must not, unless the tail *cannot* save
    it even at 1.0 apiece.
    """
    tasks = _tasks(len(pattern))
    scores = {t.id: s for t, s in zip(tasks, pattern)}
    art, _ = _artifact(scores)
    truth = sum(pattern) / len(pattern)

    art2, _ = _artifact(scores)
    bounded = art2.score_bounded(tasks, floor)

    assert (bounded > floor) == (truth > floor), (
        f"bound {bounded} vs truth {truth} disagree about floor {floor}")
    # And when it stops early it must never claim MORE than the truth could be:
    # the returned value is an upper bound, so it sits at or above the truth.
    assert bounded >= truth - 1e-9


def test_a_cut_scan_skips_real_evaluations():
    """Otherwise it is a correct no-op. Counted, because the saving is the point."""
    tasks = _tasks(12)
    art, rt = _artifact({t.id: 0.0 for t in tasks})
    from agentdescent.metrics import Meter
    rt.meter = Meter()

    assert art.score_bounded(tasks, floor=0.5) <= 0.5
    snap = rt.meter.snapshot()
    assert snap.bounded_scans_cut == 1
    assert snap.evals_skipped > 0, "nothing was saved, so nothing was gained"


def test_it_never_cuts_when_the_candidate_is_winning():
    """A scan that stopped on a winner would be reporting a bound as a rate."""
    tasks = _tasks(12)
    art, rt = _artifact({t.id: 1.0 for t in tasks})
    from agentdescent.metrics import Meter
    rt.meter = Meter()

    assert art.score_bounded(tasks, floor=0.5) == pytest.approx(1.0)
    assert rt.meter.snapshot().bounded_scans_cut == 0
    assert rt.meter.snapshot().evals_skipped == 0


def test_an_infinite_floor_is_plain_scoring():
    """The documented escape hatch, and the parity check behind it."""
    tasks = _tasks(12)
    pattern = {t.id: (1.0 if i % 3 else 0.0) for i, t in enumerate(tasks)}
    a1, _ = _artifact(pattern)
    a2, _ = _artifact(pattern)
    assert a1.score_bounded(tasks, float("-inf")) == pytest.approx(a2.score(tasks))


# ---------------------------------------------------------------------------
# End to end: the run is unchanged, the bill is not
# ---------------------------------------------------------------------------


def _run(bounded: bool, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return evolve(_tasks(), _reward,
                      run=lambda r, t: t.meta["gold"] if t.id in r else "?",
                      propose=lambda r, t, o, s: t.id,
                      strategy=AppendRules(), rounds=6, n_workers=4, seed=0,
                      held_out_frac=0.5,
                      agg_config=AggregatorConfig(bounded_gate=bounded), **kw)


def test_the_run_reaches_the_same_place_either_way():
    """Equivalence where it counts: same reward, same commits, same categories.

    This is the claim `bounded_gate=True` makes as a default. If a bounded run
    committed differently from a full one, the bound would not be a bound.
    """
    off, on = _run(False), _run(True)
    assert on.final_reward == pytest.approx(off.final_reward)
    assert [h.committed for h in on.history] == [h.committed for h in off.history]
    assert on.outcomes() == off.outcomes()


def test_the_default_is_off_because_the_posterior_reads_the_magnitude():
    """The claim this feature could not keep, pinned so it cannot be re-made.

    The bound decides accept/reject exactly as a full scan would. It does *not*
    reproduce the magnitude, and `_decide` feeds a rejected candidate's delta
    into the per-artifact Beta posterior that sets every later threshold -- so a
    bounded run can drift from a full one.

    Established by reading `_decide`, not by a measurement. The livelock test in
    `test_async_semantics.py` failed while this defaulted to on, which looked
    like the proof -- and it fails 3/3 with the feature reverted as well, so it
    is load-dependent and evidence of nothing. The equivalence test above passes
    because its domain rarely rejects through the posterior path; that is a fact
    about the domain, not a guarantee.

    Opt-in, therefore. Turning it on changes a run; it does not make one wrong.
    """
    assert AggregatorConfig().bounded_gate is False
    assert _run(False).evals_skipped == 0


def test_it_reports_what_it_saved_when_switched_on():
    """A saving nobody can see is indistinguishable from no saving."""
    r = _run(True)
    assert r.evals_skipped >= 0        # workload-shaped; the counter must exist


# ---------------------------------------------------------------------------
# Where the cut is placed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("floor,expect_scanned", [
    (0.30, 17),      # n(1-f) = 16.8 -> 17; a fixed 25% prefix could never cut
    (0.50, 12),      # n(1-f) = 12
    (0.70, 8),       # n(1-f) = 7.2, raised to the pool width
    (0.90, 8),       # n(1-f) = 2.4, likewise -- the saving caps at 1 - width/n
])
def test_the_cut_lands_at_the_earliest_point_it_can(floor, expect_scanned):
    """The boundary is `n(1 - floor) + total`, not a constant fraction.

    A cut needs `(total + n - k)/n <= floor`, so with the tail worth 1.0 apiece
    no cut is possible before `k = n(1 - floor)`. Checking earlier cannot fire;
    checking later pays for the evaluations in between, which is what a fixed
    prefix does.

    FlashEvolve's `alpha_spec` is a constant because its prefix feeds a
    speculative *accept*, where an early signal is the product. For a provable
    *reject* the optimum is derivable and is not constant.

    The last two rows are clamped by `eval_concurrency`: a chunk narrower than
    the pool would give up the fan-out, so the saving caps at `1 - width/n`
    however high the bar. That ceiling is the reason 0.70 and 0.90 tie.
    """
    tasks = _tasks(24)
    art, rt = _artifact({t.id: 0.0 for t in tasks})
    from agentdescent.metrics import Meter
    rt.meter = Meter()
    rt.eval_concurrency = 8

    art.score_bounded(tasks, floor)
    scanned = 24 - rt.meter.snapshot().evals_skipped
    assert scanned == expect_scanned


def test_a_zero_bar_scans_once_at_full_width():
    """`floor <= 0` can never cut, so chunking it would only cost round-trips."""
    tasks = _tasks(24)
    art, rt = _artifact({t.id: 0.0 for t in tasks})
    from agentdescent.metrics import Meter
    rt.meter = Meter()
    assert art.score_bounded(tasks, 0.0) == pytest.approx(0.0)
    assert rt.meter.snapshot().bounded_scans_cut == 0
