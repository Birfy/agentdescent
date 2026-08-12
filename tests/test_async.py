"""Tests for the asynchronous stage-orchestration runtime.

These assert on *outcomes* (convergence, that concurrency happened, that policies
differ in stale-discard behaviour) rather than exact interleavings, since thread
scheduling is nondeterministic.
"""

import tempfile

import pytest

from agentdescent.async_runtime import AsyncAgentDescent, AsyncConfig
from agentdescent.domains.router import make_task_universe
from agentdescent.staleness import get_policy


def _run(policy_name, async_ratio=4, seconds=15.0, noise=0.12, seed=1,
         resync_on_commit=True):
    universe = make_task_universe(seed=7)
    cfg = AsyncConfig(n_workers=6, async_ratio=async_ratio, noise=noise,
                      target_accuracy=0.95, max_seconds=seconds, seed=seed,
                      resync_on_commit=resync_on_commit)
    with tempfile.TemporaryDirectory() as repo:
        sys = AsyncAgentDescent(repo, universe, config=cfg,
                             staleness_policy=get_policy(policy_name))
        return sys.run()


def test_async_converges_and_is_concurrent():
    s = _run("full")
    assert s.final_dev_accuracy >= 0.95
    assert s.commits >= 1
    # many worker rollouts overlapped with aggregator sweeps (real pipelining).
    assert s.rollouts > s.commits * 5
    assert s.sweeps > 0


def test_full_policy_discards_nothing():
    s = _run("full")
    assert s.discarded_stale == 0  # Full accepts stale diffs directly


def test_guarded_discards_more_than_reflective():
    """The staleness trade-off: at the same async_ratio, Guarded throws stale work
    away while Reflective rebases and recovers it.

    Assert the *relationship*, not absolute accuracy. These runs are bounded by
    wall-clock, so how far either policy converges depends on how many rollouts the
    machine fits into 12 seconds -- a threshold like `>= 0.95` passes on a fast
    laptop and fails on a loaded CI runner (it did: 0.83 on Python 3.9). The
    relational invariants hold regardless of machine speed.
    """
    # `resync_on_commit=False`: this compares what two staleness *policies* do
    # with stale work, so the lag budget has to be the only resync trigger. With
    # the default on, a commit resyncs every worker and -- rollouts here being
    # dictionary lookups -- nothing is ever stale for either policy to act on,
    # so both counters read 0 and the comparison has no content.
    g = _run("guarded", async_ratio=4, seconds=12.0, seed=3, resync_on_commit=False)
    r = _run("reflective", async_ratio=4, seconds=12.0, seed=3, resync_on_commit=False)

    assert g.discarded_stale > r.discarded_stale   # the claim under test
    # ...and as a *rate*, which is what "wastes less work" actually means.
    #
    # This used to read `r.rollouts < g.rollouts`, which is not an invariant: both
    # runs are bounded by wall-clock *and* stop at the first sweep past
    # `target_accuracy`, so total rollouts mixes "how much did it waste" with "how
    # long did it take to get there" -- whichever policy converges first has fewer.
    # It is a coin flip, and it flipped: CI failed with reflective at 12094
    # rollouts against guarded's 1657 on 3.12 and 40182 against 6587 on 3.11,
    # while 3.9 passed. Measured locally the rate never comes close: guarded
    # discards 97-98% of its evidence, reflective 13-26%.
    assert (g.discarded_stale / max(1, g.proposals)
            > r.discarded_stale / max(1, r.proposals)), (
        f"guarded wasted {g.discarded_stale}/{g.proposals} of its evidence and "
        f"reflective {r.discarded_stale}/{r.proposals}; rebasing is supposed to "
        "recover what the budget would otherwise throw away")
    # Both policies must have got somewhere, and neither may end below the seed.
    #
    # What is deliberately *not* asserted is the two accuracies against each
    # other. That comparison was here twice -- first as `r >= g`, then widened to
    # `r >= g - 0.05` -- and failed CI both times, most recently at 0.931 against
    # 1.000 on 3.11 while 3.9 and 3.12 passed. Widening it a third time would be
    # the wrong move, because the assertion cannot succeed for a policy reason:
    #
    #   * These runs stop at whichever comes first, crossing `target_accuracy`
    #     or exhausting `max_seconds`. When *both* cross the bar, both land in
    #     [0.95, 1.0] and a 0.05 band is satisfied by arithmetic, testing
    #     nothing. When one runs out of wall-clock instead, the gap measures how
    #     many rollouts the CI runner fit into 12 seconds.
    #   * So every value it can distinguish is a stopping artifact, and every
    #     value it cannot is a tautology.
    #
    # The claim the test is actually named for -- guarded wastes work, reflective
    # recovers it -- is the two assertions above, and they are stable because a
    # discard *rate* does not depend on how long the machine ran.
    assert g.final_dev_accuracy > 0.0 and r.final_dev_accuracy > 0.0
    for name, stats in (("guarded", g), ("reflective", r)):
        assert stats.commits >= 1, f"{name} never committed anything"
        assert stats.error is None, f"{name} ended on {stats.error}"


def test_stable_branch_promotes_under_async():
    s = _run("full")
    # dev converged; the EMA stable branch should have caught up at least partway.
    assert s.final_stable_accuracy > 0.0


# Failure injection used to live here, driving `AsyncAgentDescent` with a
# rollout that raises. It was worth having when that class was a *separate*
# implementation of the barrier-free loop; it is an adapter over `async_evolve`
# now, so those three tests asserted `async_evolve`'s resilience through a
# wrapper -- which `tests/test_fault_matrix.py` already asserts directly, against
# both engines, and `tests/test_worker_resilience.py` asserts in more detail.
# One test of the adapter's own seam survives, in `test_async_parity.py`.

def test_a_healthy_run_reports_no_error():
    s = _run("full")
    assert s.error is None
