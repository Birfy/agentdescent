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


def _run(policy_name, async_ratio=4, seconds=15.0, noise=0.12, seed=1):
    universe = make_task_universe(seed=7)
    cfg = AsyncConfig(n_workers=6, async_ratio=async_ratio, noise=noise,
                      target_accuracy=0.95, max_seconds=seconds, seed=seed)
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
    g = _run("guarded", async_ratio=4, seconds=12.0, seed=3)
    r = _run("reflective", async_ratio=4, seconds=12.0, seed=3)

    assert g.discarded_stale > r.discarded_stale   # the claim under test
    assert r.rollouts < g.rollouts                 # Reflective wastes far less work
    # Recovering that work cannot leave Reflective *materially* behind, and both
    # must progress. Not `r >= g`: both runs stop at the first sweep that crosses
    # `target_accuracy` (0.95 here), so the value each one *lands* on is a
    # stopping artifact rather than a measure of the policy. Guarded takes
    # hundreds of small sweeps and can land exactly on 1.000; Reflective takes
    # ~8 large ones and lands wherever its batch put it (0.966 is common). That
    # made the strict comparison fail roughly one run in six, on an assertion the
    # test is not about. Both stopped past the same bar, so the widest legitimate
    # gap is the width of the band above it.
    assert r.final_dev_accuracy >= g.final_dev_accuracy - 0.05
    assert g.final_dev_accuracy > 0.0


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
