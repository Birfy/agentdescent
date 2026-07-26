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
    # The staleness trade-off: at the same async_ratio, Guarded throws stale work
    # away while Reflective rebases and recovers it. The robust invariant is that
    # Guarded discards strictly more (Guarded may not fully converge in the time
    # bound precisely *because* it wastes that work).
    g = _run("guarded", async_ratio=4, seconds=12.0, seed=3)
    r = _run("reflective", async_ratio=4, seconds=12.0, seed=3)
    assert r.final_dev_accuracy >= 0.95        # Reflective converges efficiently
    assert g.final_dev_accuracy > 0.6          # Guarded still makes progress
    assert g.discarded_stale > r.discarded_stale
    assert r.rollouts < g.rollouts             # Reflective wastes far less work


def test_stable_branch_promotes_under_async():
    s = _run("full")
    # dev converged; the EMA stable branch should have caught up at least partway.
    assert s.final_stable_accuracy > 0.0
