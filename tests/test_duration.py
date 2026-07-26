"""Tests for duration estimation and duration-aware dispatch."""

import random

from agentdescent.scheduler import DurationEstimator, fifo_makespan, lpt_schedule


def test_estimator_returns_prior_then_mean_before_calibration():
    est = DurationEstimator(prior=0.05, min_samples=3)
    assert est.estimate(100) == 0.05          # no data -> prior
    est.observe(10, 1.0)
    est.observe(20, 3.0)
    assert est.estimate(999) == 2.0           # <min_samples -> running mean


def test_estimator_recovers_linear_law():
    # exact data: seconds = 1 + 2*cost  -> should recover intercept=1, slope=2.
    est = DurationEstimator()
    for cost in range(1, 21):
        est.observe(cost, 1.0 + 2.0 * cost)
    b, m = est.params
    assert abs(b - 1.0) < 1e-6 and abs(m - 2.0) < 1e-6
    assert abs(est.estimate(100) - 201.0) < 1e-6


def test_lpt_beats_or_matches_round_robin_and_is_near_optimal():
    rng = random.Random(0)
    weights = [rng.lognormvariate(2.0, 1.0) for _ in range(40)]  # heavy-tailed
    n = 4
    assign, lpt = lpt_schedule(weights, n)
    fifo = fifo_makespan(weights, n)
    lower_bound = sum(weights) / n

    assert len(assign) == len(weights)
    assert set(assign) <= set(range(n))
    assert lpt <= fifo + 1e-9                  # LPT never worse than round-robin
    assert lpt >= lower_bound - 1e-9           # can't beat the optimal lower bound
    assert lpt <= (4 / 3) * lower_bound + 1e-9  # LPT's worst-case guarantee


def test_lpt_assignment_load_matches_makespan():
    weights = [5.0, 4.0, 3.0, 2.0, 1.0]
    assign, makespan = lpt_schedule(weights, 2)
    loads = [0.0, 0.0]
    for w, j in zip(weights, assign):
        loads[j] += w
    assert max(loads) == makespan
