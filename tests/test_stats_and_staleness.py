from concordia.evolvable import vv_dominates, vv_staleness
from concordia.stats import (
    BetaPosterior,
    annealed_delta,
    prob_improvement,
    ucb_score,
)


def test_version_vector_helpers():
    assert vv_staleness({"a": 5}, {"a": 2}) == 3
    assert vv_staleness({"a": 5, "b": 1}, {"a": 5, "b": 0}) == 1
    assert vv_staleness({"a": 5}, {}) == 0
    assert vv_dominates({"a": 5, "b": 2}, {"a": 5, "b": 1})
    assert not vv_dominates({"a": 4}, {"a": 5})


def test_prob_improvement_orders_correctly():
    strong = BetaPosterior(successes=30, failures=2)
    weak = BetaPosterior(successes=2, failures=30)
    assert prob_improvement(strong, weak) > 0.95
    assert prob_improvement(weak, strong) < 0.05


def test_evidence_starved_is_conservative():
    # a tail artifact with a single positive observation should NOT clear a
    # high-confidence bar, unlike a well-supported one.
    baseline = BetaPosterior(successes=5, failures=5)
    thin = BetaPosterior(successes=6, failures=5)      # +1 success
    thick = BetaPosterior(successes=25, failures=5)    # many successes
    assert prob_improvement(thin, baseline) < prob_improvement(thick, baseline)


def test_annealed_delta_shrinks_with_version():
    assert annealed_delta(0.2, 0) > annealed_delta(0.2, 128)
    assert annealed_delta(0.2, 1000) >= 0.01  # floored


def test_ucb_prioritizes_unexplored():
    explored = ucb_score(value=0.5, n_s=100, total=1000)
    unexplored = ucb_score(value=0.5, n_s=1, total=1000)
    assert unexplored > explored
    assert ucb_score(0.5, 0, 1000) == float("inf")
