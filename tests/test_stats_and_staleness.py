from agentdescent.evolvable import vv_dominates, vv_staleness
from agentdescent.stats import (
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


def test_acceptance_mc_seed_varies_per_candidate_but_stays_reproducible():
    """A shared MC seed decided a round's marginal candidates as a block.

    prob_improvement is a Monte-Carlo estimate (sd ~0.003 at 4000 samples). Seeding
    it from the artifact version alone gave every candidate in the round the *same*
    stream, so on knife-edge cases one draw accepted them all and another rejected
    them all. Seeding from (version, diff_id) decorrelates the draws while keeping
    the run reproducible.
    """
    from agentdescent.evolvable import stable_hash
    from agentdescent.stats import BetaPosterior, prob_improvement

    def knife():
        c, b = BetaPosterior(), BetaPosterior()
        for _ in range(6):
            c.update(True)
        for _ in range(4):
            b.update(True)
        for _ in range(3):
            b.update(False)
        return c, b

    c, b = knife()
    seeds = [stable_hash((64, f"w{i}:rule")) & 0x7FFFFFFF for i in range(5)]
    ests = [prob_improvement(c, b, seed=s) for s in seeds]
    assert len(set(ests)) > 1, "per-candidate seeds must give independent draws"

    # ...and the same candidate always gets the same estimate
    again = [prob_improvement(c, b, seed=s) for s in seeds]
    assert ests == again


def test_acceptance_seeds_its_draw_from_the_candidate():
    """Two candidates in one round must not share a Monte-Carlo draw.

    Seeding from the version alone made the ~0.003 sampling error identical for
    every candidate in a round, so on knife-edge cases the draw decided them as a
    block: one stream accepted every marginal diff, another rejected all of them.

    Asserted behaviourally rather than by reading the source for `stable_hash`:
    the seeding used to live in `Aggregator._process` and now lives in
    `DefaultAcceptance`, and a test that greps one method goes green the moment
    the logic moves -- whether or not it survived the move.
    """
    from agentdescent.defaults import DefaultAcceptance
    from agentdescent.evolvable import Diff
    from agentdescent.policies import MergeContext

    class _Art:
        version = 3
        blast_radius = 0.2

    policy = DefaultAcceptance(base_delta=0.5, anneal_half_life=64,
                               accept_samples=400)

    def draw(diff_id):
        ctx = MergeContext(artifact=_Art(), candidate=_Art(), cards=[],
                           base_counts=(5.0, 5.0), cand_counts=(6.0, 4.0),
                           diff=Diff(diff_id=diff_id, target="a", ops={"k": "v"}))
        return policy.accept(ctx).p_improve

    assert draw("cand-A") == draw("cand-A"), "the same candidate must be reproducible"
    assert draw("cand-A") != draw("cand-B"), (
        "two candidates in one round shared a draw -- the seed is not "
        "candidate-specific")


# --- input validation (#305) -------------------------------------------------
#
# ``random.gammavariate(nan, 1.0)`` never returns, so on a build without the
# validation these cases hang rather than fail.  Every test that would reach the
# sampler with a non-finite posterior therefore runs in a child process with a
# hard timeout: the regression fails in bounded time instead of wedging the suite.

import subprocess
import sys
import textwrap

import pytest


def _run_guarded(code: str, timeout: float = 15.0) -> str:
    """Run ``code`` in a child interpreter; fail (not hang) past ``timeout``."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(code)],
            capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:  # pragma: no cover - only on a regression
        pytest.fail(f"child did not finish within {timeout}s (sampler hang?)")
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


@pytest.mark.parametrize("bad", ["float('nan')", "float('inf')", "float('-inf')"])
def test_prob_improvement_rejects_non_finite_posterior_without_hanging(bad):
    out = _run_guarded(f"""
        from agentdescent.stats import BetaPosterior, prob_improvement
        good = BetaPosterior()
        poisoned = BetaPosterior()
        poisoned.successes = {bad}          # bypasses the constructor check
        for cand, base in ((poisoned, good), (good, poisoned)):
            try:
                prob_improvement(cand, base, samples=50)
            except ValueError:
                print("ok", end=" ")
            else:
                print("no-error", end=" ")
    """)
    assert out == "ok ok"


def test_prob_improvement_rejects_non_positive_alpha():
    bad = BetaPosterior()
    bad.successes = -5.0            # alpha = -4: gammavariate would raise
    with pytest.raises(ValueError, match="alpha"):
        prob_improvement(bad, BetaPosterior(), samples=50)


@pytest.mark.parametrize("samples", [0, -3])
def test_prob_improvement_rejects_samples_below_one(samples):
    with pytest.raises(ValueError, match="samples"):
        prob_improvement(BetaPosterior(), BetaPosterior(), samples=samples)


def test_prob_improvement_seeded_output_unchanged_for_valid_input():
    strong = BetaPosterior(successes=30, failures=2)
    weak = BetaPosterior(successes=2, failures=30)
    assert prob_improvement(strong, weak, samples=1, seed=3) in (0.0, 1.0)
    assert prob_improvement(strong, weak, samples=200, seed=7) == \
        prob_improvement(strong, weak, samples=200, seed=7)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), -5.0])
def test_beta_posterior_refuses_bad_counts_at_construction(bad):
    with pytest.raises(ValueError):
        BetaPosterior(successes=bad)
    with pytest.raises(ValueError):
        BetaPosterior(failures=bad)


@pytest.mark.parametrize("weight", [float("nan"), float("inf"), -1.0])
def test_beta_posterior_update_refuses_bad_weight(weight):
    p = BetaPosterior(successes=2.0, failures=1.0)
    for success in (True, False):
        with pytest.raises(ValueError):
            p.update(success, weight=weight)
    assert (p.successes, p.failures) == (2.0, 1.0)


def test_observe_delta_ignores_nan_and_still_counts_real_deltas():
    p = BetaPosterior(successes=2.0, failures=1.0)
    p.observe_delta(float("nan"))
    assert (p.successes, p.failures) == (2.0, 1.0)
    p.observe_delta(0.5)
    assert p.successes > 2.0 and p.failures == 1.0
    p.observe_delta(float("-inf"))      # a real (if extreme) regression
    assert p.failures > 1.0


@pytest.mark.parametrize("samples", [0, -1])
def test_default_acceptance_refuses_accept_samples_below_one(samples):
    from agentdescent.aggregator import AggregatorConfig
    from agentdescent.defaults import DefaultAcceptance

    with pytest.raises(ValueError, match="accept_samples"):
        DefaultAcceptance(accept_samples=samples)
    with pytest.raises(ValueError, match="accept_samples"):
        DefaultAcceptance().configure(AggregatorConfig(accept_samples=samples))
    DefaultAcceptance().configure(AggregatorConfig(accept_samples=1))  # boundary ok
