"""The join: a store of audited pairs in, a correction the gate can apply out.

The test that matters is the last one -- a whole `evolve()` with a verifier
biased by a known amount, and a calibrator that has to recover it from nothing
but the store the run left behind. Everything above it is a guard on a way that
can fail quietly.

"Quietly" is the theme. Every failure mode here produces a number rather than an
error: a correction estimated on the wrong pool reads as *more* honest than the
truth, one estimated across a verifier change describes an instrument that no
longer exists, and a thin stratum silently dropped turns the question into "the
bias among units we sampled enough of". None of those raise.
"""

import math
import time
from difflib import SequenceMatcher

import numpy as np
import pytest

from agentdescent.audit import AuditRecord, AuditStore, Purpose
from agentdescent.audit.calibrator import (STALE_INFLATION, Calibrator,
                                           Rectification)


def _rec(f, y, *, stratum="all", version="v1", prob=1.0,
         purpose=Purpose.CALIBRATION, task="t", at=1000.0):
    import uuid
    return AuditRecord(
        record_id=uuid.uuid4().hex, task_id=task, artifact_signature="s",
        output="o", verifier_version=version, verifier_score=f,
        inclusion_prob=prob, purpose=purpose, stratum=stratum,
        oracle_score=y, dispatched_at=at, resolved_at=at + 1)


def _populated(*, n=200, bias=0.3, base=0.5, version="v1", n_unlab=600, seed=0):
    """A store carrying a verifier that is generous by a known amount.

    Directional: a wrong answer is forgiven with probability `bias`, a right one
    is never marked down, so `E[f - Y] = bias * (1 - base)` exactly.
    """
    rng = np.random.default_rng(seed)
    store = AuditStore()
    for i in range(n):
        y = 1.0 if rng.random() < base else 0.0
        f = 1.0 if (y == 0.0 and rng.random() < bias) else y
        store.append(_rec(f, y, version=version, task=f"t{i}"))
    for _ in range(n_unlab):
        y = 1.0 if rng.random() < base else 0.0
        f = 1.0 if (y == 0.0 and rng.random() < bias) else y
        store.observe_unlabelled(version, "all", f)
    return store, bias * (1 - base)


# -- the estimate ------------------------------------------------------------

def test_it_carries_the_spread_of_the_error_and_not_only_its_mean():
    """`resid_sd` is the term the acceptance gate is mostly made of.

    `delta_hat` is one number applied to both sides of a comparison, so it
    cancels; the *spread* of `f - Y` does not, and it is what
    :mod:`agentdescent.audit.gate` spends. Here the residual is Bernoulli at
    ``bias * (1 - base) = 0.15``, whose sd is ``sqrt(0.15 * 0.85) = 0.357`` --
    more than twice the bias itself.
    """
    sds = []
    for seed in range(40):
        store, truth = _populated(seed=seed)
        sds.append(Calibrator(store).current("v1").resid_sd)
    assert np.mean(sds) == pytest.approx(math.sqrt(0.15 * 0.85), abs=0.02)
    assert np.mean(sds) > 2 * 0.15, (
        "the spread is the larger fact, and the plan carries only the mean")


def test_a_stale_rectification_carries_the_spread_forward_too():
    """So a log can show what was withheld, on all the numbers rather than some."""
    store, _ = _populated()
    cal = Calibrator(store)
    fresh = cal.current("v1")
    cal.mark_stale("the judge was rewritten")
    held = cal.current("v1")
    assert held.is_stale
    assert held.resid_sd == pytest.approx(fresh.resid_sd)



def test_it_recovers_a_bias_that_was_put_there_on_purpose():
    """Averaged over draws, not asserted on one.

    A single draw sits about one standard error from the truth by definition, so
    a fixed tolerance on one draw is either loose enough to prove nothing or
    tight enough to fail on a seed. The first version of this test asserted
    `abs=0.05` against an se of 0.028 -- 1.8 sigma, so roughly one seed in
    fourteen would have failed it, and the failure would have looked like a bug
    in the estimator.
    """
    deltas, ses = [], []
    for seed in range(60):
        store, truth = _populated(seed=seed)
        r = Calibrator(store, seed=seed).current("v1")
        assert not r.is_stale
        deltas.append(r.delta_hat)
        ses.append(r.se)
    assert float(np.mean(deltas)) == pytest.approx(truth, abs=0.015)
    assert all(s > 0.0 and math.isfinite(s) for s in ses)


def test_one_draw_lands_within_its_own_interval():
    """The single-draw claim the estimator actually makes."""
    store, truth = _populated()
    r = Calibrator(store).current("v1")
    assert abs(r.delta_hat - truth) < 3.0 * r.se


def test_an_unbiased_verifier_gives_a_correction_of_about_zero():
    deltas = []
    for seed in range(40):
        store, truth = _populated(bias=0.0, seed=seed)
        assert truth == 0.0
        deltas.append(Calibrator(store, seed=seed).current("v1").delta_hat)
    assert abs(float(np.mean(deltas))) < 0.01


def test_the_unlabelled_half_is_what_buys_the_tighter_interval():
    """PPI's whole claim, measured on this path rather than asserted.

    Same labels, same everything, but one store never saw the units the sampler
    passed over. `gain_factor` is the multiplier the unlabelled half bought.
    """
    with_unlab, _ = _populated(n_unlab=3000)
    without, _ = _populated(n_unlab=0)
    a = Calibrator(with_unlab).current("v1")
    b = Calibrator(without).current("v1")
    assert a.gain_factor > b.gain_factor
    assert b.gain_factor == pytest.approx(1.0, abs=0.02)
    assert a.se < b.se


def test_delta_and_theta_share_a_standard_error_because_only_one_is_estimated():
    """`E[f]` is a count over units the tap saw, not a sample statistic."""
    store, _ = _populated()
    r = Calibrator(store).current("v1")
    assert r.delta_se == r.se == pytest.approx(
        0.5 * (r.theta_ci[1] - r.theta_ci[0]) / _crit(r), rel=0.02)


def _crit(r):
    from agentdescent.audit.ppi import t_ppf
    # df is not on Rectification; recover the critical value from the interval
    return 0.5 * (r.theta_ci[1] - r.theta_ci[0]) / r.se


# -- the three silent failures -----------------------------------------------

def test_it_never_reads_the_improvement_pool():
    """Labels used to edit the verifier were chosen to agree with it.

    Asserted against the *uncontaminated* answer rather than against the truth,
    so the claim is exact rather than within a tolerance: adding 400 flattering
    labels must not move the correction by a single bit.
    """
    store, _ = _populated()
    before = Calibrator(store).current("v1")
    for i in range(400):                       # a flood of flattering labels
        store.append(_rec(1.0, 1.0, purpose=Purpose.IMPROVEMENT, task=f"i{i}"))
    after = Calibrator(store).current("v1")
    assert after.delta_hat == before.delta_hat, (
        "the improvement pool moved the correction, so it was read")
    assert after.n == before.n == 200


def test_it_never_mixes_verifier_versions():
    """Exact, for the same reason: a second version must change nothing."""
    store, _ = _populated(version="v1")
    before = Calibrator(store).current("v1")
    other, _ = _populated(version="v2", bias=0.0, seed=9)
    for rec in other.all():
        store.append(rec)
    after = Calibrator(store).current("v1")
    assert after.delta_hat == before.delta_hat
    assert after.n == before.n == 200


def test_a_thin_stratum_is_merged_rather_than_dropped():
    """Dropping it would change the population, and read as a smaller bias.

    Here the thin layer is where the verifier is *worst*: three units it got
    entirely wrong. Drop them and the correction shrinks; merge them and it does
    not.
    """
    store = AuditStore()
    for i in range(120):
        store.append(_rec(1.0, 1.0, stratum="bulk", task=f"b{i}"))
    for i in range(3):
        store.append(_rec(1.0, 0.0, stratum="thin", task=f"h{i}"))
    for _ in range(300):
        store.observe_unlabelled("v1", "bulk", 1.0)
    for _ in range(200):
        store.observe_unlabelled("v1", "thin", 1.0)

    r = Calibrator(store, min_labels=10, min_per_stratum=5).current("v1")
    assert r.n == 123, "the thin stratum's labels must still be counted"
    assert r.n_unlab == 500, (
        "the thin stratum's unlabelled units were dropped -- merging the "
        "records without merging their moments is the same silent substitution "
        "the merge exists to prevent")
    assert r.delta_hat > 0.0


def test_merging_pools_the_moments_exactly():
    """Two layers, each internally uniform, at different means.

    Adding the variances would give zero; the pooled variance is not zero, and
    the difference is the term that makes an interval too narrow.
    """
    a = {"n": 50, "mean": 0.0, "var": 0.0}
    b = {"n": 50, "mean": 1.0, "var": 0.0}
    pooled = Calibrator._pool(a, b)
    direct = np.array([0.0] * 50 + [1.0] * 50)
    assert pooled["n"] == 100
    assert pooled["mean"] == pytest.approx(float(np.mean(direct)))
    assert pooled["var"] == pytest.approx(float(np.var(direct, ddof=1)))
    assert pooled["var"] > 0.0


# -- staleness ---------------------------------------------------------------

def test_too_few_labels_is_stale_rather_than_wide():
    """A very wide interval and "we do not know" are different claims."""
    store = AuditStore()
    for i in range(5):
        store.append(_rec(1.0, 0.0, task=f"t{i}"))
    r = Calibrator(store).current("v1")
    assert r.is_stale and "labels" in r.stale_reason
    assert math.isnan(r.delta_hat), "a stale correction must not carry a usable number"


def test_an_unknown_verifier_version_is_stale():
    store, _ = _populated()
    r = Calibrator(store).current("never-seen")
    assert r.is_stale


def test_mark_stale_withholds_the_correction_until_recompute():
    store, _ = _populated()
    cal = Calibrator(store)
    assert not cal.current("v1").is_stale

    fresh = cal.current("v1")
    cal.mark_stale("the verifier prompt changed")
    held = cal.current("v1")
    assert held.is_stale and held.stale_reason == "the verifier prompt changed"
    # the previous numbers are carried, so a log can say what was withheld
    assert held.delta_hat == fresh.delta_hat

    assert not cal.recompute("v1").is_stale
    assert cal.stale_reason is None


def test_a_stale_result_never_raises_because_a_merge_has_to_be_decided():
    """Every failure mode is a value, not an exception."""
    cal = Calibrator(AuditStore())
    for version in ("", "v1", "whatever"):
        r = cal.current(version)
        assert isinstance(r, Rectification) and r.is_stale


def test_the_stale_inflation_widens_rather_than_corrects():
    """A stale rectifier has no number to correct with, so the gate widens."""
    assert STALE_INFLATION > 1.0


# -- end to end --------------------------------------------------------------

def test_a_whole_run_produces_a_store_the_calibrator_can_read():
    """The claim, end to end: inject a known bias, recover it from the run.

    Nothing is handed to the calibrator except the store `evolve()` left behind
    -- the audited pairs and the running moments of everything the sampler passed
    over. If the tap, the store and the estimator disagree about anything, this
    is where it shows.
    """
    from agentdescent import AuditedReward, GoldAnswer, Task, evolve

    # Half the tasks ask for something the agent's two rules cannot produce, so
    # the run converges to *partly* right rather than perfect. A workload the
    # agent solves completely leaves `f` and `Y` both saturated at 1.0, which has
    # no variance and therefore no bias to estimate -- see
    # `test_a_converged_run_is_stale_rather_than_a_fabricated_zero`.
    tasks = [Task(id=f"t{i}",
                  prompt=f"  MiXeD Case {i}!  ",
                  meta={"expected": f"mixed case {i}!" + ("" if i % 2 else "?")})
             for i in range(90)]

    def truth(task, output):
        return 1.0 if output.strip().lower() == task.meta["expected"] else 0.0

    def generous(task, output):
        """Scores up and never down: full credit for anything close."""
        if truth(task, output) == 1.0:
            return 1.0
        ratio = SequenceMatcher(None, output.strip().lower(),
                                task.meta["expected"]).ratio()
        return 1.0 if ratio > 0.6 else 0.0

    class Agent:
        def solve(self, text, task):
            out = task.prompt
            if "lowercase" in text:
                out = out.lower()
            if "Strip" in text:
                out = out.strip()
            return out

        def propose(self, text, task, output, reward):
            if "lowercase" not in text:
                return "Convert the text to lowercase."
            if "Strip" not in text:
                return "Strip surrounding whitespace."
            return None

    store = AuditStore()
    audited = AuditedReward(generous, oracle=GoldAnswer(truth), store=store,
                            sample_rate=0.5, seed=3)
    evolve(tasks, audited, agent=Agent(), rounds=8, n_workers=3, seed=3)

    assert audited.audited > 0 and audited.seen > audited.audited, (
        "the run must produce both halves for this to test anything")
    moments = store.unlabelled_moments(audited.verifier_version)
    assert moments and sum(m["n"] for m in moments.values()) > 0

    r = Calibrator(store, min_labels=10).current(audited.verifier_version)
    assert not r.is_stale, r.stale_reason
    assert r.n == len(store.for_calibration(audited.verifier_version))
    assert r.n_unlab == sum(m["n"] for m in moments.values())
    # the verifier only ever scores up, so the correction has a known sign
    assert r.delta_hat > 0.0
    assert 0.0 <= r.theta <= 1.0
    assert r.gain_factor >= 1.0
    # and it must land near the residual the labels themselves show
    labelled = store.for_calibration(audited.verifier_version)
    naive = float(np.mean([x.verifier_score - x.oracle_score for x in labelled]))
    assert abs(r.delta_hat - naive) < 4.0 * r.se


def test_a_converged_run_is_stale_rather_than_a_fabricated_zero():
    """When the agent solves everything, there is no bias to estimate.

    Both scorers saturate at 1.0, every variance is zero, and there is nothing to
    put an interval around. The estimator refuses and the calibrator reports
    stale -- which is the right answer, and is *not* the same as a correction of
    zero. A converged run has no evidence about the verifier either way.

    Found by writing the end-to-end test above on a workload the agent could
    solve completely.
    """
    store = AuditStore()
    for i in range(80):
        store.append(_rec(1.0, 1.0, task=f"t{i}"))
    for _ in range(200):
        store.observe_unlabelled("v1", "all", 1.0)
    r = Calibrator(store, min_labels=10).current("v1")
    assert r.is_stale and "variance" in r.stale_reason
    assert math.isnan(r.delta_hat)
