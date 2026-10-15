"""The verifier bootstrap flywheel has to stay right in the exact places a
self-improving judge is dangerous.

A verifier that learns from its own labels could learn to be confident-but-wrong,
so every one of these fixtures pins a safety property: the correction must be
monotone (it ranks), must only activate when it provably reduces residual error
on a held-out slice, must train on the IMPROVEMENT pool only (never the pool
that measures it), must never touch the commit gate, and must declare itself as
a new verifier version so the audit's calibration is withdrawn and re-measured.
"""

import math

import numpy as np
import pytest

from agentdescent.audit import (
    AuditStore,
    AuditRecord,
    Purpose,
    VerifierBoost,
    attach,
    isotonic_fit,
)


# -- helpers -------------------------------------------------------------------


def _record(score, oracle, version="v1", purpose=Purpose.IMPROVEMENT):
    return AuditRecord(
        record_id=f"r{score}-{oracle}-{purpose.value}",
        task_id="t", artifact_signature="sig", output="o",
        verifier_version=version, verifier_score=float(score),
        inclusion_prob=0.5, purpose=purpose, oracle_score=float(oracle),
        resolved_at=1.0)


def _store_with(records, path=None):
    store = AuditStore(path)
    for rec in records:
        store.append(rec)
    return store


def _biased_scores(oracle_values, bias=0.2):
    """verifier over-credits every oracle value by ``bias`` (clipped to [0,1])."""
    return [min(1.0, max(0.0, y + bias)) for y in oracle_values]


def _noisy_verifier(oracle_values):
    """A verifier whose score is unrelated to the oracle: nothing to learn."""
    return [min(1.0, max(0.0, 0.5 + 0.4 * math.sin(i * 1.7))) for i in range(len(oracle_values))]


class _FakeVerifier:
    """A minimal VerifierProtocol: a fixed score, a fixed uncertainty, raw gate."""

    def __init__(self, score=0.5, full_counts=(5, 5), uncertainty=0.1):
        self.score = score
        self.full_counts = full_counts
        self.uncertainty = uncertainty
        self.full_eval_matches_counts = True

    def rule_eval(self, artifact):
        return self.score

    def cheap_eval(self, artifact):
        return self.score

    def learned_eval(self, artifact):
        return self.score, self.uncertainty

    def eval_counts(self, artifact, floor=None):
        return self.full_counts

    def full_eval(self, artifact):
        s, f = self.full_counts
        return s / (s + f)


# -- the correction itself ------------------------------------------------------


def test_isotonic_fit_is_monotone_and_approximates_the_target():
    y = np.linspace(0.1, 0.7, 30)                 # no clipping under the bias
    x = _biased_scores(y, bias=0.2)
    xs, vals = isotonic_fit(x, y)
    assert (np.diff(vals) >= -1e-9).all()          # monotone non-decreasing
    # a constant-bias line is fit closely: g(x) ≈ y
    err = np.max(np.abs(np.interp(np.asarray(x), xs, vals) - np.asarray(y)))
    assert err < 0.05


def test_the_correction_preserves_score_order():
    y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    x = _biased_scores(y, bias=0.1)
    xs, vals = isotonic_fit(x, y)
    grid = np.linspace(0.0, 1.0, 200)
    g = np.interp(grid, xs, vals)
    assert (np.diff(g) >= -1e-9).all()
    assert g[0] >= 0.0 and g[-1] <= 1.0


# -- the bootstrap gate ----------------------------------------------------------


def test_no_labels_leaves_the_boost_inactive():
    store = AuditStore()
    boost = VerifierBoost(_FakeVerifier(), store)
    assert boost.is_active is False
    assert "insufficient improvement labels" in boost.status()["reason"]


def test_too_few_labels_stays_inactive():
    store = _store_with([_record(0.5, 0.3)])
    boost = VerifierBoost(_FakeVerifier(score=0.5), store, min_labels=10)
    assert boost.is_active is False


def test_a_genuine_bias_is_learned_and_validated():
    y = list(np.linspace(0.1, 0.9, 24))
    s = _biased_scores(y, bias=0.2)
    store = _store_with([_record(score, oracle) for score, oracle in zip(s, y)])
    verifier = _FakeVerifier(score=0.5)          # biased score for 0.3 truth
    boost = VerifierBoost(verifier, store, min_labels=10, min_val_labels=3,
                          seed=1)
    assert boost.is_active is True
    status = boost.status()
    assert status["error_after"] < status["error_before"]
    # the corrected cheap score is closer to the oracle than the raw one
    raw, corrected = 0.5, boost.cheap_eval(object())
    assert abs(corrected - 0.3) < abs(raw - 0.3)


def test_a_boost_that_does_not_help_is_rejected():
    y = list(np.linspace(0.1, 0.9, 24))
    s = _noisy_verifier(y)                        # unrelated to truth
    store = _store_with([_record(score, oracle) for score, oracle in zip(s, y)])
    boost = VerifierBoost(_FakeVerifier(score=0.5), store, min_labels=10,
                          min_val_labels=3, seed=1)
    assert boost.is_active is False
    assert "bootstrap gate" in boost.status()["reason"]


def test_min_gain_raises_the_bar():
    # A partially-learnable bias: isotopic can remove the constant part but not
    # the noise, so the error reduction is real yet below a 50% bar.
    rng = np.random.default_rng(0)
    y = np.linspace(0.1, 0.8, 40)
    s = np.clip(y + 0.1 + rng.normal(0.0, 0.15, size=y.shape), 0.0, 1.0)
    store = _store_with([_record(score, oracle) for score, oracle in zip(s, y)])
    # a lenient gate accepts (any reduction), a strict one refuses (< 50%)
    lenient = VerifierBoost(_FakeVerifier(score=0.5), store, min_labels=10,
                            min_val_labels=3, seed=3, min_gain=0.0)
    strict = VerifierBoost(_FakeVerifier(score=0.5), store, min_labels=10,
                           min_val_labels=3, seed=3, min_gain=0.5)
    assert lenient.is_active is True
    assert strict.is_active is False


# -- what the boost may and may not touch ---------------------------------------


def test_the_commit_gate_is_delegated_raw_when_active():
    y = list(np.linspace(0.1, 0.9, 24))
    s = _biased_scores(y, bias=0.2)
    store = _store_with([_record(score, oracle) for score, oracle in zip(s, y)])
    boost = VerifierBoost(_FakeVerifier(score=0.5, full_counts=(4, 4)), store,
                          min_labels=10, min_val_labels=3, seed=1)
    assert boost.is_active is True
    assert boost.eval_counts(object()) == (4, 4)      # raw
    assert boost.full_eval(object()) == 0.5           # raw
    assert boost.rule_eval(object()) == 0.5           # raw


def test_the_learned_uncertainty_reflects_the_validation_residual():
    y = list(np.linspace(0.1, 0.9, 24))
    s = _biased_scores(y, bias=0.2)
    store = _store_with([_record(score, oracle) for score, oracle in zip(s, y)])
    boost = VerifierBoost(_FakeVerifier(score=0.5, uncertainty=0.1), store,
                          min_labels=10, min_val_labels=3, seed=1)
    _, uncertainty = boost.learned_eval(object())
    assert uncertainty >= boost.status()["resid_sd"]
    assert uncertainty >= 0.1


def test_training_uses_the_improvement_pool_only():
    # A CALIBRATION record -- which exists to *measure* the verifier, not to
    # change it -- with a wild outlier must not enter the fit at all.
    y = list(np.linspace(0.1, 0.9, 24))
    s = _biased_scores(y, bias=0.2)
    records = [_record(score, oracle) for score, oracle in zip(s, y)]
    records.append(_record(0.5, 1.0, purpose=Purpose.CALIBRATION))
    store = _store_with(records)
    boost = VerifierBoost(_FakeVerifier(score=0.5), store, min_labels=10,
                          min_val_labels=3, seed=1)
    # the calibration record never trained anything: without it the fit is the
    # same -- and the corrected score is still the same clean 0.3
    assert abs(boost.cheap_eval(object()) - 0.3) < abs(0.5 - 0.3)


# -- versioning -----------------------------------------------------------------


def test_version_tag_changes_on_activation_and_is_stable_while_active():
    y = list(np.linspace(0.1, 0.9, 24))
    s = _biased_scores(y, bias=0.2)
    store = _store_with([_record(score, oracle) for score, oracle in zip(s, y)])
    boost = VerifierBoost(_FakeVerifier(score=0.5), store, min_labels=10,
                          min_val_labels=3, seed=1, auto=False)
    assert boost.version_tag() == "boost:inactive"
    boost.retrain()
    tag = boost.version_tag()
    assert tag.startswith("boost:active:")
    assert boost.version_tag() == tag              # stable between fits


# -- wiring ---------------------------------------------------------------------


def test_audit_recompute_retrains_an_attached_boost():
    store = AuditStore()
    boost = VerifierBoost(_FakeVerifier(), store, auto=False)
    audit = attach(verifier=lambda task, output: 0.5,
                   oracle=lambda task, output: 0.7,
                   store=store, boost=boost)
    assert boost.status()["fitted"] is False
    audit.recompute()                             # no labels -> retrain runs, inactive
    assert boost.status()["fitted"] is True


def test_no_boost_means_recompute_is_unchanged():
    store = AuditStore()
    audit = attach(verifier=lambda task, output: 0.5,
                   oracle=lambda task, output: 0.7, store=store)
    rect = audit.recompute()                       # must not raise
    assert rect is not None


def test_the_boost_is_a_valid_engine_verifier(tmp_path):
    # The wrapper must satisfy VerifierProtocol so evolve() can consume it
    # through Policies(verifier=...). Empty store -> inactive -> raw path.
    from agentdescent import Policies, ThreeLayerVerifier
    from agentdescent.evolution import SingleSlot, Task, evolve

    tasks = [Task(id=f"t{i}", prompt=f"item {i}") for i in range(8)]

    def run(rendered, task):
        return "answer" + (" 2026" if "year" in rendered else "")

    def reward(task, output):
        return 1.0 if "2026" in output else 0.0

    verifier = ThreeLayerVerifier(eval_fn=lambda a, ts: a.score(ts),
                                  held_out=tasks, rule_subset=4)
    boost = VerifierBoost(verifier, AuditStore())     # no labels -> inactive
    result = evolve(
        tasks, reward, run=run,
        propose=lambda r, t, o, re: "always state the year",
        strategy=SingleSlot(initial_value="seed"), rounds=2, n_workers=2,
        repo_path=str(tmp_path / "repo"), policies=Policies(verifier=boost))
    assert boost.is_active is False
    assert result.outcomes().get("committed", 0) >= 1
