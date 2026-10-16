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
import warnings

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
    boost = VerifierBoost(_FakeVerifier(score=0.5), store, min_labels=10,
                          min_val_labels=3)
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


# -- review #273 regressions ----------------------------------------------------


def test_empty_train_split_does_not_crash():
    """min_labels > min_val_labels is required; equal values refuse at init
    (review #273 blocking item 3)."""
    store = _store_with([_record(0.5, 0.3) for _ in range(10)])
    with pytest.raises(ValueError, match="min_labels"):
        VerifierBoost(_FakeVerifier(), store, min_labels=10, min_val_labels=10)


def test_training_filters_to_the_current_verifier_version():
    """Labels from an older verifier must not train the current judge
    (review #273 blocking item 2)."""
    # 200 old biased labels + 200 new unbiased labels under different versions.
    old = [_record(min(1.0, y + 0.4), y, version="old")
           for y in list(__import__("numpy").linspace(0.1, 0.9, 200))]
    new = [_record(y, y, version="new")
           for y in list(__import__("numpy").linspace(0.1, 0.9, 200))]
    store = _store_with(old + new)
    # Train only on "new": unbiased data should not activate a correction that
    # pulls 0.5 toward the old bias.
    boost = VerifierBoost(
        _FakeVerifier(score=0.5), store, min_labels=50, min_val_labels=10,
        seed=1, min_gain=0.0, auto=False,
        version_for_training=lambda: "new")
    boost.retrain()
    # With only unbiased new labels there is nothing useful to learn; either
    # inactive or a near-identity curve. Critically, g(0.5) must not be the
    # old-bias correction (~0.1).
    if boost.is_active:
        assert abs(boost.cheap_eval(object()) - 0.5) < 0.15


def test_audit_version_changes_when_boost_activates(tmp_path):
    """attach + recompute must refresh verifier_version on activation so
    VerifierWatch withdraws the old calibration (review #273 blocking item 1)."""
    y = list(__import__("numpy").linspace(0.1, 0.9, 80))
    s = _biased_scores(y, bias=0.2)
    store = _store_with(
        [_record(score, oracle, version="will-refresh")
         for score, oracle in zip(s, y)],
        path=str(tmp_path / "audit.jsonl"))
    # Pre-seed records under a fingerprint we will replace; attach will set a
    # live version_extra from the boost tag. Use an explicit store and boost.
    boost = VerifierBoost(_FakeVerifier(score=0.5), store, min_labels=20,
                          min_val_labels=5, seed=1, min_gain=0.0, auto=False)
    audit = attach(verifier=lambda task, output: 0.5,
                   oracle=lambda task, output: 0.3,
                   store=store, boost=boost)
    before = audit.verifier_version
    assert boost.version_tag() == "boost:inactive"
    # Retrain against whatever version the reward currently reports; seed the
    # store with records under that version so the filter finds them.
    current = audit.verifier_version
    for score, oracle in zip(s, y):
        store.append(_record(score, oracle, version=current))
    audit.recompute()
    assert boost.is_active is True
    after = audit.verifier_version
    assert after != before, (
        "activation must change the audit fingerprint so the old calibration "
        f"is withdrawn; before={before} after={after}")
    assert boost.version_tag().startswith("boost:active:")


def test_wrapper_preserves_legacy_false_full_eval_declaration():
    """A wrapped verifier with oracle_shares_full_set=False must stay
    independent after wrapping (review #273 P1)."""
    class Legacy:
        oracle_shares_full_set = False
        def rule_eval(self, a): return 0.5
        def cheap_eval(self, a): return 0.5
        def learned_eval(self, a): return 0.5, 0.1
        def eval_counts(self, a, floor=None): return (5, 5)
        def full_eval(self, a): return 0.4  # independent

    from agentdescent.verifier import shares_eval_counts
    inner = Legacy()
    assert shares_eval_counts(inner) is False
    boost = VerifierBoost(inner, AuditStore(), auto=False)
    assert shares_eval_counts(boost) is False
    assert boost.full_eval_matches_counts is False


def test_wrapper_preserves_missing_full_eval_declaration():
    """No declaration on the inner verifier => independent measurements."""
    class Bare:
        def rule_eval(self, a): return 0.5
        def cheap_eval(self, a): return 0.5
        def learned_eval(self, a): return 0.5, 0.1
        def eval_counts(self, a, floor=None): return (5, 5)
        def full_eval(self, a): return 0.5

    from agentdescent.verifier import shares_eval_counts
    inner = Bare()
    assert shares_eval_counts(inner) is False
    boost = VerifierBoost(inner, AuditStore(), auto=False)
    assert shares_eval_counts(boost) is False


# -- #304: recompute must not oscillate; wrapping must not mask legacy evals ----


def _stamped(score, oracle, version, scorer, n):
    rec = _record(score, oracle, version=version)
    return AuditRecord(**{**rec.to_dict(), "purpose": rec.purpose,
                          "record_id": f"{rec.record_id}-{n}",
                          "scorer_version": scorer})


def _attached_boost(tmp_path, n_labels=80, **kw):
    """An audit with a boost, plus `n_labels` biased IMPROVEMENT labels written
    while the correction was inactive (the first-activation situation)."""
    y = list(np.linspace(0.1, 0.9, n_labels))
    s = _biased_scores(y, bias=0.2)
    store = AuditStore(str(tmp_path / "audit.jsonl"))
    boost = VerifierBoost(_FakeVerifier(score=0.5), store, min_labels=20,
                          min_val_labels=5, seed=1, min_gain=0.0, auto=False,
                          **kw)
    audit = attach(verifier=lambda task, output: 0.5,
                   oracle=lambda task, output: 0.3, store=store, boost=boost)
    inactive = audit.verifier_version
    for i, (score, oracle) in enumerate(zip(s, y)):
        store.append(_stamped(score, oracle, inactive,
                              audit.reward.scorer_version, i))
    return audit, boost, store, inactive, (s, y)


def test_recompute_keeps_a_validated_correction_without_new_data(tmp_path):
    audit, boost, store, inactive, _ = _attached_boost(tmp_path)
    audit.recompute()
    assert boost.is_active is True
    tag, version = boost.version_tag(), audit.verifier_version
    assert version != inactive, "activation is a new measurement identity"
    for _ in range(4):                       # no records added in between
        audit.recompute()
        assert boost.is_active is True, boost.status()["reason"]
        assert boost.version_tag() == tag
        assert audit.verifier_version == version


def test_the_scorer_identity_survives_the_boost_activating(tmp_path):
    audit, boost, _, inactive, _ = _attached_boost(tmp_path)
    scorer = audit.reward.scorer_version
    audit.recompute()
    assert audit.verifier_version != inactive
    assert audit.reward.scorer_version == scorer


def test_labels_written_after_activation_keep_training_the_curve(tmp_path):
    audit, boost, store, _, (s, y) = _attached_boost(tmp_path)
    audit.recompute()
    first = boost.status()
    assert first["active"]
    # a sub-min_labels batch written under the *active* measurement version
    for i in range(5):
        store.append(_stamped(s[i], y[i], audit.verifier_version,
                              audit.reward.scorer_version, 1000 + i))
    audit.recompute()
    assert boost.is_active is True
    assert boost.status()["n_train"] + boost.status()["n_val"] >= 80, (
        "the post-activation labels join the same scorer's pool")


def test_unstamped_older_labels_do_not_oscillate_either(tmp_path):
    """A store written before records carried a scorer stamp: the labels sit
    under the old measurement version. Activation then finds nothing under the
    new one, and must keep the curve (the scorer is unchanged), not flip."""
    y = list(np.linspace(0.1, 0.9, 80))
    s = _biased_scores(y, bias=0.2)
    store = AuditStore(str(tmp_path / "audit.jsonl"))
    boost = VerifierBoost(_FakeVerifier(score=0.5), store, min_labels=20,
                          min_val_labels=5, seed=1, min_gain=0.0, auto=False)
    audit = attach(verifier=lambda task, output: 0.5,
                   oracle=lambda task, output: 0.3, store=store, boost=boost)
    for i, (score, oracle) in enumerate(zip(s, y)):
        store.append(_stamped(score, oracle, audit.verifier_version, "", i))
    audit.recompute()
    assert boost.is_active is True
    tag = boost.version_tag()
    for _ in range(3):
        audit.recompute()
        assert boost.is_active is True and boost.version_tag() == tag


def test_a_real_scorer_change_still_withdraws_the_correction(tmp_path):
    audit, boost, store, _, _ = _attached_boost(tmp_path)
    audit.recompute()
    assert boost.is_active is True
    # the scorer itself moves: a different identity, with too few labels of its
    # own, so the old scorer's curve no longer applies
    audit.reward.scorer_version = "a-different-scorer"
    boost.retrain(verifier_version=audit.verifier_version,
                  scorer_version=audit.reward.scorer_version)
    assert boost.is_active is False
    assert boost.version_tag() == "boost:inactive"


def test_stamped_records_round_trip_and_unstamped_lines_are_unchanged(tmp_path):
    plain = _record(0.5, 0.3)
    assert "scorer_version" not in plain.to_dict()
    stamped = _stamped(0.5, 0.3, "v1", "scorer", 0)
    assert stamped.to_dict()["scorer_version"] == "scorer"
    assert AuditRecord.from_dict(stamped.to_dict()) == stamped
    assert AuditRecord.from_dict(plain.to_dict()) == plain


class _OracleEvalOnly:
    """A pre-0.5.1 verifier: `oracle_eval` is the whole-set scorer, no full_eval."""
    oracle_shares_full_set = False

    def __init__(self, value=0.4):
        self.value, self.calls = value, 0

    def rule_eval(self, a): return 0.5
    def cheap_eval(self, a): return 0.5
    def learned_eval(self, a): return 0.5, 0.1
    def eval_counts(self, a, floor=None): return (5, 5)

    def oracle_eval(self, a):
        self.calls += 1
        return self.value


@pytest.mark.parametrize("active", [False, True])
def test_a_wrapped_oracle_eval_only_verifier_is_still_evaluated(active):
    from agentdescent.verifier import full_eval_of, shares_eval_counts
    inner = _OracleEvalOnly(0.4)
    if active:
        y = list(np.linspace(0.1, 0.9, 24))
        store = _store_with([_record(sc, o) for sc, o in
                             zip(_biased_scores(y, 0.2), y)])
        boost = VerifierBoost(inner, store, min_labels=10, min_val_labels=3,
                              seed=1)
        assert boost.is_active is True
    else:
        boost = VerifierBoost(inner, AuditStore())
        assert boost.is_active is False
    assert shares_eval_counts(boost) is False        # independence preserved
    with pytest.warns(DeprecationWarning, match="oracle_eval"):
        assert full_eval_of(boost)(object()) == 0.4
    with pytest.warns(DeprecationWarning, match="oracle_eval"):
        assert boost.oracle_eval(object()) == 0.4    # the wrapper's own alias
    assert inner.calls == 2


def test_wrapped_and_unwrapped_independent_gates_refuse_the_same_regression(
        tmp_path):
    """The engine's independent gate runs the scorer through the wrapper exactly
    as it does unwrapped, and refuses a candidate whose independent score
    regresses (rather than dying with AttributeError)."""
    from agentdescent.aggregator import Aggregator, AggregatorConfig, _Candidate
    from agentdescent.evolution import EvolvingArtifact
    from agentdescent.evolvable import Diff
    from agentdescent.ledger import Ledger
    from agentdescent.scheduler import AuditScheduler

    class Legacy(_OracleEvalOnly):
        def eval_counts(self, a, floor=None):
            return (7, 3)                          # cheap sweep says "improved"
        def oracle_eval(self, a):                  # ...the independent one differs
            return 0.2 if a.state.get("k") == "v" else 0.9

    def run(wrap):
        def serialize(a): return {"state": a.state, "blast_radius": a.blast_radius}
        def deserialize(aid, version, state):
            return EvolvingArtifact(aid, state.get("state", {}), version, 0.6)
        lg = Ledger(str(tmp_path / f"repo-{wrap}"), serialize, deserialize)
        lg.register(EvolvingArtifact("artifact", {"k": "v0"}, blast_radius=0.6))
        v = Legacy()
        verifier = VerifierBoost(v, AuditStore(), auto=False) if wrap else v
        agg = Aggregator(lg, verifier, AuditScheduler(),
                         AggregatorConfig(batch_trigger=1))
        snap = lg.snapshot(Ledger.DEV)
        base = snap.get("artifact")
        diff = Diff(diff_id="d", target="artifact", ops={"k": "v"}, author="w")
        c = _Candidate(
            artifact_id="artifact", artifact=base, candidate=base.apply(diff),
            diff=diff, cards=[], survivor_cards=[], head=snap.version,
            fused=False, considered=1, survived=1, discarded=0, conflicts=0,
            base_counts=(2.0, 8.0), cand_counts=(8.0, 2.0),
            base_cheap=0.5, cand_cheap=0.8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return agg._decide(c)

    plain, wrapped = run(False), run(True)
    assert plain.committed_version is None, plain.category
    assert wrapped.committed_version is None
    assert wrapped.category == plain.category


# -- the gate must beat a constant predictor (#306) ------------------------------


class _ListStore:
    """The slice of AuditStore the boost reads: just ``for_improvement``."""

    def __init__(self, recs):
        self._recs = recs

    def for_improvement(self, version=None, scorer_version=None):
        return self._recs


def _namespace_records(pairs):
    import types
    return [types.SimpleNamespace(resolved=True, verifier_score=f, oracle_score=y)
            for f, y in pairs]


def test_a_pure_bias_shift_without_discrimination_is_not_validated():
    """Score = noise in [0.8, 1.0], oracle passes 30% of the time, uncorrelated.

    The raw verifier is badly biased (MSE ~0.58), so *any* curve that moves the
    mean down beats it -- yet the curve carries no ranking information and is
    worse than predicting the training mean. The gate must say so."""
    import random

    rng = random.Random(1)
    recs = _namespace_records(
        (0.8 + 0.2 * rng.random(), 1.0 if rng.random() < 0.3 else 0.0)
        for _ in range(400))
    boost = VerifierBoost(_FakeVerifier(score=0.9), _ListStore(recs))
    status = boost.status()
    assert boost.is_active is False
    assert "constant" in status["reason"]
    assert boost.version_tag() == "boost:inactive"
    # rejection still exposes the measured baseline (and the MSEs that
    # justified it) rather than the default "not measured" inf
    assert math.isfinite(status["error_const"]) and status["error_const"] > 0.0
    assert math.isfinite(status["error_before"])
    assert math.isfinite(status["error_after"])
    assert status["n_train"] > 0 and status["n_val"] > 0
    # the curve is not applied to the ranking layers
    assert boost.cheap_eval(object()) == 0.9


def test_a_biased_but_informative_verifier_still_activates():
    """Control: monotone-informative yet miscalibrated (oracle = 1 iff f > 0.9,
    scores in [0.8, 1.0]) is the case the boost exists for."""
    import random

    rng = random.Random(2)
    pairs = []
    for _ in range(400):
        f = 0.8 + 0.2 * rng.random()
        pairs.append((f, 1.0 if f > 0.9 else 0.0))
    boost = VerifierBoost(_FakeVerifier(score=0.95), _ListStore(_namespace_records(pairs)))
    status = boost.status()
    assert boost.is_active is True, status["reason"]
    assert status["error_after"] <= status["error_const"] * (1.0 - boost.min_gain)
    assert status["error_after"] <= status["error_before"] * (1.0 - boost.min_gain)
    assert boost.version_tag().startswith("boost:active:")
    assert boost.cheap_eval(object()) > 0.5      # f=0.95 is in the passing region


def test_status_exposes_the_constant_baseline_and_keeps_existing_keys():
    y = list(np.linspace(0.1, 0.9, 24))
    s = _biased_scores(y, bias=0.2)
    store = _store_with([_record(score, oracle) for score, oracle in zip(s, y)])
    boost = VerifierBoost(_FakeVerifier(score=0.5), store, min_labels=10,
                          min_val_labels=3, seed=1)
    status = boost.status()
    assert {"active", "fitted", "n_train", "n_val", "error_before", "error_after",
            "resid_sd", "reason", "fit_at", "version_tag"} <= set(status)
    assert math.isfinite(status["error_const"]) and status["error_const"] > 0.0
    # a state built the old way (no baseline) reports "not measured"
    from agentdescent.audit import boost as boost_mod
    old = boost_mod.BoostState(False, False, 0, 0, math.inf, math.inf, "x", 0.0)
    assert old.error_const == math.inf
    # before any fit (or after an early rejection with no measured baseline)
    # the baseline is "not measured"
    fresh = VerifierBoost(_FakeVerifier(), _store_with([]), auto=False)
    assert fresh.status()["error_const"] == math.inf


def test_a_degenerate_single_level_curve_is_rejected_with_a_reason():
    # min_gain=0 lets a one-level (constant) curve tie the constant baseline.
    recs = _namespace_records([(0.5 + 0.001 * i, 0.5) for i in range(100)])
    boost = VerifierBoost(_FakeVerifier(), _ListStore(recs), min_gain=0.0)
    assert boost.is_active is False
    assert "degenerate" in boost.status()["reason"]
