"""A verifier cascade must buy the expensive vote only when it matters."""

from types import SimpleNamespace

import pytest

from agentdescent.verifier_ensemble import (
    VerifierEnsemble,
    VerifierSpec,
)


def _spec(name, value, cost=1.0, weight=1.0):
    return VerifierSpec(name, lambda _candidate: value, cost=cost, weight=weight)


def test_cascade_stops_when_cheap_verifiers_are_confident():
    calls = []

    def oracle(_candidate):
        calls.append("oracle")
        return True

    ensemble = VerifierEnsemble([
        _spec("unit", 0.95, cost=1),
        _spec("judge", 0.9, cost=10),
        VerifierSpec("oracle", oracle, cost=100),
    ], pass_threshold=0.5, min_confidence=0.5, min_votes=2)
    report = ensemble.evaluate(object())
    assert report.stopped_early
    assert report.names == ("unit", "judge")
    assert calls == []
    assert report.cost == 11
    assert report.passed


def test_cascade_buys_more_votes_when_uncertain():
    ensemble = VerifierEnsemble([
        _spec("a", 0.6, cost=1),
        _spec("b", 0.55, cost=2),
        _spec("c", 0.95, cost=4),
    ], min_confidence=0.5, min_votes=2)
    report = ensemble.evaluate(object())
    assert report.names == ("a", "b", "c")
    assert not report.stopped_early
    assert report.passed


def test_weighted_majority_and_unanimous_run_every_verifier():
    specs = [_spec("a", 1.0, cost=1), _spec("b", 0.0, cost=1)]
    weighted = VerifierEnsemble(specs, strategy="weighted", pass_threshold=0.51).evaluate(object())
    assert weighted.names == ("a", "b")
    assert weighted.score == pytest.approx(0.5)
    assert not weighted.passed

    majority = VerifierEnsemble([_spec("yes", True), _spec("no", False)],
                                strategy="majority").evaluate(object())
    assert majority.passed is False
    unanimous = VerifierEnsemble([_spec("yes", True), _spec("no", False)],
                                 strategy="unanimous").evaluate(object())
    assert unanimous.passed is False


def test_weighted_vote_can_override_majority():
    ensemble = VerifierEnsemble([
        VerifierSpec("cheap", lambda _c: 0.2, cost=1, weight=1),
        VerifierSpec("oracle", lambda _c: 1.0, cost=10, weight=5),
    ], strategy="weighted")
    report = ensemble.evaluate(object())
    assert report.score > 0.5
    assert report.passed


def test_max_cost_stops_before_an_expensive_vote():
    ensemble = VerifierEnsemble([
        _spec("a", 0.55, cost=2),
        _spec("b", 0.95, cost=100),
    ], min_confidence=0.9)
    report = ensemble.evaluate(object(), max_cost=10)
    assert report.names == ("a",)
    assert report.stopped_early
    assert report.cost == 2


def test_bool_float_mapping_and_object_are_normalised():
    values = [True, 0.25, {"score": 0.8, "reason": "judge"},
              SimpleNamespace(score=0.6, passed=True, reason="custom")]
    ensemble = VerifierEnsemble([
        VerifierSpec("bool", lambda _c: values[0]),
        VerifierSpec("float", lambda _c: values[1]),
        VerifierSpec("mapping", lambda _c: values[2]),
        VerifierSpec("object", lambda _c: values[3]),
    ], strategy="weighted")
    report = ensemble.evaluate(object())
    assert [vote.score for vote in report.votes] == [1.0, 0.25, 0.8, 0.6]
    assert [vote.passed for vote in report.votes] == [True, False, True, True]
    assert report.votes[2].reason == "judge"
    assert report.votes[3].reason == "custom"


def test_a_verifier_exception_is_a_zero_vote_not_a_crash():
    def broken(_candidate):
        raise RuntimeError("tool timeout")

    ensemble = VerifierEnsemble([VerifierSpec("broken", broken),
                                 _spec("ok", 1.0)], strategy="weighted")
    report = ensemble.evaluate(object())
    assert report.votes[0].score == 0.0
    assert "RuntimeError" in report.votes[0].reason
    assert report.score == pytest.approx(0.5)


def test_disagreement_and_confidence_are_reported():
    ensemble = VerifierEnsemble([_spec("a", 1.0), _spec("b", 0.0)],
                                strategy="weighted")
    report = ensemble.evaluate(object())
    assert report.disagreement == pytest.approx(0.5)
    assert 0.0 <= report.confidence <= 1.0
    assert set(report.to_dict()) == {"score", "passed", "confidence",
                                     "disagreement", "cost", "stopped_early",
                                     "complete", "errors", "votes"}


def test_invalid_specs_and_arguments_are_refused():
    with pytest.raises(ValueError):
        VerifierEnsemble([])
    with pytest.raises(ValueError):
        VerifierEnsemble([_spec("a", 1), _spec("a", 1)])
    with pytest.raises(TypeError):
        VerifierSpec("a", None)
    with pytest.raises(ValueError):
        VerifierSpec("a", lambda _c: 1, cost=-1)
    with pytest.raises(ValueError):
        VerifierSpec("a", lambda _c: 1, weight=0)
    with pytest.raises(ValueError):
        VerifierEnsemble([_spec("a", 1)], pass_threshold=2)
    with pytest.raises(ValueError):
        VerifierEnsemble([_spec("a", 1)], min_confidence=2)
    with pytest.raises(ValueError):
        VerifierEnsemble([_spec("a", 1)], min_votes=0)
    with pytest.raises(ValueError):
        VerifierEnsemble([_spec("a", 1)], strategy="best")
    with pytest.raises(ValueError):
        VerifierEnsemble([_spec("a", 1)]).evaluate(object(), max_cost=-1)
    with pytest.raises(TypeError):
        VerifierEnsemble([VerifierSpec("bad", lambda _c: object())]).evaluate(object())


def test_nan_scores_are_rejected_on_every_normalisation_path():
    for value in ({"score": float("nan")},
                  SimpleNamespace(score=float("nan"))):
        ensemble = VerifierEnsemble([VerifierSpec("bad", lambda _c, v=value: v)])
        with pytest.raises(ValueError):
            ensemble.evaluate(object())


def test_partial_non_cascade_panel_fails_closed():
    ensemble = VerifierEnsemble([
        _spec("a", True, cost=1),
        _spec("b", False, cost=50),
    ], strategy="unanimous")
    report = ensemble.evaluate(object(), max_cost=10)
    assert report.names == ("a",)
    assert not report.complete
    assert not report.passed


def test_budget_is_a_real_cap_even_for_the_first_verifier():
    ensemble = VerifierEnsemble([_spec("expensive", True, cost=100)])
    report = ensemble.evaluate(object(), max_cost=10)
    assert report.names == ()
    assert report.cost == 0
    assert not report.complete and not report.passed


def test_cheaper_later_verifier_is_not_skipped_by_an_expensive_one():
    ensemble = VerifierEnsemble([
        _spec("a", True, cost=1),
        _spec("x", False, cost=50),
        _spec("z", False, cost=1),
    ], strategy="weighted")
    report = ensemble.evaluate(object(), max_cost=10)
    assert report.names == ("a", "z")
    assert not report.complete
    assert not report.passed


def test_explicit_fail_vetoes_a_high_score():
    ensemble = VerifierEnsemble([
        VerifierSpec("veto", lambda _c: {"score": 0.9, "passed": False}),
        _spec("ok", 0.9),
    ], strategy="weighted")
    report = ensemble.evaluate(object())
    assert not report.passed
    assert next(v for v in report.votes if v.name == "veto").explicit


def test_capability_report_plugs_in_without_a_wrapper():
    from agentdescent.capability_guard import CapabilityReport

    ensemble = VerifierEnsemble([
        VerifierSpec("cap", lambda _c: CapabilityReport()),
    ])
    assert ensemble.evaluate(object()).passed
