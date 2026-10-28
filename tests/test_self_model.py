"""A self-model has to know when it is overconfident, not just be monotone."""

import math
import random

import pytest

from agentdescent.self_model import (
    SelfModel,
    brier_score,
    expected_calibration_error,
    select_by_confidence,
)


def test_ece_and_brier_on_a_degenerate_distribution():
    assert expected_calibration_error([0.0] * 5, [0] * 5, bins=4) == 0.0
    conf = [0.0, 0.25, 0.5, 0.75, 1.0]
    out = [0, 0, 1, 1, 1]
    assert brier_score(conf, out) == pytest.approx(0.075)


def test_ece_detects_systematic_overconfidence():
    assert expected_calibration_error([0.9] * 10, [0] * 10) == pytest.approx(0.9)
    report = SelfModel().raw_report([0.9] * 10, [0] * 10)
    assert report.verdict == "overconfident"
    assert report.gap > 0.8


def test_platt_scaling_reduces_brier_on_overconfident_rows():
    conf = [0.9] * 10 + [0.8] * 10 + [0.2] * 10
    out = [0] * 10 + [1] * 10 + [0] * 10
    model = SelfModel()
    raw = model.raw_report(conf, out)
    fitted = model.fit(conf, out)
    assert fitted.brier <= raw.brier
    assert fitted.mean_confidence <= raw.mean_confidence + 1e-9


def test_a_calibrated_input_stays_calibrated_on_held_out_rows():
    rng = random.Random(11)

    def rows(n):
        conf = [rng.random() for _ in range(n)]
        out = [1 if rng.random() < x else 0 for x in conf]
        return conf, out

    train_conf, train_out = rows(500)
    test_conf, test_out = rows(500)
    model = SelfModel()
    raw = model.raw_report(test_conf, test_out)
    model.fit(train_conf, train_out)
    fitted = model.report
    # Re-score the held-out rows through the fitted map.
    held = [model.predict(x) for x in test_conf]
    held_ece = expected_calibration_error(held, test_out)
    assert held_ece <= raw.ece + 0.02
    assert raw.ece < 0.08


def test_fit_is_deterministic_and_predict_is_monotone():
    conf = [0.1, 0.4, 0.6, 0.9]
    out = [0, 0, 1, 1]
    one = SelfModel()
    two = SelfModel()
    one.fit(conf, out, steps=200)
    two.fit(conf, out, steps=200)
    assert (one.a, one.b) == pytest.approx((two.a, two.b))
    values = one.predict_many([0.0, 0.2, 0.5, 0.8, 1.0])
    assert values == tuple(sorted(values))


def test_unfitted_model_refuses_to_predict_or_route():
    model = SelfModel()
    with pytest.raises(RuntimeError):
        model.predict(0.5)
    with pytest.raises(RuntimeError):
        model.route(0.5)
    with pytest.raises(RuntimeError):
        model.coverage([0.5])


def test_route_and_coverage_use_the_calibrated_probability():
    model = SelfModel()
    model.fit([0.1, 0.2, 0.3, 0.4, 0.5], [0, 0, 1, 1, 1])
    model.a, model.b = 2.0, -1.0
    assert model.route(0.9, threshold=0.5) == "answer"
    assert model.route(0.1, threshold=0.5) == "abstain"
    assert model.coverage([0.1, 0.9], threshold=0.5) == 0.5
    with pytest.raises(ValueError):
        model.route(0.5, threshold=1.5)


def test_select_by_confidence_keeps_alignment_and_model_routing():
    items = ["a", "b", "c"]
    answered, abstained = select_by_confidence(items, [0.9, 0.2, 0.6], threshold=0.5)
    assert answered == ("a", "c")
    assert abstained == ("b",)
    model = SelfModel()
    model.fit([0.1, 0.2, 0.3, 0.4, 0.5], [0, 0, 1, 1, 1])
    model.a, model.b = -20.0, 5.0  # raw confidence is anti-correlated here
    answered, abstained = select_by_confidence(items, [0.9, 0.2, 0.6],
                                               threshold=0.5, model=model)
    assert answered == ("b",)
    assert abstained == ("a", "c")


def test_malformed_rows_are_refused():
    with pytest.raises(ValueError):
        expected_calibration_error([], [])
    with pytest.raises(ValueError):
        expected_calibration_error([0.1], [0, 1])
    with pytest.raises(ValueError):
        expected_calibration_error([1.2], [1])
    with pytest.raises(ValueError):
        expected_calibration_error([0.5], [0.5])
    with pytest.raises(ValueError):
        select_by_confidence(["a"], [])
    with pytest.raises(ValueError):
        select_by_confidence(["a"], [0.5], threshold=-0.1)


def test_fit_can_report_anti_correlation_instead_of_hiding_it():
    model = SelfModel()
    report = model.fit([0.9] * 10 + [0.1] * 10, [0] * 10 + [1] * 10)
    assert model.a < 0
    assert report.anti_correlated
    assert model.predict(0.9) < model.predict(0.1)


def test_short_rows_are_insufficient_not_falsely_calibrated():
    report = SelfModel().raw_report([0.5, 0.5], [0, 1])
    assert report.verdict == "insufficient"


def test_select_by_confidence_validates_raw_confidences():
    with pytest.raises(ValueError):
        select_by_confidence(["a"], [math.nan])
    with pytest.raises(ValueError):
        select_by_confidence(["a"], [1.7])


def test_more_bins_do_not_change_a_degenerate_distribution():
    assert expected_calibration_error([0.0] * 4, [0] * 4, bins=1) == 0.0
    assert expected_calibration_error([0.0] * 4, [0] * 4, bins=50) == 0.0
    with pytest.raises(ValueError):
        expected_calibration_error([0.1], [1], bins=0)


def test_saturated_confidences_converge_instead_of_diverging():
    rng = random.Random(3)
    conf, out = [], []
    for _ in range(200):
        value = rng.choice([0.0, 1.0])
        conf.append(value)
        out.append(value if rng.random() < 0.8 else 1.0 - value)

    model = SelfModel()
    model.fit(conf, out)
    assert 0.7 <= model.predict(1.0) <= 0.9
    assert model.predict(0.0) <= 0.4

    def log_loss(probabilities):
        total = 0.0
        for p, y in zip(probabilities, out):
            p = min(1 - 1e-12, max(1e-12, p))
            total -= y * math.log(p) + (1 - y) * math.log(1 - p)
        return total / len(out)

    raw = log_loss(conf)
    fitted = log_loss([model.predict(value) for value in conf])
    assert fitted <= raw + 1e-9
