"""The 2F1 task: reference integrity, scoring, gate, runner, evaluator.

The first block is the one that carries the result. This task has no
leaderboard, so what makes a number from it worth reporting is that the
reference is independent of everything being scored and that the point set was
not drawn to flatter anybody. Those are properties of the committed data file,
so they are tested as such: the stored values are re-derived from mpmath where
mpmath is installed, and the file's own claims about how it was made are
checked against what is in it either way.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys

import pytest

from examples.era import _era_hyp2f1 as hyp
from examples.era import _era_support as support


SUITE = json.loads(hyp.SUITE_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The reference
# ---------------------------------------------------------------------------


def test_the_stress_file_records_how_it_was_made():
    """A data file whose provenance is not in it cannot be audited later."""
    reference = SUITE["reference"]
    assert reference["library"] == "mpmath"
    assert reference["precisions_dps"] == [30, 60]
    assert reference["kept_when_they_agree_to_digits"] == 25
    assert SUITE["distribution"]["a"] == "U(-30.0, 30.0)"
    assert SUITE["distribution"]["z"] == "U(-40.0, 0.999)"
    assert isinstance(SUITE["seed"], int)
    assert len(SUITE["shards"]) == 12
    assert all(len(shard) == 20 for shard in SUITE["shards"])


def test_no_point_is_degenerate():
    """The two rejections the generator claims, checked on what it emitted."""
    for shard in SUITE["shards"]:
        for row in shard:
            c = float(row["c"])
            assert not (abs(c - round(c)) < 1e-3 and round(c) <= 0), (
                f"c={c} sits on a pole of 2F1")
            assert abs(float(row["value"])) >= 1e-250
            # The stored reference must carry more digits than a float64 can,
            # or the scoring would be measuring its truncation.
            assert len(row["value"].replace("-", "").replace(".", "").lstrip("0")) >= 20


@pytest.mark.parametrize("shard", range(12))
def test_the_stored_reference_matches_mpmath_at_higher_precision(shard):
    """Re-derive the file's values, at a precision above the one it kept."""
    mpmath = pytest.importorskip("mpmath",
                                 reason="no mpmath to check the references with")
    mpmath.mp.dps = 60
    for row in SUITE["shards"][shard]:
        value = mpmath.hyp2f1(float(row["a"]), float(row["b"]),
                              float(row["c"]), float(row["z"]))
        stored = mpmath.mpf(row["value"])
        relative = abs(value - stored) / abs(value)
        assert float(relative) < 1e-24, (
            f"stored reference for {row} is off by {float(relative):.2e}")


def test_the_committed_stress_file_is_exactly_what_the_generator_produces():
    """The points were drawn, not chosen -- and this is how a reader confirms it.

    Re-runs the generator from its declared seed and distribution and demands
    the committed file back, byte for byte. That covers the half the
    value-by-value check above cannot: that nobody picked *which* points to
    include after seeing how an implementation did on them.
    """
    pytest.importorskip("mpmath", reason="the generator needs mpmath")
    completed = subprocess.run(
        [sys.executable, "-m", "tools.gen_hyp2f1_stress", "--check"],
        capture_output=True, text=True, timeout=900)
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_the_baseline_is_not_a_strawman_and_not_perfect_either():
    """Both halves matter, and both are properties of this specific data file.

    If SciPy solved everything there would be nothing to search for; if it
    solved nothing the task would be measuring a broken call rather than a hard
    problem. This is the check that the reported gain is a gain over the state
    of the practice.
    """
    scipy_special = pytest.importorskip("scipy.special")
    suite = hyp.load_suite()
    total = solved = 0
    digits_sum = 0.0
    for shard in range(12):
        for point, truth in zip(suite.shard_points[shard], suite.shard_truth[shard]):
            value = scipy_special.hyp2f1(float(point["a"]), float(point["b"]),
                                         float(point["c"]), float(point["z"]))
            digits = hyp.digits_of(repr(float(value)), truth)
            digits_sum += digits
            solved += digits >= hyp.SOLVED_DIGITS
            total += 1
    assert total == 240
    assert 8.0 < digits_sum / total < 11.0, "the baseline moved; rereport the numbers"
    assert 0.5 < solved / total < 0.8, "the baseline moved; rereport the numbers"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_digits_are_measured_against_the_full_precision_reference():
    """The comparison happens in `decimal`, so it does not round away the answer."""
    truth = "1.000000000000000000000000"
    assert hyp.digits_of("1.0", truth) == hyp.DIGIT_CAP
    assert hyp.digits_of("1.000001", truth) == pytest.approx(6.0, abs=1e-6)
    assert hyp.digits_of("2.0", truth) == 0.0
    # A value that differs in the 20th digit is beyond float64 and beyond the
    # cap; it must saturate rather than report a number the metric cannot mean.
    assert hyp.digits_of("1.0000000000000000000001", truth) == hyp.DIGIT_CAP


def test_a_failed_point_scores_zero_rather_than_raising():
    for value in (None, "nan", "inf", "-inf", "not a number", ""):
        assert hyp.digits_of(value, "3.14159265358979323846") == 0.0


def test_the_reward_is_order_preserving_with_the_metric():
    low = hyp.framework_score({"mean_digits": 3.0})
    high = hyp.framework_score({"mean_digits": 9.0})
    assert 0.0 < low < high <= 1.0
    assert hyp.framework_score({"mean_digits": None}) == 0.0
    assert hyp.framework_score({"mean_digits": float("-inf")}) == 0.0


# ---------------------------------------------------------------------------
# The suite as the example sees it
# ---------------------------------------------------------------------------


def test_the_suite_splits_into_scored_and_held_back_sets():
    suite = hyp.load_suite(shards=8, test_shards=4)
    assert suite.test_range() == (8, 9, 10, 11)
    assert suite.size(0) == 20
    assert len(suite.shard_points) == len(suite.shard_truth) == 12
    for points, truth in zip(suite.shard_points, suite.shard_truth):
        assert len(points) == len(truth)
        # What goes into the sandbox is four parameters, and nothing else.
        assert all(set(point) == {"a", "b", "c", "z"} for point in points)


def test_a_suite_that_cannot_be_scored_is_refused():
    with pytest.raises(ValueError):
        hyp.load_suite(shards=1, test_shards=1)
    with pytest.raises(ValueError):
        hyp.load_suite(shards=4, test_shards=0)
    with pytest.raises(ValueError):
        hyp.load_suite(shards=11, test_shards=4)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def _gate(source: str):
    return support.validate_source(
        source, entrypoint="hyp2f1", allowed_imports=hyp.ALLOWED_IMPORTS,
        literal_top_level=False)


def test_the_gate_accepts_the_baseline_and_rejects_the_obvious_accidents():
    assert _gate(hyp.INITIAL_PROGRAM)[0]
    assert not _gate("def other(a, b, c, z):\n    return 0.0\n")[0]
    assert not _gate("import os\ndef hyp2f1(a, b, c, z):\n    return 0.0\n")[0]
    assert not _gate("def hyp2f1(a, b, c, z):\n    return eval('1')\n")[0]


@pytest.mark.parametrize("module", ["mpmath", "decimal", "fractions", "sympy"])
def test_arbitrary_precision_arithmetic_is_off_the_allowlist(module):
    """The deliverable is a float64 routine; this is what makes that true.

    A candidate that reached for arbitrary precision would be answering a
    different question -- and would be scored against a reference produced the
    same way, which is not a comparison with the practice.
    """
    source = f"import {module}\ndef hyp2f1(a, b, c, z):\n    return 0.0\n"
    ok, reason = _gate(source)
    assert not ok and module in reason


# ---------------------------------------------------------------------------
# The runner, without a sandbox
# ---------------------------------------------------------------------------


runner = pytest.importorskip("examples.era._era_hyp2f1_runner")

POINTS = [{"a": "0.5", "b": "1.5", "c": "2.5", "z": "-3.0"}] * 3


def test_one_failing_point_does_not_take_the_rest_of_the_shard_down():
    calls = {"n": 0}

    def flaky(a, b, c, z):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ZeroDivisionError("boom")
        return 1.25

    results = runner.evaluate(flaky, POINTS, deadline=float("inf"))
    assert results[0]["estimate"] is None and "ZeroDivisionError" in results[0]["error"]
    assert results[1]["estimate"] == repr(1.25) and results[2]["estimate"] == repr(1.25)


def test_a_non_finite_result_is_recorded_as_a_failure_not_serialised():
    results = runner.evaluate(lambda *args: float("nan"), POINTS[:1],
                              deadline=float("inf"))
    assert results[0]["estimate"] is None and "non-finite" in results[0]["error"]
    json.dumps(results, allow_nan=False)


def test_the_estimate_survives_the_trip_as_an_exact_float():
    """`repr`, not a JSON number: the scorer compares against 25 digits."""
    value = 0.1 + 0.2  # 0.30000000000000004, and the last digits are the point
    results = runner.evaluate(lambda *args: value, POINTS[:1], deadline=float("inf"))
    assert float(results[0]["estimate"]) == value


def test_a_shard_that_runs_out_of_wall_clock_scores_its_remaining_points_zero():
    results = runner.evaluate(lambda *args: 1.0, POINTS, deadline=-1.0)
    assert all(row["estimate"] is None for row in results)
    assert all("TimeLimit" in row["error"] for row in results)


def test_the_candidate_is_handed_floats_and_not_the_problem_row():
    seen = []
    runner.evaluate(lambda *args: (seen.append(args), 1.0)[1], POINTS[:1],
                    deadline=float("inf"))
    assert seen == [(0.5, 1.5, 2.5, -3.0)]
    assert all(isinstance(value, float) for value in seen[0])


# ---------------------------------------------------------------------------
# What the model is shown
# ---------------------------------------------------------------------------


def test_the_prompt_states_the_distribution_and_withholds_every_answer():
    suite = hyp.load_suite()
    metrics = {
        "mean_digits": 9.73, "solved": 158, "points": 240,
        "worst": [{"shard": 0, "index": 3, "a": 1.5, "b": -2.5, "c": 3.5,
                   "z": -20.0, "digits": 0.0, "error": ""}],
    }
    program = support.Program("id", 0, None, hyp.INITIAL_PROGRAM, "baseline",
                              metrics, True, "")
    prompt = hyp.mutation_prompt(program, preview=hyp.suite_preview(suite))
    assert "U(-30.0, 30.0)" in prompt and "9.73" in prompt
    assert "0.0 digits" in prompt
    # Every reference value in the file, absent from what the model reads.
    for shard in SUITE["shards"][:2]:
        for row in shard:
            assert row["value"] not in prompt
    assert "mpmath" in prompt and "NOT available" in prompt


# ---------------------------------------------------------------------------
# End to end, through the real sandbox
# ---------------------------------------------------------------------------


@pytest.mark.skipif(support.sandbox_backend() is None,
                    reason="no Bubblewrap or Seatbelt on this host")
def test_the_baseline_scores_through_the_sandbox_and_leaves_room_to_improve():
    pytest.importorskip("scipy", reason="the baseline program imports scipy")
    suite = hyp.load_suite()
    valid, metrics, error = hyp.evaluate_source(
        hyp.INITIAL_PROGRAM, suite=suite, shards=(0, 1), timeout=60.0)
    assert valid, error
    assert 6.0 < metrics["mean_digits"] < hyp.DIGIT_CAP - 0.5
    assert 0 < metrics["solved"] < metrics["points"]
    assert metrics["score"] == metrics["mean_digits"]


@pytest.mark.skipif(support.sandbox_backend() is None,
                    reason="no Bubblewrap or Seatbelt on this host")
def test_a_program_that_cannot_be_imported_fails_the_whole_evaluation():
    suite = hyp.load_suite()
    valid, metrics, error = hyp.evaluate_source(
        "import scipy.not_a_module\ndef hyp2f1(a, b, c, z):\n    return 0.0\n",
        suite=suite, shards=(0,), timeout=30.0)
    assert not valid and metrics["score"] == -math.inf
    assert "Error" in error


def test_the_gate_rejects_before_any_process_is_started(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("spawned"))
    suite = hyp.load_suite()
    valid, metrics, error = hyp.evaluate_source(
        "import os\ndef hyp2f1(a, b, c, z):\n    return 0.0\n",
        suite=suite, shards=(0,), timeout=5.0)
    assert not valid and error.startswith("gate:")


def test_the_port_runs_from_the_command_line_without_touching_anything():
    completed = subprocess.run(
        [sys.executable, "-m", "examples.era.era_hypergeometric", "--dry-run"],
        capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0
    assert "hypergeometric" in completed.stdout.lower()
    assert "scipy.special.hyp2f1" in completed.stdout
    assert "dry-run" in completed.stdout
