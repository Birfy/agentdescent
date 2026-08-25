"""The sweep-chosen tasks: the sweep itself, reference integrity, scoring, gate.

Two blocks here carry the result rather than the plumbing.

The first is the **sweep**. `tools/scan_numeric_precision.py` is what chose
`pbdv` and `hyperu` over the other forty-five entry points it measures, so the
claim "these two are the inaccurate ones" is only as good as that tool. It is
tested the way a measuring instrument is: on inputs whose answer is known
independently -- a library that is exactly right must score the cap, one that is
deliberately wrong in a known digit must score that digit -- and by checking
that it rejects a point whose reference has not converged instead of scoring it.

The second is the **committed data**. These tasks have no leaderboard, so what
makes a number worth reporting is that the reference is independent of
everything being scored and that the points were not drawn to flatter anybody.
Both are properties of the files, so both are tested as properties of the files.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys

import pytest

from examples.era import _era_special as special
from examples.era import _era_support as support


TARGET_KEYS = ("pbdv", "hyperu")
SUITES = {key: json.loads(special.suite_path(key).read_text(encoding="utf-8"))
          for key in TARGET_KEYS}


# ---------------------------------------------------------------------------
# The sweep that chose the targets
# ---------------------------------------------------------------------------


def test_the_sweep_scores_an_exact_library_at_the_cap():
    """A probe whose library call *is* the reference must come back perfect.

    Without this, a bug that made every probe look inaccurate would read as a
    discovery about SciPy.
    """
    pytest.importorskip("mpmath")
    import mpmath as mp

    from tools import scan_numeric_precision as scan

    probe = scan.Probe(
        key="exact", target="exact", domain="x ~ U(1, 2)",
        draw=lambda r: {"x": r.uniform(1.0, 2.0)},
        library=lambda p: math.sqrt(p["x"]),
        reference=lambda p: mp.sqrt(p["x"]),
    )
    result = scan.run_probe(probe, points=40, dps=30, agree=22)
    assert result.scored == 40
    assert min(result.digits) > 15.0


def test_the_sweep_measures_a_known_error_where_it_was_planted():
    """Perturb the library by 1e-9 relative and the probe must say ~9 digits."""
    pytest.importorskip("mpmath")
    import mpmath as mp

    from tools import scan_numeric_precision as scan

    probe = scan.Probe(
        key="planted", target="planted", domain="x ~ U(1, 2)",
        draw=lambda r: {"x": r.uniform(1.0, 2.0)},
        library=lambda p: math.sqrt(p["x"]) * (1.0 + 1e-9),
        reference=lambda p: mp.sqrt(p["x"]),
    )
    result = scan.run_probe(probe, points=40, dps=30, agree=22)
    assert all(8.9 < value < 9.1 for value in result.digits)


def test_the_sweep_rejects_a_point_instead_of_scoring_an_unconverged_reference():
    """The rule that makes "SciPy got zero digits" a claim about SciPy.

    A reference that disagrees with itself between the two precisions is not
    evidence, and the sweep must drop the point rather than score against
    either value.
    """
    pytest.importorskip("mpmath")
    import mpmath as mp

    from tools import scan_numeric_precision as scan

    def unstable(point):
        # Returns a different answer at each precision, which is exactly the
        # condition the agreement gate exists to catch.
        return mp.mpf(1) + mp.mpf(10) ** (-mp.mp.dps // 2)

    probe = scan.Probe(
        key="unstable", target="unstable", domain="x ~ U(1, 2)",
        draw=lambda r: {"x": r.uniform(1.0, 2.0)},
        library=lambda p: 1.0,
        reference=unstable,
    )
    result = scan.run_probe(probe, points=20, dps=30, agree=22)
    assert result.scored == 0
    assert result.reference_rejected > 0


def test_the_two_targets_are_the_ones_the_sweep_singled_out():
    """The example's targets and the sweep's findings must not drift apart.

    Re-measures both probes rather than reading the committed report, so the
    table in the docs is checked against SciPy as installed. Bounds are wide:
    this asserts *that* these functions are the inaccurate ones, not the exact
    figures, which move with the SciPy version.
    """
    pytest.importorskip("mpmath")
    pytest.importorskip("scipy")
    from tools import scan_numeric_precision as scan

    for key in TARGET_KEYS:
        result = scan.run_probe(scan.PROBES_BY_KEY[key], points=250,
                                dps=30, agree=22)
        summary = result.summary()
        assert summary["mean_digits"] < 15.0, (
            f"{key} is no longer inaccurate; rereport the numbers")
        assert summary["frac_below_1"] > 0.005, (
            f"{key} no longer has points with no correct digit; rereport")


# ---------------------------------------------------------------------------
# The reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", TARGET_KEYS)
def test_the_stress_file_records_how_it_was_made(key):
    """A data file whose provenance is not in it cannot be audited later."""
    suite = SUITES[key]
    reference = suite["reference"]
    assert reference["library"] == "mpmath"
    assert reference["precisions_dps"] == [30, 60]
    assert reference["kept_when_they_agree_to_digits"] == 25
    assert suite["entrypoint"] == key
    assert suite["params"] == {"pbdv": ["v", "x"],
                               "hyperu": ["a", "b", "x"]}[key]
    for name in suite["params"]:
        assert name in suite["distribution"], f"{name} has no declared range"
    assert "the two precisions disagreeing" in suite["distribution"]["rejected"]


def _bounds(spec: str):
    """The numbers in a declared range such as ``U(-20.0, 20.0)``, as floats."""
    import re
    return [float(match) for match in
            re.findall(r"-?\d+\.?\d*(?:[eE]-?\d+)?", spec)]


@pytest.mark.parametrize("key", TARGET_KEYS)
def test_the_stress_set_draws_the_sweeps_range_and_not_a_narrower_one(key):
    """The stress set must not be drawn around the failures the sweep found.

    This is the property that keeps every reported number honest. The sweep
    located where SciPy fails; if the suite then narrowed onto that region,
    every "mean digits" figure in the docs would be inflated and the measured
    gain would be a gain against a region chosen for being bad. So the two
    declared ranges are compared numerically, and narrowing either end of
    either interval fails this test.
    """
    from tools import gen_special_stress as generator
    from tools import scan_numeric_precision as scan

    declared = generator.TARGETS_BY_KEY[key].distribution
    # The sweep states shared ranges as one clause ("a,b ~ U(-20,20)"), so a
    # clause names one or more parameters.
    swept = {}
    for clause in scan.PROBES_BY_KEY[key].domain.split(";"):
        names, _, spec = clause.partition("~")
        for name in names.split(","):
            swept[name.strip()] = spec.strip()
    for name, spec in declared.items():
        assert name in swept, f"{key}.{name} is not a parameter the sweep drew"
        assert _bounds(spec) == _bounds(swept[name]), (
            f"{key}.{name}: the suite draws {spec} but the sweep declared "
            f"{swept[name]} -- a suite narrowed onto the failure region would "
            f"inflate every number reported from it")


def test_the_range_check_would_notice_a_narrowed_suite():
    """The guard above must be able to fail, or it is decoration.

    Narrows one bound by a hair and requires the comparison to reject it.
    """
    assert _bounds("U(-20.0, 20.0)") == _bounds("U(-20,20)")
    assert _bounds("U(-15.0, 20.0)") != _bounds("U(-20,20)")
    assert _bounds("logU(1e-3, 100.0)") == _bounds("logU(1e-3,100)")


@pytest.mark.parametrize("key", TARGET_KEYS)
def test_no_point_is_degenerate(key):
    """The rejections the generator claims, checked on what it emitted."""
    suite = SUITES[key]
    for shard in suite["shards"]:
        assert len(shard) == 250
        for row in shard:
            value = abs(float(row["value"]))
            assert 1e-250 <= value <= 1e250
            # The stored reference must carry more digits than a float64 can,
            # or the scoring would be measuring its truncation.
            digits = row["value"].split("e")[0]
            assert len(digits.replace("-", "").replace(".", "").lstrip("0")) >= 20


@pytest.mark.parametrize("key", TARGET_KEYS)
def test_the_stored_reference_matches_mpmath_at_higher_precision(key):
    """Re-derive the file's values at a precision above the one it kept.

    Every 25th point rather than every point: 3000 values at 60 digits is slow
    enough that the test would get skipped, and
    `test_the_committed_stress_file_redraws_identically` is the exhaustive half.
    """
    mpmath = pytest.importorskip("mpmath",
                                 reason="no mpmath to check the references with")
    from tools import gen_special_stress as generator

    target = generator.TARGETS_BY_KEY[key]
    mpmath.mp.dps = 60
    for shard in SUITES[key]["shards"]:
        for row in shard[::25]:
            point = {name: float(row[name]) for name in target.params}
            value = target.reference(point, mpmath)
            stored = mpmath.mpf(row["value"])
            relative = abs(value - stored) / abs(value)
            assert float(relative) < 1e-24, (
                f"stored reference for {row} is off by {float(relative):.2e}")


@pytest.mark.parametrize("key", TARGET_KEYS)
def test_the_committed_stress_file_redraws_identically(key):
    """The points were drawn, not chosen -- and this is how a reader confirms it.

    Redraws **shard 0** from the declared seed and distribution and demands the
    committed 250 points back exactly. That covers the half a value-by-value
    check cannot: that nobody picked *which* points to include after seeing how
    an implementation did on them. One shard because the generator is seeded per
    shard for this purpose; the whole file is
    `python -m tools.gen_special_stress --all --check`.
    """
    pytest.importorskip("mpmath", reason="the generator needs mpmath")
    from tools import gen_special_stress as generator
    assert generator.build_shard(generator.TARGETS_BY_KEY[key], 0) == \
        SUITES[key]["shards"][0]


def test_the_baselines_are_not_strawmen_and_not_perfect_either():
    """Both halves matter, and both are properties of these specific files.

    If SciPy solved everything there would be nothing to search for; if it
    solved nothing the task would be measuring a broken call rather than a hard
    problem. This is the check that a reported gain is a gain over the state of
    the practice.
    """
    scipy_special = pytest.importorskip("scipy.special")
    calls = {
        "pbdv": lambda p: scipy_special.pbdv(p["v"], p["x"])[0],
        "hyperu": lambda p: scipy_special.hyperu(p["a"], p["b"], p["x"]),
    }
    for key in TARGET_KEYS:
        suite = special.load_suite(key)
        total = solved = 0
        digits_sum = 0.0
        for shard in range(12):
            for point, truth in zip(suite.shard_points[shard],
                                    suite.shard_truth[shard]):
                value = float(calls[key]({name: float(point[name])
                                          for name in suite.params}))
                estimate = repr(value) if math.isfinite(value) else None
                digits = special.digits_of(estimate, truth)
                digits_sum += digits
                solved += digits >= special.SOLVED_DIGITS
                total += 1
        assert total == 3000
        assert 8.0 < digits_sum / total < 12.0, (
            f"{key}: the baseline moved; rereport the numbers")
        assert 0.6 < solved / total < 0.99, (
            f"{key}: the baseline moved; rereport the numbers")


def test_the_baseline_actually_returns_nan_on_part_of_the_hyperu_range():
    """The specific defect the `hyperu` task exists to fix.

    Not an inaccurate value -- no value at all, where the function is finite.
    If SciPy ever stops doing this the task's premise has changed and the
    numbers here need restating.
    """
    scipy_special = pytest.importorskip("scipy.special")
    suite = special.load_suite("hyperu")
    nans = sum(
        1
        for shard in range(12)
        for point in suite.shard_points[shard]
        if not math.isfinite(float(scipy_special.hyperu(
            float(point["a"]), float(point["b"]), float(point["x"]))))
    )
    assert nans > 20, "scipy.special.hyperu no longer returns nan here"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_digits_are_measured_against_the_full_precision_reference():
    """The comparison happens in `decimal`, so it does not round away the answer."""
    truth = "1.000000000000000000000000"
    assert special.digits_of("1.0", truth) == special.DIGIT_CAP
    assert 8.9 < special.digits_of("1.000000001", truth) < 9.1


def test_a_failed_point_scores_zero_rather_than_raising():
    assert special.digits_of(None, "1.0") == 0.0
    assert special.digits_of("nan", "1.0") == 0.0
    assert special.digits_of("not a number", "1.0") == 0.0


def test_the_reward_is_order_preserving_with_the_metric():
    assert special.framework_score({"mean_digits": None}) == 0.0
    assert special.framework_score({"mean_digits": -math.inf}) == 0.0
    lower = special.framework_score({"mean_digits": 4.0})
    higher = special.framework_score({"mean_digits": 8.0})
    assert 0.0 <= lower < higher <= 1.0


@pytest.mark.parametrize("key", TARGET_KEYS)
def test_the_suite_splits_into_scored_and_held_back_sets(key):
    suite = special.load_suite(key, shards=8, test_shards=4)
    assert suite.test_range() == (8, 9, 10, 11)
    assert suite.entrypoint == key
    assert suite.signature().startswith(f"{key}(")


@pytest.mark.parametrize("key", TARGET_KEYS)
def test_a_suite_that_cannot_be_scored_is_refused(key):
    with pytest.raises(ValueError):
        special.load_suite(key, shards=1, test_shards=1)
    with pytest.raises(ValueError):
        special.load_suite(key, shards=11, test_shards=4)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", TARGET_KEYS)
def test_the_gate_accepts_the_baseline_and_rejects_the_obvious_accidents(key):
    target = special.TARGETS_BY_KEY[key]
    valid, reason = support.validate_source(
        target.initial_program, entrypoint=key,
        allowed_imports=special.ALLOWED_IMPORTS, literal_top_level=False)
    assert valid, reason
    valid, reason = support.validate_source(
        "import os\ndef f():\n    return 1\n", entrypoint=key,
        allowed_imports=special.ALLOWED_IMPORTS, literal_top_level=False)
    assert not valid


@pytest.mark.parametrize("module", ["mpmath", "decimal", "fractions"])
def test_arbitrary_precision_arithmetic_is_off_the_allowlist(module):
    """The deliverable is a float64 routine; this is what makes that true.

    A candidate that reimplemented the reference's own arithmetic would be
    answering a different question, and would be scored against a value
    produced the way it produced its own.
    """
    source = f"import {module}\ndef pbdv(v, x):\n    return 0.0\n"
    valid, reason = support.validate_source(
        source, entrypoint="pbdv", allowed_imports=special.ALLOWED_IMPORTS,
        literal_top_level=False)
    assert not valid and module in reason


# ---------------------------------------------------------------------------
# What the model is shown
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", TARGET_KEYS)
def test_the_prompt_states_the_distribution_and_withholds_every_answer(key):
    target = special.TARGETS_BY_KEY[key]
    suite = special.load_suite(key)
    worst = {"shard": 0, "index": 3, "digits": 0.0, "error": ""}
    worst.update({name: 1.5 for name in suite.params})
    metrics = {"mean_digits": 9.73, "solved": 158, "points": 240,
               "worst": [worst]}
    program = support.Program("id", 0, None, target.initial_program, "baseline",
                              metrics, True, "")
    prompt = special.mutation_prompt(
        program, target=target, suite=suite, preview=special.suite_preview(suite))
    assert "9.73" in prompt and "0.0 digits" in prompt
    assert suite.signature() in prompt
    for name in suite.params:
        assert SUITES[key]["distribution"][name] in prompt
    # Every reference value in the file, absent from what the model reads.
    for shard in SUITES[key]["shards"][:2]:
        for row in shard:
            assert row["value"] not in prompt
    assert "mpmath" in prompt and "NOT available" in prompt


def test_each_target_gets_its_own_prompt_and_not_a_shared_one():
    """The tasks share code; they must not share the mathematics they state."""
    prompts = {}
    for key in TARGET_KEYS:
        target = special.TARGETS_BY_KEY[key]
        suite = special.load_suite(key)
        program = support.Program("id", 0, None, target.initial_program,
                                  "baseline", {}, True, "")
        prompts[key] = special.mutation_prompt(
            program, target=target, suite=suite,
            preview=special.suite_preview(suite))
    assert prompts["pbdv"] != prompts["hyperu"]
    assert "parabolic cylinder" in prompts["pbdv"].lower()
    assert "Tricomi" in prompts["hyperu"]
    assert "scipy.special.pbdv" not in prompts["hyperu"]


# ---------------------------------------------------------------------------
# End to end, through the real sandbox
# ---------------------------------------------------------------------------


@pytest.mark.skipif(support.sandbox_backend() is None,
                    reason="no Bubblewrap or Seatbelt on this host")
@pytest.mark.parametrize("key", TARGET_KEYS)
def test_the_baseline_scores_through_the_sandbox_and_leaves_room_to_improve(key):
    pytest.importorskip("scipy", reason="the baseline program imports scipy")
    target = special.TARGETS_BY_KEY[key]
    suite = special.load_suite(key)
    valid, metrics, error = special.evaluate_source(
        target.initial_program, suite=suite, shards=(0, 1), timeout=60.0)
    assert valid, error
    assert 6.0 < metrics["mean_digits"] < special.DIGIT_CAP - 0.2
    assert 0 < metrics["solved"] < metrics["points"]
    assert metrics["score"] == metrics["mean_digits"]


@pytest.mark.skipif(support.sandbox_backend() is None,
                    reason="no Bubblewrap or Seatbelt on this host")
def test_one_failing_point_does_not_take_the_rest_of_the_shard_down():
    suite = special.load_suite("pbdv")
    source = (
        "def pbdv(v, x):\n"
        "    if v > 0:\n"
        "        raise ValueError('no')\n"
        "    return 1.0\n"
    )
    valid, metrics, error = special.evaluate_source(
        source, suite=suite, shards=(0,), timeout=60.0)
    assert valid, error
    assert metrics["points"] == 250
    assert any(row["error"] for row in metrics["worst"])


@pytest.mark.skipif(support.sandbox_backend() is None,
                    reason="no Bubblewrap or Seatbelt on this host")
def test_a_non_finite_result_is_recorded_as_a_failure_not_serialised():
    suite = special.load_suite("hyperu")
    source = "def hyperu(a, b, x):\n    return float('nan')\n"
    valid, metrics, error = special.evaluate_source(
        source, suite=suite, shards=(0,), timeout=60.0)
    assert valid, error
    assert metrics["mean_digits"] == 0.0
    assert all("non-finite" in row["error"] for row in metrics["worst"])


@pytest.mark.skipif(support.sandbox_backend() is None,
                    reason="no Bubblewrap or Seatbelt on this host")
def test_a_program_that_cannot_be_imported_fails_the_whole_evaluation():
    suite = special.load_suite("pbdv")
    valid, metrics, error = special.evaluate_source(
        "import scipy.not_a_module\ndef pbdv(v, x):\n    return 0.0\n",
        suite=suite, shards=(0,), timeout=30.0)
    assert not valid and metrics["score"] == -math.inf
    assert "Error" in error


def test_the_gate_rejects_before_any_process_is_started(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("spawned"))
    suite = special.load_suite("pbdv")
    valid, metrics, error = special.evaluate_source(
        "import os\ndef pbdv(v, x):\n    return 0.0\n",
        suite=suite, shards=(0,), timeout=5.0)
    assert not valid and error.startswith("gate:")


@pytest.mark.parametrize("key", TARGET_KEYS)
def test_the_port_runs_from_the_command_line_without_touching_anything(key):
    completed = subprocess.run(
        [sys.executable, "-m", "examples.era.era_special_precision",
         "--function", key, "--dry-run"],
        capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0
    assert f"scipy.special.{key}" in completed.stdout
    assert "dry-run" in completed.stdout


def test_the_two_functions_are_two_trees_and_not_one():
    """`--function` must change the whole task, not only a label."""
    outputs = {}
    for key in TARGET_KEYS:
        completed = subprocess.run(
            [sys.executable, "-m", "examples.era.era_special_precision",
             "--function", key, "--dry-run"],
            capture_output=True, text=True, timeout=120)
        outputs[key] = completed.stdout
    assert outputs["pbdv"] != outputs["hyperu"]
    assert special.TARGETS_BY_KEY["pbdv"].title in outputs["pbdv"]
    assert special.TARGETS_BY_KEY["hyperu"].title in outputs["hyperu"]
