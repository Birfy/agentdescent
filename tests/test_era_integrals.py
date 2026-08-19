"""The ERA numerical-integration task: references, gate, runner, evaluator.

The first block is the one that matters most. Every other test here checks that
the machinery does what it says; ``test_the_closed_form_matches_high_precision_quadrature``
checks that the *benchmark* is right, because a task scored against a wrong
reference measures nothing at all. It runs against mpmath at 30 digits, through
substitutions chosen per family so that mpmath's own quadrature is not being
asked to solve the hard problem -- these integrands defeat `mp.quad` on default
settings, which is why they are in the suite.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys

import pytest

from examples.era import _era_integrals as catalogue
from examples.era import _era_integration as integration
from examples.era import _era_support as support
from examples.era._era_integrals import (
    DIFFICULTIES,
    DIGIT_CAP,
    FAMILIES,
    Problem,
    digits_of,
    draw_shard,
    exact,
    integrand,
)


# ---------------------------------------------------------------------------
# The references
# ---------------------------------------------------------------------------


def _high_precision(problem: Problem, mpmath):
    """mpmath's value for one problem, through a family-specific substitution.

    Each branch integrates something *equivalent and smooth* rather than the
    integrand itself: `beta_endpoints` is mapped so each endpoint singularity
    becomes a finite limit, the two logarithmic families become damped
    exponentials under x = e^-t, the peak is rescaled to unit width, the
    oscillatory tails go through `quadosc`, and the heavy tail is folded onto
    [0, 1] by x -> 1/x. Handing mpmath the raw integrand instead disagrees with
    the closed form in the third decimal place on `fresnel` -- which is the
    point of the benchmark, not a defect in the reference.
    """
    mp = mpmath.mp
    mpf, quad, quadosc = mpmath.mpf, mpmath.quad, mpmath.quadosc
    exp, sin, cos, log, sqrt, atan = (mpmath.exp, mpmath.sin, mpmath.cos,
                                      mpmath.log, mpmath.sqrt, mpmath.atan)
    pi, inf = mpmath.pi, mpmath.inf
    p = problem.params
    family = problem.family

    if family == "beta_endpoints":
        a, b = mpf(p["alpha"]), mpf(p["beta"])
        left = quad(lambda t: (1 - t ** (1 / a)) ** (b - 1) / a, [0, (mpf(1) / 2) ** a])
        right = quad(lambda u: (1 - u ** (1 / b)) ** (a - 1) / b, [0, (mpf(1) / 2) ** b])
        return left + right
    if family == "log_power":
        s, n = mpf(p["s"]), int(p["n"])
        return quad(lambda t: t ** n * exp(-s * t), [0, inf])
    if family == "log_oscillation":
        s, w = mpf(p["s"]), mpf(p["omega"])
        return quadosc(lambda t: exp(-s * t) * cos(w * t), [0, inf], period=2 * pi / w)
    if family == "lorentz_peak":
        eps, x0 = mpf(p["eps"]), mpf(p["x0"])
        return quad(lambda u: 1 / (u * u + 1), [-x0 / eps, 0, (1 - x0) / eps])
    if family == "dirichlet_decay":
        decay, freq = mpf(p["p"]), mpf(p["q"])
        return quadosc(lambda x: exp(-decay * x) * sin(freq * x) / x, [0, inf],
                       period=2 * pi / freq)
    if family == "fresnel":
        scale = mpf(p["a"])
        g = ((lambda u: cos(u) / sqrt(u)) if int(p["cosine"])
             else (lambda u: sin(u) / sqrt(u)))
        head = quad(g, [0, 2 * pi])
        tail = quadosc(g, [2 * pi, inf], period=2 * pi)
        return (head + tail) / (2 * sqrt(scale))
    if family == "gamma_oscillatory":
        s, b = mpf(p["s"]), mpf(p["b"])
        f = lambda x: x ** (s - 1) * exp(-x) * cos(b * x)  # noqa: E731
        period = 2 * pi / b
        return quad(f, [0, period]) + quadosc(f, [period, inf], period=period)
    if family == "gauss_oscillatory":
        b, shift = mpf(p["b"]), mpf(p["shift"])
        return quad(lambda y: exp(-y * y) * cos(b * (y + shift)), [-inf, 0, inf])
    if family == "algebraic_tail":
        # x -> t^(1/s) on [0, 1] and x -> u^(1/(1-s)) on [1, inf) cancel *both*
        # algebraic factors exactly, leaving two bounded integrands. Folding the
        # tail without them leaves x^(-s) at the origin, and mpmath at 30 digits
        # is then a part in 1e-6 out -- which says how hard this family is, not
        # that the identity is wrong.
        s = mpf(p["s"])
        head = quad(lambda t: 1 / (1 + t ** (1 / s)), [0, 1]) / s
        tail = quad(lambda u: 1 / (1 + u ** (1 / (1 - s))), [0, 1]) / (1 - s)
        return head + tail
    raise AssertionError(f"no reference route for {family!r}")


@pytest.mark.parametrize("family", FAMILIES)
def test_the_closed_form_matches_high_precision_quadrature(family):
    # Imported here rather than at module scope: mpmath is not in `[dev]`, and a
    # module-level skip would take the gate, runner and scoring tests with it on
    # every CI run.
    mpmath = pytest.importorskip("mpmath",
                                 reason="no mpmath to check the references with")
    mpmath.mp.dps = 30
    problem = next(p for p in draw_shard(0, 0) if p.family == family)
    reference = _high_precision(problem, mpmath)
    closed = exact(problem)
    relative = abs(mpmath.mpf(closed) - reference) / abs(reference)
    assert float(relative) < 1e-13, f"{family}: closed form off by {float(relative):.2e}"


def test_the_whole_line_family_is_not_symmetric_about_the_origin():
    """Closing a hole the first live run walked straight through.

    Centred at the origin this integrand is even, and "map both halves onto
    [0, inf) and double" -- a bug -- scores full marks. The offset makes the
    two halves genuinely different, so a program has to integrate both.
    """
    problem = next(p for p in draw_shard(0, 0) if p.family == "gauss_oscillatory")
    f = integrand(problem)
    assert problem.params["shift"] != 0.0
    assert f(0.7) != pytest.approx(f(-0.7), rel=1e-6)


def test_every_family_has_a_reference_a_body_and_a_description():
    """A family that is drawn but not described is a family nobody can solve."""
    for family in FAMILIES:
        assert family in DIFFICULTIES and DIFFICULTIES[family].strip()
        problem = next(p for p in draw_shard(0, 0) if p.family == family)
        assert math.isfinite(exact(problem))
        assert abs(exact(problem)) >= catalogue.VALUE_FLOOR
        assert callable(integrand(problem))


# ---------------------------------------------------------------------------
# The integrand's contract with the candidate
# ---------------------------------------------------------------------------


def test_the_integrand_refuses_points_outside_its_interval():
    problem = Problem("lorentz_peak", {"eps": 1e-5, "x0": 0.5}, 0.0, 1.0)
    f = integrand(problem)
    assert f(0.5) == pytest.approx(1e5)
    for outside in (-1e-9, 1.0000001, math.inf, math.nan):
        with pytest.raises(ValueError):
            f(outside)


def test_a_singular_endpoint_is_infinite_or_refused_but_never_a_number():
    """Both are honest; a finite value there would not be."""
    beta = integrand(Problem("beta_endpoints", {"alpha": 0.2, "beta": 0.3}, 0.0, 1.0))
    assert beta(0.0) == math.inf and beta(1.0) == math.inf
    oscillating = integrand(
        Problem("log_oscillation", {"s": 0.5, "omega": 10.0}, 0.0, 1.0))
    with pytest.raises(ValueError):
        oscillating(0.0)


def test_the_removable_singularity_is_resolved_rather_than_left_to_divide_by_zero():
    f = integrand(Problem("dirichlet_decay", {"p": 0.01, "q": 1.5}, 0.0, math.inf))
    assert f(0.0) == pytest.approx(1.5)
    assert f(1e-12) == pytest.approx(1.5, rel=1e-9)


def test_a_problem_round_trips_through_json_including_infinite_limits():
    for problem in draw_shard(3, 1):
        restored = Problem.from_dict(json.loads(json.dumps(problem.to_dict())))
        assert restored == problem
        assert integrand(restored)(0.5) == integrand(problem)(0.5)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_digits_are_correct_significant_digits_and_saturate_at_the_cap():
    assert digits_of(1.0, 1.0) == DIGIT_CAP
    assert digits_of(1.000001, 1.0) == pytest.approx(6.0, abs=1e-6)
    assert digits_of(2.0, 1.0) == 0.0
    assert digits_of(-1.0, 1.0) == 0.0
    assert digits_of(1e-30, 1.0) == 0.0


def test_a_failed_problem_scores_zero_rather_than_raising():
    for value in (None, float("nan"), float("inf"), "not a number", [1.0]):
        assert digits_of(value, 3.0) == 0.0


def test_the_reward_is_order_preserving_with_the_metric():
    low = integration.framework_score({"mean_digits": 3.0})
    high = integration.framework_score({"mean_digits": 9.0})
    assert 0.0 < low < high <= 1.0
    assert integration.framework_score({"mean_digits": None}) == 0.0
    assert integration.framework_score(
        {"mean_digits": float("-inf")}) == 0.0


# ---------------------------------------------------------------------------
# The suite
# ---------------------------------------------------------------------------


def test_a_shard_holds_one_instance_of_every_family_and_redraws_are_identical():
    first, second = draw_shard(0, 2), draw_shard(0, 2)
    assert [p.family for p in first] == [p.family for p in second]
    assert sorted(p.family for p in first) == sorted(FAMILIES)
    assert first == second


def test_shards_do_not_share_instances_and_do_not_share_an_order():
    orders = {tuple(p.family for p in draw_shard(0, index)) for index in range(12)}
    assert len(orders) > 1, "position within a shard is the family, and it leaks"
    first = {(p.family, tuple(sorted(p.params.items()))) for p in draw_shard(0, 0)}
    second = {(p.family, tuple(sorted(p.params.items()))) for p in draw_shard(0, 1)}
    assert not first & second


def test_the_suite_writes_one_problem_file_per_shard(tmp_path, monkeypatch):
    monkeypatch.setattr(integration, "cache_path",
                        lambda subdir, name: str(tmp_path / name))
    suite = integration.prepare_suite(seed=1, shards=3, test_shards=2)
    assert len(suite.shard_paths) == 5
    assert suite.test_range() == (3, 4)
    for path, problems in zip(suite.shard_paths, suite.shard_problems):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert [Problem.from_dict(row) for row in payload] == list(problems)
        # The file the sandbox reads must not carry the answers.
        assert "exact" not in path.read_text(encoding="utf-8")


def test_a_suite_that_cannot_be_scored_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(integration, "cache_path",
                        lambda subdir, name: str(tmp_path / name))
    with pytest.raises(ValueError):
        integration.prepare_suite(seed=0, shards=1, test_shards=1)
    with pytest.raises(ValueError):
        integration.prepare_suite(seed=0, shards=4, test_shards=0)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def _gate(source: str):
    return support.validate_source(
        source, entrypoint="integrate",
        allowed_imports=integration.ALLOWED_IMPORTS, literal_top_level=False)


def test_the_gate_accepts_the_baseline_and_rejects_the_obvious_accidents():
    assert _gate(integration.INITIAL_PROGRAM)[0]
    assert not _gate("def other(f, a, b):\n    return 0.0\n")[0]
    assert not _gate("import os\ndef integrate(f, a, b):\n    return 0.0\n")[0]
    assert not _gate("import pandas\ndef integrate(f, a, b):\n    return 0.0\n")[0]
    assert not _gate(
        "def integrate(f, a, b):\n    return eval('1')\n")[0]
    assert not _gate(
        "def integrate(f, a, b):\n    return f.__globals__\n")[0]


def test_a_computed_node_table_is_allowed_here_and_refused_by_the_tabular_gate():
    """The one deliberate loosening, and the proof it is scoped to this task."""
    source = ("import numpy as np\n"
              "NODES, WEIGHTS = np.polynomial.legendre.leggauss(64)\n"
              "def integrate(f, a, b):\n    return 0.0\n")
    assert _gate(source)[0]
    assert not support.validate_source(source)[0]


# ---------------------------------------------------------------------------
# The runner, without a sandbox
# ---------------------------------------------------------------------------


runner = pytest.importorskip("examples.era._era_integration_runner")


class _Catalogue:
    """The two names `solve` uses, so the runner can be exercised in-process."""

    Problem = Problem

    @staticmethod
    def integrand(problem):
        return integrand(problem)


def _one_problem():
    return [Problem("lorentz_peak", {"eps": 1e-3, "x0": 0.5}, 0.0, 1.0).to_dict()]


def test_the_runner_stops_a_candidate_that_overruns_its_evaluation_budget():
    def greedy(f, a, b):
        total = 0.0
        for index in range(10_000):
            total += f((index + 0.5) / 10_000)
        return total / 10_000

    results = runner.solve(greedy, _Catalogue, _one_problem(), budget=100, seconds=30.0)
    assert results[0]["estimate"] is None
    assert "BudgetExceeded" in results[0]["error"]
    assert results[0]["calls"] == 101


def test_one_failing_problem_does_not_take_the_rest_of_the_shard_down():
    calls = {"n": 0}

    def flaky(f, a, b):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ZeroDivisionError("boom")
        return float(f(0.5))

    problems = _one_problem() * 3
    results = runner.solve(flaky, _Catalogue, problems, budget=10, seconds=30.0)
    assert results[0]["estimate"] is None and "ZeroDivisionError" in results[0]["error"]
    assert results[1]["estimate"] is not None and results[2]["estimate"] is not None


def test_a_non_finite_estimate_is_recorded_as_a_failure_not_serialised():
    results = runner.solve(lambda f, a, b: float("nan"), _Catalogue, _one_problem(),
                           budget=10, seconds=30.0)
    assert results[0]["estimate"] is None
    assert "non-finite" in results[0]["error"]
    json.dumps(results, allow_nan=False)  # the payload has to survive this


def test_the_candidate_cannot_read_the_problem_specification_off_its_integrand():
    """It is handed a callable; the family and its parameters stay on this side."""
    seen = {}

    def peeking(f, a, b):
        seen["attributes"] = [name for name in dir(f) if not name.startswith("_")]
        return 0.0

    runner.solve(peeking, _Catalogue, _one_problem(), budget=10, seconds=30.0)
    assert "family" not in seen["attributes"]
    assert "params" not in seen["attributes"]


# ---------------------------------------------------------------------------
# What the model is shown
# ---------------------------------------------------------------------------


def test_the_prompt_describes_the_difficulties_and_names_no_family(tmp_path, monkeypatch):
    monkeypatch.setattr(integration, "cache_path",
                        lambda subdir, name: str(tmp_path / name))
    suite = integration.prepare_suite(seed=0, shards=2, test_shards=1)
    metrics = {
        "mean_digits": 5.5, "solved": 3, "problems": 9,
        "worst": [{"shard": 0, "index": 4, "interval": "[0, 1]", "digits": 0.0,
                   "calls": 12, "error": ""}],
    }
    program = support.Program("id", 0, None, integration.INITIAL_PROGRAM,
                              "baseline", metrics, True, "")
    prompt = integration.mutation_prompt(program, preview=integration.suite_preview(suite))
    for difficulty in DIFFICULTIES.values():
        assert difficulty.split(" -- ")[0].split(",")[0] in prompt
    for family in FAMILIES:
        assert family not in prompt, f"the prompt names {family}"
    assert "5.5" in prompt and "0.0 digits" in prompt


@pytest.mark.skipif(support.sandbox_backend() is None,
                    reason="no Bubblewrap or Seatbelt on this host")
def test_the_failure_report_never_carries_a_family_name(tmp_path, monkeypatch):
    monkeypatch.setattr(integration, "cache_path",
                        lambda subdir, name: str(tmp_path / name))
    suite = integration.prepare_suite(seed=0, shards=2, test_shards=1)
    valid, metrics, error = integration.evaluate_source(
        "def integrate(f, a, b):\n    return 1.0\n",
        suite=suite, shards=(0,), timeout=20.0)
    assert valid, error
    for row in metrics["worst"]:
        assert set(row) == {"shard", "index", "interval", "digits", "calls", "error"}


# ---------------------------------------------------------------------------
# End to end, through the real sandbox
# ---------------------------------------------------------------------------


@pytest.mark.skipif(support.sandbox_backend() is None,
                    reason="no Bubblewrap or Seatbelt on this host")
def test_the_baseline_scores_through_the_sandbox_and_leaves_room_to_improve(tmp_path,
                                                                           monkeypatch):
    pytest.importorskip("scipy", reason="the baseline program imports scipy")
    monkeypatch.setattr(integration, "cache_path",
                        lambda subdir, name: str(tmp_path / name))
    suite = integration.prepare_suite(seed=0, shards=2, test_shards=1)
    valid, metrics, error = integration.evaluate_source(
        integration.INITIAL_PROGRAM, suite=suite, shards=(0,), timeout=60.0)
    assert valid, error
    # `quad` is a real baseline: it solves several of the nine outright. It also
    # fails outright on others, and a suite it swept would have nothing to search.
    assert 3.0 < metrics["mean_digits"] < DIGIT_CAP - 0.5
    assert 0 < metrics["solved"] < metrics["problems"]
    assert metrics["score"] == metrics["mean_digits"]


@pytest.mark.skipif(support.sandbox_backend() is None,
                    reason="no Bubblewrap or Seatbelt on this host")
def test_a_program_the_gate_admits_but_the_import_breaks_fails_the_whole_evaluation(
        tmp_path, monkeypatch):
    monkeypatch.setattr(integration, "cache_path",
                        lambda subdir, name: str(tmp_path / name))
    suite = integration.prepare_suite(seed=0, shards=2, test_shards=1)
    valid, metrics, error = integration.evaluate_source(
        "import scipy.not_a_module\ndef integrate(f, a, b):\n    return 0.0\n",
        suite=suite, shards=(0,), timeout=30.0)
    assert not valid and metrics["score"] == -math.inf
    assert "Error" in error


def test_the_gate_rejects_before_any_process_is_started(tmp_path, monkeypatch):
    monkeypatch.setattr(integration, "cache_path",
                        lambda subdir, name: str(tmp_path / name))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("spawned"))
    suite = integration.prepare_suite(seed=0, shards=2, test_shards=1)
    valid, metrics, error = integration.evaluate_source(
        "import os\ndef integrate(f, a, b):\n    return 0.0\n",
        suite=suite, shards=(0,), timeout=5.0)
    assert not valid and error.startswith("gate:")


def test_the_port_runs_from_the_command_line_without_touching_anything():
    completed = subprocess.run(
        [sys.executable, "-m", "examples.era.era_hard_integrals", "--dry-run"],
        capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0
    assert "numerical solution of integrals" in completed.stdout
    assert "dry-run" in completed.stdout
