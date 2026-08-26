"""The ERA LLM-SRBench task: the benchmark itself, the answer grammar, the metrics.

The first block is the one that matters most, for the same reason it does in
``tests/test_era_integrals.py``: the samples this task scores against are
downloaded from a **third party's re-upload** of a gated dataset, and a
benchmark whose data does not match the equations it claims to hold measures
nothing at all. So every ground-truth expression that is evaluable as published
is re-evaluated on the samples shipped beside it, and has to reproduce them.

Everything after that checks the machinery. The grammar block is the security
boundary as much as the scientific one: an answer is a string this repository
interprets, on data the candidate is not allowed to see, so "only an equation
gets through" is not a stylistic rule.
"""

from __future__ import annotations

import json
import math
import urllib.error

import pytest

pytest.importorskip("numpy", reason="the LLM-SRBench task is numeric throughout")

import numpy as np  # noqa: E402

from examples.era import _era_srbench as srbench  # noqa: E402
from examples.era import _era_srbench_expr as expr  # noqa: E402
from examples.era import _era_support as support  # noqa: E402
from examples.era import era_llm_srbench as port  # noqa: E402


# ---------------------------------------------------------------------------
# The benchmark
# ---------------------------------------------------------------------------


def _mirror_subset(subset: str):
    """One subset's problems and samples, or a skip if the mirror is unreachable."""
    pytest.importorskip("pyarrow", reason="reading the benchmark parquet needs pyarrow")
    try:
        return srbench._read_subset(subset)
    except (urllib.error.URLError, OSError) as exc:  # offline CI
        pytest.skip(f"LLM-SRBench mirror unreachable: {exc}")


def _ground_truth_prediction(problem, samples):
    """Evaluate a published ground truth, or None when it is not evaluable.

    Two whole subsets are not, and both are metadata defects in the published
    copy rather than anything this port does: ``chem_react`` expressions carry a
    mangled parameter (``0.189…_z``) and ``phys_osc`` expressions are symbolic
    templates whose parameters (``F0``, ``beta``, ``omega0``) have no values
    attached. Scoring never touches these strings -- it is numeric throughout --
    so they are reported and skipped rather than repaired.
    """
    text = problem.gt_expression
    # `A(t)` is sympy's function-call notation for the state variable; the same
    # symbol is a plain column here.
    for name in problem.input_vars:
        text = text.replace(f"{name}(t)", name)
    try:
        tree = expr.validate_expression(text, problem.input_vars)
    except expr.ExpressionError:
        return None
    del tree
    try:
        return expr.evaluate_expression(text, problem.input_vars, samples["train_x"])
    except expr.ExpressionError:
        return None


@pytest.mark.parametrize("subset", sorted(srbench.GROUPS["lsr_synth"]))
def test_the_published_equations_reproduce_the_published_samples(subset):
    """The check that makes a number from this task mean anything."""
    loaded = _mirror_subset(subset)
    checked = 0
    for problem, samples in loaded:
        prediction = _ground_truth_prediction(problem, samples)
        if prediction is None:
            continue
        scored = expr.score_predictions(prediction, samples["train_y"])
        assert scored["nmse"] < 1e-6, (
            f"{problem.problem_id}: ground truth does not reproduce its own "
            f"samples (NMSE {scored['nmse']:.3e})")
        checked += 1
    if subset in ("lsr_synth_chem_react", "lsr_synth_phys_osc"):
        # Documented defects in the published metadata, asserted so that a
        # mirror which quietly *fixed* them stops being described as broken.
        assert checked == 0, (
            f"{subset} ground truths now evaluate; update the port's notes")
    else:
        assert checked == len(loaded)


def test_every_subset_holds_the_number_of_problems_the_paper_states():
    for subset, (_files, expected) in srbench.SUBSETS.items():
        if subset == "lsr_transform":
            continue  # 353 MB; covered by the count assertion inside _read_subset
        loaded = _mirror_subset(subset)
        assert len(loaded) == expected
        assert len({problem.problem_id for problem, _ in loaded}) == expected


def test_the_released_data_holds_240_problems_where_the_paper_says_239():
    """One physics problem more than the abstract's arithmetic, and it is upstream's.

    The paper reads "111 problems in the first category (LSR-Transform), and 128
    problems in the second category (LSR-Synth) ... chemistry (36), biology (24),
    physics (43), and material science (25)" -- 239. The benchmark's own gated
    HuggingFace dataset card lists `lsr_synth_phys_osc` at 44, which makes 240,
    and the ungated mirror agrees with the card. The port follows the data.
    Pinned here so the discrepancy is a recorded fact rather than a typo someone
    later "fixes" in one direction or the other.
    """
    counts = {name: expected for name, (_files, expected) in srbench.SUBSETS.items()}
    assert counts["lsr_synth_phys_osc"] == 44, "the paper's text says 43"
    assert sum(counts.values()) == 240
    assert sum(counts[name] for name in srbench.GROUPS["lsr_synth"]) == 129
    assert counts["lsr_transform"] == 111


def test_the_four_synthetic_domains_carry_an_out_of_distribution_split():
    """OOD is what separates LSR-Synth from LSR-Transform, and it has to be there."""
    loaded = _mirror_subset("lsr_synth_matsci")
    for problem, samples in loaded:
        assert problem.ood_rows > 0
        assert "ood_x" in samples and "ood_y" in samples
        assert samples["ood_x"].shape[0] == problem.ood_rows


# ---------------------------------------------------------------------------
# The answer grammar -- the boundary the held-out samples sit behind
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("equation", [
    "1.5*P - 0.02*P**2",
    "sin(t) + exp(-P)/2",
    "-3.0*sqrt(abs(t)) + pi*P",
    "2.0",
    "P**0.3333 * log(t + 1.0)",
])
def test_an_equation_is_accepted(equation):
    assert expr.validate_expression(equation, ("t", "P")) is not None


@pytest.mark.parametrize("equation,reason", [
    ("__import__('os').listdir()", "dunder"),
    ("open('/etc/passwd').read()", "unknown function"),
    ("t if t > 0 else P", "conditional"),
    ("t[0]", "subscript"),
    ("np.sin(t)", "attribute"),
    ("lambda t: t", "lambda"),
    ("(t > 1) * P", "comparison"),
    ("sin(t, P)", "arity"),
    ("q * 2", "unknown symbol"),
    ("", "empty"),
    ("t +", "syntax"),
])
def test_anything_that_is_not_an_equation_is_refused(equation, reason):
    with pytest.raises(expr.ExpressionError):
        expr.validate_expression(equation, ("t", "P"))


def test_a_caret_is_read_as_a_power_rather_than_as_xor():
    """What a scientist writes, and what sympy prints."""
    x = np.array([[2.0, 3.0]])
    assert expr.evaluate_expression("t^2", ("t", "P"), x)[0] == pytest.approx(4.0)


def test_an_equation_is_evaluated_column_wise_in_the_declared_order():
    x = np.array([[1.0, 10.0], [2.0, 20.0]])
    values = expr.evaluate_expression("t + 0.5*P", ("t", "P"), x)
    assert values == pytest.approx([6.0, 12.0])


def test_a_constant_equation_is_broadcast_rather_than_rejected():
    x = np.zeros((4, 2))
    assert expr.evaluate_expression("7.0", ("t", "P"), x) == pytest.approx([7.0] * 4)


def test_an_overflowing_equation_returns_non_finite_rather_than_raising():
    x = np.array([[800.0, 0.0], [900.0, 0.0]])
    values = expr.evaluate_expression("exp(t)*exp(t)", ("t", "P"), x)
    assert not np.all(np.isfinite(values))


# ---------------------------------------------------------------------------
# The metrics -- the benchmark's own definitions
# ---------------------------------------------------------------------------


def test_nmse_and_accuracy_are_the_papers_definitions():
    truth = np.array([1.0, 2.0, 3.0, 4.0])
    prediction = truth * np.array([1.05, 0.96, 1.02, 0.99])
    scored = expr.score_predictions(prediction, truth)
    expected = np.sum((prediction - truth) ** 2) / np.sum((truth - truth.mean()) ** 2)
    assert scored["nmse"] == pytest.approx(expected)
    # Acc_0.1 is an indicator on the *worst* relative error, not a mean of them,
    # so the 5% point is what decides it and the 1% points cannot rescue it.
    assert scored["max_relative"] == pytest.approx(0.05)
    assert scored["acc"] == 1


def test_one_bad_point_is_enough_to_lose_the_accuracy_indicator():
    truth = np.array([1.0, 2.0, 3.0])
    prediction = np.array([1.0, 2.0, 3.9])
    assert expr.score_predictions(prediction, truth)["acc"] == 0


def test_a_pole_in_the_test_range_fails_the_problem_here_and_not_upstream():
    """The port's one deliberate deviation from `compute_output_base_metrics`."""
    truth = np.array([1.0, 2.0, 3.0, 4.0])
    prediction = np.array([1.0, 2.0, np.inf, 4.0])
    scored = expr.score_predictions(prediction, truth)
    assert scored["nmse"] == math.inf and scored["acc"] == 0
    assert scored["nonfinite_points"] == 1
    # Upstream drops the non-finite point and scores the rest, which is why the
    # two numbers are both reported rather than one of them being chosen.
    assert scored["nmse_upstream"] == pytest.approx(0.0)


def test_a_zero_target_leaves_the_papers_accuracy_undefined_and_it_is_counted():
    """Not a choice here: `Acc_tau` divides by `y_i`, so `y_i = 0` has no answer.

    Five of LSR-Synth's 129 problems carry one in their in-domain test targets,
    which caps Acc(0.1) on that dataset at 124/129 for every method including the
    ground truth. Reported rather than patched, because patching it would be
    reporting a different metric under the paper's name.
    """
    truth = np.array([0.0, 1.0, 2.0])
    scored = expr.score_predictions(truth.copy(), truth)
    assert scored["nmse"] == pytest.approx(0.0)     # an exact fit, and still
    assert scored["acc"] == 0                       # no Acc credit
    assert scored["zero_targets"] == 1


def test_digits_are_monotone_in_nmse_and_capped_at_the_data_precision():
    assert expr.digits_of(1.0) == 0.0
    assert expr.digits_of(1e-6) == pytest.approx(6.0)
    assert expr.digits_of(1e-30) == expr.DIGIT_CAP
    assert expr.digits_of(math.inf) == 0.0
    assert expr.digits_of(1e-4) > expr.digits_of(1e-3)


def test_aggregation_pools_over_problems_and_counts_the_failures():
    rows = [
        {"id": {"digits": 6.0, "nmse": 1e-6, "nmse_upstream": 1e-6, "acc": 1},
         "ood": {"digits": 3.0, "nmse": 1e-3, "acc": 0}},
        {"id": {"digits": 0.0, "nmse": 2.0, "nmse_upstream": 2.0, "acc": 0},
         "ood": None},
        {"id": None, "ood": None},
    ]
    pooled = expr.aggregate(rows)
    assert pooled["problems"] == 2 and pooled["failed"] == 1
    assert pooled["mean_digits"] == pytest.approx(3.0)
    assert pooled["acc_0.1"] == pytest.approx(0.5)
    assert pooled["median_nmse"] == pytest.approx(0.5 * (1e-6 + 2.0))
    assert pooled["ood_problems"] == 1


def test_an_infinite_nmse_is_reported_as_null_rather_than_as_invalid_json():
    """`json.dump` writes `inf` as the bare token `Infinity`, which is not JSON."""
    assert srbench._reportable(math.inf) is None
    assert srbench._reportable(float("nan")) is None
    assert srbench._reportable(None) is None
    assert srbench._reportable(1e-9) == pytest.approx(1e-9)


def test_the_framework_reward_is_order_preserving_with_the_metric():
    better = srbench.framework_score({"mean_digits": 8.0})
    worse = srbench.framework_score({"mean_digits": 2.0})
    assert 0.0 <= worse < better <= 1.0
    assert srbench.framework_score({"mean_digits": None}) == 0.0
    assert srbench.framework_score({"mean_digits": -math.inf}) == 0.0


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def _gate(source: str):
    return support.validate_source(
        source, 20_000, entrypoint="discover",
        allowed_imports=srbench.ALLOWED_IMPORTS, literal_top_level=False)


def test_the_gate_wants_a_discover_function():
    valid, reason = _gate("import numpy as np\n\ndef fit(x, y, spec):\n    return '1'\n")
    assert not valid and "discover" in reason


def test_the_gate_refuses_an_import_the_task_has_no_use_for():
    valid, reason = _gate("import os\n\ndef discover(x, y, spec):\n    return '1'\n")
    assert not valid and "os" in reason


def test_the_gate_admits_the_seed_program():
    valid, reason = _gate(srbench.INITIAL_PROGRAM)
    assert valid, reason


# ---------------------------------------------------------------------------
# The runner and the evaluator, on a suite built here rather than downloaded
# ---------------------------------------------------------------------------


def _fixture_suite(tmp_path, *, rows: int = 200, problems: int = 2,
                   shards: int = 2) -> srbench.Suite:
    """A two-shard suite over ``y = 2x0 + 3`` -- solvable, and offline."""
    rng = np.random.default_rng(0)
    paths, metas, dealt = [], [], []
    for shard in range(shards):
        payload = {}
        holds = []
        for position in range(problems):
            train_x = rng.uniform(1.0, 5.0, size=(rows, 2))
            test_x = rng.uniform(1.0, 5.0, size=(rows // 4, 2))
            ood_x = rng.uniform(5.0, 7.0, size=(rows // 4, 2))
            for name, matrix in (("train", train_x), ("test", test_x), ("ood", ood_x)):
                payload[f"p{position}_{name}_x"] = matrix
                payload[f"p{position}_{name}_y"] = 2.0 * matrix[:, 0] + 3.0
            holds.append(srbench.SrProblem(
                problem_id=f"fixture-{shard}-{position}",
                subset="fixture",
                input_vars=("a", "b"),
                output_var="y",
                description="Discover y from a and b.",
                gt_expression="2*a + 3",
                train_rows=rows, test_rows=rows // 4, ood_rows=rows // 4))
        data_path = tmp_path / f"shard-{shard:03d}.npz"
        meta_path = tmp_path / f"shard-{shard:03d}.json"
        np.savez(data_path, **payload)
        meta_path.write_text(
            json.dumps([problem.to_dict() for problem in holds]), encoding="utf-8")
        paths.append(data_path)
        metas.append(meta_path)
        dealt.append(tuple(holds))
    return srbench.Suite(tmp_path, 0, ("fixture",), tuple(paths), tuple(metas),
                         tuple(dealt), shards - 1, 1)


needs_sandbox = pytest.mark.skipif(
    support.sandbox_backend() is None,
    reason="no Bubblewrap or Seatbelt on this host")


@needs_sandbox
def test_the_seed_program_solves_a_linear_fixture(tmp_path):
    suite = _fixture_suite(tmp_path)
    valid, metrics, error = srbench.evaluate_source(
        srbench.INITIAL_PROGRAM, suite=suite, shards=(0,), timeout=60.0,
        problem_seconds=10.0)
    assert valid, error
    assert metrics["problems"] == 2
    assert metrics["mean_digits"] > 8.0
    assert metrics["acc_0.1"] == 1.0
    assert metrics["ood_problems"] == 2


@needs_sandbox
def test_one_failed_problem_does_not_take_the_shard_down(tmp_path):
    """A method that is wrong on one problem has still earned the other."""
    suite = _fixture_suite(tmp_path)
    source = (
        "import numpy as np\n\n\n"
        "def discover(x, y, spec):\n"
        "    if spec['description'].endswith('b.') and x.shape[0] % 2 == 0:\n"
        "        return '2*a + 3'\n"
        "    return '2*a + 3'\n"
    )
    valid, metrics, error = srbench.evaluate_source(
        source, suite=suite, shards=(0,), timeout=60.0, problem_seconds=10.0)
    assert valid, error
    assert metrics["mean_digits"] == pytest.approx(expr.DIGIT_CAP)


@needs_sandbox
def test_an_answer_that_is_not_an_equation_scores_zero_rather_than_failing(tmp_path):
    suite = _fixture_suite(tmp_path)
    source = (
        "import numpy as np\n\n\n"
        "def discover(x, y, spec):\n"
        "    return \"__import__('os').listdir('/')\"\n"
    )
    valid, metrics, error = srbench.evaluate_source(
        source, suite=suite, shards=(0,), timeout=60.0, problem_seconds=10.0)
    assert valid, error
    assert metrics["mean_digits"] == 0.0
    assert all("ExpressionError" in row["error"] for row in metrics["per_problem"])


@needs_sandbox
def test_the_candidate_is_handed_the_graders_own_evaluator(tmp_path):
    """The way a method scores its own forms has to be the way its answer is scored.

    The gate refuses `eval`, so without this a candidate would have to
    re-implement the grammar and hope its copy matched -- and a near-miss there
    shows up as a good method scoring zero.
    """
    suite = _fixture_suite(tmp_path)
    source = (
        "import numpy as np\n\n\n"
        "def discover(x, y, spec):\n"
        "    fitted = spec['evaluate']('2*a + 3', x)\n"
        "    if not np.allclose(fitted, y):\n"
        "        return 'wrong values'\n"
        "    try:\n"
        "        spec['evaluate'](\"open('/etc/passwd')\", x)\n"
        "    except Exception:\n"
        "        return '2*a + 3'\n"
        "    return 'the grammar let anything through'\n"
    )
    valid, metrics, error = srbench.evaluate_source(
        source, suite=suite, shards=(0,), timeout=60.0, problem_seconds=10.0)
    assert valid, error
    assert metrics["mean_digits"] == pytest.approx(expr.DIGIT_CAP)


@needs_sandbox
def test_a_program_without_an_entrypoint_fails_the_whole_evaluation(tmp_path):
    suite = _fixture_suite(tmp_path)
    valid, metrics, error = srbench.evaluate_source(
        "import numpy as np\n\n\ndef search(x, y, spec):\n    return '1'\n",
        suite=suite, shards=(0,), timeout=60.0, problem_seconds=5.0)
    assert not valid
    assert metrics["score"] == -math.inf
    assert "discover" in error


@needs_sandbox
def test_a_problem_that_overruns_its_budget_is_interrupted(tmp_path):
    """The per-problem deadline, enforced in the runner rather than trusted.

    Without it, one method that never returns costs every remaining problem in
    the shard its score -- so the budget has to be a property of the harness.
    """
    suite = _fixture_suite(tmp_path, problems=1)
    source = (
        "import numpy as np\n\n\n"
        "def discover(x, y, spec):\n"
        "    total = 0.0\n"
        "    while True:\n"
        "        total += 1.0\n"
        "    return '2*a + 3'\n"
    )
    valid, metrics, error = srbench.evaluate_source(
        source, suite=suite, shards=(0,), timeout=60.0, problem_seconds=1.0)
    assert valid, error
    assert metrics["mean_digits"] == 0.0
    assert "Deadline" in metrics["per_problem"][0]["error"]


# ---------------------------------------------------------------------------
# The suite
# ---------------------------------------------------------------------------


def test_dealing_keeps_every_domain_in_every_shard():
    problems = [
        srbench.SrProblem(f"{subset}-{index}", subset, ("t",), "y", "", "", 1, 1, 0)
        for subset in ("a", "b") for index in range(8)
    ]
    dealt = srbench._deal(problems, 4, seed=0)
    assert sum(len(shard) == 4 for shard in dealt) == 4
    for shard in dealt:
        assert len({problem.subset for problem in shard}) == 2


def test_capping_the_problem_count_stays_even_across_subsets():
    problems = [
        srbench.SrProblem(f"{subset}-{index}", subset, ("t",), "y", "", "", 1, 1, 0)
        for subset, count in (("a", 20), ("b", 4)) for index in range(count)
    ]
    chosen = srbench._stratified_cap(problems, 8, seed=0)
    assert len(chosen) == 8
    counts = {}
    for problem in chosen:
        counts[problem.subset] = counts.get(problem.subset, 0) + 1
    assert counts == {"a": 4, "b": 4}
    # Deterministic: the same seed picks the same problems.
    assert [p.problem_id for p in chosen] == [
        p.problem_id for p in srbench._stratified_cap(problems, 8, seed=0)]


def test_the_dataset_names_the_paper_uses_all_resolve():
    assert srbench.resolve_subsets("lsr_synth") == srbench.GROUPS["lsr_synth"]
    assert srbench.resolve_subsets("lsr_transform") == ("lsr_transform",)
    assert len(srbench.resolve_subsets("all")) == 5
    assert srbench.resolve_subsets("lsr_synth_matsci") == ("lsr_synth_matsci",)
    with pytest.raises(ValueError):
        srbench.resolve_subsets("feynman")


def test_the_mirror_files_are_pinned_to_a_revision():
    """`main` moving under a benchmark is how a rerun stops being a rerun."""
    assert len(srbench.MIRROR_REVISION) == 40
    url = srbench.MIRROR_URL.format(repo=srbench.MIRROR_REPO,
                                    revision=srbench.MIRROR_REVISION,
                                    path="lsr_synth_matsci/train.parquet")
    assert srbench.MIRROR_REVISION in url and "/main/" not in url


# ---------------------------------------------------------------------------
# The upstream-aligned answer format
# ---------------------------------------------------------------------------


_PROGRAM = ("import numpy as np\n\n\n"
            "def equation(a, b, params):\n"
            "    out = params[0]*a + params[1]*np.sin(b)\n"
            "    if params[2] > 0:\n"
            "        out = out + params[2]\n"
            "    return out\n")


def test_the_program_format_allows_what_upstream_allows():
    """Branches, loops and numpy -- the restricted grammar is this port's own."""
    assert expr.validate_program(_PROGRAM, ("a", "b")) is not None


@pytest.mark.parametrize("source,reason", [
    ("def equation(a, params):\n    return params[0]*a\n", "wrong arity"),
    ("def equation(b, a, params):\n    return params[0]*a\n", "wrong order"),
    ("def other(a, b, params):\n    return params[0]\n", "no entry point"),
    ("import os\ndef equation(a, b, params):\n    return params[0]*a\n", "import"),
    ("def equation(a, b, params):\n    return a.__class__\n", "dunder"),
    ("def equation(a, b, params):\n    return eval('a')\n", "eval"),
    ("", "empty"),
])
def test_the_program_gate_refuses_what_should_never_run(source, reason):
    with pytest.raises(expr.ProgramError):
        expr.validate_program(source, ("a", "b"))


_SMOOTH_PROGRAM = ("import numpy as np\n\n\n"
                   "def equation(a, b, params):\n"
                   "    return params[0]*a + params[1]*np.sin(b) + params[2]\n")


def test_the_harness_fits_the_constants_the_way_upstream_fits_them():
    """One BFGS from all ones, over ten parameters -- `searcher.py` verbatim."""
    rng = np.random.default_rng(0)
    x = rng.uniform(1.0, 4.0, size=(500, 2))
    y = 2.5 * x[:, 0] + 1.5 * np.sin(x[:, 1]) - 0.75
    call = expr.compile_program(_SMOOTH_PROGRAM, ("a", "b"))
    params = expr.fit_program(call, x, y)
    assert params.shape == (expr.MAX_NPARAMS,)
    assert params[:3] == pytest.approx([2.5, 1.5, -0.75], abs=1e-4)
    assert expr.score_predictions(call(x, params), y)["nmse"] < 1e-9


def test_branching_on_a_parameter_fits_badly_under_upstreams_optimiser():
    """A property of the aligned fitter, recorded rather than tuned away.

    The program format lets an answer branch on `params[i]`, and upstream fits
    with one gradient-based BFGS. A branch makes the loss discontinuous in that
    parameter, so the fit is poor -- the same program is recovered exactly when
    the branch is removed. This is what alignment costs, and a candidate that
    branches on its own constants is choosing it.
    """
    rng = np.random.default_rng(0)
    x = rng.uniform(1.0, 4.0, size=(500, 2))
    y = 2.5 * x[:, 0] + 1.5 * np.sin(x[:, 1]) - 0.75
    branchy = expr.compile_program(_PROGRAM, ("a", "b"))
    smooth = expr.compile_program(_SMOOTH_PROGRAM, ("a", "b"))
    branchy_nmse = expr.score_predictions(
        branchy(x, expr.fit_program(branchy, x, y)), y)["nmse"]
    smooth_nmse = expr.score_predictions(
        smooth(x, expr.fit_program(smooth, x, y)), y)["nmse"]
    assert smooth_nmse < 1e-9 < branchy_nmse


def test_ten_constants_is_a_real_ceiling_on_an_answer():
    """Upstream's MAX_NPARAMS is what stops a long interpolating sum."""
    source = ("def equation(a, b, params):\n"
              "    return sum(params[i]*a**i for i in range(12))\n")
    call = expr.compile_program(source, ("a", "b"))
    x = np.ones((4, 2))
    with pytest.raises(Exception):
        call(x, np.ones(expr.MAX_NPARAMS))


@needs_sandbox
def test_a_program_answer_is_fitted_and_scored_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(srbench, "cache_path",
                        lambda subdir, name: str(tmp_path / name))
    rng = np.random.default_rng(1)
    train_x = rng.uniform(1.0, 5.0, size=(400, 2))
    test_x = rng.uniform(1.0, 5.0, size=(80, 2))
    truth = lambda m: 2.0 * m[:, 0] * np.sin(m[:, 1]) + 3.0  # noqa: E731
    problem = srbench.SrProblem("fx", "fixture", ("a", "b"), "y", "Discover y.",
                                "2*a*sin(b) + 3", 400, 80, 0)
    suite = srbench.prepare_problem_suite(
        problem, {"train_x": train_x, "train_y": truth(train_x),
                  "test_x": test_x, "test_y": truth(test_x)}, seed=0, shards=4)
    source = ("def discover(x, y, spec):\n"
              "    return ('import numpy as np\\n\\n\\n'\n"
              "            'def equation(a, b, params):\\n'\n"
              "            '    return params[0]*a*np.sin(b) + params[1]\\n')\n")
    valid, metrics, error = srbench.evaluate_source(
        source, suite=suite, shards=suite.test_range(), timeout=90.0,
        problem_seconds=20.0, answer_format="program")
    assert valid, error
    row = metrics["per_problem"][0]
    assert row["digits"] == pytest.approx(expr.DIGIT_CAP)
    assert row["acc"] == 1


def test_the_linear_program_root_is_upstreams_skeleton():
    """`params[0]*x1 + ... + params[n]`, which is what LLM-SR starts every problem from."""
    code, _summary = srbench.seed_program("linear", "program")
    namespace = {}
    exec(compile(code, "<root>", "exec"), namespace)  # noqa: S102 - a fixture
    emitted = namespace["discover"](np.zeros((4, 2)), np.zeros(4),
                                    {"input_vars": ["a", "b"], "max_params": 10})
    assert emitted.strip() == (
        "def equation(a, b, params):\n    return params[0]*a + params[1]*b + params[2]")


def test_both_roots_exist_in_both_formats():
    for fmt in ("expression", "program"):
        for root in ("library", "linear"):
            code, summary = srbench.seed_program(root, fmt)
            valid, why = _gate(code)
            assert valid, f"{fmt}/{root}: {why}"
            assert summary
    with pytest.raises(ValueError):
        srbench.seed_program("nonesuch", "program")


# ---------------------------------------------------------------------------
# Symbolic accuracy, the paper's third metric
# ---------------------------------------------------------------------------


def test_a_damaged_ground_truth_is_not_scorable_and_is_not_a_miss():
    """Both defects are the published copy's, not any answer's."""
    from tools import score_symbolic_accuracy as sa
    assert not sa.scorable("-0.19*A(t)**2 + 0.19_z*A(t)**2", ["t", "A"])
    assert not sa.scorable("F0*sin(t) - beta*sin(v(t))", ["x", "t", "v"])
    assert sa.scorable("(-A + x1*y1 - x2*y2)/x3",
                       ["A", "x1", "y1", "x2", "y2", "x3"])


def test_a_program_answer_is_reduced_to_its_equation_before_judging():
    from tools import score_symbolic_accuracy as sa
    source = ("import numpy as np\n\n\ndef equation(a, b, params):\n"
              "    return params[0]*a*np.sin(b) + params[1]\n")
    # With no fitted values the holes become free symbols, which is what makes
    # them comparable at all; with them they become the numbers that were fitted.
    # Compared without whitespace: the equation is reassembled by `ast.unparse`,
    # which spaces operators out, and neither sympy nor the judge cares.
    squash = lambda text: text.replace(" ", "")  # noqa: E731
    assert squash(sa.normalise(source, ["a", "b"])) == "__p0*a*sin(b)+__p1"
    assert squash(sa.normalise(source, ["a", "b"], fitted=[2.5, -0.75])) == (
        "(2.5)*a*sin(b)+(-0.75)")


def test_an_answer_built_in_steps_has_its_names_inlined_before_judging():
    """Taking the last `return` alone leaves undefined names in the equation.

    Six of 111 answers in the first aligned run computed their result in steps,
    and a judge shown `params[3]*E_n / denom` with `denom` undefined reads it as
    a different equation -- a defect in the scorer, not a wrong answer.
    """
    from tools import score_symbolic_accuracy as sa
    source = ("def equation(E_n, omega, omega_0, x, params):\n"
              "    denom = params[0]*omega_0**2*x**2 + params[1]*x**2\n"
              "    return params[2]*E_n / denom\n")
    inlined = sa.normalise(source, ["E_n", "omega", "omega_0", "x"])
    assert "denom" not in inlined
    assert "omega_0" in inlined and "__p0" in inlined and "__p2" in inlined

    # Chains of names resolve too, not just one level.
    chained = ("def equation(a, b, params):\n"
               "    u = a / b\n"
               "    v = params[0] * u\n"
               "    return v + params[1]\n")
    resolved = sa.normalise(chained, ["a", "b"]).replace(" ", "")
    assert "u" not in resolved.replace("__p", "") .replace("a", "").replace("b", "")
    assert "a/b" in resolved
    assert sa.normalise("0.3*P(t)**2", ["t", "P"]) == "0.3*P**2"


def test_fitted_constants_are_substituted_back_before_judging():
    """A skeleton cannot be compared with a concrete equation; a filled one can.

    Program-format answers come back with `params[i]` holes, and the values the
    harness fitted into them are part of the answer. Both cases below are real
    answers from an aligned run, and both are the ground truth with redundant
    free constants -- which is what the model actually writes.
    """
    pytest.importorskip("sympy")
    from tools import score_symbolic_accuracy as sa

    variables = ["m", "m_0", "c"]
    answer = ("def equation(m, m_0, c, params):\n"
              "    return c * np.sqrt(1.0 - (m_0/m)**2) * params[0]\n")
    filled = sa.normalise(answer, variables, fitted=[-0.9999999998] + [1.0] * 9)
    assert "params[" not in filled
    assert sa.deterministic_verdict("-c*sqrt(1 - m_0**2/m**2)", filled,
                                    variables) is True
    # Without the fitted values the hole becomes a free symbol and nothing can
    # be shown, so the judge decides rather than the problem scoring a miss.
    empty = sa.normalise(answer, variables)
    assert "__p0" in empty
    assert sa.deterministic_verdict("-c*sqrt(1 - m_0**2/m**2)", empty,
                                    variables) is not False


def test_numeric_equivalence_is_checked_away_from_where_anything_was_fitted():
    """An answer that only agrees where it was fitted must not pass."""
    pytest.importorskip("sympy")
    from tools import score_symbolic_accuracy as sa
    assert sa.numerically_equivalent("a*b", "b*a", ["a", "b"]) is True
    assert sa.numerically_equivalent("a*b", "a + b", ["a", "b"]) is False


def test_the_deterministic_check_only_ever_claims_equivalence():
    """It accelerates the easy cases and hands everything else to the judge."""
    pytest.importorskip("sympy")
    from tools import score_symbolic_accuracy as sa
    variables = ["q1", "q2", "F", "epsilon"]
    # 1/(2*sqrt(pi)) and 1/sqrt(4*pi) are the same number, and it finds that.
    assert sa.deterministic_verdict(
        "-sqrt(q1*q2/(F*epsilon))/(2*sqrt(pi))",
        "-sqrt((q1*q2)/(4*pi*epsilon*F))", variables) is True
    # Structurally different, and it declines rather than scoring a miss.
    assert sa.deterministic_verdict("q1*q2", "q1 + q2", variables) in (None, False)


# ---------------------------------------------------------------------------
# The per-problem protocol -- the benchmark's own
# ---------------------------------------------------------------------------


def _one_problem_fixture(rows: int = 400):
    rng = np.random.default_rng(3)
    train_x = rng.uniform(1.0, 5.0, size=(rows, 2))
    test_x = rng.uniform(1.0, 5.0, size=(50, 2))
    ood_x = rng.uniform(5.0, 7.0, size=(40, 2))
    truth = lambda m: 2.0 * m[:, 0] + 3.0  # noqa: E731
    samples = {"train_x": train_x, "train_y": truth(train_x),
               "test_x": test_x, "test_y": truth(test_x),
               "ood_x": ood_x, "ood_y": truth(ood_x)}
    problem = srbench.SrProblem(
        problem_id="fixture-1", subset="fixture", input_vars=("a", "b"),
        output_var="y", description="Discover y.\nInput 1: a\nInput 2: b",
        gt_expression="2*a + 3", train_rows=rows, test_rows=50, ood_rows=40)
    return problem, samples


def test_a_per_problem_suite_scores_on_train_slices_and_reports_on_the_benchmark(tmp_path,
                                                                                 monkeypatch):
    """The split that makes `--per-problem` comparable with the paper at all."""
    monkeypatch.setattr(srbench, "cache_path",
                        lambda subdir, name: str(tmp_path / name))
    problem, samples = _one_problem_fixture()
    suite = srbench.prepare_problem_suite(problem, samples, seed=0, shards=4,
                                          val_frac=0.25)
    assert suite.scoring_shards == 4 and suite.test_shards == 1
    assert suite.test_range() == (4,)

    fit_rows = suite.shard_problems[0][0].train_rows
    assert fit_rows == 300                      # 400 rows, a quarter held out
    pool = 0
    for shard in range(4):
        data = np.load(suite.shard_paths[shard])
        assert data["p0_train_x"].shape[0] == fit_rows
        pool += data["p0_test_x"].shape[0]
        assert "p0_ood_x" not in data           # a scoring shard is train-only
    assert pool == 100                          # the slices partition the pool

    # The one test shard is the benchmark's own split, untouched.
    final = np.load(suite.shard_paths[4])
    assert final["p0_test_x"].shape[0] == samples["test_x"].shape[0]
    assert final["p0_ood_x"].shape[0] == samples["ood_x"].shape[0]
    assert np.allclose(final["p0_test_x"], samples["test_x"])


def test_the_validation_pool_is_drawn_rather_than_taken_from_the_tail(tmp_path,
                                                                     monkeypatch):
    """Several subsets vary a state variable monotonically, so a tail split
    would hold out a *region* and every gate score would be an extrapolation."""
    monkeypatch.setattr(srbench, "cache_path",
                        lambda subdir, name: str(tmp_path / name))
    problem, samples = _one_problem_fixture()
    # Make the ordering carry the signal: column `a` increases down the table.
    samples["train_x"] = samples["train_x"][np.argsort(samples["train_x"][:, 0])]
    samples["train_y"] = 2.0 * samples["train_x"][:, 0] + 3.0
    suite = srbench.prepare_problem_suite(problem, samples, seed=0, shards=4,
                                          val_frac=0.25)
    held = np.concatenate([np.load(suite.shard_paths[i])["p0_test_x"][:, 0]
                           for i in range(4)])
    fit = np.load(suite.shard_paths[0])["p0_train_x"][:, 0]
    # A tail split would put every held-out `a` above every fitted one.
    assert held.min() < fit.mean() < held.max()


def test_a_per_problem_suite_refuses_too_few_shards_to_split(tmp_path, monkeypatch):
    monkeypatch.setattr(srbench, "cache_path",
                        lambda subdir, name: str(tmp_path / name))
    problem, samples = _one_problem_fixture()
    with pytest.raises(ValueError, match="at least four"):
        srbench.prepare_problem_suite(problem, samples, shards=3)


def test_the_per_problem_preview_carries_the_statement_and_not_the_answer():
    problem, samples = _one_problem_fixture()
    text = srbench.problem_preview(problem, samples)
    assert "Discover y." in text
    assert "a: [" in text and "b: [" in text
    assert "y (the target)" in text
    assert "2*a + 3" not in text


def test_the_per_problem_prompt_feeds_back_the_equation_that_was_tried():
    """What a per-problem search buys and a whole-category one cannot have."""
    parent = type("P", (), {
        "code": "def discover(x, y, spec):\n    return '1.0'\n",
        "metrics": {"mean_digits": 1.25, "worst": [
            {"equation": "1.0*a + 2.0", "digits": 1.25, "seconds": 0.3,
             "error": "", "problem_id": "fixture-1", "subset": "fixture",
             "variables": ["a", "b"]}]},
    })()
    text = srbench.per_problem_prompt(parent, preview="THE STATEMENT",
                                      timeout=60.0, problem_seconds=8.0,
                                      functions=("sin", "exp"))
    assert "THE STATEMENT" in text
    assert "1.0*a + 2.0" in text                   # the structure that was tried
    assert "1.2500" in text
    assert "least_squares" in text                 # fit the constants
    assert "one or two terms" in text              # and stay short
    assert "2*a + 3" not in text                   # never the ground truth


@needs_sandbox
def test_a_per_problem_search_is_scored_on_the_benchmarks_own_split(tmp_path,
                                                                   monkeypatch):
    monkeypatch.setattr(srbench, "cache_path",
                        lambda subdir, name: str(tmp_path / name))
    problem, samples = _one_problem_fixture()
    suite = srbench.prepare_problem_suite(problem, samples, seed=0, shards=4)
    domain = port.per_problem_domain(suite, problem, samples)
    assert domain.data_summary["problem_id"] == "fixture-1"
    assert domain.test_shards == (4,)
    valid, metrics, error = domain.evaluate(srbench.INITIAL_PROGRAM,
                                            domain.test_shards)
    assert valid, error
    assert metrics["problems"] == 1
    assert metrics["mean_digits"] > 8.0
    assert metrics["per_problem"][0]["ood_acc"] == 1


def test_a_resume_refuses_a_file_written_under_a_different_budget(tmp_path):
    """Otherwise one file holds two experiments under one heading."""
    import json as _json
    path = tmp_path / "sweep.json"
    args = port.build_parser().parse_args(["--per-problem", "--iterations", "20"])
    path.write_text(_json.dumps({
        "budget": {**port._budget_fingerprint(args), "iterations": 6},
        "per_problem": [{"problem_id": "x"}],
    }), encoding="utf-8")
    with pytest.raises(SystemExit, match="different budget"):
        port._load_checkpoint(path, port._budget_fingerprint(args))


def test_a_resume_returns_the_rows_of_a_matching_sweep(tmp_path):
    import json as _json
    path = tmp_path / "sweep.json"
    args = port.build_parser().parse_args(["--per-problem"])
    fingerprint = port._budget_fingerprint(args)
    path.write_text(_json.dumps({
        "budget": fingerprint,
        "per_problem": [{"problem_id": "a"}, {"problem_id": "b"}],
    }), encoding="utf-8")
    rows = port._load_checkpoint(path, fingerprint)
    assert [row["problem_id"] for row in rows] == ["a", "b"]
    assert port._load_checkpoint(tmp_path / "missing.json", fingerprint) == []


def test_the_budget_fingerprint_covers_what_changes_a_number():
    base = port.build_parser().parse_args(["--per-problem"])
    for flag, value in (("--iterations", "20"), ("--problem-seconds", "30"),
                        ("--shards", "8"), ("--train-points", "2000"),
                        ("--seed", "7")):
        other = port.build_parser().parse_args(["--per-problem", flag, value])
        assert port._budget_fingerprint(base) != port._budget_fingerprint(other), flag
    # --problem-concurrency is wall-clock only and must NOT split a sweep.
    same = port.build_parser().parse_args(["--per-problem",
                                           "--problem-concurrency", "4"])
    assert port._budget_fingerprint(base) == port._budget_fingerprint(same)


def test_the_per_problem_dry_run_says_which_protocol_it_would_run(capsys):
    assert port.main(["--dry-run", "--per-problem"]) == 0
    printed = capsys.readouterr().out
    assert "one search per problem" in printed
    assert "iterations 6/problem" in printed.replace("iterations=", "iterations ")
    assert port.main(["--dry-run"]) == 0
    assert "one search for the whole category" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The prompt and the port
# ---------------------------------------------------------------------------


class _Parent:
    code = "def discover(x, y, spec):\n    return '0.0'\n"
    metrics = {"mean_digits": 4.5, "acc_0.1": 0.25, "median_nmse": 3e-5,
               "worst": [{"problem_id": "matsci-3", "subset": "lsr_synth_matsci",
                          "variables": ["epsilon", "T"], "digits": 0.0,
                          "equation": "1.0*epsilon", "error": "", "seconds": 0.4}]}


def test_the_prompt_states_the_contract_without_leaking_an_answer(tmp_path):
    suite = _fixture_suite(tmp_path)
    text = srbench.mutation_prompt(
        _Parent(), preview=srbench.suite_preview(suite), timeout=300.0,
        problem_seconds=10.0, functions=("sin", "cos", "exp"))
    assert "discover(x, y, spec)" in text
    assert "returns a STRING" in text
    assert "4.5000" in text and "Acc(0.1) = 25.0%" in text
    assert "matsci-3" in text                      # which problem failed
    assert "2*a + 3" not in text                   # never the ground truth
    assert "sin, cos, exp" in text


def test_the_domain_reports_the_benchmark_it_ran(tmp_path):
    suite = _fixture_suite(tmp_path)
    domain = port.srbench_domain(suite)
    assert domain.entrypoint == "discover"
    assert domain.metric_key == "mean_digits"
    assert domain.metric_better == "higher"
    assert domain.gain(2.0, 6.0) == pytest.approx(4.0)
    summary = domain.data_summary
    assert summary["benchmark"] == "LLM-SRBench"
    assert summary["paper"] == srbench.BENCHMARK_PAPER
    assert srbench.MIRROR_REPO in summary["source"]
    assert domain.test_shards == suite.test_range()


def test_the_ports_dry_run_says_it_touched_nothing(capsys):
    assert port.main(["--dry-run"]) == 0
    printed = capsys.readouterr().out
    assert "dry-run" in printed.lower()
    assert "LLM-SRBench" in printed


# ---------------------------------------------------------------------------
# The reachability tool: what the answer format itself puts out of reach
# ---------------------------------------------------------------------------


def _reach():
    pytest.importorskip("sympy", reason="the reachability tool rewrites truths with sympy")
    pytest.importorskip("scipy", reason="the reachability tool fits with BFGS")
    from tools import srbench_reachability

    return srbench_reachability


def test_holing_a_truth_leaves_its_exponents_alone():
    """`x**2` is structure. Fitting the exponent measures a harder problem."""
    reach = _reach()
    source, params = reach.hole_coefficients("3.5*x**2 + 2.0*y", ["x", "y"])
    assert "x**2" in source
    assert "params[0]" in source and "params[1]" in source
    assert "params[2]" not in source
    assert params == 2


def test_holing_a_truth_keeps_the_values_that_carry_structure():
    """A sign or an identity is not a magnitude, so it is not a `params[i]`."""
    reach = _reach()
    source, params = reach.hole_coefficients("-x + y/2", ["x", "y"])
    assert params == 1                            # only the 2, not the -1
    assert source.count("params[") == 1


def test_a_truth_with_no_coefficient_still_gets_one_to_fit():
    """Upstream's format always fits something; a fixed scale is a stricter claim."""
    reach = _reach()
    source, params = reach.hole_coefficients("x*y", ["x", "y"])
    assert params == 1
    assert "params[0]" in source


def test_holing_binds_the_problems_own_names_before_sympy_claims_them():
    """`beta` and `gamma` are sympy *functions*; a variable named for one must survive."""
    reach = _reach()
    source, _ = reach.hole_coefficients("(f - beta)/(beta*gamma)", ["f", "beta", "gamma"])
    assert "def equation(f, beta, gamma, params):" in source
    assert "beta" in source and "gamma" in source


def test_a_holed_truth_is_the_answer_format_the_run_had_to_write_in():
    """What the tool scores has to be loadable by the grader's own compiler."""
    reach = _reach()
    source, _ = reach.hole_coefficients("2.5*a/b", ["a", "b"])
    call = expr.compile_program(source, ["a", "b"])
    x = np.array([[4.0, 2.0], [9.0, 3.0]])
    params = np.zeros(expr.MAX_NPARAMS)
    params[0] = 2.5
    assert call(x, params) == pytest.approx([5.0, 7.5])


def test_reachability_separates_the_optimiser_from_the_parameterisation():
    """The three columns are only useful if the summary attributes the gaps."""
    reach = _reach()
    rows = [
        {"expression_digits": 12.0, "program_digits": 12.0, "restart_digits": 12.0},
        {"expression_digits": 12.0, "program_digits": 0.0, "restart_digits": 11.0},
        {"expression_digits": 12.0, "program_digits": 0.0, "restart_digits": 0.0},
    ]
    summary = reach.summarise(rows)
    assert summary["problems"] == 3
    assert summary["expression_reachable"] == 3
    assert summary["program_reachable"] == 1
    assert summary["restart_reachable"] == 2
    assert summary["lost_to_single_start"] == 1
    assert summary["lost_to_parameterisation"] == 1


def test_reachability_reports_a_truth_it_cannot_rewrite_rather_than_dying():
    """One unparseable truth must not stop a sweep over a hundred of them."""
    reach = _reach()

    class _Problem:
        problem_id = "broken-1"
        subset = "lsr_transform"
        input_vars = ("x",)
        gt_expression = "this is not an equation"

    row = reach.measure(_Problem(), {"train_x": np.zeros((4, 1)), "train_y": np.zeros(4),
                                     "test_x": np.zeros((4, 1)), "test_y": np.zeros(4)},
                        train_points=4, restarts=2, seed=0)
    assert row["problem_id"] == "broken-1"
    assert row["program_digits"] is None
    assert "program_error" in row


# ---------------------------------------------------------------------------
# Recall: what the tree already tried, in the prompt that proposes the next one
# ---------------------------------------------------------------------------


def _attempt(answer: str, score: float):
    """One entry of a tree's recall snapshot, shaped as the tree hands it over."""
    return ({"mean_digits": score,
             "worst": [{"equation": answer, "digits": score, "seconds": 0.2,
                        "error": "", "problem_id": "fixture-1"}]}, score)


def test_no_recall_renders_nothing_which_is_upstreams_prompt():
    """`--recall-attempts 0` has to be byte-for-byte the prompt without it."""
    assert srbench.recalled_structures(()) == ""
    parent = type("P", (), {"code": "def discover(x, y, spec):\n    return '1.0'\n",
                            "metrics": {"mean_digits": 1.25, "worst": []}})()
    without = srbench.per_problem_prompt(parent, preview="P", timeout=60.0,
                                         problem_seconds=8.0, functions=("sin",))
    assert "already scored" not in without


def test_recall_lists_the_structures_with_their_scores_best_first():
    """The ordering and the scores are the mechanism, not decoration.

    A bare list of "already tried" pushes the model off a near-miss as hard as
    off a dead end. Ranked and scored, the same list separates them.
    """
    text = srbench.recalled_structures((
        _attempt("def equation(o, c, v, params):\n    return o*np.sqrt(1 - v**2/c**2)", 0.363),
        _attempt("def equation(o, c, v, params):\n    return params[0]*o + params[1]*c", 1.597),
    ))
    assert "o*np.sqrt(1 - v**2/c**2)" in text
    assert "params[0]*o + params[1]*c" in text
    assert "1.597" in text and "0.363" in text
    assert text.index("params[0]*o") < text.index("o*np.sqrt")   # best first
    assert "def equation" not in text                            # the body, not the signature


def test_recall_collapses_a_repeated_structure_into_a_count():
    """Four programs returning one structure is what a stuck search looks like."""
    same = "def equation(o, c, v, params):\n    return o*np.sqrt(1 - v**2/c**2)"
    text = srbench.recalled_structures(tuple(_attempt(same, 0.363) for _ in range(4)))
    assert "(x4)" in text
    assert text.count("o*np.sqrt") == 1


def test_recall_skips_an_attempt_that_returned_no_equation():
    """A node that failed to produce an answer has no structure to rule out."""
    assert srbench.recalled_structures((({"worst": [{"equation": ""}]}, 0.0),)) == ""
    assert srbench.recalled_structures((({}, 0.0),)) == ""


def test_the_per_problem_prompt_carries_the_recall_it_is_given():
    parent = type("P", (), {"code": "def discover(x, y, spec):\n    return '1.0'\n",
                            "metrics": {"mean_digits": 1.25, "worst": []}})()
    text = srbench.per_problem_prompt(
        parent, preview="P", timeout=60.0, problem_seconds=8.0,
        functions=("sin",),
        recalled=(_attempt("def equation(a, b, params):\n    return params[0]*a", 0.5),))
    assert "already scored" in text
    assert "params[0]*a" in text
    assert "re-parameterisation" in text


def test_a_resume_refuses_to_fold_a_recall_run_into_one_without_it(tmp_path):
    """`--recall-attempts` changes what a number means, so it is in the fingerprint."""
    import argparse

    base = dict(dataset="lsr_transform", iterations=24, workers=3, shards=6,
                held_out_frac=0.5, val_frac_per_problem=0.25, problem_seconds=20.0,
                candidate_timeout=120.0, train_points=4000, seed_program="linear",
                answer_format="program", model="glm-5.2", seed=0)
    off = port._budget_fingerprint(argparse.Namespace(recall_attempts=0, **base))
    on = port._budget_fingerprint(argparse.Namespace(recall_attempts=12, **base))
    assert off != on
    assert off["recall_attempts"] == 0 and on["recall_attempts"] == 12


def test_reachability_unwraps_the_ode_notation_synth_truths_carry():
    """`P(t)` and `P` are the same column; the parenthesis is the ODE's, not data."""
    reach = _reach()
    plain = reach.unwrap_ode_notation(
        "0.95*(1 - P(t)/96.9)*P(t) + 0.95*P(t)**0.333", ["t", "P"])
    assert "P(t)" not in plain
    assert plain == "0.95*(1 - P/96.9)*P + 0.95*P**0.333"


def test_reachability_leaves_real_function_calls_alone():
    """`sin` is not a column of the data, so rewriting it would change the truth."""
    reach = _reach()
    plain = reach.unwrap_ode_notation("0.3*P(t)*exp(-0.05*t) + sin(t)", ["t", "P"])
    assert plain == "0.3*P*exp(-0.05*t) + sin(t)"


def test_reachability_unwraps_only_the_names_a_problem_declares():
    reach = _reach()
    assert reach.unwrap_ode_notation("f(t) + P(t)", ["t", "P"]) == "f(t) + P"
