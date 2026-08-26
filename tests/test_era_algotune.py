"""The ERA port's AlgoTune task: the derivation, the gate, the runner, the score.

The block that matters most is the first one. Everything else here checks that
the machinery does what it says; ``test_the_derived_program_computes_what_the_
reference_computes`` checks that the *benchmark* is right, because a speedup
measured against a reference that is not the task's reference is a measurement
of nothing.

Offline, like the rest of the suite: the task file, its description and
upstream's published problem sizes are all served from fixtures here, and the
one test that reaches the real network is skipped unless it is asked for.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import json
import math
import os
import textwrap

import pytest

from examples.era import _algotune_tasks as tasks
from examples.era import _era_algotune as algotune
from examples.era import _era_support as support
from examples.era import era_algotune as port
from examples.era._algotune_tasks import DerivationError, derive_seed_program


# ---------------------------------------------------------------------------
# Fixtures: an AlgoTune-shaped task, without AlgoTune
# ---------------------------------------------------------------------------


FIXTURE_TASK = '''
import logging
from typing import Any

import numpy as np

from AlgoTuneTasks.base import register_task, Task


@register_task("fixture_norm")
class FixtureNorm(Task):
    """Row-wise 2-norms of a random matrix -- the shape of a real task, no more."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tolerance = 1e-09

    def generate_problem(self, n: int, random_seed: int = 1) -> dict[str, Any]:
        rng = np.random.default_rng(random_seed)
        return {"matrix": rng.standard_normal((n, n))}

    def _reference(self, matrix):
        return np.sqrt((matrix * matrix).sum(axis=1))

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        logging.debug("solving")
        return {"norms": self._reference(problem["matrix"])}

    def is_solution(self, problem: dict[str, Any], solution: dict[str, Any]) -> bool:
        if not isinstance(solution, dict) or "norms" not in solution:
            return False
        expected = np.linalg.norm(problem["matrix"], axis=1)
        got = np.asarray(solution["norms"], dtype=float)
        if got.shape != expected.shape:
            return False
        return bool(np.allclose(got, expected, atol=self.tolerance))
'''

FIXTURE_DESCRIPTION = "FixtureNorm Task:\n\nCompute the 2-norm of every row.\n"

FIXTURE_SIZES = {
    "svd": {"target_time_ms": 100, "n": 474,
            "baseline_runs": {"0": {"avg_min_ms": 117.0}, "1": {"avg_min_ms": 119.0}}},
}


@pytest.fixture
def fixture_suite(tmp_path, monkeypatch):
    """A :class:`~examples.era._era_algotune.Suite` over the fixture task.

    ``svd`` is borrowed as the name so the ``TASKS`` membership check -- which is
    a real guard, not a formality -- is exercised rather than bypassed. What is
    behind the name is the fixture above, so nothing here reaches the network.
    """
    def fake_fetch(url, **_kwargs):
        if url.endswith("generation.json"):
            return json.dumps(FIXTURE_SIZES)
        if url.endswith("description.txt"):
            return FIXTURE_DESCRIPTION
        return FIXTURE_TASK

    monkeypatch.setattr(algotune, "fetch_text", fake_fetch)
    monkeypatch.setattr(algotune, "cache_path",
                        lambda subdir, name: str(tmp_path / subdir / name))
    suite = algotune.prepare_suite("svd", shards=2, test_shards=1, problems=1)
    # The fixture's class registers itself as `fixture_norm`, so the shard spec
    # has to name what the runner will look up inside the sandbox.
    return dataclasses.replace(suite, task="fixture_norm")


# ---------------------------------------------------------------------------
# The benchmark: does the derived program compute the reference?
# ---------------------------------------------------------------------------


def test_the_derived_program_computes_what_the_reference_computes():
    """The root node has to *be* the reference, not resemble it.

    The whole metric is a ratio against this program. If lifting ``solve`` out of
    its class changed what it computed -- dropped a helper, inlined the wrong
    constant -- every speedup the search reported would be measured against
    something upstream never wrote, and would still look perfectly plausible.
    """
    numpy = pytest.importorskip("numpy")
    tasks.install_shim()
    namespace: dict = {}
    exec(compile(derive_seed_program(FIXTURE_TASK), "<derived>", "exec"), namespace)

    module = {}
    exec(compile(FIXTURE_TASK, "<fixture>", "exec"), module)
    reference = module["FixtureNorm"]()

    problem = reference.generate_problem(16, random_seed=3)
    expected = reference.solve({"matrix": problem["matrix"].copy()})
    derived = namespace["solve"]({"matrix": problem["matrix"].copy()})
    assert numpy.allclose(derived["norms"], expected["norms"])
    assert reference.is_solution(problem, derived)


def test_the_derivation_lifts_helpers_and_constants_and_drops_the_class():
    derived = derive_seed_program(FIXTURE_TASK)
    tree = ast.parse(derived)
    assert not [node for node in tree.body if isinstance(node, ast.ClassDef)]
    top_level = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert "solve" in top_level and "_ref_reference" in top_level
    assert "AlgoTuneTasks" not in derived
    assert "self" not in {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def test_a_constant_read_off_self_becomes_a_module_constant():
    source = textwrap.dedent('''
        from AlgoTuneTasks.base import register_task, Task

        @register_task("k")
        class K(Task):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.mode = "full"

            def solve(self, problem):
                return (problem, self.mode)
    ''')
    derived = derive_seed_program(source)
    assert "_REF_MODE = 'full'" in derived
    namespace: dict = {}
    exec(compile(derived, "<derived>", "exec"), namespace)
    assert namespace["solve"](1) == (1, "full")


def test_state_built_outside_init_is_refused_rather_than_guessed():
    """A silent wrong answer is the failure mode this whole file exists to stop.

    ``self.cache`` assigned inside ``solve`` has no value to lift, and a
    derivation that invented one -- ``None``, an empty dict -- would produce a
    program that imports, runs, and computes something else.
    """
    source = textwrap.dedent('''
        from AlgoTuneTasks.base import register_task, Task

        @register_task("k")
        class K(Task):
            def prepare(self):
                self.cache = 1

            def solve(self, problem):
                return self.cache
    ''')
    with pytest.raises(DerivationError) as excinfo:
        derive_seed_program(source)
    assert "cache" in str(excinfo.value)


def test_the_task_class_is_the_decorated_one_not_the_first_one():
    source = textwrap.dedent('''
        from AlgoTuneTasks.base import register_task, Task

        class Helper:
            def solve(self, problem):
                return "helper"

        @register_task("k")
        class K(Task):
            def solve(self, problem):
                return "task"
    ''')
    namespace: dict = {}
    exec(compile(derive_seed_program(source), "<derived>", "exec"), namespace)
    assert namespace["solve"](None) == "task"


# ---------------------------------------------------------------------------
# The suite
# ---------------------------------------------------------------------------


def test_the_suite_reads_upstreams_published_problem_size(fixture_suite):
    assert fixture_suite.n == 474
    assert fixture_suite.published_n == 474
    assert fixture_suite.target_time_ms == 100
    assert fixture_suite.published_ms == pytest.approx(118.0)
    assert fixture_suite.source_path.exists()


def test_shards_draw_disjoint_seeds_and_a_redraw_is_identical(fixture_suite):
    seen = [set(fixture_suite.seeds(shard)) for shard in range(3)]
    assert seen[0].isdisjoint(seen[1]) and seen[1].isdisjoint(seen[2])
    assert fixture_suite.seeds(1) == fixture_suite.seeds(1)


def test_two_seeds_draw_different_problem_sets(tmp_path, monkeypatch):
    monkeypatch.setattr(algotune, "fetch_text",
                        lambda url, **k: (json.dumps(FIXTURE_SIZES)
                                          if url.endswith("generation.json")
                                          else FIXTURE_TASK))
    monkeypatch.setattr(algotune, "cache_path",
                        lambda subdir, name: str(tmp_path / subdir / name))
    first = algotune.prepare_suite("svd", seed=0, shards=2, test_shards=1)
    second = algotune.prepare_suite("svd", seed=1, shards=2, test_shards=1)
    assert set(first.seeds(0)).isdisjoint(second.seeds(0))


def test_a_wider_reported_split_leaves_the_search_seeing_the_same_problems(tmp_path,
                                                                          monkeypatch):
    """`--test-problems` widens what is *reported on*, not what is searched.

    Two properties, and both are load-bearing. The scoring sets keep the seeds
    they had, so a run with a wider held-back split is still a rerun of the same
    search rather than a different one. And no seed the search could score
    against appears in the reported split -- widening the measurement must not
    quietly start measuring the sets the optimiser was allowed to fit.
    """
    monkeypatch.setattr(algotune, "fetch_text",
                        lambda url, **k: (json.dumps(FIXTURE_SIZES)
                                          if url.endswith("generation.json")
                                          else FIXTURE_TASK))
    monkeypatch.setattr(algotune, "cache_path",
                        lambda subdir, name: str(tmp_path / subdir / name))
    narrow = algotune.prepare_suite("svd", shards=6, test_shards=3, problems=2)
    wide = algotune.prepare_suite("svd", shards=6, test_shards=3, problems=2,
                                  test_problems=50)

    assert [narrow.seeds(s) for s in range(6)] == [wide.seeds(s) for s in range(6)]
    assert narrow.size(0) == wide.size(0) == 2
    assert wide.size(6) == 50 and narrow.size(6) == 2

    searchable = set().union(*(set(wide.seeds(s)) for s in range(6)))
    reported = set().union(*(set(wide.seeds(s)) for s in wide.test_range()))
    assert not searchable & reported
    assert len(reported) == 150


def test_a_task_outside_the_runnable_set_is_refused_by_name():
    with pytest.raises(ValueError) as excinfo:
        algotune.prepare_suite("max_common_subgraph")
    assert "max_common_subgraph" in str(excinfo.value)


def test_every_runnable_task_name_is_unique_and_sorted():
    assert list(algotune.TASKS) == sorted(algotune.TASKS)
    assert len(set(algotune.TASKS)) == len(algotune.TASKS)
    assert set(algotune.DEFAULT_TASKS) <= set(algotune.TASKS)


def test_lqr_is_excluded_and_the_exclusion_is_explained():
    """The one task dropped for a reason a reader could not re-derive.

    ``lqr`` clears both mechanical filters and is still absent, because its own
    ``is_solution`` calls ``float()`` on a 1x1 array -- which NumPy has refused
    since 1.25, so the reference is invalid by the task's own oracle. A silent
    omission would look like an oversight and get "fixed".
    """
    assert "lqr" not in algotune.TASKS
    assert "lqr" in inspect.getsource(algotune)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def test_the_gate_accepts_a_derived_reference_and_rejects_the_obvious_accidents():
    derived = derive_seed_program(FIXTURE_TASK)
    valid, reason = support.validate_source(
        derived, entrypoint="solve", allowed_imports=algotune.ALLOWED_IMPORTS,
        literal_top_level=False)
    assert valid, reason

    for source, expected in (
        ("import os\ndef solve(problem):\n    return 1\n", "os"),
        ("def helper(problem):\n    return 1\n", "solve"),
        ("def solve(problem):\n    return problem.__class__\n", "dunder"),
    ):
        valid, reason = support.validate_source(
            source, entrypoint="solve", allowed_imports=algotune.ALLOWED_IMPORTS,
            literal_top_level=False)
        assert not valid and expected in reason


def test_the_allowlist_admits_the_compiler_directive_two_references_open_with():
    """`from __future__ import annotations` is not a module, and it cost a task.

    Left off the allowlist, `prepare_suite` still succeeds -- it parses the task
    file, it does not execute it -- so the refusal arrives much later, as "the
    initial ERA program failed to run", and the run reports one fewer task than
    it was asked for with no obvious cause. Measured that way on
    `sparse_lowest_eigenvalues_posdef`, in a 20-task run.
    """
    assert "__future__" in algotune.ALLOWED_IMPORTS


@pytest.mark.skipif(not os.getenv("AGENTDESCENT_ALGOTUNE_NETWORK"),
                    reason="set AGENTDESCENT_ALGOTUNE_NETWORK=1 to fetch the "
                           "task files from upstream")
def test_every_runnable_reference_derives_and_passes_this_tasks_own_gate():
    """The sweep that would have caught the allowlist hole before a run did.

    A root node the gate refuses is a task that cannot be searched at all, and
    nothing else here checks the 72 real references against the gate that has to
    admit them -- the fixtures above are this file's own task, not upstream's.
    Opt-in because it fetches 72 files; the offline suite stays offline.
    """
    failures = []
    for task in algotune.TASKS:
        try:
            derived = derive_seed_program(algotune.task_source(task))
        except DerivationError as exc:
            failures.append(f"{task}: derivation: {exc}")
            continue
        valid, reason = support.validate_source(
            derived, entrypoint="solve", allowed_imports=algotune.ALLOWED_IMPORTS,
            literal_top_level=False)
        if not valid:
            failures.append(f"{task}: gate: {reason}")
    assert not failures, "\n".join(failures)


def test_the_gate_admits_a_jit_warm_up_and_the_other_era_tasks_still_refuse_one():
    """Warming a JIT is a bare call at module level, and the gate refused it.

    ``@njit`` compiles on first call. Without a module-level ``_kernel(0.0)`` that
    first call is the first *timed* call, and the candidate is charged for the
    compiler -- which on a task where numba is the whole point turns a 9000x
    program into a slow one. The capability is not new: ``literal_top_level=False``
    already admits ``TABLE = build()``, which is the same call with its result
    bound. So this was friction, not a boundary.

    Still off by default, and this asserts that: a port with no reason to compile
    keeps the narrower gate.
    """
    source = ("import numba\n"
              "@numba.njit\n"
              "def _k(x):\n    return x + 1\n"
              "_k(1.0)\n"
              "def solve(problem):\n    return _k(problem)\n")
    valid, reason = support.validate_source(
        source, entrypoint="solve", allowed_imports=algotune.ALLOWED_IMPORTS,
        literal_top_level=False, allow_top_level_calls=True)
    assert valid, reason

    valid, reason = support.validate_source(
        source, entrypoint="solve", allowed_imports=algotune.ALLOWED_IMPORTS,
        literal_top_level=False)
    assert not valid and "top-level expression" in reason


def test_the_compiled_toolchain_is_on_the_allowlist_and_the_prompt_says_so():
    """AlgoTune's own results make these two load-bearing, not optional.

    Across upstream's 2076 published solutions numba and Cython are 21% of
    everything and **half of the results at 100x or better**: the reference on an
    ODE task pays a Python callback per derivative evaluation, and nothing
    written in NumPy closes that. A run without them is not a harder run, it is a
    run over the half of the benchmark where the large wins are not.
    """
    assert {"numba", "cython"} <= algotune.ALLOWED_IMPORTS
    text = algotune.mutation_prompt(
        support.Program("i", 0, None, "def solve(p):\n    return p\n", "",
                        {"speedup": 1.0}, True),
        suite=_bare_suite())
    assert "numba" in text and "cython" in text
    assert "warm-up" in text


def _bare_suite():
    from pathlib import Path
    return algotune.Suite(
        task="svd", source_path=Path("."), description="d", initial_program="p",
        n=1, published_n=1, target_time_ms=100, published_ms=1.0, problems=2,
        scoring_shards=6, test_shards=3, seed=0)


def test_a_precomputed_table_is_allowed_here_and_refused_by_the_tabular_gate():
    """Module-level setup is the point of the task, not a smell.

    A cached plan, a precomputed twiddle table or a preallocated workspace is
    exactly what makes a numerical routine fast, and the tabular task's gate --
    which requires literal top-level assignments -- would reject every one of
    them.
    """
    source = ("import numpy as np\n"
              "TABLE = np.arange(1024, dtype=float)\n"
              "def solve(problem):\n    return TABLE\n")
    assert support.validate_source(
        source, entrypoint="solve", allowed_imports=algotune.ALLOWED_IMPORTS,
        literal_top_level=False)[0]
    assert not support.validate_source(
        source, entrypoint="solve", allowed_imports=algotune.ALLOWED_IMPORTS,
        literal_top_level=True)[0]


# ---------------------------------------------------------------------------
# The score
# ---------------------------------------------------------------------------


def test_the_reward_is_order_preserving_with_the_metric():
    """The tree ranks on `score` and the engine's gate ranks on this.

    A port where those two disagreed would be selecting against its own
    acceptance rule -- and unlike a rescale by an assumed maximum, this one never
    saturates, so a 40x candidate still outranks a 20x one.
    """
    speedups = [0.1, 0.5, 1.0, 2.0, 8.0, 40.0, 200.0]
    rewards = [algotune.framework_score({"speedup": value}) for value in speedups]
    assert rewards == sorted(rewards)
    assert all(0.0 <= reward <= 1.0 for reward in rewards)
    assert algotune.framework_score({"speedup": 1.0}) == pytest.approx(0.5)
    assert algotune.framework_score({"speedup": None}) == 0.0
    assert algotune.framework_score({"speedup": float("nan")}) == 0.0


def test_a_failed_metric_carries_upstreams_minus_infinity_sentinel():
    metrics = algotune._zero_metrics("boom")
    assert metrics["score"] == -math.inf
    assert metrics["speedup"] is None
    assert metrics["error"] == "boom"


def test_the_geometric_mean_is_used_because_speedups_are_ratios():
    """4x on one task and 0.25x on another is no change, not 2.1x."""
    assert port.geometric_mean([4.0, 0.25]) == pytest.approx(1.0)
    assert port.geometric_mean([2.0, 8.0]) == pytest.approx(4.0)
    assert port.geometric_mean([]) is None
    assert port.geometric_mean([None, float("inf"), 0.0]) is None


# ---------------------------------------------------------------------------
# The evaluator, through the sandbox
# ---------------------------------------------------------------------------


needs_sandbox = pytest.mark.skipif(
    support.sandbox_backend() is None,
    reason="no candidate isolation backend on this host")


@needs_sandbox
def test_the_reference_scores_about_one_through_the_sandbox(fixture_suite):
    """The root node is the reference, so it must measure as the reference.

    Not exactly 1.0 -- it is two independent timings of the same code, and the
    band here is the noise a shared machine adds. A root that came out at 0.5x
    or 2x would mean the two sides are not being timed alike, which would make
    every number this task reports meaningless.
    """
    pytest.importorskip("numpy")
    valid, metrics, error = algotune.evaluate_source(
        fixture_suite.initial_program, suite=fixture_suite, shards=(0,),
        timeout=60.0, repeats=2)
    assert valid, error
    assert 0.5 < metrics["speedup"] < 2.0
    assert metrics["valid_problems"] == metrics["problems"] == fixture_suite.problems
    assert metrics["baseline_ms"] > 0.0


@needs_sandbox
def test_one_wrong_answer_invalidates_the_whole_evaluation(fixture_suite):
    """AlgoTune's own rule: not all valid, no speedup at all.

    It is what keeps the benchmark about speed. A program a thousand times
    faster on nine problems and wrong on the tenth has not sped anything up.
    """
    pytest.importorskip("numpy")
    code = ("import numpy as np\n"
            "def solve(problem):\n"
            "    return {'norms': np.zeros(len(problem['matrix']))}\n")
    valid, metrics, error = algotune.evaluate_source(
        code, suite=fixture_suite, shards=(0,), timeout=60.0, repeats=1)
    assert not valid
    assert metrics["score"] == -math.inf
    assert "not solved correctly" in error


@needs_sandbox
def test_a_faster_program_scores_above_the_reference(fixture_suite):
    pytest.importorskip("numpy")
    code = ("import numpy as np\n"
            "def solve(problem):\n"
            "    return {'norms': np.linalg.norm(problem['matrix'], axis=1)}\n")
    valid, metrics, error = algotune.evaluate_source(
        code, suite=fixture_suite, shards=(0,), timeout=60.0, repeats=3)
    assert valid, error
    assert metrics["speedup"] > 0.0


@needs_sandbox
def test_a_program_that_raises_is_a_scored_failure_not_a_crash(fixture_suite):
    pytest.importorskip("numpy")
    code = "def solve(problem):\n    raise RuntimeError('nope')\n"
    valid, metrics, error = algotune.evaluate_source(
        code, suite=fixture_suite, shards=(0,), timeout=60.0, repeats=1)
    assert not valid
    assert "RuntimeError" in error


@needs_sandbox
def test_compilation_is_not_charged_wherever_the_author_put_it(fixture_suite):
    """Two identical programs must not differ by where their JIT compiles.

    A `@numba.njit` function compiles on its first call. When the first call was
    also the first *timed* call -- and the call the slow-check read -- an
    identical program measured 0.052x compiling inside `solve` and 0.947x
    compiling at import. An 18x swing that reads where the author put a line, not
    how fast the program is, and one the search learns from: a few of those and
    it concludes compiling makes things twenty times slower and steers away from
    the only lever that wins on this benchmark.

    AlgoTune's rule is that compilation is not charged. Honouring it cannot
    depend on the model knowing the trick, so the candidate gets the same untimed
    warm-up the reference already got.

    The fixture task runs in microseconds, so *any* compile exceeds ten times its
    baseline -- which is exactly the regime the old code got wrong.
    """
    pytest.importorskip("numba")
    lazy = ("import numpy as np\n"
            "import numba\n"
            "@numba.njit\n"
            "def _k(a):\n"
            "    s = 0.0\n"
            "    for i in range(a.shape[0]):\n"
            "        s += a[i]\n"
            "    return s\n"
            "def solve(problem):\n"
            "    m = problem['matrix']\n"
            "    _k(np.zeros(2))\n"
            "    return {'norms': np.linalg.norm(m, axis=1)}\n")
    warmed = lazy.replace("def solve(problem):",
                          "_k(np.zeros(2))\n\ndef solve(problem):")

    scored = {}
    for label, code in (("lazy", lazy), ("warmed", warmed)):
        valid, metrics, error = algotune.evaluate_source(
            code, suite=fixture_suite, shards=(0,), timeout=120.0, repeats=3)
        assert valid, f"{label}: {error}"
        scored[label] = metrics["speedup"]

    ratio = max(scored.values()) / min(scored.values())
    assert ratio < 3.0, (
        f"where the compile happens moved the score by {ratio:.1f}x: {scored}")


@needs_sandbox
def test_the_gate_rejects_before_any_process_is_started(fixture_suite, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("the gate let a rejected program reach the sandbox")

    monkeypatch.setattr(algotune.subprocess, "run", forbidden)
    valid, _metrics, error = algotune.evaluate_source(
        "import socket\ndef solve(problem):\n    return 1\n",
        suite=fixture_suite, shards=(0,), timeout=10.0)
    assert not valid and error.startswith("gate:")


@needs_sandbox
def test_the_candidate_is_timed_against_a_reference_measured_beside_it(fixture_suite):
    """Both timings come out of one runner invocation, on the same problem.

    A baseline measured once on the host and reused would fold the whole run's
    scheduling weather into the score, so it would move when the machine got
    busy rather than when the program got faster.
    """
    pytest.importorskip("numpy")
    payload = algotune.run_candidate(
        fixture_suite.initial_program, suite=fixture_suite, shard=0,
        timeout=60.0, repeats=2)
    assert payload["ok"], payload.get("error")
    row = payload["results"][0]
    assert row["baseline_ms"] > 0.0 and row["candidate_ms"] > 0.0
    assert row["valid"] is True
    assert row["seed"] in fixture_suite.seeds(0)


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------


def test_the_prompt_is_upstreams_system_message_and_carries_upstreams_eval_block(
        fixture_suite):
    """Aligned means aligned: the framing is AlgoTuner's, not this port's.

    A speedup measured under a prompt that also named the levers is not
    comparable with upstream's, whose system message describes the setting and
    the rules and then stops. So the default carries upstream's wording -- the
    10x rule, "setup is not charged", "Be creative" -- and its post-eval summary
    verbatim, and says nothing about how to make code fast.
    """
    parent = support.Program(
        "id", 0, None, "def solve(problem):\n    return 1\n", "baseline",
        {"speedup": 1.25, "problems": 4, "valid_problems": 4, "slowest": [
            {"seed": 7, "baseline_ms": 10.0, "candidate_ms": 8.0, "speedup": 1.25,
             "valid": True, "error": ""}]}, True)
    text = algotune.mutation_prompt(parent, suite=fixture_suite)
    # Upstream's wording is hard-wrapped; compare on a whitespace-normalised
    # copy so a rewrap does not read as a change of meaning.
    flat = " ".join(text.split())

    assert "You're an autonomous programmer" in flat
    assert "at most 10x the reference runtime" in flat
    assert "Be creative and optimize your approach!" in flat
    assert "Speedup: 1.250x" in text
    assert "(Speedup = Baseline Time / Your Time; Higher is better)" in text
    assert "Valid Solutions: 100% (4/4)" in text
    assert "Compute the 2-norm of every row." in text
    assert "def solve(problem):" in text
    assert "seed 7" in text
    # The levers upstream does not name, and this prompt must not either.
    assert "numba.njit" not in text
    assert "WHERE THE LARGE WINS" not in text


def test_the_three_styles_are_nested_and_only_guided_names_the_levers(fixture_suite):
    """Each style adds exactly one thing, and what each is worth is measured.

    Drawing from the root on ode_stiff_vanderpol, counting how often a draw
    replaces the library call with a compiled loop: aligned 0/6, hinted 1/8,
    guided 3/8. Over a whole 46-node tree those became 1.04x (hinted) and
    39.65x (guided). An effect that large cannot be baked silently into every
    number, so the arms stay separately runnable.
    """
    parent = support.Program("id", 0, None, "def solve(problem):\n    return 1\n",
                             "", {"speedup": 1.0, "problems": 2,
                                  "valid_problems": 2}, True)
    texts = {style: algotune.mutation_prompt(parent, suite=fixture_suite, style=style)
             for style in algotune.PROMPT_STYLES}

    # aligned: upstream's bare package list, nothing else.
    assert "just-in-time compiler" not in texts["aligned"]
    assert "Compile an interpreted loop" not in texts["aligned"]
    # hinted: what the compilers are and how to call them -- not what to point
    # them at.
    assert "just-in-time compiler" in texts["hinted"]
    assert "Compile an interpreted loop" not in texts["hinted"]
    # guided: the techniques, in upstream's own TIPS slot.
    assert "Compile an interpreted loop" in texts["guided"]
    assert "**TIPS:**" in texts["guided"]

    assert len(texts["aligned"]) < len(texts["hinted"]) < len(texts["guided"])
    with pytest.raises(ValueError):
        algotune.mutation_prompt(parent, suite=fixture_suite, style="nonsense")


def test_a_rejected_answer_reaches_the_prompt_as_upstreams_invalid_example(
        fixture_suite):
    """Upstream shows up to three rejected instances; so does this.

    What goes inside the block is this port's own: upstream can only print the
    checker's source context when the checker raised, and a checker that returns
    False leaves it nothing. The distance from the reference is the thing that
    separates "structurally wrong" from "a factor of three short".
    """
    metrics = algotune._zero_metrics("1/2 problems were not solved correctly")
    metrics.update({"problems": 2, "valid_problems": 1, "slowest": [
        {"seed": 3, "baseline_ms": 10.0, "candidate_ms": None, "speedup": None,
         "valid": False,
         "error": "is_solution rejected the output (largest relative "
                  "difference from the reference 7.050e-03)"}]})
    parent = support.Program("id", 1, None, "def solve(problem):\n    return 1\n",
                             "", metrics, False)
    text = algotune.mutation_prompt(parent, suite=fixture_suite)
    assert "Invalid Example #1:" in text
    assert "Error in 'is_solution':" in text
    assert "7.050e-03" in text
    assert "Valid Solutions: 50% (1/2)" in text


def test_the_profile_reaches_the_prompt_when_one_was_taken(fixture_suite):
    """Upstream's `profile` command, in the one place this port can put it."""
    parent = support.Program(
        "id", 0, None, "def solve(problem):\n    return 1\n", "",
        {"speedup": 1.0, "problems": 1, "valid_problems": 1,
         "profile": "Line #      Hits         Time\n     8   1   706.5"}, True)
    text = algotune.mutation_prompt(parent, suite=fixture_suite)
    assert "Line-level profile" in text and "706.5" in text
    without = algotune.mutation_prompt(
        support.Program("id", 0, None, "def solve(problem):\n    return 1\n", "",
                        {"speedup": 1.0, "problems": 1, "valid_problems": 1}, True),
        suite=fixture_suite)
    assert "Line-level profile" not in without


def test_the_slow_factor_matches_upstreams_per_instance_rule():
    """AlgoTune: "your function can run for at most 10x the reference runtime"."""
    assert algotune.SLOW_FACTOR == 10.0


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def test_tasks_resolves_the_three_forms_and_refuses_an_unknown_name():
    assert port.resolve_tasks("default") == algotune.DEFAULT_TASKS
    assert port.resolve_tasks("") == algotune.DEFAULT_TASKS
    assert port.resolve_tasks("all") == algotune.TASKS
    assert port.resolve_tasks("svd, qr_factorization") == ("svd", "qr_factorization")
    with pytest.raises(SystemExit) as excinfo:
        port.resolve_tasks("svd,not_a_task")
    assert "not_a_task" in str(excinfo.value)


def test_list_tasks_prints_the_runnable_set_and_touches_nothing(capsys, monkeypatch):
    monkeypatch.setattr(port, "prepare_suite", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("--list-tasks loaded a task")))
    assert port.main(["--list-tasks"]) == 0
    printed = capsys.readouterr().out.split()
    assert printed == list(algotune.TASKS)


def test_dry_run_touches_no_network_task_file_or_sandbox(capsys, monkeypatch):
    monkeypatch.setattr(port, "prepare_suite", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("dry-run crossed a boundary")))
    monkeypatch.setattr(port, "completion_for", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("dry-run called a model")))
    assert port.main(["--dry-run", "--tasks", "svd"]) == 0
    out = capsys.readouterr().out
    assert "dry-run" in out.lower()
    assert "AlgoTune" in out and "svd" in out


def test_the_domain_reports_its_own_metric_under_its_own_name(fixture_suite):
    domain = port.algotune_domain(fixture_suite)
    assert domain.metric_key == "speedup"
    assert domain.metric_better == "higher"
    assert domain.entrypoint == "solve"
    assert domain.gain(1.0, 3.0) == pytest.approx(2.0)
    assert domain.data_summary["published_n"] == 474
    assert domain.test_shards == fixture_suite.test_range()
