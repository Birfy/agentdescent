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
    """A name this port cannot score must fail loudly, not run and mislead.

    `aes_gcm_encryption` is a real AlgoTune task that this port deliberately
    does not carry -- its reference wants `os.urandom`, and `os` is out because
    the gate's forbidden-name check cannot tell `os.urandom` from `os.system`.
    """
    with pytest.raises(ValueError) as excinfo:
        algotune.prepare_suite("aes_gcm_encryption")
    assert "aes_gcm_encryption" in str(excinfo.value)


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


def _runner_module():
    """The sandbox runner, imported on the host so its helpers can be tested.

    It is written to be loadable by path from inside the sandbox, so importing
    it here costs nothing and needs no sandbox.
    """
    import importlib.util
    import sys
    path = algotune.RUNNER
    spec = importlib.util.spec_from_file_location("_algotune_runner_under_test",
                                                  str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except SystemExit:  # its __main__ guard, if it ever grows one
        pass
    return module


def test_inviting_the_packages_adds_a_sentence_and_not_a_technique(fixture_suite):
    """`--packages invited` adds one sentence. It may not add a second thing.

    Two heavier versions were measured and both are gone. `ACCELERATION_TIPS`
    named four techniques next to the parent's profile -- worth 3/8 draws
    reaching for a compiler against 0/8, and 2785x on ode_stiff_vanderpol -- and
    it made the number incomparable with upstream's, whose agent is told none of
    it. A middle version that glossed each library ("numba, a just-in-time
    compiler") bought nothing at all: 0/8 against the bare list's 1/8 on
    polynomial_real, where upstream's field splits into 70x-138x with numba and
    about 1.0x without.

    What is left is an invitation, not advice: the packages are there, you may
    try them. The test exists because the difference between those three is one
    edit, and only the first of them is honest to report under AlgoTune's name.
    """
    parent = support.Program("id", 0, None, "def solve(problem):\n    return 1\n",
                             "", {"speedup": 1.0, "problems": 2,
                                  "valid_problems": 2}, True)
    bare = algotune.mutation_prompt(parent, suite=fixture_suite, packages="bare")
    invited = algotune.mutation_prompt(parent, suite=fixture_suite, packages="invited")

    assert "free to use any of these" in invited
    assert "free to use any of these" not in bare
    # One sentence, not a gloss on each library.
    assert len(invited) - len(bare) < 120, "the invitation grew into a paragraph"

    # Neither arm may say what a library is, name a technique, or say when to
    # reach for one.
    for text, arm in ((bare, "bare"), (invited, "invited")):
        for banned in ("just-in-time compiler", "tracing JIT", "ahead-of-time",
                       "Compile an interpreted loop", "Skip work the answer",
                       "Do less arithmetic", "Pick the specialised routine",
                       "routinely buys 100x"):
            assert banned not in text, f"{arm} says too much: {banned!r}"

    with pytest.raises(ValueError):
        algotune.mutation_prompt(parent, suite=fixture_suite, packages="nonsense")


def test_an_optional_import_guard_survives_derivation_and_the_gate():
    """`try: from x import y / except: y = None` must reach the root node intact.

    This is the third time upstream's own reference has failed this port's gate
    and taken a task silently out of the runnable set: first `__future__` was
    missing from the allowlist, then `contextlib` and `numbers`, and then the
    derivation dropped module-level `ast.Try` outright. Dropping it does not
    drop a feature -- it leaves the *name* unbound, so the reference raises
    NameError from whatever used it, which reads as a broken task.

    Both halves are checked because they failed separately: the derivation kept
    the statement and the gate then refused it.
    """
    source = textwrap.dedent('''
        import numpy as np

        from AlgoTuneTasks.base import register_task, Task

        try:
            from threadpoolctl import threadpool_limits
        except Exception:
            threadpool_limits = None

        def _limit():
            return 1 if threadpool_limits is None else 2

        @register_task("guarded")
        class Guarded(Task):
            def solve(self, problem):
                return _limit()
    ''')
    derived = derive_seed_program(source)
    assert "threadpool_limits = None" in derived, derived
    assert "from threadpoolctl import threadpool_limits" in derived

    valid, reason = support.validate_source(
        derived, entrypoint="solve", allowed_imports=algotune.ALLOWED_IMPORTS,
        literal_top_level=False, allow_top_level_calls=True,
        allow_top_level_try=True)
    assert valid, reason

    # The guard is not a hole: the allowlist walks the whole tree, so an import
    # inside the block is refused exactly as one outside it would be.
    smuggled = derived.replace("from threadpoolctl import threadpool_limits",
                               "from subprocess import run")
    valid, reason = support.validate_source(
        smuggled, entrypoint="solve", allowed_imports=algotune.ALLOWED_IMPORTS,
        literal_top_level=False, allow_top_level_calls=True,
        allow_top_level_try=True)
    assert not valid and "subprocess" in reason, reason


def test_the_rejection_note_measures_against_a_tolerance_not_a_bare_ratio():
    """A near-zero reference must not produce a number with no meaning.

    This is a real defect this file did not catch. The note divided by the
    reference element, so on `kalman_filter` -- whose solution vectors are full
    of legitimately near-zero entries -- it told the model "largest relative
    difference 5.033e+23, at element 2095 of 3197: reference 1.19584887991e-23,
    yours 6.01848096735". The reference there is zero to within floating point
    and the candidate produced 6. "Off by 6" is the fact; "off by twenty-three
    orders of magnitude" is an artefact of the denominator, and it both
    overstates the error and hides what it is.

    The tolerance multiple is the part that has to be right: it is what
    separates "tighten the step" from "this is structurally wrong", and those
    need opposite next moves.
    """
    pytest.importorskip("numpy")
    note = _runner_module()._accuracy_note

    near_zero = note([1.0, 2.0, 1.19584887991e-23], [1.0, 2.0, 6.01848096735])
    assert "off by 6.018e+00" in near_zero, near_zero
    assert "5.033e+23" not in near_zero.split("a relative")[0], (
        f"the meaningless ratio leads the note again: {near_zero}")

    # Ten times over tolerance reads as ten times over, not as a catastrophe.
    close = note([1.0, 100.0], [1.0, 100.01])
    assert "10x the tolerance" in close, close

    # A structural miss still reads as one.
    far = note([1.15341132656, -1.24798805191], [-1.81837003711, -1.24798805191])
    assert "2.57e+05x the tolerance" in far, far

    # And a shape miss still reads as one rather than as an accuracy miss.
    mismatched = note([1.0, 2.0, 3.0], [1.0, 2.0])
    assert "structure differs" in mismatched, mismatched
    assert "tolerance" not in mismatched, mismatched


def test_the_rejection_note_never_tells_a_rejected_answer_it_is_correct():
    """Right numbers in the wrong box must not read as "0x the tolerance".

    The second real defect this file did not catch, and the same shape as the
    first: the note flattens both sides before comparing, so a solver returning
    the correct values in the wrong container looked *identical* to it.
    `affine_transform_2d`'s `is_solution` checks `proposed.shape != image.shape`
    before it compares a single value, so a flat list of the correct 20000
    pixels is rejected -- and the note said "worst element 0 of 20000: reference
    0, yours 0 -- off by 0.000e+00, 0x the tolerance", which a model can only
    read as "your answer is right, the harness is broken". 13 of 29 rejections
    in one run of that task were of exactly this kind.

    Two ways out, both needed: describe the container when it differs, and
    refuse to report a passing tolerance multiple on an answer that was
    rejected.
    """
    np = pytest.importorskip("numpy")
    note = _runner_module()._accuracy_note

    # Right values, wrong container: the note must name the structure.
    flat = note({"image": np.zeros((2, 3))}, {"image": [0.0] * 6})
    assert "the structure differs" in flat, flat
    assert "array(2, 3)" in flat and "list[6" in flat, flat
    assert "0x the tolerance" not in flat, flat

    # Same numbers, same container, and is_solution still said no -- the note
    # must send the model looking somewhere other than accuracy.
    identical = note([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert "other than accuracy" in identical, identical
    assert "off by" not in identical, identical

    # A genuine miss is still reported as one, structure equal on both sides.
    real = note([1.0, 2.0], [1.0, 2.5])
    assert "the structure differs" not in real, real
    assert "x the tolerance" in real, real


@needs_sandbox
def test_a_solver_that_caches_on_its_input_wins_nothing(fixture_suite):
    """The warm-up runs on a different instance, so a cache cannot be primed.

    Upstream picks `warmup_idx = (idx - 1) % problem_count` -- the previous
    record in the dataset -- and logs an error if warm-up and timed problem ever
    coincide. This port warmed on a deepcopy of the timed problem, which is a
    hole as much as a deviation: a solver that memoises on the input answers the
    timed call out of a dictionary, in nanoseconds, and is indistinguishable
    from one that is genuinely fast.

    The program here is that solver, written as plainly as possible. It has to
    come out at about 1x, not at a thousand.
    """
    pytest.importorskip("numpy")
    suite = dataclasses.replace(fixture_suite, problems=3)
    code = (
        "import numpy as np\n"
        "_CACHE = {}\n"
        "def solve(problem):\n"
        "    key = problem['matrix'].tobytes()\n"
        "    hit = _CACHE.get(key)\n"
        "    if hit is None:\n"
        "        hit = np.sqrt((problem['matrix'] * problem['matrix']).sum(axis=1))\n"
        "        _CACHE[key] = hit\n"
        "    return {'norms': hit}\n")
    valid, metrics, error = algotune.evaluate_source(
        code, suite=suite, shards=(0,), timeout=120.0, repeats=3)
    assert valid, error
    assert metrics["speedup"] < 20.0, (
        f"the cache was primed on the timed problem: {metrics['speedup']}x")


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


def test_the_prompt_names_no_acceleration_technique(fixture_suite):
    """Upstream's system message says nothing about how to make code fast.

    This is the property the whole comparison rests on, and it is the one that
    decays quietly. A block naming four techniques used to live here and it
    worked -- against the evidence rather than at the top of the prompt it took
    draws reaching for a compiler from 0/8 to 3/8, and on ode_stiff_vanderpol it
    produced 2785x against an upstream field topping out at 2062x. That is
    exactly why it cannot come back without being noticed: a search told which
    levers to pull is not measuring what an agent that had to find them measured,
    and the number would still be reported under AlgoTune's name.

    The package list may name numba and cython, because upstream's does -- it
    substitutes one bare bullet per installed extra. What it may not do is
    explain them or say when to reach for one.
    """
    parent = support.Program("id", 0, None, "def solve(problem):\n    return 1\n",
                             "", {"speedup": 1.0, "problems": 2,
                                  "valid_problems": 2}, True)
    text = algotune.mutation_prompt(parent, suite=fixture_suite)

    for banned in ("just-in-time compiler", "Compile an interpreted loop",
                   "Skip work the answer does not need", "Do less arithmetic",
                   "Pick the specialised routine", "routinely buys 100x"):
        assert banned not in text, f"the prompt names a technique: {banned!r}"

    # The bare list itself is upstream's, and stays.
    assert " - numba\n" in text and " - cython\n" in text
    assert "**TIPS:**" in text          # upstream has one; it is not about speed
    assert "Be creative and optimize your approach!" in text


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


def test_a_failed_draw_is_retried_with_its_error_and_a_working_one_is_not():
    """The fix-it loop, without a model or a sandbox.

    A `-inf` node is a permanent dead end: rank_score 0, never selected again,
    and the direction it was exploring dies with it. On this benchmark that is
    precisely the direction that wins -- a compiled kernel whose first attempt
    misses the tolerance or fails to compile -- so the retry is what keeps the
    route alive. Upstream ERA appends the failure and moves on, which is what
    `repair_attempts=1` still does.
    """
    from examples.era import era_empirical_software as era

    tree = era.EraTree(c_puct=1.0, candidate_limit=10, metric_key="speedup")
    tree.seed(support.Program("root", 0, None, "def solve(problem):\n    return 1\n",
                              "root", {"speedup": 1.0, "score": 1.0}, True), 1.0)

    drawn = []
    replies = iter(["```python\ndef solve(problem):\n    return 'bad'\n```",
                    "```python\ndef solve(problem):\n    return 'bad2'\n```",
                    "```python\ndef solve(problem):\n    return 'good'\n```"])

    def complete(prompt):
        drawn.append(prompt)
        return next(replies)

    def repair(code):
        ok = "good" in code
        return ok, {"speedup": 2.0 if ok else None}, "" if ok else "it raised TypeError"

    counter = {}
    propose = era.make_propose(
        tree, complete, prompt_for=lambda program: "MUTATE",
        repair=repair,
        repair_prompt=lambda program, code, error, attempt: f"FIX {attempt}: {error}",
        repair_attempts=4, repair_counter=counter)

    payload = json.loads(propose("rendered", _task(), "", 0.0))
    assert "good" in payload["code"]
    assert len(drawn) == 3
    assert drawn[0] == "MUTATE"
    assert drawn[1].startswith("FIX 1:") and "TypeError" in drawn[1]
    # Four attempts allowed, third draw works: draws 1-3 all still had a retry
    # in hand so all three were checked; only the last of four would be skipped.
    assert counter == {"drawn": 3, "failed": 2, "repaired": 1}


def test_without_the_loop_a_failure_goes_forward_as_upstream_requires():
    """`repair_attempts=1` is upstream ERA: one draw, and a failure is a node."""
    from examples.era import era_empirical_software as era

    tree = era.EraTree(c_puct=1.0, candidate_limit=10, metric_key="speedup")
    tree.seed(support.Program("root", 0, None, "def solve(problem):\n    return 1\n",
                              "root", {"speedup": 1.0, "score": 1.0}, True), 1.0)
    drawn = []

    def complete(prompt):
        drawn.append(prompt)
        return "```python\ndef solve(problem):\n    return 'bad'\n```"

    def forbidden(_code):
        raise AssertionError("repair ran when no loop was asked for")

    propose = era.make_propose(tree, complete, prompt_for=lambda p: "MUTATE",
                               repair=forbidden, repair_prompt=lambda *a: "FIX",
                               repair_attempts=1)
    payload = json.loads(propose("rendered", _task(), "", 0.0))
    assert "bad" in payload["code"] and len(drawn) == 1


def test_the_loop_gives_up_and_lets_the_failure_become_a_node():
    """Out of attempts, the last draw goes forward. The loop buys retries, not
    a guarantee -- a program that never works still becomes the `-inf` node."""
    from examples.era import era_empirical_software as era

    tree = era.EraTree(c_puct=1.0, candidate_limit=10, metric_key="speedup")
    tree.seed(support.Program("root", 0, None, "def solve(problem):\n    return 1\n",
                              "root", {"speedup": 1.0, "score": 1.0}, True), 1.0)
    counter = {}
    propose = era.make_propose(
        tree, lambda prompt: "```python\ndef solve(problem):\n    return 'bad'\n```",
        prompt_for=lambda p: "MUTATE",
        repair=lambda code: (False, {}, "still broken"),
        repair_prompt=lambda program, code, error, attempt: "FIX",
        repair_attempts=3, repair_counter=counter)
    payload = json.loads(propose("rendered", _task(), "", 0.0))
    assert "bad" in payload["code"]
    assert counter["gave_up"] == 1 and counter["failed"] == 2


def _task():
    from agentdescent.evolution import Task
    return Task(id="shard-0", prompt="p", meta={"shard": 0})


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


def test_the_model_prior_aims_the_exploration_term_rather_than_widening_it():
    """`P(s,a)` must lift a promising node and push a dead-end one down.

    Upstream ERA leaves the prior slot uniform at `1/N`, which spreads the
    exploration budget evenly over a tree that is mostly dead ends. Measured on
    AlgoTune's polynomial_real: at c_puct=1.0 in a 46-node tree the exploration
    term is worth 5.6 rank positions out of 45, and a first-draft Newton solver
    -- slower than LAPACK, and the only route to upstream's 138x -- sits past
    30th. It needs 25 positions and cannot get them.

    Raising c_puct alone reaches it and lifts every dead end with it. The prior
    is what makes the reach selective, and squaring is what makes the
    separation large enough to matter: rated 8 against a mean of 5.5 a node gets
    2.12x the term, rated 2 it gets 0.13x.
    """
    from agentdescent.selection import Candidate, FlatPuct

    rows = [Candidate(artifact_id="a", version=i, score=float(-i), prior=prior)
            for i, prior in enumerate([5.5] * 20 + [8.0, 2.0])]
    uniform = FlatPuct(c_puct=1.0, prior_exponent=0.0)._priors(rows)
    aimed = FlatPuct(c_puct=1.0, prior_exponent=2.0)._priors(rows)

    assert len(set(round(p, 12) for p in uniform)) == 1, "upstream's prior is uniform"
    assert abs(sum(aimed) - 1.0) < 1e-9, "priors must normalise, or c_puct changes meaning"
    promising, dead_end, typical = aimed[20], aimed[21], aimed[0]
    assert promising > typical > dead_end
    assert promising / typical > 2.0, (promising, typical)
    assert dead_end / typical < 0.2, (dead_end, typical)


def test_an_unrated_candidate_is_not_barred_from_selection():
    """No rating means no opinion, not a rating of zero.

    A prior of zero would make the absence of a number the reason a direction
    is never explored, which is worse than the uniform prior it replaced. On
    polynomial_real 5 of 30 replies carried no rating.
    """
    from agentdescent.selection import Candidate, FlatPuct

    rows = [Candidate(artifact_id="a", version=0, score=1.0, prior=8.0),
            Candidate(artifact_id="a", version=1, score=1.0, prior=2.0),
            Candidate(artifact_id="a", version=2, score=1.0, prior=None)]
    priors = FlatPuct(c_puct=1.0, prior_exponent=2.0)._priors(rows)
    assert priors[2] > 0.0
    assert priors[0] > priors[2] > priors[1], priors

    # And with nobody rated at all it degenerates to upstream exactly.
    bare = [Candidate(artifact_id="a", version=i, score=1.0) for i in range(4)]
    assert FlatPuct(c_puct=1.0, prior_exponent=2.0)._priors(bare) == [0.25] * 4


def test_the_promise_line_is_read_and_a_missing_one_is_not_a_zero():
    from examples.era.era_empirical_software import _read_promise

    assert _read_promise("```python\ncode\n```\n\nPROMISE: 8") == 8.0
    assert _read_promise("PROMISE:  3.5\n") == 3.5
    assert _read_promise("promise: 7") == 7.0            # case-insensitive
    assert _read_promise("no rating here") is None
    assert _read_promise("PROMISE: 0") is None           # zero would bar it
    assert _read_promise("PROMISE: banana") is None


def test_a_slower_than_parent_candidate_is_retried_only_when_asked():
    """The regression retry is off by default, because it changes the loop.

    With it off the repair loop rescues programs that failed, which is upstream
    ERA plus a retry. With it on the loop also keeps working on one that runs
    and is slower -- the case a flat tree cannot recover, since such a node
    ranks last and is never selected again.
    """
    from examples.era.era_empirical_software import _is_regression

    slower = {"speedup": 0.5}
    faster = {"speedup": 2.0}
    assert _is_regression(slower, 1.0, "higher", True) is True
    assert _is_regression(slower, 1.0, "higher", False) is False   # off by default
    assert _is_regression(faster, 1.0, "higher", True) is False
    # a metric where lower wins reverses, and an unmeasurable one never fires
    assert _is_regression({"rmse": 2.0}, 1.0, "lower", True) is True
    assert _is_regression({"rmse": 0.5}, 1.0, "lower", True) is False
    assert _is_regression({}, 1.0, "higher", True) is False
    assert _is_regression(slower, None, "higher", True) is False


def test_asking_for_a_rating_changes_only_the_tail_of_the_prompt(fixture_suite):
    parent = support.Program("id", 0, None, "def solve(problem):\n    return 1\n",
                             "", {"speedup": 1.0, "problems": 2,
                                  "valid_problems": 2}, True)
    plain = algotune.mutation_prompt(parent, suite=fixture_suite)
    asked = algotune.mutation_prompt(parent, suite=fixture_suite, ask_promise=True)
    assert asked.startswith(plain.rstrip("\n"))
    assert "PROMISE:" in asked and "PROMISE:" not in plain
    # it asks about the approach after tuning, not about this draft
    assert "after further tuning" in asked


def test_a_rating_survives_the_whole_path_from_reply_to_node():
    """The end-to-end path, because every link of it was already tested and the
    chain still broke.

    `make_propose` parsed the rating and wrote it into its JSON, and `to_diff`
    built its ops dict field by field and silently dropped it -- so a 2x2
    ablation ran a whole arm with `--prior-exponent 2` and every node unrated,
    which reads in the result file as the mechanism doing nothing rather than
    as the mechanism never firing. The unit tests all passed.
    """
    import json as _json
    from examples.era.era_empirical_software import EraStrategy, EraTree, _read_promise

    reply = "```python\ndef solve(p):\n    return p\n```\n\nPROMISE: 8"
    assert _read_promise(reply) == 8.0

    strategy = EraStrategy()
    assert "promise" in strategy.keys()
    proposal = _json.dumps({"code": "def solve(p):\n    return p\n",
                            "change_summary": "", "iteration": 1,
                            "parent_index": 0, "parent_id": "r",
                            "promise": repr(_read_promise(reply))})
    ops = strategy.to_diff({}, proposal, "w", 0, "era_program").ops
    assert ops["promise"] == "8.0", ops

    # ...and reaches the node, where FlatPuct reads it as the prior.
    tree = EraTree(prior_exponent=2.0, metric_key="speedup")
    root = tree.seed(support.Program("r", 0, None, "x", "", {"speedup": 1.0}, True), 1.0)
    node = tree.add_node(
        support.Program("c", 1, "r", "y", "", {"speedup": 0.5}, True), 0.5,
        root.index, promise=float(ops["promise"]))
    assert node.promise == 8.0
    assert node.summary()["promise"] == 8.0

    # An unrated proposal reaches the node as None, not as 0.0.
    bare = _json.dumps({"code": "def solve(p):\n    return p\n",
                        "iteration": 2, "parent_index": 0})
    assert strategy.to_diff({}, bare, "w", 0, "era_program").ops["promise"] == ""


def test_the_result_file_reports_every_repair_counter():
    """A hand-listed subset drops the counter that matters and reads as a zero.

    The `repair` block used to name four keys, so `slower` -- candidates that
    ran and were worse than their parent, the entire point of
    `--repair-regressions` -- never reached the file. An arm that was firing
    read as an arm that never fired, and I reported it as one.
    """
    import argparse
    import inspect
    source = inspect.getsource(port.main)
    assert '"repair": {' in source
    block = source[source.index('"repair": {'):]
    block = block[:block.index("},")]
    assert "sorted(repairs.items())" in block, (
        "the repair block lists keys by hand again; a new counter will vanish")
    assert "regressions_retried" in block


def test_the_sandbox_gives_the_reference_and_the_candidate_the_same_cores():
    """Whatever cores a candidate can reach, the reference must reach too.

    This is a real defect this file did not catch, and it inflated a published
    number. The sandbox pinned OpenMP, OpenBLAS, MKL, NumExpr and vecLib to one
    thread, so the reference's `np.roots` -- a LAPACK eigenvalue solve -- ran on
    one core. None of those variables reach numba: with `OMP_NUM_THREADS=1` set
    and the OpenMP threading layer selected, `numba.get_num_threads()` still
    returned 4, because numba reads `NUMBA_NUM_THREADS` and defaults it to the
    core count. A candidate compiled `@njit(parallel=True)` was therefore timed
    on four cores against a one-core reference -- 2.95x on `polynomial_real`'s
    Aberth winner, inside a number reported as 962x.

    The first fix pinned numba too, and that was wrong the other way: writing a
    parallel implementation *is* an optimisation, upstream sets no thread policy,
    and forbidding it invents a rule AlgoTune does not have. So neither side is
    pinned. What this test defends is the symmetry -- a half-applied cap is the
    only outcome that silently mismeasures.
    """
    env = support._THREAD_ENV
    capped = {k for k, v in env.items() if k.endswith("NUM_THREADS") and v == "1"}
    assert not capped, (
        "a thread cap is back in _THREAD_ENV. It has to reach numba as well as "
        "BLAS or reach neither, because capping one side times a parallel "
        f"candidate against a serial reference; capped: {sorted(capped)}")
    _, built = support.sandbox_wrapper(["/bin/true"], scratch="/tmp")
    assert not [k for k in built if k.endswith("NUM_THREADS")], (
        "the sandbox environment caps threads for one side again")


def test_the_cpu_limit_scales_with_the_cores_a_candidate_may_use():
    """`RLIMIT_CPU` sums CPU seconds over threads, so it has to scale with them.

    With threads uncapped, a candidate using every core burns a 300-second
    budget in 300/cores wall seconds and is killed for using exactly what the
    sandbox gave it -- reported as "the candidate crashed". The cap is a
    runaway ceiling, not a budget anyone should reach.
    """
    assert support.THREAD_BUDGET_FACTOR >= 1
    source = inspect.getsource(algotune.run_candidate)
    assert "THREAD_BUDGET_FACTOR" in source, (
        "--cpu-seconds no longer scales with the core count, so an honestly "
        "parallel candidate is killed for being parallel")


def test_only_one_sandbox_is_timed_at_a_time():
    """Concurrency has to change the wall clock and not the score.

    This task's metric *is* a wall-clock ratio, and a candidate and its
    reference do not respond to contention alike. Measured on `polynomial_real`
    at the harness's own `--repeats 3`, the winning `@njit(parallel=True)`
    solver against its `np.roots` reference, with N spinning processes beside
    it: 834.4x at 0, **3.2x at 2**, 2.6x at 5. The single-threaded solver on the
    same task holds at 535x / 539x / 457x, so it is the parallel candidate being
    measured on a machine that is not there -- an OpenMP region that cannot get
    its threads spins rather than runs.

    With `--workers 3` the search runs three sandboxes, so without this lock a
    parallel candidate's score is a lottery on whether its evaluation collided
    with another. That is not a noisy measurement, it is a different one.
    """
    import inspect
    import threading as _threading

    assert isinstance(algotune._EVALUATION, type(_threading.Lock())), (
        "the evaluation mutex is gone or is no longer a lock")

    source = inspect.getsource(algotune.run_candidate)
    assert "with _EVALUATION:" in source, (
        "run_candidate no longer serialises the sandbox, so three workers time "
        "three candidates against each other")
    # It has to be around the timed run and not around the whole function, or it
    # serialises the search rather than the measurement.
    held = source[source.index("with _EVALUATION:"):]
    held = held[:held.index("except subprocess.TimeoutExpired")]
    assert "subprocess.run(" in held, "the lock no longer covers the timed run"
    assert "validate_source" not in held and "write_text" not in held, (
        "the lock now covers the gate or the file writes; that serialises work "
        "that does not need exclusion")
