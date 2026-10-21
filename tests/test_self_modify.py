"""Self-modifying framework has to be right where the framework edits its own code.

The candidate is a function's source; the gate is the AST walk every other
self-edit runs; the tests must all pass before any benchmark score counts; and
`improve_source` searches the source under that gate. The fixtures pin the
surfaces that make this honest: patches round-trip and bad ones are refused,
the gate blocks imports / dunder / interpreter calls, a failing test is a zero
(never a regression with a number attached), and the run accepts a shorter,
faster source that passes every test.
"""

import pytest

from agentdescent import (
    SelfModifyReport,
    apply_patch,
    evaluate_source,
    gate_source,
    improve_source,
    patch_source,
)


_VERBOSE = "def fn(x):\n    y = x\n    z = y * y\n    return z\n"
_SLIM = "def fn(x):\n    return x * x\n"
_TESTS = [lambda fn: fn(3) == 9, lambda fn: fn(0) == 0, lambda fn: fn(-2) == 4]
_BENCH = lambda fn: 1.0 / (1.0 + len(fn.__code__.co_code))   # bytecode size ~ speed


def test_patches_round_trip_and_bad_ones_are_refused():
    patch = patch_source(_VERBOSE, _SLIM)
    assert "fn(x)" in patch
    assert apply_patch(_VERBOSE, patch).strip() == _SLIM.strip()
    with pytest.raises(ValueError):
        apply_patch(_VERBOSE, "not a patch")


def test_apply_patch_preserves_untouched_lines():
    # a long source with one middle-of-file edit must keep every other line
    old = "def fn(x):\n" + "".join(f"    a{i} = {i}\n" for i in range(10)) \
        + "    return x\n"
    new = old.replace("a5 = 5", "a5 = 50")
    applied = apply_patch(old, patch_source(old, new))
    assert applied == new


def test_apply_patch_refuses_a_mismatched_source():
    old = "def fn(x):\n    return 1\n"
    new = "def fn(x):\n    return 2\n"
    patch = patch_source(old, new)
    with pytest.raises(ValueError):
        apply_patch("def fn(x):\n    return 'totally different'\n", patch)


def test_the_gate_blocks_unsafe_edits():
    with pytest.raises(ValueError):
        gate_source("import os\ndef fn(x):\n    return x\n")
    with pytest.raises(ValueError):
        gate_source("def fn(x):\n    return eval('1')\n")
    with pytest.raises(ValueError):
        gate_source("def fn(x):\n    return __import__('os')\n")
    with pytest.raises(ValueError):
        gate_source("def wrong(x):\n    return x\n")       # wrong function name


def test_a_failing_test_is_a_zero_never_a_number():
    passed, total, score = evaluate_source(
        "def fn(x):\n    return x + 1\n", _TESTS, _BENCH)
    assert passed < total
    assert score == 0.0                                    # gate, not a score


def test_an_unparseable_candidate_is_counted_not_crashed():
    passed, total, score = evaluate_source("def fn(x:\n", _TESTS, _BENCH)
    assert (passed, total, score) == (0, 3, 0.0)


def test_module_level_code_that_raises_is_counted_not_crashed():
    # defaults and module-level statements run at compile time; a raise there
    # used to escape evaluate_source, now it is (0, total, 0.0)
    for bad in ("def fn(x):\n    return x * x\ny = 1/0\n",
                "def fn(x):\n    return x * x\nprint('hi')\n",
                "def fn(x, y=[][0]):\n    return x\n"):
        assert evaluate_source(bad, _TESTS, _BENCH) == (0, 3, 0.0)


def test_the_gate_refuses_a_frame_escape():
    with pytest.raises(ValueError):
        gate_source("def fn(x):\n"
                    "    box = []\n"
                    "    g = (box[0].gi_frame.f_back.f_globals for _ in [1])\n"
                    "    box.append(g)\n"
                    "    return 0\n")
    with pytest.raises(ValueError):
        gate_source("def fn(x):\n"
                    "    return __builtins__['__import__']('os')\n")


def test_improve_source_accepts_a_faster_patch_that_passes_every_test():
    def slim_propose(rendered, task, output, reward):
        return _SLIM

    report, result = improve_source(
        _VERBOSE, _TESTS, _BENCH, propose=slim_propose,
        seeds=(0, 1, 2, 3), rounds=1, n_workers=1, repo_path=None)
    assert isinstance(report, SelfModifyReport)
    assert report.evolved_tests == 3                        # all tests passed
    assert report.evolved_benchmark > report.seed_benchmark
    assert report.gain > 0
    assert "return x * x" in report.value
    assert result.meta["gain"] == pytest.approx(report.gain)


def test_improve_source_needs_a_proposer():
    with pytest.raises(ValueError):
        improve_source(_VERBOSE, _TESTS, _BENCH)


def test_improve_source_model_path_is_wired():
    # the model replies the way real models do: prose plus a ```python fence
    class _FakeCompletion:
        def __call__(self, prompt):
            return ('Here is the improved source:\n```python\n'
                    + _SLIM + '```\nIt is faster and passes the tests.')

    report, result = improve_source(
        _VERBOSE, _TESTS, _BENCH, model=_FakeCompletion(),
        seeds=(0, 1, 2, 3), rounds=1, n_workers=1, repo_path=None)
    assert isinstance(report, SelfModifyReport)
    assert report.evolved_tests == 3
    assert report.gain > 0
    assert "return x * x" in report.value
