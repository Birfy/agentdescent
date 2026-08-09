"""The real DGM objective: what it must reject, and what it must let through.

The surrogate objective cannot fail in these ways -- adding a string to a set
never breaks the agent, never escapes a directory, and never scores anything but
monotonically upward. Every test here exists because the real one can.
"""
from __future__ import annotations

import json
import os

import pytest

from agentdescent.treestrategy import FileTree
from examples.dgm import real_objective as R


# -- the vendored tasks ------------------------------------------------------


def test_every_task_starts_with_exactly_one_failing_test():
    """A task with no failing test measures nothing, and one that fails on
    import measures the fixture rather than the agent."""
    import subprocess
    import sys

    cases = R.load_tasks()
    assert len(cases) >= 4, "the split needs at least four"
    for case in cases:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--no-header",
             "-rN", "--tb=line", "test_lib.py"],
            cwd=str(case.path), capture_output=True, text=True, timeout=120)
        out = proc.stdout + proc.stderr
        # pytest's own count line ("1 failed, 1 passed") is not emitted by every
        # configuration -- this environment suppresses it -- so the check reads
        # the two signals that are always there: the exit status, and the
        # per-test progress characters on the first line.
        progress = out.strip().splitlines()[0] if out.strip() else ""
        assert proc.returncode != 0, f"{case.id}: nothing fails, nothing to learn"
        assert progress.count("F") == 1, (
            f"{case.id}: expected exactly one failing test, saw {progress!r}")
        assert progress.count(".") >= 1, (
            f"{case.id}: everything fails -- a broken fixture, not a bug")
        assert "E" not in progress, (
            f"{case.id}: collection error rather than a test failure")


# -- the evaluator -----------------------------------------------------------


def _llm(reply):
    return lambda _prompt: reply


def test_a_correct_fix_resolves_and_a_wrong_one_does_not():
    cases = {c.id: c for c in R.load_tasks()}
    good = ('def deep_merge(a, b):\n    out = dict(a)\n'
            '    for k, v in b.items():\n'
            '        if isinstance(v, dict) and isinstance(out.get(k), dict):\n'
            '            out[k] = deep_merge(out[k], v)\n'
            '        else:\n            out[k] = v\n    return out\n')
    ok, _ = R.run_agent_on_task(R.SEED_AGENT, cases["merge-nested"], _llm(good))
    assert ok
    bad = 'def deep_merge(a, b):\n    return dict(b)\n'
    ok, detail = R.run_agent_on_task(R.SEED_AGENT, cases["merge-nested"], _llm(bad))
    assert not ok and "failed" in detail


def test_a_self_edit_that_breaks_the_agent_scores_zero_rather_than_raising():
    """The whole reason for a non-monotone objective.

    Under the surrogate this case does not exist: a capability string cannot
    contain a syntax error. Here it can, and the search has to be able to score
    it -- an exception would end the run instead of recording a bad child.
    """
    cases = {c.id: c for c in R.load_tasks()}
    broken = dict(R.SEED_AGENT)
    broken["agent/tools.py"] = "this is not python("
    ok, detail = R.run_agent_on_task(broken, cases["range-end"], _llm("whatever"))
    assert not ok
    assert detail, "a failure with no explanation is not scoreable"


def test_an_agent_file_cannot_escape_the_workspace():
    with pytest.raises(ValueError, match="escapes"):
        R._materialise({"../escape.py": "x = 1\n"},
                       __import__("pathlib").Path(
                           __import__("tempfile").mkdtemp()))


# -- the self-modification ---------------------------------------------------


def test_an_empty_proposal_warns_rather_than_looking_like_no_proposal():
    """An empty completion and a deliberate "nothing to suggest" are the same
    return value and opposite diagnoses. Measured on `deepseek-v4-flash` at
    `max_tokens=4096`: 5 of 6 proposals came back empty, and the run ended
    byte-identical to its seed with nothing in the log to say why."""
    with pytest.warns(RuntimeWarning, match="empty completion"):
        assert R.propose_self_edit(R.SEED_AGENT, [("t", False, "x")], _llm("")) is None


@pytest.mark.parametrize("proposal", [
    "here is my fix\ndef x(): pass",           # no PATH header
    "PATH: ../../etc/passwd\nboom",            # escapes
    "PATH: agent/tools.py\n   ",               # empty body
    "PATH: agent/nope.py\nx = 1",              # outside the editable set
])
def test_a_malformed_self_edit_is_refused_not_applied(proposal):
    assert R.parse_self_edit(proposal) is None


def test_a_well_formed_self_edit_becomes_a_diff():
    """`FileTree.to_diff` parses the shipped `<EDITS>` protocol, not a rendered
    tree. Handing it the wrong shape returns `None` on every proposal -- silently
    -- which is indistinguishable from a search that found nothing."""
    body = 'def read_source(x):\n    return "patched"\n'
    proposal = json.dumps({"edits": [{"path": "agent/tools.py", "content": body}]})
    tree = FileTree(editable=list(R.EDITABLE))
    diff = tree.to_diff(R.SEED_AGENT, proposal, "w1", 1, "coding_agent")
    assert diff is not None and list(diff.ops) == ["agent/tools.py"]


@pytest.mark.parametrize("path", ["agent/nope.py", "../escape.py"])
def test_the_editable_boundary_is_the_engines_not_a_hand_check(path):
    proposal = json.dumps({"edits": [{"path": path, "content": "x = 1\n"}]})
    tree = FileTree(editable=list(R.EDITABLE))
    assert tree.to_diff(R.SEED_AGENT, proposal, "w", 1, "coding_agent") is None


def test_a_no_op_edit_is_not_a_diff():
    proposal = json.dumps({"edits": [
        {"path": "agent/tools.py", "content": R.SEED_AGENT["agent/tools.py"]}]})
    tree = FileTree(editable=list(R.EDITABLE))
    assert tree.to_diff(R.SEED_AGENT, proposal, "w", 1, "coding_agent") is None


# -- the loop ----------------------------------------------------------------


def test_the_loop_commits_self_edits_and_grows_the_archive():
    """End to end, offline: a scripted actor that fixes two tasks and appends a
    comment to its own tools on every proposal."""
    fixes = {
        "expand": ('def expand(spec):\n    if "-" not in spec:\n'
                   '        return [int(spec)]\n    lo, hi = spec.split("-")\n'
                   '    return list(range(int(lo), int(hi) + 1))\n'),
        "sort_versions": ('def sort_versions(vs):\n'
                          '    return sorted(vs, key=lambda v: '
                          '[int(p) for p in v.split(".")])\n'),
    }
    seen = {"n": 0}

    def actor(prompt):
        if prompt.startswith("You are a self-improving coding agent"):
            seen["n"] += 1
            return ("PATH: agent/tools.py\n"
                    + R.SEED_AGENT["agent/tools.py"]
                    + f"\n# improvement {seen['n']}\n")
        for name, body in fixes.items():
            if f"def {name}" in prompt:
                return body
        return "# unchanged\n"

    result = R.run_dgm_real(actor, generations=3, selfimprove_size=2,
                            max_rollouts=6)
    assert result.rollouts > 0
    assert seen["n"] > 0, "the proposer was never asked for a self-modification"
    committed = sum(h.committed for h in result.history)
    assert committed > 0, "no self-edit ever reached the head"


def test_an_agent_that_retries_gets_more_than_one_model_call():
    """The bridge must not decide how many calls an agent may make.

    It used to: the harness ran the agent once to capture a prompt, answered it,
    and re-ran with that single reply served from a file. "One call per task" was
    then a property of the harness, and the first self-modification this
    objective ever produced was a retry loop -- the one improvement that scheme
    made impossible. It scored 0.875 -> 0.500 and the regression was the
    harness's.
    """
    cases = {c.id: c for c in R.load_tasks()}
    retrying = dict(R.SEED_AGENT)
    retrying["agent/solve.py"] = (
        'from tools import read_source, run_tests, write_source\n\n'
        'def solve(task_dir, llm):\n'
        '    for _ in range(3):\n'
        '        reply = llm("attempt")\n'
        '        write_source(task_dir, reply)\n'
        '        passed, failed, _ = run_tests(task_dir)\n'
        '        if failed == 0 and passed > 0:\n'
        '            return\n')
    seen = {"n": 0}
    good = ('def expand(spec):\n    if "-" not in spec:\n        return [int(spec)]\n'
            '    lo, hi = spec.split("-")\n    return list(range(int(lo), int(hi) + 1))\n')

    def flaky(_prompt):
        seen["n"] += 1
        return "# not python at all(" if seen["n"] < 3 else good

    ok, _ = R.run_agent_on_task(retrying, cases["range-end"], flaky)
    assert seen["n"] == 3, f"the agent got {seen['n']} call(s), not the 3 it asked for"
    assert ok, "the third attempt was correct and should have resolved the task"


def test_a_runaway_agent_is_cut_off_rather_than_billed_forever():
    cases = {c.id: c for c in R.load_tasks()}
    looping = dict(R.SEED_AGENT)
    looping["agent/solve.py"] = (
        'def solve(task_dir, llm):\n'
        '    while True:\n        llm("again")\n')
    seen = {"n": 0}

    def counting(_prompt):
        seen["n"] += 1
        return "# nope\n"

    ok, detail = R.run_agent_on_task(looping, cases["range-end"], counting,
                                     max_calls=4)
    assert not ok
    assert seen["n"] <= 4, f"the budget did not hold: {seen['n']} calls"
    assert "exceeded" in detail


def test_the_baseline_is_measured_even_when_no_self_edit_ever_lands():
    """`seed_score` used to be set inside `step()`, which only runs when a card
    reaches the merger -- so a run whose self-modifications all failed reported
    `DGMContext.seed_score`'s 0.0 default as though it had measured it. The
    final number came from the engine and was right; only the thing it was
    compared against was invented, which is the more dangerous half.
    """
    cases = R.load_tasks()[:6]
    # An actor that fixes nothing and proposes nothing: no card can ever arrive.
    result = R.run_dgm_real(lambda _p: "", cases=cases, generations=1,
                            selfimprove_size=2, max_rollouts=2)
    assert getattr(result, "dgm_seed_score", None) is not None, (
        "the baseline was never measured")
    assert len(getattr(result, "dgm_archive", [])) == 1, (
        "the seed itself belongs in the archive before round 0")


HASH_ORDER_TASKS = ("cache-eviction", "dedupe-order", "dict-invert-dupes",
                    "set-difference-order", "strip-prefix", "sum-nested-dict",
                    "unique-preserve-key")


@pytest.mark.parametrize("task_id", HASH_ORDER_TASKS)
def test_a_bug_about_ordering_fails_on_every_hash_seed(task_id):
    """These seven fixtures are the ones whose `lib.py` or `test_lib.py` touches
    a `set` or a `dict`, which is where a task can stop being a fixed measurement
    and become a coin flip.

    `dedupe-order` was one. Its bug was `list(set(xs))` losing order, asserted on
    strings -- and Python randomises string hashing per process, so on about one
    seed in fifteen the buggy code came out in the right order, the task had
    nothing failing, and `test_every_task_starts_with_exactly_one_failing_test`
    failed in CI while passing on every local run. `set-difference-order` was
    worse at roughly one seed in five. Both now assert on integers, which hash to
    themselves.

    A flaky fixture is not only a flaky test: the DGM agent is scored on these,
    so a seed that hid the bug would have handed it a free solve.
    """
    import subprocess
    import sys

    case = {c.id: c for c in R.load_tasks()}[task_id]
    for seed in ("0", "11", "24", "48", "55"):   # seeds that used to hide it
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--no-header", "-rN",
             "--tb=no", "test_lib.py"],
            cwd=str(case.path), capture_output=True, text=True, timeout=120,
            env={**os.environ, "PYTHONHASHSEED": seed})
        assert proc.returncode != 0, (
            f"{task_id} passes under PYTHONHASHSEED={seed}: the bug is hidden by "
            f"hash order, so this task measures the seed rather than the agent")
