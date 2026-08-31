"""The ERA SWE-bench Science task: the split, the evaluator, the patch, the tree.

The first block is the one that matters most. This port searches over patches
while a *held-out programmatic verifier* decides the result, and the thing that
would quietly destroy the number it reports is the search reading the tests it
is measured on. So the split is checked from both ends -- what a shard can
select, and what a prompt can quote -- rather than described in a docstring.

The second block holds the evaluator to the release's own grader. The port does
not run that grader for a node (it needs a *selection*, which the grader has no
way to express) so it runs a copy with one thing added, and a copy that had
drifted would report a different number than the benchmark does for the same
patch.

The rest is machinery: a patch survives the round trip that a Python-shaped
reply parser would have destroyed, an agent session that changed nothing is a
node rather than a crash, and the whole flat-PUCT loop runs over patch-shaped
candidates with no Docker daemon, no release download and no agent.

The tests that need the release or a Docker daemon are marked and skipped:
``AGENTDESCENT_SWE_SCIENCE_NETWORK=1`` turns on the ones that fetch the
published bundles, and the container tests skip themselves when no daemon
answers.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re

import pytest

from examples.era import _era_swe_science as swe
from examples.era import _era_swe_science_runner as runner
from examples.era import era_swe_science as port
from examples.era._era_domain import Domain
from examples.era._era_support import Program, extract_program
from examples.era.era_empirical_software import run_agentdescent_era


needs_release = pytest.mark.skipif(
    not os.getenv("AGENTDESCENT_SWE_SCIENCE_NETWORK"),
    reason="set AGENTDESCENT_SWE_SCIENCE_NETWORK=1 to check the published release")
needs_docker = pytest.mark.skipif(
    swe.docker_backend() is None,
    reason="SWE-bench Science is distributed as Docker images; no daemon here")

FOUR_TESTS = ("/tests/private_tests/t.py::t1", "/tests/private_tests/t.py::t2",
              "/tests/private_tests/t.py::t3", "/tests/private_tests/t.py::t4")


def make_suite(tests=FOUR_TESTS, *, held_back_frac=0.25, shards=4, test_shards=2,
               seed=0, instruction="Repair the inconsistent workflow."):
    """A `Suite` built by hand, so the split and the prompts are testable offline."""
    visible, held_back = swe.split_tests(tests, held_back_frac=held_back_frac,
                                         seed=seed)
    held_back_shards = test_shards if held_back else 0
    return swe.Suite(
        task_id="001", title="Repair an inconsistent workflow",
        domain="computational-reaction-chemistry", language="python",
        repository_url="https://example.invalid/repo", base_commit="0" * 40,
        source_license="MIT", license_gate="none",
        environment_image="example.invalid/env@sha256:" + "0" * 64,
        verifier_image="example.invalid/verifier@sha256:" + "1" * 64,
        instruction=instruction, workdir="/app/task_001",
        tests=tuple(tests), held_back=held_back,
        shard_tests=swe.build_shards(visible, held_back, scoring_shards=shards,
                                     held_back_shards=held_back_shards),
        scoring_shards=shards, held_back_shards=held_back_shards, seed=seed)


# ---------------------------------------------------------------------------
# The held-back split. Everything this port reports rests on it.
# ---------------------------------------------------------------------------


def test_no_scoring_shard_can_select_a_held_back_test():
    """The search picks shards; if a scoring shard held one, it would be seen."""
    suite = make_suite()
    assert suite.held_back, "the fixture must hold something back to test this"
    visible = suite.tests_for(range(suite.scoring_shards))
    assert set(visible).isdisjoint(suite.held_back)
    assert set(visible) == set(suite.visible)


def test_the_held_back_shards_hold_only_held_back_tests():
    suite = make_suite()
    assert set(suite.tests_for(suite.test_range())) == set(suite.held_back)


def _scored(report_private, *, public_passed=0):
    """Metrics shaped like a real evaluation, with a private failure report."""
    return {"passed": 1, "collected": 3, "public": public_passed, "checks": 4,
            "pass_rate": 0.5, "report": report_private,
            "report_public": "" if public_passed else
                             "public reproduction (reproduce.py) exited 1:\n{}"}


def test_a_mutation_prompt_quotes_no_held_back_test():
    """The prompt is the other end of the leak: an agent that read a held-out
    test's name could write to it directly. Checked at the *most* permissive
    feedback level, because that is where there is something to leak."""
    suite = make_suite()
    metrics = _scored("FAILED " + suite.visible[0])
    program = Program("p", 3, None, "diff --git a/x b/x\n", "tried a thing",
                      metrics, True, "")
    text = swe.mutation_prompt(program, suite=suite, agent_timeout=600,
                               feedback="tests")
    for node in suite.held_back:
        assert node not in text
        assert node.rsplit("::", 1)[-1] not in text
    assert suite.instruction in text, "the benchmark's own instruction must go in whole"


def test_the_default_feedback_quotes_nothing_from_the_private_suite():
    """The defect the first runs shipped with, and the reason `--feedback`
    exists.

    pytest's traceback embeds the **body** of the failing test, so a prompt
    built from the private suite hands over its assertions and expected values
    before the agent makes an edit. On the benchmark an agent has `reproduce.py`
    and nothing else. The default must therefore quote the public reproduction
    and not one word of the private run -- not the source, not a node id, not
    even how many checks there were.
    """
    suite = make_suite()
    metrics = _scored(
        "=== FAILURES ===\n"
        "def test_secret_invariant():\n>   assert rotor_number(g) == 3\n"
        "FAILED /tests/private_tests/test_secret.py::test_secret_invariant")
    program = Program("p", 1, None, "diff --git a/x b/x\n", "a change", metrics,
                      True, "")
    text = swe.mutation_prompt(program, suite=suite, agent_timeout=600)

    assert "test_secret_invariant" not in text
    assert "rotor_number(g) == 3" not in text
    assert "/tests/private_tests" not in text
    assert "3 private check" not in text and "1 of them passed" not in text
    # ...but the benchmark's own signal is still there.
    assert "public reproduction" in text
    assert "does NOT exit 0" in text


def test_counts_says_how_many_and_never_which():
    suite = make_suite()
    metrics = _scored("def test_secret_invariant(): assert x == 3")
    program = Program("p", 1, None, "", "", metrics, True, "")
    text = swe.mutation_prompt(program, suite=suite, agent_timeout=600,
                               feedback="counts")
    assert "3 private check(s)" in text and "1 of them passed" in text
    assert "test_secret_invariant" not in text


def test_tests_level_says_out_loud_that_more_are_hidden():
    """At `tests` the agent has seen the visible split's source, so telling it
    the verifier's tests "cannot be read" would be false."""
    suite = make_suite()
    program = Program("p", 1, None, "", "", _scored("boom"), True, "")
    shown = swe.mutation_prompt(program, suite=suite, agent_timeout=600,
                               feedback="tests")
    assert "further tests you have not been shown" in shown
    assert "cannot see and cannot read" not in shown
    hidden = swe.mutation_prompt(program, suite=suite, agent_timeout=600)
    assert "cannot see and cannot read" in hidden


def test_an_unknown_feedback_level_is_refused():
    program = Program("p", 0, None, "", "", {}, True, "")
    with pytest.raises(ValueError):
        swe.mutation_prompt(program, suite=make_suite(), agent_timeout=60,
                            feedback="everything")


def test_the_split_is_deterministic_in_the_seed():
    assert swe.split_tests(FOUR_TESTS, held_back_frac=0.5, seed=7) == \
        swe.split_tests(FOUR_TESTS, held_back_frac=0.5, seed=7)
    assert swe.split_tests(FOUR_TESTS, held_back_frac=0.5, seed=7) != \
        swe.split_tests(FOUR_TESTS, held_back_frac=0.5, seed=8)


def test_the_split_always_leaves_something_visible():
    """A task whose whole suite was held back would give the tree a constant
    score, and the search nothing at all to do."""
    for frac in (0.5, 0.9, 1.0, 4.0):
        visible, held_back = swe.split_tests(FOUR_TESTS, held_back_frac=frac, seed=0)
        assert visible
        assert set(visible).isdisjoint(held_back)
        assert set(visible) | set(held_back) == set(FOUR_TESTS)


def test_a_one_test_suite_holds_nothing_back_rather_than_pretending():
    suite = make_suite(tests=("a::only",))
    assert suite.held_back == ()
    assert suite.held_back_shards == 0
    assert suite.test_range() == ()
    assert suite.data_summary()["held_back_tests"] == 0


def test_a_big_enough_suite_is_cut_into_disjoint_shards():
    suite = make_suite(tests=tuple(f"t.py::t{i}" for i in range(16)),
                       held_back_frac=0.0, shards=4)
    buckets = suite.shard_tests[:suite.scoring_shards]
    assert all(buckets), "an empty shard is a rollout that scores nothing"
    assert set().union(*map(set, buckets)) == set(suite.visible)
    assert sum(len(b) for b in buckets) == len(suite.visible), "shards overlapped"
    assert suite.shard_rule == "subset"
    assert suite.data_summary()["shard_rule"] == "subset"


def test_a_suite_too_small_to_subsample_is_replicated_rather_than_shredded():
    """`evolve()` needs four rollout tasks and most of this benchmark's private
    suites hold three to five tests. Round-robining those is not sampling: a
    node's score would come from whichever single test the engine's held-out
    split landed on."""
    suite = make_suite(tests=("t.py::a", "t.py::b", "t.py::c"), shards=4)
    buckets = suite.shard_tests[:4]
    assert all(buckets)
    assert all(bucket == buckets[0] for bucket in buckets)
    assert set(buckets[0]) == set(suite.visible)
    assert suite.shard_rule == "replicate"


def test_a_replicated_shard_still_never_holds_a_held_back_test():
    suite = make_suite(tests=("t.py::a", "t.py::b", "t.py::c"), shards=4)
    assert suite.shard_rule == "replicate"
    assert set(suite.tests_for(range(suite.scoring_shards))).isdisjoint(suite.held_back)


# ---------------------------------------------------------------------------
# The evaluator is the release's own grader, plus a selection
# ---------------------------------------------------------------------------


def _junit(tests, failures=0, errors=0, skipped=0):
    return (f'<testsuites><testsuite tests="{tests}" failures="{failures}" '
            f'errors="{errors}" skipped="{skipped}"/></testsuites>')


def test_the_runner_counts_the_way_the_release_grader_counts(tmp_path):
    """`passed = tests - failures - errors`, skips included, is the grader's own
    arithmetic. A runner that counted a skip as anything else would report a
    different number than the benchmark does for the same run."""
    path = tmp_path / "junit.xml"
    path.write_text(_junit(5, failures=1, errors=1, skipped=1), encoding="utf-8")
    assert runner._counts(str(path)) == (5, 3, 2)
    path.write_text(_junit(3), encoding="utf-8")
    assert runner._counts(str(path)) == (3, 3, 0)


def test_a_missing_junit_file_is_zero_rather_than_a_crash(tmp_path):
    assert runner._counts(str(tmp_path / "absent.xml")) == (0, 0, 0)


def test_the_runner_reanchors_node_ids_onto_the_private_test_root(monkeypatch):
    """pytest prints ids relative to the rootdir it computed, which for a suite
    under /tests is not the directory the grader runs from. Handing those back
    unchanged selects nothing, and a selection of nothing scores vacuously."""
    monkeypatch.setattr(runner, "_run", lambda *a, **k: (
        0, "t_sym.py::test_one\nt_sym.py::test_two\n2 tests collected\n", 0.1))
    payload = runner._collect(
        {"workdir": "/app/task_001", "tests_root": "/tests/private_tests"}, {})
    assert payload["tests"] == ["/tests/private_tests/t_sym.py::test_one",
                               "/tests/private_tests/t_sym.py::test_two"]


def test_an_absolute_node_id_is_left_alone(monkeypatch):
    monkeypatch.setattr(runner, "_run", lambda *a, **k: (
        0, "/tests/private_tests/t.py::test_one\n", 0.1))
    payload = runner._collect(
        {"workdir": "/app/task_001", "tests_root": "/tests/private_tests"}, {})
    assert payload["tests"] == ["/tests/private_tests/t.py::test_one"]


def test_the_score_counts_the_public_reproduction_as_a_check():
    """It is the benchmark's own first condition: a patch that breaks
    `reproduce.py` has broken the task however many private tests still pass."""
    valid, metrics, error = swe._metrics_from(
        {"applied": True, "public": {"passed": 0, "return_code": 1, "output": "boom"},
         "private": {"collected": 3, "passed": 3, "failed": 0, "output": ""}},
        selection=("a::1", "a::2", "a::3"), public=True, seconds=1.0)
    assert valid and not error
    assert metrics["pass_rate"] == pytest.approx(3 / 4)
    assert metrics["subset_resolved"] == 0


def test_every_check_passing_is_the_benchmark_rule_on_the_visible_subset():
    _, metrics, _ = swe._metrics_from(
        {"applied": True, "public": {"passed": 1, "return_code": 0, "output": ""},
         "private": {"collected": 2, "passed": 2, "failed": 0, "output": ""}},
        selection=("a::1", "a::2"), public=True, seconds=1.0)
    assert metrics["pass_rate"] == 1.0
    assert metrics["subset_resolved"] == 1


def test_a_patch_that_does_not_apply_is_an_invalid_node_not_a_zero():
    """Upstream's sentinel. A candidate that never ran is not a candidate that
    ran and failed, and `-inf` is what keeps FlatPuct from selecting it again."""
    valid, metrics, error = swe._metrics_from(
        {"applied": False, "apply_error": "error: patch does not apply"},
        selection=("a::1",), public=True, seconds=1.0)
    assert valid is False
    assert metrics["score"] == float("-inf")
    assert "does not apply" in error


def test_a_selection_that_collected_nothing_is_invalid_rather_than_perfect():
    """Vacuous success is the failure mode of a graded score: zero of zero tests
    failing would otherwise rank above a patch that fixed three of four."""
    valid, metrics, error = swe._metrics_from(
        {"applied": True, "public": {"passed": 1, "return_code": 0, "output": ""},
         "private": {"collected": 0, "passed": 0, "failed": 0,
                     "output": "ERROR collecting"}},
        selection=("a::1", "a::2"), public=True, seconds=1.0)
    assert valid is False
    assert metrics["score"] == float("-inf")
    assert "collected none" in error


def test_the_reward_the_gate_uses_is_the_score_the_tree_ranks_on():
    """A port whose gate disagreed with its tree would select against itself."""
    rates = [0.0, 0.25, 0.5, 0.75, 1.0]
    scored = [swe.framework_score({"pass_rate": rate}) for rate in rates]
    assert scored == rates
    assert swe.framework_score({"pass_rate": float("-inf")}) == 0.0
    assert swe.framework_score({}) == 0.0


def test_the_warnings_summary_is_dropped_from_what_a_child_is_shown():
    """On a scientific stack it is routinely longer than the traceback, and a
    truncated raw tail hands the agent a page of DeprecationWarning where the
    assertion should be."""
    output = "\n".join([
        "=================================== FAILURES ===================================",
        "____ test_rotor ____",
        "E   ValueError: too many values to unpack",
        "=============================== warnings summary ===============================",
        "source/x.py:1: PyparsingDeprecationWarning: 'delimitedList' deprecated",
        "=========================== short test summary info ============================",
        "FAILED t.py::test_rotor",
    ])
    report = swe.pytest_failures(output)
    assert "ValueError" in report
    assert "FAILED t.py::test_rotor" in report
    assert "PyparsingDeprecationWarning" not in report


def test_a_pytest_run_with_no_sections_keeps_its_output():
    assert swe.pytest_failures("boom") == "boom"
    assert swe.pytest_failures("") == ""


def test_grade_runs_the_bundle_grader_and_not_the_image_s(monkeypatch):
    """The release mounts its own grader over the image's, and it means it:
    some published verifier images carry a stale grader that collects nothing
    and would report a floor of zero as a result."""
    seen = {}

    def fake_release_file(relative, release=None, timeout=120.0):
        seen["relative"] = relative
        return "print('the bundle grader')\n"

    class FakeProc:
        returncode = 0
        stderr = ""
        stdout = '{"reward": 1, "private": {"collected": 3, "passed": 3}}'

    def fake_docker(args, **_kwargs):
        seen["args"] = list(args)
        mount = [a for a in args if a.endswith(":/era")][0]
        seen["grader"] = (pathlib.Path(mount.split(":/era")[0])
                          / "grader.py").read_text(encoding="utf-8")
        return FakeProc()

    monkeypatch.setattr(swe, "_release_file", fake_release_file)
    monkeypatch.setattr(swe, "_docker", fake_docker)
    result = swe.grade("", suite=make_suite())
    assert result["reward"] == 1
    assert seen["relative"] == "tasks/task_001/tests/grader.py"
    assert seen["grader"] == "print('the bundle grader')\n"
    assert "python /era/grader.py" in " ".join(seen["args"])
    assert "/tests/grader.py" not in " ".join(seen["args"])


@needs_docker
@needs_release
def test_the_bundle_grader_is_the_one_that_runs():
    """Task 034 is the case that makes this necessary: its verifier image runs
    pytest on `/tests/private_tests/test_task_034.py`, which does not exist
    beside the `test_stochastic_orientation.py` that does -- so the image's
    grader collects nothing, exits 4, and scores every submission 0."""
    bundle = swe._release_file("tasks/task_034/tests/grader.py", release=None,
                               timeout=120)
    assert '"/tests/private_tests",' in bundle
    manifest = swe.load_manifest()
    suite = swe.prepare_suite("034", manifest=manifest, shards=4, test_shards=2,
                              held_back_frac=0.0, seed=0)
    graded = swe.grade("", suite=suite)
    assert graded["private"]["collected"] == len(suite.tests) > 0, (
        "the grader that ran collected nothing, so it was the image's")


# ---------------------------------------------------------------------------
# The candidate is a patch, not a Python file
# ---------------------------------------------------------------------------


DIFF_WITH_FENCES = (
    "diff --git a/README.md b/README.md\n"
    "--- a/README.md\n+++ b/README.md\n@@ -1,3 +1,3 @@\n"
    "-```python\n-print(1)\n-```\n+```python\n+print(2)\n+```\n")


def test_a_patch_survives_the_round_trip_that_breaks_the_python_parser():
    """Why the sentinel exists. `extract_program` takes the longest ```fenced
    block, and a diff of a markdown file carries fences of its own."""
    reply = swe.wrap_reply("Bumped the printed value.", DIFF_WITH_FENCES)
    patch, summary = swe.extract_patch(reply)
    assert patch == DIFF_WITH_FENCES.strip("\n")
    assert summary == "Bumped the printed value."
    assert extract_program(reply)[0] != patch, (
        "if upstream's parser handled this, the port would not need its own")


def test_extract_patch_skips_the_promise_line_when_looking_for_a_summary():
    reply = swe.wrap_reply("PROMISE: 8\nRewrote the axis selection.", "diff --git\n")
    assert swe.extract_patch(reply)[1] == "Rewrote the axis selection."


def test_a_session_that_changed_nothing_is_an_empty_patch_not_a_crash():
    patch, summary = swe.extract_patch(swe.wrap_reply("I could not find it.", ""))
    assert patch == ""
    assert summary == "I could not find it."


def test_a_reply_with_no_patch_markers_yields_no_patch():
    assert swe.extract_patch("the agent said something else")[0] == ""


def test_the_trailing_newline_git_apply_needs_survives_the_ledger():
    """`Diff.ops` are strings and `to_diff` strips them, and `git apply` refuses
    a patch whose last hunk has no newline."""
    assert swe._normalise_patch("diff --git a/x b/x").endswith("\n")
    assert swe._normalise_patch("") == ""
    assert swe.patch_id("a\n") == swe.patch_id("a")


def test_the_prompt_carries_the_parent_patch_to_the_workspace():
    """The mutation needs the parent as a checkout, not as text, so the domain's
    prompt carries both and the mutation opens the one and reads the other."""
    suite = make_suite()
    program = Program("p", 2, None, DIFF_WITH_FENCES, "a change", {}, True, "")
    payload = json.loads(swe.envelope(program, suite=suite, agent_timeout=60))
    assert payload["patch"] == DIFF_WITH_FENCES
    assert suite.instruction in payload["text"]


def test_the_root_prompt_says_it_is_the_baseline():
    suite = make_suite()
    root = Program("p", 0, None, "", "", {}, True, "")
    text = swe.mutation_prompt(root, suite=suite, agent_timeout=60)
    assert "unmodified baseline" in text
    assert "PROMISE" not in text, "upstream's prior is uniform unless asked for"
    asked = swe.mutation_prompt(root, suite=suite, agent_timeout=60, ask_promise=True)
    assert "PROMISE: <n>" in asked


# ---------------------------------------------------------------------------
# The release catalogue
# ---------------------------------------------------------------------------


FAKE_MANIFEST = {
    "001": {"license_gate": "none", "title": "a"},
    "002": {"license_gate": "none", "title": "b"},
    "003": {"license_gate": "gpl-family", "title": "c"},
    "004": {"license_gate": "none", "title": "d"},
}


def test_selection_understands_the_release_range_syntax():
    assert swe.parse_selection("002,003-004", FAKE_MANIFEST) == ("002", "003", "004")
    assert swe.parse_selection("2", FAKE_MANIFEST) == ("002",)
    assert swe.parse_selection("all", FAKE_MANIFEST) == ("001", "002", "003", "004")
    assert swe.parse_selection("unrestricted", FAKE_MANIFEST) == ("001", "002", "004")
    assert swe.parse_selection("", FAKE_MANIFEST) == swe.DEFAULT_TASKS


def test_selection_names_an_unknown_task_rather_than_running_the_rest():
    with pytest.raises(SystemExit) as excinfo:
        swe.parse_selection("001,999", FAKE_MANIFEST)
    assert "999" in str(excinfo.value)


def test_a_repeated_id_is_run_once():
    assert swe.parse_selection("001,001,002", FAKE_MANIFEST) == ("001", "002")


def test_the_workdir_is_derived_from_the_task_id():
    assert swe.WORKDIR_TEMPLATE.format(task_id="007") == "/app/task_007"


@needs_release
def test_the_release_manifest_is_the_release_this_port_pins():
    manifest = swe.load_manifest()
    assert len(manifest) == swe.RELEASE_TASKS
    assert len(swe.unrestricted(manifest)) == swe.UNRESTRICTED_TASKS
    assert set(swe.DEFAULT_TASKS) <= set(swe.unrestricted(manifest)), (
        "the default selection must not need --allow-restricted-licenses")
    for row in manifest.values():
        assert row["image_platform"] == "linux/amd64"
        assert "@sha256:" in row["environment_image"], "images must be digest-pinned"
        assert "@sha256:" in row["verifier_image"]


@needs_release
def test_the_licence_gate_names_the_same_96_tasks_the_release_selects():
    """One source of truth: the port derives the default selection from
    `license_gate`, and the release ships it as a file."""
    manifest = swe.load_manifest()
    published = json.loads(swe._release_file(
        "selections/default-96.json", release=None, timeout=120))["task_ids"]
    assert list(swe.unrestricted(manifest)) == sorted(published)


@needs_release
def test_every_published_grader_is_the_same_grader():
    """The claim the runner rests on. If the graders differed per task, one
    subset evaluator could not stand in for all of them."""
    digests = set()
    for task_id in sorted(swe.load_manifest()):
        text = swe._release_file(f"tasks/task_{task_id}/tests/grader.py",
                                 release=None, timeout=120)
        normalised = re.sub(r"task_\d\d\d", "task_NNN", text)
        digests.add(hashlib.sha256(normalised.encode("utf-8")).hexdigest())
    assert len(digests) == 1


@needs_release
def test_the_bundles_agree_with_the_constants_this_port_derives():
    for task_id in sorted(swe.load_manifest()):
        toml = swe._release_file(f"tasks/task_{task_id}/task.toml",
                                 release=None, timeout=120)
        assert f'workdir = "{swe.WORKDIR_TEMPLATE.format(task_id=task_id)}"' in toml
        assert f'artifacts = ["{swe.MODEL_PATCH_ARTIFACT}"]' in toml
        assert f"timeout_sec = {swe.RELEASE_AGENT_TIMEOUT}" in toml
        assert f"timeout_sec = {swe.RELEASE_VERIFIER_TIMEOUT}" in toml


# ---------------------------------------------------------------------------
# The whole flat-PUCT loop, over patch-shaped candidates
# ---------------------------------------------------------------------------


HUNK = "diff --git a/s.py b/s.py\n--- a/s.py\n+++ b/s.py\n@@\n+fix{n}\n"


def _toy_domain(suite):
    """A task whose score is "how many of the fixes are in the patch".

    No container, no agent, no release: what is under test is that the search
    ranks, selects and commits *patches* -- which upstream's loop has never seen
    -- and that the `extract` seam puts the patch rather than the reply into the
    node.
    """
    def evaluate(patch, shards):
        selected = suite.tests_for(shards)
        if not selected:
            return True, {"score": 0.0, "pass_rate": 0.0, "passed": 0,
                          "collected": 0, "checks": 0, "report": ""}, ""
        passed = sum(1 for node in selected if f"fix{node[-1]}" in patch)
        # The extra check is the public reproduction, which nothing here fixes.
        # It keeps the toy off 1.0, where `evolve(solved_threshold=1.0)` would
        # stop the run early and the budget under test would go unspent.
        checks = len(selected) + 1
        rate = passed / checks
        return True, {"score": rate, "pass_rate": rate, "passed": passed,
                      "collected": len(selected), "checks": checks,
                      "report": ""}, ""

    return Domain(
        name="toy", entrypoint="model.patch", metric_key="pass_rate",
        metric_better="higher", initial_program="",
        initial_summary="the unmodified baseline repository (empty patch)",
        evaluate=evaluate, reward=swe.framework_score,
        prompt=lambda program: swe.envelope(program, suite=suite, agent_timeout=1),
        task_prompt=lambda index: f"score shard {index}",
        test_shards=suite.test_range(), data_summary=suite.data_summary())


def test_the_search_runs_over_patches_and_keeps_the_best_one():
    # No held-back split here on purpose: which test the seeded shuffle holds
    # back is checked above, and leaving all four visible makes "the best node
    # is the one with the most fixes" a deterministic claim rather than one
    # that depends on the draw.
    suite = make_suite(tests=tuple(f"t.py::t{i}" for i in range(1, 5)),
                       held_back_frac=0.0, shards=4, test_shards=2)
    replies = iter([
        swe.wrap_reply("added the first fix", HUNK.format(n=1)),
        swe.wrap_reply("added the first two", HUNK.format(n=1) + HUNK.format(n=2)),
        swe.wrap_reply("broke it", ""),
        swe.wrap_reply("added three", HUNK.format(n=1) + HUNK.format(n=2)
                       + HUNK.format(n=3)),
    ])
    seen = []

    def agent(prompt):
        seen.append(json.loads(prompt))
        return next(replies)

    run = run_agentdescent_era(
        agent, mode="sync", iterations=4, workers=2, shards=4, test_shards=2,
        held_out_frac=0.5, domain=_toy_domain(suite), extract=swe.extract_patch,
        empty_warning=port.EMPTY_WARNING, max_seconds=60.0)

    assert len(run.tree.nodes) == 5, "every expansion is a node, upstream's rule"
    assert run.tree.root().program.code == "", "the root is the unmodified repository"
    best = run.tree.best().program
    assert "fix1" in best.code and best.code.startswith("diff --git")
    assert best.change_summary == "added three"
    # The seam: what reached the tree is the patch, not the reply that wrapped it.
    assert swe.PATCH_BEGIN not in best.code
    # And every expansion was handed its parent's patch to open as a checkout.
    assert seen and all("patch" in envelope for envelope in seen)
    assert seen[0]["patch"] == "", "the first expansion starts from the baseline"


def test_an_agent_that_returns_nothing_still_appends_a_node():
    """Upstream appends a node for every expansion, including a failed one; a
    session that timed out or refused must not shrink the rank denominator."""
    suite = make_suite(tests=tuple(f"t.py::t{i}" for i in range(1, 5)))
    run = run_agentdescent_era(
        lambda prompt: swe.wrap_reply("[agent failed: AgentError: timeout]", ""),
        mode="serial", iterations=2, workers=1, shards=4, test_shards=2,
        held_out_frac=0.5, domain=_toy_domain(suite), extract=swe.extract_patch,
        empty_warning=port.EMPTY_WARNING, max_seconds=60.0)
    assert len(run.tree.nodes) == 3
    assert all(node.program.code == "" for node in run.tree.nodes)


def test_a_shard_that_scores_perfectly_still_gets_expanded():
    """`evolve` skips a worker's proposal when its rollout already scored 1.0.
    Here a shard is a handful of tests and the benchmark's verdict is a
    different set, so a perfect shard is not a finished task -- and the default
    would silently spend less of the budget than the result file reports."""
    suite = make_suite(tests=("t.py::a", "t.py::b"), held_back_frac=0.0, shards=4)
    solved = Domain(
        name="solved", entrypoint="model.patch", metric_key="pass_rate",
        metric_better="higher", initial_program="", initial_summary="baseline",
        evaluate=lambda patch, shards: (
            True, {"score": 1.0, "pass_rate": 1.0, "passed": 2, "collected": 2,
                   "checks": 2, "report": ""}, ""),
        reward=swe.framework_score,
        prompt=lambda program: swe.envelope(program, suite=suite, agent_timeout=1),
        task_prompt=lambda index: f"score shard {index}",
        test_shards=suite.test_range(), data_summary=suite.data_summary())

    def once(threshold):
        return len(run_agentdescent_era(
            lambda prompt: swe.wrap_reply("a change", "diff --git a/s.py b/s.py\n"),
            mode="serial", iterations=2, workers=1, shards=4, test_shards=2,
            held_out_frac=0.5, domain=solved, extract=swe.extract_patch,
            solved_threshold=threshold, max_seconds=60.0).tree.nodes)

    assert once(1.0) == 1, "the default stops at the root, which is the defect"
    assert once(float("inf")) == 3, "the budget the run reports is the budget it spends"


def test_the_agent_mutation_diffs_a_workspace_even_when_the_agent_raises(monkeypatch):
    """The release's own rule: `pre_artifacts.sh` writes the artifact "including
    for a clean or timed-out agent run", so a session that died still becomes a
    node carrying whatever it had written."""
    suite = make_suite()

    class FakeWorkspace:
        work = "/tmp/nowhere"
        applied = []

        def env(self):
            return {"PATH": "/bin"}

        def diff(self):
            return "diff --git a/s.py b/s.py\n+half a fix\n"

    import contextlib

    @contextlib.contextmanager
    def fake_open(_suite, patch="", **_kwargs):
        FakeWorkspace.applied.append(patch)
        yield FakeWorkspace()

    monkeypatch.setattr(swe, "open_workspace", fake_open)
    counter = {}

    def launch(_workspace, _env):
        def run(_prompt):
            raise RuntimeError("agent exceeded timeout=600s")
        return run

    mutate = swe.make_agent_mutation(suite, launch=launch, counter=counter)
    program = Program("p", 1, None, "parent-patch", "", {}, True, "")
    reply = mutate(swe.envelope(program, suite=suite, agent_timeout=60))
    patch, summary = swe.extract_patch(reply)
    assert patch == "diff --git a/s.py b/s.py\n+half a fix"
    assert counter == {"sessions": 1, "failed": 1}
    assert FakeWorkspace.applied == ["parent-patch"], "the parent patch is the checkout"
    assert "agent failed" in summary or "agent failed" in reply


def _git_repo(root):
    """A checkout shaped like the one the environment image ships."""
    import subprocess
    (root / "source").mkdir(parents=True)
    (root / "source" / "s.py").write_text("value = 1\n", encoding="utf-8")
    (root / "reproduce.py").write_text("print('ok')\n", encoding="utf-8")
    for args in (["init", "-b", "main"], ["config", "user.email", "t@example.invalid"],
                 ["config", "user.name", "t"], ["add", "-A"],
                 ["commit", "-m", "baseline"]):
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)
    return root


def test_the_workspace_diff_is_the_artifact_the_release_collects(tmp_path):
    """`git add -A && git diff --cached --binary <root commit>` -- including
    files the agent added, which is why the prompt tells it to clean up after
    itself."""
    work = _git_repo(tmp_path / "work")
    space = swe.Workspace(make_suite(), tmp_path, work, tmp_path / "bin", "")
    assert space.diff() == "", "an untouched checkout is the empty patch"
    (work / "source" / "s.py").write_text("value = 2\n", encoding="utf-8")
    (work / "scratch.py").write_text("print('debug')\n", encoding="utf-8")
    patch = space.diff()
    assert "-value = 1" in patch and "+value = 2" in patch
    assert "scratch.py" in patch, "an untracked file is part of model.patch"


def test_a_workspace_round_trips_a_patch_through_apply_and_diff(tmp_path):
    work = _git_repo(tmp_path / "work")
    space = swe.Workspace(make_suite(), tmp_path, work, tmp_path / "bin", "")
    (work / "source" / "s.py").write_text("value = 2\n", encoding="utf-8")
    patch = space.diff()

    fresh = _git_repo(tmp_path / "fresh")
    other = swe.Workspace(make_suite(), tmp_path / "other", fresh,
                          tmp_path / "bin", "")
    (tmp_path / "other").mkdir()
    other.apply(patch)
    assert (fresh / "source" / "s.py").read_text() == "value = 2\n"
    assert other.diff() == patch, "a node's patch has to survive being a parent"


def test_a_checkout_with_two_root_commits_is_refused(tmp_path):
    """The release's own hook refuses it, because the baseline it diffs against
    would be ambiguous and the artifact would be a different patch."""
    import subprocess
    work = _git_repo(tmp_path / "work")
    subprocess.run(["git", "checkout", "--orphan", "second"], cwd=work, check=True,
                   capture_output=True)
    subprocess.run(["git", "commit", "-m", "another root", "--allow-empty"],
                   cwd=work, check=True, capture_output=True)
    subprocess.run(["git", "checkout", "main"], cwd=work, check=True,
                   capture_output=True)
    subprocess.run(["git", "merge", "second", "--allow-unrelated-histories",
                    "-m", "merge"], cwd=work, check=True, capture_output=True)
    space = swe.Workspace(make_suite(), tmp_path, work, tmp_path / "bin", "")
    with pytest.raises(swe.BenchmarkError):
        space.diff()


def test_the_completion_control_arm_applies_the_diff_it_was_given(monkeypatch,
                                                                  tmp_path):
    """The arm exists to be beaten, but it has to actually run: a control that
    never applied anything would make the agent look good for free."""
    work = _git_repo(tmp_path / "work")
    space = swe.Workspace(make_suite(), tmp_path, work, tmp_path / "bin", "")
    (work / "source" / "s.py").write_text("value = 2\n", encoding="utf-8")
    proposed = space.diff()
    (work / "source" / "s.py").write_text("value = 1\n", encoding="utf-8")

    import contextlib

    @contextlib.contextmanager
    def fake_open(_suite, patch="", **_kwargs):
        yield space

    monkeypatch.setattr(swe, "open_workspace", fake_open)
    seen = {}

    def complete(prompt):
        seen["prompt"] = prompt
        return "Here is the fix.\n```diff\n" + proposed + "\n```\n"

    counter = {}
    mutate = swe.make_completion_mutation(make_suite(), complete, counter=counter)
    program = Program("p", 0, None, "", "", {}, True, "")
    reply = mutate(swe.envelope(program, suite=make_suite(), agent_timeout=60))
    patch, _ = swe.extract_patch(reply)
    assert "+value = 2" in patch
    assert "## The repository" in seen["prompt"], "the model gets the file list"
    assert "source/s.py" in seen["prompt"]
    assert counter == {"sessions": 1}


def test_the_control_arm_records_a_diff_that_would_not_apply(monkeypatch, tmp_path):
    work = _git_repo(tmp_path / "work")
    space = swe.Workspace(make_suite(), tmp_path, work, tmp_path / "bin", "")

    import contextlib

    @contextlib.contextmanager
    def fake_open(_suite, patch="", **_kwargs):
        yield space

    monkeypatch.setattr(swe, "open_workspace", fake_open)
    counter = {}
    mutate = swe.make_completion_mutation(
        make_suite(),
        lambda prompt: "```diff\ndiff --git a/absent b/absent\n@@ -1 +1 @@\n-a\n+b\n```",
        counter=counter)
    program = Program("p", 0, None, "", "", {}, True, "")
    reply = mutate(swe.envelope(program, suite=make_suite(), agent_timeout=60))
    assert swe.extract_patch(reply)[0] == "", "nothing was applied, so nothing changed"
    assert counter.get("unapplied") == 1


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def test_the_agent_command_forwards_the_model_only_when_one_was_given():
    parser = port.build_parser()
    assert port.agent_command(parser.parse_args([])) == port.CLAUDE_CODE
    with_model = port.agent_command(parser.parse_args(["--model", "claude-opus-5"]))
    assert with_model[-2:] == ("--model", "claude-opus-5")
    assert port.agent_command(parser.parse_args(
        ["--agent", "command", "--agent-command", "my-agent --go"])) == \
        ("my-agent", "--go")


def test_thinking_disabled_reaches_the_agent_cli_as_its_own_budget_knob(monkeypatch):
    """Extended thinking is most of a session's wall-clock, and the CLI takes it
    from an environment variable the launching shell may already have set -- so
    the flag has to *override* it, not merely decline to set it, and the run has
    to record what was in force."""
    parser = port.build_parser()
    monkeypatch.setenv("MAX_THINKING_TOKENS", "31999")
    assert port.agent_environment(parser.parse_args([])) == {}
    assert port.agent_environment(parser.parse_args(["--thinking", "disabled"])) == {
        "MAX_THINKING_TOKENS": "0"}
    monkeypatch.delenv("MAX_THINKING_TOKENS")
    assert port.agent_environment(parser.parse_args(["--thinking", "enabled"])) == {
        "MAX_THINKING_TOKENS": "31999"}


def test_the_agent_session_env_carries_the_workspace_path_and_the_override(monkeypatch):
    seen = {}

    def fake_cli_agent(command, **kwargs):
        seen.update(kwargs)
        return lambda prompt: "ok"

    monkeypatch.setattr(port, "cli_agent", fake_cli_agent)
    args = port.build_parser().parse_args(["--thinking", "disabled"])
    port.build_launch(args, port.Usage())("/tmp/ws", {"PATH": "/era/bin:/usr/bin"})
    assert seen["env"]["PATH"] == "/era/bin:/usr/bin", "the run-in-env shims must survive"
    assert seen["env"]["MAX_THINKING_TOKENS"] == "0"
    assert seen["via_stdin"] is True


def test_an_agent_command_arm_with_no_command_is_refused():
    parser = port.build_parser()
    with pytest.raises(SystemExit):
        port.agent_command(parser.parse_args(["--agent", "command"]))


def test_the_completion_control_arm_needs_a_model(capsys):
    with pytest.raises(SystemExit) as excinfo:
        port.main(["--agent", "completion", "--yes", "--tasks", "001"])
    assert "--model" in str(excinfo.value) or "Docker" in str(excinfo.value)


def test_the_dry_run_names_the_release_it_pins(capsys):
    port.main(["--dry-run"])
    out = capsys.readouterr().out
    assert swe.RELEASE_REPO in out
    assert swe.RELEASE_REVISION[:12] in out
    assert "one tree each" in out


def test_too_few_shards_is_refused_before_an_image_is_pulled():
    with pytest.raises(SystemExit) as excinfo:
        port.main(["--shards", "2", "--yes"])
    assert "at least 4" in str(excinfo.value) or "Docker" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Against the real images
# ---------------------------------------------------------------------------


@needs_docker
@needs_release
def test_the_baseline_of_task_001_scores_what_the_release_grader_says_it_does():
    """End to end on one published task: the empty patch, the release's own
    grader, and this port's subset evaluator have to agree about the same repo."""
    manifest = swe.load_manifest()
    suite = swe.prepare_suite("001", manifest=manifest, shards=4, test_shards=2,
                              held_back_frac=0.0, seed=0)
    assert suite.held_back == ()
    valid, metrics, error = swe.evaluate_patch(
        "", suite=suite, shards=range(suite.scoring_shards))
    assert valid, error
    graded = swe.grade("", suite=suite)
    assert metrics["collected"] == graded["private"]["collected"]
    assert metrics["passed"] == graded["private"]["passed"]
    assert metrics["public"] == graded["public"]["passed"]
    assert metrics["subset_resolved"] == graded["reward"]


@needs_docker
@needs_release
def test_a_patch_that_does_not_apply_reaches_the_tree_as_an_invalid_node():
    manifest = swe.load_manifest()
    suite = swe.prepare_suite("001", manifest=manifest, shards=4, test_shards=2,
                              held_back_frac=0.25, seed=0)
    valid, metrics, error = swe.evaluate_patch(
        "diff --git a/absent.py b/absent.py\n--- a/absent.py\n+++ b/absent.py\n"
        "@@ -1 +1 @@\n-nothing\n+something\n",
        suite=suite, shards=(0,))
    assert valid is False
    assert metrics["score"] == float("-inf")
    assert error
