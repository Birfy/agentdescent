"""Offline tests for the gskill microport: no network, no Docker, no API key.

The fake model below plays both actors. The coding agent applies a repository
convention only when the skill it was handed states that convention; the skill
optimizer writes the convention the failing task is an instance of (or, in the
control, a note that helps nothing). Everything between them is real: the
FileTree strategy and its skill validator, the fixture verifier, the strict
held-out gate, and the engine in every scheduler.
"""

import re

import pytest

from agentdescent.agents import Usage, metered
from agentdescent.filetree import load_tree
from agentdescent.skillspec import SKILL_FILE, validate_skill_tree

from examples._measure import MODES, Recorder
from examples._method_runner import run_port
from examples.gskill import _fixtures
from examples.gskill import gskill_repo_skill as gs


CONVENTION_NOTE = {
    "half-open": "Ranges are half-open: lo is inside the window and hi is not.",
    "config-error": "Invalid user values raise ConfigError, never ValueError, KeyError or None.",
    "partial-counts": "A partial page, chunk or bucket counts as one: round counts up.",
    "missing-is-none": "A missing value is None -- not an exception and not an empty string.",
}


def _edits(path_to_content, rationale="learned a convention"):
    import json

    edits = [{"path": p, "content": c} if c is not None
             else {"path": p, "delete": True} for p, c in path_to_content.items()]
    return "<EDITS>" + json.dumps({"rationale": rationale, "edits": edits}) + "</EDITS>"


def _task_instance(prompt):
    for item in _fixtures.load_instances():
        if item.issue in prompt:
            return item
    return None


def _skill_block(prompt):
    match = re.search(r"=== SKILL ===\n(.*?)\n=== END SKILL ===", prompt, re.S)
    return match.group(1) if match else ""


def gskill_fake_completion(prompt, *, helpful=True):
    """Answer a gskill prompt, or ``None`` when the prompt is not gskill's."""
    lower = prompt.lower()
    item = _task_instance(prompt)
    if "gskill's mini-agent" in lower and item is not None:
        skill = _skill_block(prompt)
        if CONVENTION_NOTE[item.convention] in skill:
            source = item.gold_source()
        else:                                    # a patch that exists and is wrong
            source = item.buggy_source() + "\n# tried a change\n"
        return f"```python\n{source}```"
    if "gskill's skill optimizer" in lower and item is not None:
        current = re.search(r"--- SKILL.md ---\n(.*?)\n\n## Evaluation",
                            prompt, re.S).group(1) + "\n"
        note = (CONVENTION_NOTE[item.convention] if helpful
                else "Run the tests before and after every change.")
        return _edits({SKILL_FILE: current.rstrip() + f"\n\n- {note}\n"})
    return None


def _offline_run(mode="serial", *, helpful=True, export_dir="", seed=0):
    usage = Usage()
    recorder = Recorder(
        metered(lambda prompt: gskill_fake_completion(prompt, helpful=helpful)
                or "no idea", usage), usage)
    policy = gs.build(seed, export_dir=export_dir)
    payload = run_port(policy, recorder, mode=mode, seed=seed, workers=2,
                       candidate_budget=2, max_seconds=60.0,
                       shutdown_grace=10.0).compact()
    return policy, payload


# ---------------------------------------------------------------------------
# The vendored surrogate for SWE-smith
# ---------------------------------------------------------------------------

def test_fixtures_are_wellformed():
    """Clean repo passes everything; each bug fails exactly its FAIL_TO_PASS."""
    assert _fixtures.verify_fixtures() == []
    assert len(_fixtures.load_instances()) == 12


def test_gold_patch_scores_one_and_the_prompt_never_carries_it():
    for item in _fixtures.load_instances():
        assert _fixtures.judge(item, item.gold_source()).score == 1.0
        assert item.old not in item.prompt, "the clean code is in the prompt"
        assert item.gold_source() not in item.prompt


def test_verdicts_are_told_apart_as_gskills_harness_does():
    item = _fixtures.instance("in_window")
    assert _fixtures.judge(item, None).status == _fixtures.NO_PATCH
    assert _fixtures.judge(item, item.buggy_source()).status == _fixtures.NO_PATCH
    wrong = item.buggy_source() + "\n# tried\n"
    assert _fixtures.judge(item, wrong).status == _fixtures.F2P_FAILED
    # Fixes the target but breaks a neighbour in the same file: regression.
    regress = item.gold_source().replace("a_lo < b_hi and b_lo < a_hi",
                                         "a_lo <= b_hi and b_lo <= a_hi")
    outcome = _fixtures.judge(item, regress)
    assert outcome.status == _fixtures.P2P_REGRESSION and outcome.score == 0.0
    assert "PASS_TO_PASS test_ranges::test_overlaps_adjacent_is_disjoint" in outcome.report()
    broken = _fixtures.judge(item, "def in_window(:\n")
    assert broken.status == _fixtures.F2P_FAILED


def test_a_runaway_candidate_costs_its_score_and_not_the_run():
    item = _fixtures.instance("in_window")
    hang = item.gold_source() + "\nwhile True:\n    pass\n"
    results = _fixtures.run_tests(_fixtures.attempt_files(item, hang), timeout=1.0)
    assert list(results) == ["<workspace>"] and "timed out" in results["<workspace>"]


def test_splits_are_frozen_per_seed_disjoint_and_stratified():
    first = gs.gskill_splits(0)
    assert [[t.id for t in s] for s in first] == [[t.id for t in s] for s in gs.gskill_splits(0)]
    train, held_out, test = first
    ids = [t.id for split in first for t in split]
    assert len(ids) == len(set(ids)) == 12
    assert len(train) == len(held_out) == len(test) == 4
    for split in first:     # one bug of every convention in every split
        assert sorted(t.meta["convention"] for t in split) == sorted(CONVENTION_NOTE)
    assert [t.id for t in gs.gskill_splits(1)[0]] != [t.id for t in train]


# ---------------------------------------------------------------------------
# Structural validation: a broken skill is rejected before it is measured
# ---------------------------------------------------------------------------

def _diff(strategy, content_by_path):
    return strategy.to_diff(strategy.initial(), _edits(content_by_path), "w", 1, "a")


def test_a_skill_that_deletes_skill_md_is_structurally_rejected():
    strategy = gs.make_strategy()
    assert _diff(strategy, {SKILL_FILE: None}) is None
    assert strategy.structural_rejections["skill-missing-skill-md"] == 1
    assert strategy.invalid_proposals == 1


@pytest.mark.parametrize("broken, reason", [
    ("# no frontmatter at all\n", "skill-no-frontmatter"),
    (gs.INITIAL_SKILL.replace(f"name: {gs.SKILL_NAME}", "name: other-skill"),
     "skill-name-mismatch"),
    (gs.INITIAL_SKILL.replace("description: Repository", "description: [x"),
     "skill-frontmatter-malformed"),
    (re.sub(r"description: .*\n", "description:\n", gs.INITIAL_SKILL),
     "skill-description-missing"),
])
def test_a_skill_with_broken_frontmatter_is_structurally_rejected(broken, reason):
    strategy = gs.make_strategy()
    assert _diff(strategy, {SKILL_FILE: broken}) is None
    assert strategy.structural_rejections[reason] == 1


def test_valid_edits_and_reference_files_pass_and_other_paths_do_not():
    strategy = gs.make_strategy()
    note = gs.INITIAL_SKILL + "\n- Ranges are half-open.\n"
    assert _diff(strategy, {SKILL_FILE: note}).ops == {SKILL_FILE: note}
    ref = _diff(strategy, {"references/conventions.md": "# Conventions\n"})
    assert ref is not None and "references/conventions.md" in ref.ops
    assert _diff(strategy, {"kvtools/ranges.py": "x = 1\n"}) is None      # not the skill
    assert strategy.invalid_proposals == 1 and not strategy.structural_rejections


# ---------------------------------------------------------------------------
# The gate: a skill that raises the pass rate is accepted, one that does not is not
# ---------------------------------------------------------------------------

def test_a_skill_that_raises_the_held_out_pass_rate_is_accepted():
    policy, payload = _offline_run("serial", helpful=True)
    assert payload["accepted"] >= 1
    assert payload["final_quality"] > payload["baseline_quality"]
    assert payload["validation_quality"] > payload["baseline_validation_quality"]
    assert payload["budget"]["matched"] is True


def test_a_skill_that_does_not_raise_the_pass_rate_is_rejected():
    policy, payload = _offline_run("serial", helpful=False)
    assert payload["accepted"] == 0
    assert payload["final_quality"] == payload["baseline_quality"]
    assert payload["validation_quality"] == payload["baseline_validation_quality"]
    assert payload["framework"]["acceptance"] == "StrictHeldOutGate"
    assert payload["framework"]["outcomes"].get("committed", 0) == 0
    assert payload["framework"]["outcomes"].get("oracle-rejected", 0) >= 1


def test_the_gate_is_strict_not_statistical():
    """``(successes, failures)`` counts on the full held-out split: only a
    strictly higher pass rate commits -- a tie is a rejection."""
    from types import SimpleNamespace

    gate = gs.StrictHeldOutGate()
    for cand, accepted in (((2, 2), True), ((1, 3), False), ((0, 4), False)):
        decision = gate.accept(SimpleNamespace(base_counts=(1, 3), cand_counts=cand))
        assert decision.accept is accepted
        assert (decision.category == "") is accepted


@pytest.mark.parametrize("mode", MODES)
def test_runs_through_every_scheduler(mode):
    policy, payload = _offline_run(mode)
    assert payload["candidates"] == 2 and payload["budget"]["matched"] is True
    assert payload["final_quality"] >= payload["baseline_quality"]
    assert policy.reflective is False
    assert policy.fidelity == "mechanism_microport"


# ---------------------------------------------------------------------------
# Deploying the best skill
# ---------------------------------------------------------------------------

def test_the_best_skill_is_exported_as_a_valid_claude_code_skill(tmp_path):
    policy, payload = _offline_run("serial", export_dir=str(tmp_path))
    assert payload["accepted"] >= 1
    message = policy.report()
    assert "[export-skill] wrote" in message
    root = tmp_path / ".claude" / "skills" / gs.SKILL_NAME
    tree = load_tree(str(root))
    assert validate_skill_tree(tree, root_name=gs.SKILL_NAME) == []
    assert any(note in tree[SKILL_FILE] for note in CONVENTION_NOTE.values())


def test_export_refuses_a_tree_a_host_could_not_load(tmp_path):
    with pytest.raises(ValueError, match="missing-skill-md"):
        gs.export_skill({"references/x.md": "x"}, str(tmp_path))
    assert not list(tmp_path.iterdir())


def test_the_initial_skill_is_itself_a_valid_skill():
    assert validate_skill_tree(gs.make_strategy().initial(), root_name=gs.SKILL_NAME) == []


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------

def test_dry_run_needs_no_network_no_docker_and_no_key(monkeypatch, capsys):
    import subprocess

    import examples._method_runner as runner

    def refuse(*_a, **_k):
        raise AssertionError("a dry run must not reach a model or spawn a process")

    monkeypatch.setattr(runner, "completion_for", refuse)
    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert gs.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "gskill [mechanism_microport]" in out
    assert "No SWE-smith, no Docker" in out


def test_the_module_is_runnable_as_python_dash_m():
    import subprocess
    import sys

    done = subprocess.run([sys.executable, "-m", "examples.gskill", "--dry-run"],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0 and "gskill [mechanism_microport]" in done.stdout
