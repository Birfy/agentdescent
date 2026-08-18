"""Evolving a skill library: add, edit, retire.

`evolve_skill_dir` refines a skill someone already chose to have. A library run
has to be able to invent one -- so the test that matters here starts from a
library that cannot answer the question and ends with a skill that did not
exist when the run began.

Offline throughout: the agent is a real subprocess bound to the workspace, and
only the reflector is a stub.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

from agentdescent.agents import cli_agent
from agentdescent.dsh.library import (
    LIBRARY_LAYOUT,
    evolve_skill_library,
    library_root,
)
from agentdescent.runners import layout_prefix

# The agent reads whatever skills are staged and looks for an ANSWER line. It
# genuinely opens the files: that is what makes the run measure the library
# rather than the model's priors.
AGENT = """
import glob, os, re, sys
answer = None
for path in sorted(glob.glob(os.path.join('.dsh', 'skills', '*', 'SKILL.md'))):
    found = re.search(r'ANSWER:\\s*(\\S+)', open(path).read())
    if found:
        answer = found.group(1)
print(answer or 'i do not know')
"""

START = {
    "csv/SKILL.md": "---\ndescription: total a column\n---\n\nSum the column.\n",
}


def reflector_adding(name: str, answer: str):
    """What a model would return: a brand new skill, as a file that did not exist."""
    def complete(_prompt: str) -> str:
        return "<EDITS>" + json.dumps({
            "rationale": f"nothing in the library answers this; add {name}",
            "edits": [{
                "path": f"{name}/SKILL.md",
                "content": f"---\ndescription: answer the year question\n---\n\nANSWER: {answer}\n",
            }],
        }) + "</EDITS>"
    return complete


def write_library(root, tree):
    for path, body in tree.items():
        full = os.path.join(str(root), path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as fh:
            fh.write(body)
    return str(root)


def test_the_layout_is_the_root_the_harness_scans():
    """A single skill stages to `.dsh/skills/<name>`; a library *is* the root."""
    assert layout_prefix(LIBRARY_LAYOUT, "anything") == ".dsh/skills"


def test_a_run_can_add_a_skill_that_did_not_exist(tmp_path):
    """Discovery, not refinement -- the reason a library is its own artifact.

    The library starts unable to answer, so every rollout scores 0 until a skill
    that was never written appears in the tree.
    """
    root = write_library(tmp_path / "skills", START)
    tasks = [{"prompt": "what year is it?", "gold": "2026"} for _ in range(8)]

    result = evolve_skill_library(
        root, tasks,
        agent=cli_agent([sys.executable, "-c", AGENT]),
        reflect_with=reflector_adding("year", "2026"),
        score="contains", rounds=3, n_workers=2, max_concurrency=2, seed=0)

    assert result.error is None, result.error
    assert "year/SKILL.md" in result.state, sorted(result.state)
    assert "ANSWER: 2026" in result.state["year/SKILL.md"]
    # The skill that was already there is untouched by an unrelated addition.
    assert result.state["csv/SKILL.md"] == START["csv/SKILL.md"]
    assert result.final_reward > 0.0


def test_the_existing_library_is_the_starting_state(tmp_path):
    root = write_library(tmp_path / "skills", START)
    result = evolve_skill_library(
        root, [{"prompt": "q", "gold": "2026"} for _ in range(4)],
        agent=cli_agent([sys.executable, "-c", AGENT]),
        reflect_with=reflector_adding("year", "2026"),
        score="contains", rounds=1, n_workers=1, max_concurrency=1)
    assert "csv/SKILL.md" in result.state


def test_frozen_paths_keep_a_subtree_out_of_reach(tmp_path):
    """`frozen=("private/**",)` is how a user keeps part of a library off-limits."""
    tree = dict(START)
    tree["private/SKILL.md"] = "---\ndescription: do not touch\n---\n\nsecret\n"
    root = write_library(tmp_path / "skills", tree)
    result = evolve_skill_library(
        root, [{"prompt": "q", "gold": "2026"} for _ in range(4)],
        agent=cli_agent([sys.executable, "-c", AGENT]),
        reflect_with=reflector_adding("private", "leaked"),
        score="contains", frozen=("private/**",),
        rounds=2, n_workers=1, max_concurrency=1)
    assert result.state["private/SKILL.md"] == tree["private/SKILL.md"]


def test_library_root_reports_a_root_that_exists(tmp_path):
    env = {"DSH_HOME": str(tmp_path / "home"), "DSH_AGENTS_HOME": str(tmp_path / "ag")}
    assert library_root("user-dsh", str(tmp_path), environ=env) is None
    os.makedirs(os.path.join(str(tmp_path / "home"), "skills"))
    assert library_root("user-dsh", str(tmp_path), environ=env) == \
        os.path.join(str(tmp_path / "home"), "skills")


def test_an_absent_root_is_none_rather_than_a_path_that_would_be_created(tmp_path):
    """Creating one silently would put skills somewhere the user never chose."""
    env = {"DSH_HOME": str(tmp_path / "nope"), "DSH_AGENTS_HOME": str(tmp_path / "ag")}
    assert library_root("user-dsh", str(tmp_path), environ=env) is None


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


def test_the_runner_dry_runs_a_library_and_lists_what_it_holds(tmp_path, capsys, monkeypatch):
    from examples.dsh.evolve_dsh_skill import main

    home = tmp_path / "home"
    write_library(home / "skills", START)
    tasks = str(tmp_path / "t.jsonl")
    with open(tasks, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"prompt": "q", "gold": "2026"}) + "\n")
    monkeypatch.setenv("DSH_HOME", str(home))
    monkeypatch.setenv("DSH_AGENTS_HOME", str(tmp_path / "none"))

    code = main(["--library", "--tasks", tasks, "--cwd", str(tmp_path), "--dry-run"])
    out = capsys.readouterr().out
    assert code == 0
    assert "user-dsh" in out
    assert "csv" in out                              # what the root already holds
    assert "add, edit and retire" in out
    assert "nothing was written" in out


def test_the_runner_says_where_it_looked_when_no_root_exists(tmp_path, capsys, monkeypatch):
    from examples.dsh.evolve_dsh_skill import main

    monkeypatch.setenv("DSH_HOME", str(tmp_path / "nope"))
    monkeypatch.setenv("DSH_AGENTS_HOME", str(tmp_path / "none"))
    code = main(["--library", "--tasks", "/nonexistent", "--cwd", str(tmp_path), "--dry-run"])
    assert code == 1
    assert "absent" in capsys.readouterr().err


def test_skill_names_uses_the_same_rule_the_harness_does():
    from examples.dsh.evolve_dsh_skill import _skill_names

    state = {
        "sql/SKILL.md": "x",
        "sql/references/style.md": "y",          # a resource, not a skill
        "README.md": "z",                        # a stray root file, not a skill
        "csv/SKILL.md": "w",
    }
    assert _skill_names(state) == ["csv", "sql"]
