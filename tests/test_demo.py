"""`agentdescent demo`: the one command a newcomer runs before anything else.

If this breaks, the first thing a new user sees is a failure, so it is tested
like a feature rather than a sample.
"""

import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stdout

import pytest

from agentdescent import cli, demo, runstore

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_build_writes_a_skill_that_is_wrong_on_purpose(tmp_path):
    spec = demo.build(str(tmp_path))
    rules = tmp_path / demo.DEMO_SKILL / "references" / "rules.md"
    assert (tmp_path / demo.DEMO_SKILL / "SKILL.md").exists()
    assert rules.read_text().strip() == "COLUMN: id"     # the run has to fix this
    rows = [json.loads(l) for l in (tmp_path / "cases.jsonl").read_text().splitlines()]
    assert len(rows) == 12
    assert "data.csv" in rows[0]["fixtures"] and rows[0]["gold"].isdigit()
    assert json.loads((tmp_path / "spec.json").read_text()) == spec


def test_the_demo_spec_needs_no_credentials_and_no_allowlist_widening(tmp_path):
    spec = demo.build(str(tmp_path))
    # both refs live inside the package, so `allow` stays empty -- a newcomer
    # never has to widen the import allowlist to run the demo
    assert spec["agent"]["ref"].startswith("agentdescent.")
    assert spec["reflect"]["ref"].startswith("agentdescent.")
    assert "allow" not in spec
    from agentdescent import EvolveSpec, compose

    comp = compose(EvolveSpec.from_dict(spec))
    assert comp.kwargs["rounds"] == 4 and len(comp.tasks) == 12


def test_the_offline_agent_obeys_the_skill_it_is_given(tmp_path):
    """It is a real workspace agent: the skill on disk changes its answer."""
    from agentdescent import Task
    from agentdescent.filetree import load_tree
    from agentdescent.runners import tree_runner
    from agentdescent.treestrategy import FileTree

    demo.build(str(tmp_path))
    tree = load_tree(str(tmp_path / demo.DEMO_SKILL))
    strategy = FileTree(tree)
    run = tree_runner(demo.offline_agent(), layout="claude_skill",
                      name=demo.DEMO_SKILL, prompt_template="{prompt}")
    task = Task(id="t", prompt="What is the total?",
                meta={"gold": "30", "fixtures": {"data.csv": "id,amount\n1,10\n2,20\n"}})
    assert run(strategy.render(tree), task) == "3"          # sums `id`: 1 + 2
    fixed = dict(tree, **{"references/rules.md": "COLUMN: amount\n"})
    assert run(strategy.render(fixed), task) == "30"        # sums `amount`


def test_the_offline_reflector_proposes_the_fix_from_what_it_was_shown():
    prompt = ("--- references/rules.md ---\nCOLUMN: id\n"
              "--- other ---\nTASK THE AGENT WAS GIVEN: ...")
    out = demo.offline_reflector(prompt)
    assert out.startswith("<EDITS>") and out.endswith("</EDITS>")
    payload = json.loads(out[len("<EDITS>"):-len("</EDITS>")])
    assert payload["edits"] == [{"path": "references/rules.md",
                                 "content": "COLUMN: amount\n"}]
    # it reads the prompt rather than emitting a constant, so a prompt without
    # the file still yields a valid edit
    assert "COLUMN: amount" in demo.offline_reflector("nothing here")


def test_the_demo_command_runs_a_real_evolution_and_fixes_the_skill(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTDESCENT_HOME", str(tmp_path / "home"))
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = cli.main(["--store", str(tmp_path / "runs"), "demo",
                         "--dir", str(tmp_path / "work")])
    out = buf.getvalue()
    assert code == 0, out
    assert "held-out reward: 1.000" in out, out
    assert "'COLUMN: amount'" in out, out
    assert "agentdescent apply" in out and "--dry-run" in out

    # the run is in the store, and the user's skill on disk is untouched
    runs = runstore.list_runs(store=str(tmp_path / "runs"))
    assert len(runs) == 1 and runs[0].state == "done"
    rules = tmp_path / "work" / demo.DEMO_SKILL / "references" / "rules.md"
    assert rules.read_text().strip() == "COLUMN: id"


def test_demo_is_reachable_from_a_bare_install():
    """A newcomer's first command must work from the console script alone."""
    proc = subprocess.run([sys.executable, "-m", "agentdescent.cli", "--help"],
                          capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert proc.returncode == 0
    assert "demo" in proc.stdout


def test_init_says_what_to_do_when_the_cases_file_is_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "s").mkdir()
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert cli.main(["init", str(tmp_path / "s"), "--out", "spec.json"]) == 0
    out = buf.getvalue()
    assert "create" in out and "cases.jsonl" in out
    assert "agentdescent demo" in out          # the way to see a working one
