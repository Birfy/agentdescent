"""The risk gate: a pull request's tier T0-T3 from its diff (AGENTS.md › Merge policy).

The tier decides whether an agent may auto-merge (T0/T1) or must wait for
Danyang's click (T2/T3), so a rule that silently drops a core-loop path to T0
is a merge-policy bug. Offline: the PR file lists are fixtures in the shape of
``GET /pulls/{n}/files``.
"""

import importlib.util
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RULES = ROOT / ".github" / "agent" / "risk-rules.json"
WORKFLOW = ROOT / ".github" / "workflows" / "risk-tier.yml"


def _mod():
    spec = importlib.util.spec_from_file_location("risk_gate", ROOT / ".github" / "scripts" / "risk_gate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rg = _mod()
GATE = rg.Gate(json.loads(RULES.read_text(encoding="utf-8")))


def f(name, add=1, delete=0, patch=None, previous=None):
    out = {"filename": name, "additions": add, "deletions": delete, "changes": add + delete,
           "patch": "@@ -0,0 +1 @@\n+x" if patch is None else patch}
    if previous:
        out["previous_filename"] = previous
    return out


def test_glob_semantics():
    g = rg.glob_to_re
    assert g("docs/**").match("docs/a/b.md")
    assert not g("docs/*").match("docs/a/b.md")
    assert g("**/package.json").match("package.json")
    assert g("**/package.json").match("integrations/dsh-agentdescent/package.json")
    assert not g("**/package.json").match("x/not-package.json")
    assert g("agentdescent/evolution*.py").match("agentdescent/evolution.py")
    assert not g("agentdescent/evolution*.py").match("agentdescent/sub/evolution.py")
    assert g("**/*.test.{js,cjs}").match("tests/javascript/experience.test.cjs")
    assert not g("AGENTS.md").match("AGENTSxmd"), "a dot is a literal dot"


def test_tier_of_follows_the_agentdescent_layout():
    t = GATE.tier_of
    for path in ("docs/evolution.md", "README.md"):
        assert t(path) == "T0", path
    for path in ("tests/test_runstore.py", "examples/gepa/run.py", "changelog.d/1-x.md", "docs/api.md"):
        assert t(path) == "T1", path
    for path in ("agentdescent/cli.py", "agentdescent/audit/gate.py", "bench/run.py", "tools/gen_api_docs.py",
                 "mkdocs.yml", "some/new/top-level.txt"):
        assert t(path) == "T2", path
    for path in ("agentdescent/evolution.py", "agentdescent/async_evolve.py", "agentdescent/runstore.py",
                 "agentdescent/agents.py", "agentdescent/integrations/__init__.py",
                 "integrations/dsh-agentdescent/lib/index.js", "pyproject.toml", "CHANGELOG.md",
                 ".github/workflows/tests.yml", ".github/agent/risk-rules.json", "AGENTS.md", "LESSONS.md",
                 "CONTRIBUTING.md", "CODE_STANDARDS.md", ".claude/skills/review-pr/SKILL.md",
                 "scripts/install.sh"):
        assert t(path) == "T3", path


def test_unknown_paths_are_t2_never_t0_or_t1():
    assert json.loads(RULES.read_text())["unknownPathTier"] == "T2"
    assert GATE.tier_of("brand/new/thing.py") == "T2"


def test_the_tier_is_the_maximum_and_a_renames_old_path_counts():
    assert GATE.judge([f("docs/a.md"), f("tests/test_x.py")])["tier"] == "T1"
    assert GATE.judge([f("docs/a.md"), f("agentdescent/runstore.py")])["tier"] == "T3"
    moved = GATE.judge([f("agentdescent/store2.py", previous="agentdescent/runstore.py")])
    assert moved["tier"] == "T3" and "path agentdescent/runstore.py" in moved["why"]


def test_content_hits_in_added_code_lines_raise_to_t3():
    for line in ("subprocess.run(cmd)", "x = eval(s)", "os.environ['ARK_API_KEY']", "requests.get(url)",
                 "pickle.load(fh)", "check(shell=True)", '__version__ = "9.9.9"'):
        r = GATE.judge([f("agentdescent/cli.py", patch="@@\n+" + line)])
        assert r["tier"] == "T3", line
    skip = GATE.judge([f("tests/test_x.py", patch="@@\n+@pytest.mark.skip(reason='later')")])
    assert skip["tier"] == "T3"


def test_removed_lines_prose_and_test_only_patterns_are_not_hits():
    assert GATE.judge([f("agentdescent/cli.py", patch="@@\n-subprocess.run(x)")])["tier"] == "T2"
    assert GATE.judge([f("docs/x.md", patch="@@\n+run subprocess.run(x) with eval(")])["tier"] == "T0"
    # A test may spawn a process or read the environment; skipif is a condition, not a skip.
    test = f("tests/test_x.py", patch="@@\n+subprocess.run(x)\n+os.environ['ARK_API_KEY']\n"
                                       "+@pytest.mark.skipif(True, reason='x')")
    assert GATE.judge([test])["tier"] == "T1"


def test_unreadable_diffs_are_t3_and_generated_files_are_exempt():
    blind = {"filename": "agentdescent/cli.py", "additions": 5, "deletions": 0, "changes": 5}
    assert GATE.judge([blind])["tier"] == "T3"
    api_page = {"filename": "docs/api.md", "additions": 5000, "deletions": 10, "changes": 5010}
    r = GATE.judge([api_page, f("docs/x.md")])
    assert r["tier"] == "T1" and r["lines"] == 1 and r["generated"] == 1


def test_size_caps_jump_straight_to_t3():
    assert GATE.judge([f("docs/x.md", add=300)])["tier"] == "T0"
    assert GATE.judge([f("docs/x.md", add=301)])["tier"] == "T3"
    assert GATE.judge([f("agentdescent/cli.py", add=251)])["tier"] == "T3"
    many = [f(f"tests/test_{i}.py") for i in range(21)]
    assert GATE.judge(many)["tier"] == "T3"


def test_truncated_or_empty_file_lists():
    assert GATE.judge([f("docs/x.md")], truncated=True)["tier"] == "T3"
    assert GATE.judge([])["tier"] == "T0"


def test_render_says_report_only():
    text = rg.render(GATE.judge([f("pyproject.toml")]), GATE.meaning)
    assert "### Risk tier: T3" in text and "path pyproject.toml" in text and "Report only" in text


def test_rules_file_is_consistent():
    rules = json.loads(RULES.read_text())
    assert set(rules["tiers"]) == {"T0", "T1", "T2", "T3"}
    assert set(rules["limits"]) == {"T0", "T1", "T2"}
    assert ".github/**" in rules["tiers"]["T3"]["paths"]
    assert GATE.tier_of(".github/agent/risk-rules.json") == "T3", "editing the rules is itself T3"


def test_the_workflow_stays_report_only():
    text = WORKFLOW.read_text(encoding="utf-8")
    wf = yaml.safe_load(text)
    on = wf.get("on", wf.get(True))
    assert set(on) == {"pull_request"}, "never pull_request_target"
    assert wf["permissions"] == {"contents": "read", "pull-requests": "read"}
    job = wf["jobs"]["report"]
    assert job["continue-on-error"] is True
    assert job["steps"][0]["with"]["ref"] == "${{ github.event.pull_request.base.sha }}"
    assert "${{ secrets." not in text

