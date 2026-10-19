"""The label set in ``.github/labels.json`` and the script that syncs it.

The labels are the process's state machine (AGENTS.md › Labels): a typo in a
name silently splits one state into two, and a sync that deleted labels would
strip them from every open issue. Both halves are checked here, offline.
"""

import importlib.util
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / ".github" / "labels.json"
WORKFLOW = ROOT / ".github" / "workflows" / "labels.yml"


def _sync():
    spec = importlib.util.spec_from_file_location(
        "sync_labels", ROOT / ".github" / "scripts" / "sync_labels.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _wanted():
    return json.loads(LABELS.read_text(encoding="utf-8"))


def test_labels_json_is_valid():
    assert _sync().validate(_wanted()) == []


def test_the_process_labels_are_all_there():
    names = {w["name"] for w in _wanted()}
    by_group = {}
    for w in _wanted():
        by_group.setdefault(w["group"], set()).add(w["name"])
    assert {"bug", "enhancement", "documentation"} <= by_group["type"]
    assert by_group["source"] == {"self-test"}
    assert by_group["state"] == {"needs-info", "needs-human", "agent-ready", "agent-working",
                                 "agent-pr", "awaiting-approval", "hold"}
    assert by_group["risk"] == {"risk:T0", "risk:T1", "risk:T2", "risk:T3"}
    assert {"area:plugins", "area:cli", "area:evolve", "area:runstore", "area:audit",
            "area:ports", "area:docs", "area:ci"} <= names


def test_validate_catches_the_mistakes_that_split_a_state():
    bad = [
        {"group": "state", "name": "hold", "color": "000000", "description": "x"},
        {"group": "state", "name": "Hold", "color": "000000", "description": "x"},
        {"group": "area", "name": "plugins", "color": "#00000", "description": "x" * 101},
        {"group": "risk", "name": "risk:T4", "color": "000000", "description": ""},
        {"group": "kind", "name": "x", "color": "000000", "description": ""},
    ]
    problems = "\n".join(_sync().validate(bad))
    for needle in ("listed twice", "6 hex digits", "100 characters", "start with 'area:'",
                   "risk:T0..risk:T3", "group must be one of"):
        assert needle in problems


def test_plan_creates_updates_and_keeps_but_never_deletes():
    wanted = [
        {"name": "bug", "color": "d73a4a", "description": "Something isn't working"},
        {"name": "hold", "color": "000000", "description": "Frozen"},
        {"name": "needs-info", "color": "D4C5F9", "description": "Waiting"},
    ]
    existing = [
        {"name": "bug", "color": "D73A4A", "description": "Something isn't working"},
        {"name": "HOLD", "color": "000000", "description": "Frozen"},
        {"name": "wontfix", "color": "ffffff", "description": "This will not be worked on"},
    ]
    steps = {s["label"]["name"]: s["action"] for s in _sync().plan(wanted, existing)}
    # Colors compare case-insensitively; a name that differs only in case is renamed in place.
    assert steps == {"bug": "keep", "hold": "update", "needs-info": "create"}
    assert "wontfix" not in steps


def test_the_script_has_no_delete_path():
    source = (ROOT / ".github" / "scripts" / "sync_labels.py").read_text(encoding="utf-8")
    # Only POST (create) and PATCH (update) are ever sent.
    methods = set(re.findall(r'"-X", "([A-Z]+)"', source))
    assert methods == {"POST", "PATCH"}


def test_the_workflow_is_least_privilege_and_runs_main_only():
    wf = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    on = wf.get("on", wf.get(True))       # PyYAML reads a bare `on:` key as True
    assert set(on) == {"push", "workflow_dispatch"}
    assert on["push"]["branches"] == ["main"]
    assert wf["permissions"] == {}
    job = wf["jobs"]["sync"]
    assert job["permissions"] == {"contents": "read", "issues": "write"}
    checkout = job["steps"][0]
    assert checkout["with"]["ref"] == "${{ github.event.repository.default_branch }}"
    assert checkout["with"]["persist-credentials"] is False
    assert "secrets." not in WORKFLOW.read_text(encoding="utf-8")
