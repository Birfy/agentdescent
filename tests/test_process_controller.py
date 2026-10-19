"""The label-only controller (AGENTS.md › Controller).

It runs every 30 minutes with a write token, so its contract is tested
offline: it touches only the labels it manages, only in the allowed direction,
never adds ``agent-ready``, stops entirely on ``AGENT_PAUSE`` and is a dry run
unless ``CONTROLLER_LIVE`` is exactly ``true``.
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".github" / "scripts"
WORKFLOW = ROOT / ".github" / "workflows" / "controller.yml"
CONFIG = json.loads((ROOT / ".github" / "agent" / "controller.json").read_text())


def _mod():
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location("controller", SCRIPTS / "controller.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.remove(str(SCRIPTS))


c = _mod()
REPO = "Birfy/agentdescent"
ALL = ["agent-ready", "agent-working", "agent-pr", "hold", "risk:T0", "risk:T1", "risk:T2", "risk:T3"]
NOW = c.parse_time("2026-10-19T12:00:00Z")


def ev(type_, at, actor, label=None):
    return {"type": type_, "at": at, "actor": actor, "label": label}


def issue(number, labels, events=(), edits=()):
    return {"number": number, "labels": list(labels), "events": list(events), "edits": list(edits)}


def pr(number, labels=(), tier="T1", author="HugoXK", head="feat/x", same=True, body="", push="2026-10-19T11:00:00Z"):
    return {"number": number, "labels": list(labels), "tier": tier, "author": author, "headRef": head,
            "sameRepo": same, "body": body, "lastPushAt": push}


def run(issues=(), prs=(), labels=ALL):
    prs = list(prs)
    return c.plan({"repo": REPO, "config": CONFIG, "now": NOW, "repoLabels": list(labels),
                   "issues": list(issues), "prs": prs, "linked": c.linked_prs(prs, REPO)})


def test_mode_is_dry_run_by_default_and_a_pause_wins():
    assert c.decide_mode({}) == "dry-run"
    assert c.decide_mode({"CONTROLLER_LIVE": "TRUE"}) == "dry-run"
    assert c.decide_mode({"CONTROLLER_LIVE": "1"}) == "dry-run"
    assert c.decide_mode({"CONTROLLER_LIVE": "true"}) == "live"
    assert c.decide_mode({"CONTROLLER_LIVE": "true", "AGENT_PAUSE": "true"}) == "paused"
    assert c.decide_mode({"CONTROLLER_LIVE": "true", "AGENT_PAUSE": "false"}) == "live"


def test_config_names_the_maintainer_and_sane_limits():
    assert CONFIG["maintainers"] == ["Birfy"]
    assert 1 <= CONFIG["claimTtlHours"] <= 48
    assert CONFIG["maxPrsPerRun"] <= 100 and CONFIG["maxIssuesPerRun"] <= 100


def test_closing_refs_reads_this_repository_only():
    body = "Fixes #12\ncloses: #13\nResolved Birfy/agentdescent#14\nfixes other/repo#15\nsee #16"
    assert c.closing_refs(body, REPO) == [12, 13, 14]


def test_is_agent_pr():
    cfg = dict(CONFIG, agentAuthors=["bot-x"])
    assert c.is_agent_pr(pr(1, author="bot-x", same=False), cfg, REPO, set())
    assert c.is_agent_pr(pr(1, head="agent/fix-3"), cfg, REPO, set())
    assert not c.is_agent_pr(pr(1, head="agent/fix-3", same=False), cfg, REPO, set()), "fork: by author only"
    assert c.is_agent_pr(pr(1, body="Fixes #7"), cfg, REPO, {7})
    assert not c.is_agent_pr(pr(1, body="Fixes #8"), cfg, REPO, {7})


def test_judge_ready():
    ok = issue(1, ["agent-ready"], [ev("labeled", "2026-10-19T01:00:00Z", "Birfy", "agent-ready")])
    assert c.judge_ready(ok, CONFIG)["demote"] is None
    by_other = issue(1, ["agent-ready"], [ev("labeled", "2026-10-19T01:00:00Z", "HugoXK", "agent-ready")])
    assert c.judge_ready(by_other, CONFIG)["demote"] == "labeled-by-non-maintainer"
    edited = dict(ok, edits=[{"at": "2026-10-19T02:00:00Z", "editor": "someone"}])
    assert c.judge_ready(edited, CONFIG)["demote"] == "edited-by-non-maintainer"
    renamed = dict(ok, events=ok["events"] + [ev("renamed", "2026-10-19T02:00:00Z", "someone")])
    assert c.judge_ready(renamed, CONFIG)["demote"] == "edited-by-non-maintainer"
    before = dict(ok, edits=[{"at": "2026-10-18T02:00:00Z", "editor": "someone"}])
    assert c.judge_ready(before, CONFIG)["demote"] is None


def test_claim_expiry():
    claimed = issue(2, ["agent-working"], [ev("labeled", "2026-10-19T01:00:00Z", "Birfy", "agent-working")])
    assert c.claim_expired(claimed, [], CONFIG, NOW)
    assert not c.claim_expired(claimed, [{"lastPushAt": "2026-10-19T10:00:00Z"}], CONFIG, NOW)
    fresh = issue(2, ["agent-working"], [ev("labeled", "2026-10-19T09:00:00Z", "Birfy", "agent-working")])
    assert not c.claim_expired(fresh, [], CONFIG, NOW)


def test_plan_issues():
    ready_by_other = issue(3, ["agent-ready", "agent-working"],
                           [ev("labeled", "2026-10-19T01:00:00Z", "HugoXK", "agent-ready")])
    held = issue(4, ["agent-ready", "hold"], [ev("labeled", "2026-10-19T01:00:00Z", "HugoXK", "agent-ready")])
    out = run(issues=[ready_by_other, held])
    assert {(o["number"], o["op"], o["label"]) for o in out["ops"]} == {
        (3, "remove", "agent-ready"), (3, "remove", "agent-working")}
    assert {"number": 4, "label": "*", "reason": "hold"} in out["skipped"]


def test_plan_prs_get_one_risk_label_and_agent_pr_by_rule():
    out = run(prs=[pr(10, ["risk:T1", "risk:T3"], tier="T3"), pr(11, tier="T0", head="agent/x"),
                   pr(12, ["hold"], tier="T3"), pr(13, tier=None)])
    ops = {(o["number"], o["op"], o["label"]) for o in out["ops"]}
    assert ops == {(10, "remove", "risk:T1"), (11, "add", "risk:T0"), (11, "add", "agent-pr")}


def test_a_missing_label_is_skipped_never_created():
    out = run(prs=[pr(10, ["risk:T1"], tier="T2")], labels=["risk:T1"])
    assert [(o["op"], o["label"]) for o in out["ops"]] == [("remove", "risk:T1")]
    assert out["skipped"] == [{"number": 10, "label": "risk:T2", "reason": "label-missing"}]
    assert "not created" in c.render(dict(out, version=1), "live")


def test_plan_only_proposes_managed_labels_and_never_adds_agent_ready():
    ready = issue(5, ["agent-ready"], [ev("labeled", "2026-10-19T01:00:00Z", "Birfy", "agent-ready")])
    out = run(issues=[ready], prs=[pr(20, tier="T2", body="Fixes #5")])
    for o in out["ops"]:
        assert o["label"] in c.MANAGED[o["op"]]
    assert "agent-ready" not in c.MANAGED["add"]


@pytest.mark.parametrize("bad", [
    None, {}, {"version": 2, "ops": []},
    {"version": 1, "ops": [{"number": 0, "op": "add", "label": "agent-pr", "reason": "agent-pr"}]},
    {"version": 1, "ops": [{"number": True, "op": "add", "label": "agent-pr", "reason": "agent-pr"}]},
    {"version": 1, "ops": [{"number": 1, "op": "close", "label": "agent-pr", "reason": "agent-pr"}]},
    {"version": 1, "ops": [{"number": 1, "op": "add", "label": "agent-ready", "reason": "agent-pr"}]},
    {"version": 1, "ops": [{"number": 1, "op": "remove", "label": "hold", "reason": "risk-tier"}]},
    {"version": 1, "ops": [{"number": 1, "op": "add", "label": "risk:T0", "reason": "because"}]},
    {"version": 1, "ops": [{"number": 1, "op": "add", "label": "risk:T0", "reason": "risk-tier"}] * 201},
])
def test_validate_plan_refuses_anything_outside_the_contract(bad):
    with pytest.raises(ValueError):
        c.validate_plan(bad)


class FakeApi:
    def __init__(self, state, fail=None):
        self.state, self.fail, self.calls = state, fail or {}, []

    def get(self, path):
        self.calls.append(("GET", path))
        return self.state[int(path.rsplit("/", 1)[1])]

    def post(self, path, body):
        self.calls.append(("POST", path, body))
        if path in self.fail:
            raise RuntimeError(self.fail[path])

    def delete(self, path):
        self.calls.append(("DELETE", path))
        if path in self.fail:
            raise RuntimeError(self.fail[path])


def test_apply_writes_only_label_endpoints_and_rechecks_the_state():
    ops = [
        {"number": 1, "op": "add", "label": "risk:T1", "reason": "risk-tier"},
        {"number": 2, "op": "add", "label": "risk:T1", "reason": "risk-tier"},
        {"number": 3, "op": "remove", "label": "risk:T3", "reason": "risk-tier"},
        {"number": 4, "op": "remove", "label": "agent-working", "reason": "claim-expired"},
        {"number": 5, "op": "add", "label": "agent-pr", "reason": "agent-pr"},
    ]
    api = FakeApi({1: {"state": "open", "labels": []}, 2: {"state": "closed", "labels": []},
                   3: {"state": "open", "labels": [{"name": "risk:T3"}, {"name": "hold"}]},
                   4: {"state": "open", "labels": [{"name": "agent-working"}]},
                   5: {"state": "open", "labels": [{"name": "agent-pr"}]}})
    out = c.apply({"version": 1, "ops": ops}, api, REPO)
    assert [o["number"] for o in out["done"]] == [1, 4]
    assert {s["number"]: s["why"] for s in out["skipped"]} == {2: "closed", 3: "hold", 5: "already"}
    writes = [x for x in api.calls if x[0] != "GET"]
    assert writes == [("POST", f"repos/{REPO}/issues/1/labels", {"labels": ["risk:T1"]}),
                      ("DELETE", f"repos/{REPO}/issues/4/labels/agent-working")]


def test_apply_tolerates_a_vanished_label_and_reports_other_errors():
    ops = [{"number": 1, "op": "add", "label": "risk:T1", "reason": "risk-tier"},
           {"number": 2, "op": "add", "label": "risk:T1", "reason": "risk-tier"}]
    api = FakeApi({1: {"state": "open", "labels": []}, 2: {"state": "open", "labels": []}},
                  fail={f"repos/{REPO}/issues/1/labels": "HTTP 404: Not Found",
                        f"repos/{REPO}/issues/2/labels": "HTTP 500"})
    out = c.apply({"version": 1, "ops": ops}, api, REPO)
    assert [s["why"] for s in out["skipped"]] == ["label-missing", "error"]


def test_apply_refuses_unless_live_and_not_paused():
    plan = json.dumps({"version": 1, "mode": "live", "ops": []})
    for env in ({"GH_REPO": REPO, "PLAN": plan},
                {"GH_REPO": REPO, "PLAN": plan, "CONTROLLER_LIVE": "true", "AGENT_PAUSE": "true"},
                {"GH_REPO": REPO, "CONTROLLER_LIVE": "true",
                 "PLAN": json.dumps({"version": 1, "mode": "dry-run", "ops": []})}):
        with pytest.raises(ValueError):
            c.main("apply", env, api=FakeApi({}))


def test_a_paused_plan_reads_nothing(monkeypatch, tmp_path):
    out = tmp_path / "out"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary"))
    api = FakeApi({})
    c.main("plan", {"GH_REPO": REPO, "AGENT_PAUSE": "true", "CONTROLLER_LIVE": "true"}, api=api)
    assert api.calls == []
    assert "live=false" in out.read_text()
    assert "nothing was read" in (tmp_path / "summary").read_text()


def test_render_neutralizes_untrusted_characters():
    text = c.render({"ops": [{"kind": "pr", "number": 1, "op": "add", "label": "risk:T1|`x`", "reason": "r"}],
                     "skipped": [], "notes": ["::error::boom"]}, "dry-run")
    assert "Dry run" in text and "|`x`" not in text and "::error::" not in text


def test_source_has_no_merge_review_comment_close_or_code_execution():
    src = (SCRIPTS / "controller.py").read_text()
    for forbidden in ("/merge", "/reviews", "/comments", '"state": "closed"', "pull_request_target",
                      "eval(", "exec(", "shell=True"):
        assert forbidden not in src, forbidden
    assert set(re.findall(r'"-X", "([A-Z]+)"', src)) == {"POST", "DELETE"}


def test_workflow_shape():
    text = WORKFLOW.read_text(encoding="utf-8")
    wf = yaml.safe_load(text)
    on = wf.get("on", wf.get(True))
    assert set(on) == {"schedule", "workflow_dispatch"}
    assert on["schedule"] == [{"cron": "7,37 * * * *"}], "every 30 minutes"
    assert wf["permissions"] == {}
    plan, apply = wf["jobs"]["plan"], wf["jobs"]["apply"]
    assert plan["permissions"] == {"contents": "read", "issues": "read", "pull-requests": "read"}
    assert apply["permissions"] == {"contents": "read", "issues": "write", "pull-requests": "write"}
    assert "vars.CONTROLLER_LIVE == 'true'" in apply["if"] and "vars.AGENT_PAUSE != 'true'" in apply["if"]
    for job in (plan, apply):
        assert job["steps"][0]["with"]["ref"] == "${{ github.event.repository.default_branch }}"
    assert "${{ secrets." not in text
    runs = "\n".join(s.get("run", "") for j in (plan, apply) for s in j["steps"])
    assert "${{" not in runs, "values reach the shell only through env"


def test_labels_json_has_the_labels_the_controller_manages():
    labels = ROOT / ".github" / "labels.json"
    if not labels.exists():
        pytest.skip("labels.json lands in its own pull request")
    names = {w["name"] for w in json.loads(labels.read_text())}
    assert c.MANAGED["add"] | c.MANAGED["remove"] <= names
