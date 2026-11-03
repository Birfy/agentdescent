"""The meta-campaign surface: continuous meta-learning as a bounded loop."""

import json
import os

import pytest

from agentdescent import cli
from agentdescent.mcp import Tools


def _json(capsys):
    return json.loads(capsys.readouterr().out)


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_meta_campaign_init_plan_run_demo(tmp_path, capsys):
    spec = tmp_path / "meta.json"
    assert cli.main(["meta-campaign", "init", "--out", str(spec),
                     "--demo", "--json"]) == 0
    init = _json(capsys)
    assert init["ok"] is True
    assert init["demo"] is True

    assert cli.main(["meta-campaign", "plan", "--spec", str(spec),
                     "--json"]) == 0
    plan = _json(capsys)
    assert plan["ok"] is True
    assert plan["archive_exists"] is False
    assert plan["slot"] == "task_sampler"

    out = tmp_path / "meta-report.json"
    assert cli.main(["meta-campaign", "run", "--spec", str(spec),
                     "--json", "--out", str(out)]) == 0
    payload = _json(capsys)
    assert payload["schema_version"] == 1
    assert payload["report"]["complete"] is True
    assert len(payload["report"]["sessions"]) == 2
    assert payload["report"]["archive_count"] >= 1
    assert all(row["verdict"] == "adopt"
               for row in payload["report"]["sessions"])
    assert json.loads(out.read_text())["report"]["complete"] is True


def test_meta_campaign_bad_ref_is_json_error(tmp_path, capsys):
    spec_path = tmp_path / "meta.json"
    cli.main(["meta-campaign", "init", "--out", str(spec_path),
              "--demo", "--json"])
    capsys.readouterr()
    spec = json.loads(spec_path.read_text())
    spec["tasks_for"] = "nosuchmodule:tasks_for"
    path = _write(tmp_path / "bad.json", spec)
    assert cli.main(["meta-campaign", "plan", "--spec", path, "--json"]) == 2
    payload = _json(capsys)
    assert payload["ok"] is False
    assert "nosuchmodule" in payload["error"]


def test_mcp_meta_campaign_and_allowlist(tmp_path):
    spec_path = tmp_path / "meta.json"
    cli.main(["meta-campaign", "init", "--out", str(spec_path),
              "--demo", "--json"])
    spec = json.loads(spec_path.read_text())
    tools = Tools(surface_allow=("agentdescent.",),
                  surface_write_dirs=(str(tmp_path),))
    plan = tools.meta_campaign(spec, dry_run=True)
    assert plan["ok"] is True
    run = tools.meta_campaign(spec, dry_run=False)
    assert run["report"]["complete"] is True

    denied = Tools(surface_allow=())
    payload = denied.meta_campaign(spec, dry_run=True)
    assert payload["ok"] is False
    assert "allowlist" in payload["error"]


def test_mcp_meta_campaign_writes_are_confined(tmp_path):
    spec_path = tmp_path / "meta.json"
    cli.main(["meta-campaign", "init", "--out", str(spec_path),
              "--demo", "--json"])
    spec = json.loads(spec_path.read_text())
    tools = Tools(surface_allow=("agentdescent.",),
                  surface_write_dirs=(str(tmp_path),))
    outside = str(tmp_path.parent / "meta-outside")
    payload = tools.meta_campaign(dict(spec, archive=outside), dry_run=True)
    assert payload["ok"] is False
    assert "write allowlist" in payload["error"]
    payload = tools.meta_campaign(dict(spec, repo_dir=outside), dry_run=True)
    assert payload["ok"] is False
    assert "write allowlist" in payload["error"]
