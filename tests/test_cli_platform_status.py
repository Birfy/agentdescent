"""``agentdescent platform status``: inspect a finished campaign."""

import gc
import hashlib
import json

from agentdescent import cli


def _json(capsys):
    return json.loads(capsys.readouterr().out)


def _campaign(tmp_path, capsys):
    spec_path = tmp_path / "platform.json"
    assert cli.main(["platform", "init", "--out", str(spec_path),
                     "--demo", "--json"]) == 0
    spec = json.loads(spec_path.read_text())
    capsys.readouterr()
    assert cli.main(["platform", "run", "--spec", str(spec_path),
                     "--json"]) == 0
    capsys.readouterr()
    gc.collect()   # the writer process has exited; release its SQLite handle
    return spec["runs_db"]


def test_status_lists_runs_and_one_full_report(tmp_path, capsys):
    runs_db = _campaign(tmp_path, capsys)

    assert cli.main(["platform", "status", "--runs-db", runs_db,
                     "--json"]) == 0
    listing = _json(capsys)
    assert listing["schema_version"] == 1
    assert listing["ok"] is True
    assert listing["tenant"] is None
    assert listing["runs"]
    row = listing["runs"][0]
    assert set(row) == {
        "run_id", "tenant", "workspace", "status", "best_reward",
        "generations", "accepted", "stopped_reason"}
    assert row["tenant"] == "demo" and row["workspace"] == "default"

    assert cli.main(["platform", "status", "--runs-db", runs_db,
                     "--run-id", row["run_id"], "--json"]) == 0
    detail = _json(capsys)
    assert detail["run_id"] == row["run_id"]
    assert detail["status"] == row["status"]
    assert len(detail["report"]["generations"]) == row["generations"]
    assert detail["report"]["run_id"] == row["run_id"]


def test_status_limit_and_filters(tmp_path, capsys):
    runs_db = _campaign(tmp_path, capsys)
    assert cli.main(["platform", "status", "--runs-db", runs_db,
                     "--limit", "1", "--json"]) == 0
    limited = _json(capsys)
    assert len(limited["runs"]) == 1

    assert cli.main(["platform", "status", "--runs-db", runs_db,
                     "--tenant", "demo", "--workspace", "default",
                     "--json"]) == 0
    assert _json(capsys)["runs"]

    assert cli.main(["platform", "status", "--runs-db", runs_db,
                     "--tenant", "other", "--json"]) == 0
    assert _json(capsys)["runs"] == []


def test_status_missing_db_and_unknown_run_are_errors(tmp_path, capsys):
    assert cli.main(["platform", "status", "--runs-db",
                     str(tmp_path / "missing.sqlite3"), "--json"]) == 2
    payload = _json(capsys)
    assert payload["ok"] is False and "does not exist" in payload["error"]

    runs_db = _campaign(tmp_path, capsys)
    assert cli.main(["platform", "status", "--runs-db", runs_db,
                     "--run-id", "nope", "--json"]) == 2
    payload = _json(capsys)
    assert payload["ok"] is False and "unknown run" in payload["error"]


def test_status_does_not_change_the_database(tmp_path, capsys):
    runs_db = _campaign(tmp_path, capsys)
    with open(runs_db, "rb") as handle:
        before = hashlib.sha256(handle.read()).hexdigest()

    assert cli.main(["platform", "status", "--runs-db", runs_db,
                     "--json"]) == 0
    capsys.readouterr()
    with open(runs_db, "rb") as handle:
        after = hashlib.sha256(handle.read()).hexdigest()
    assert after == before


def test_mcp_platform_status_is_read_only(tmp_path, capsys):
    from agentdescent.mcp import Tools

    runs_db = _campaign(tmp_path, capsys)
    tools = Tools(surface_write_dirs=())
    payload = tools.platform_status(runs_db)
    assert payload["ok"] is True and payload["runs"]
    missing = tools.platform_status(str(tmp_path / "missing.sqlite3"))
    assert missing["ok"] is False
