"""``agentdescent ops history``: a durable log across publishes."""

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
    return spec["runs_db"]


def _publish(runs_db, history, capsys, tenant="demo", workspace="default"):
    code = cli.main([
        "ops", "publish", "--runs-db", runs_db, "--tenant", tenant,
        "--workspace", workspace, "--history", str(history), "--json"])
    assert code == 0
    return _json(capsys)


def test_publish_appends_and_history_aggregates(tmp_path, capsys):
    runs_db = _campaign(tmp_path, capsys)
    history = tmp_path / "ops.jsonl"

    first = _publish(runs_db, history, capsys)
    second = _publish(runs_db, history, capsys)
    assert first["published"] > 0
    assert first["history"] == str(history)
    lines = history.read_text().splitlines()
    assert len(lines) == 2
    record = json.loads(lines[0])
    assert record["schema_version"] == 1
    assert record["tenant"] == "demo" and record["workspace"] == "default"
    assert [row["run_id"] for row in record["runs"]] == [
        row["run"]["run_id"] for row in second["report"]["records"]]
    assert len(record["runs"]) == first["published"]
    assert all(row["metrics"] for row in record["runs"])

    assert cli.main(["ops", "history", "--history", str(history),
                     "--json"]) == 0
    summary = _json(capsys)
    assert summary["schema_version"] == 1
    assert summary["publishes"] == 2
    # The same runs were published twice; deduplicating by run id keeps the
    # totals equal to one snapshot, not two.
    assert summary["published"] == first["published"]
    assert summary["runs"] == first["published"]
    assert summary["decisions"] == first["metrics"]["decisions"]
    assert summary["accepted"] == first["metrics"]["accepted"]
    assert summary["calls"] == first["metrics"]["calls"]
    assert summary["alerts"] == len(set(first["alerts"]))
    assert len(summary["run_ids"]) == first["published"]
    assert summary["last"]["published"] == first["published"]
    assert summary["last_recorded_at"]
    assert summary["skipped"] == 0


def test_bad_lines_are_skipped_and_counted(tmp_path, capsys):
    runs_db = _campaign(tmp_path, capsys)
    history = tmp_path / "ops.jsonl"
    _publish(runs_db, history, capsys)
    with history.open("a", encoding="utf-8") as handle:
        handle.write("not json\n")
        handle.write("[1, 2, 3]\n")

    assert cli.main(["ops", "history", "--history", str(history),
                     "--json"]) == 0
    summary = _json(capsys)
    assert summary["publishes"] == 1
    assert summary["skipped"] == 2


def test_tenant_and_workspace_filter_without_rewriting(tmp_path, capsys):
    history = tmp_path / "ops.jsonl"
    rows = [
        {"schema_version": 1, "tenant": "t1", "workspace": "w1",
         "recorded_at": 100.0,
         "runs": [
             {"run_id": "a", "metrics": {"decisions": 2, "accepted": 1,
                                         "calls": 3, "categories": {"accept": 1}}},
             {"run_id": "b", "metrics": {"decisions": 1, "accepted": 0,
                                         "calls": 1}}],
         "metrics": {}, "alerts": []},
        {"schema_version": 1, "tenant": "t1", "workspace": "w2",
         "recorded_at": 200.0,
         "runs": [{"run_id": "c", "metrics": {"decisions": 1,
                                              "accepted": 1, "calls": 2}}],
         "metrics": {}, "alerts": ["x"]},
        {"schema_version": 1, "tenant": "t2", "workspace": "w1",
         "recorded_at": 300.0,
         "runs": [{"run_id": "d", "metrics": {"decisions": 5,
                                              "accepted": 5, "calls": 9}}],
         "metrics": {}, "alerts": []},
    ]
    history.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    before = history.read_text()

    assert cli.main(["ops", "history", "--history", str(history),
                     "--tenant", "t1", "--workspace", "w1", "--json"]) == 0
    summary = _json(capsys)
    assert summary["publishes"] == 1
    assert summary["published"] == 2
    assert summary["runs"] == 2
    assert summary["decisions"] == 3
    assert summary["accepted"] == 1
    assert summary["calls"] == 4
    assert summary["categories"] == {"accept": 1}
    assert summary["last_recorded_at"].startswith("1970-01-01T00:01:40")
    assert history.read_text() == before


def test_malformed_metric_fields_do_not_crash(tmp_path, capsys):
    history = tmp_path / "ops.jsonl"
    rows = [
        {"schema_version": 1, "tenant": "t1", "workspace": "w1",
         "recorded_at": 100.0,
         "runs": [{"run_id": "a", "metrics": {
             "decisions": 1, "accepted": 1, "calls": 2,
             "categories": ["not", "a", "dict"]}}],
         "metrics": {"categories": ["also", "bad"]}, "alerts": "not a list"},
        {"schema_version": 1, "tenant": "t1", "workspace": "w1",
         "recorded_at": 101.0, "runs": [], "metrics": {}, "alerts": []},
    ]
    history.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert cli.main(["ops", "history", "--history", str(history),
                     "--json"]) == 0
    summary = _json(capsys)
    assert summary["decisions"] == 1
    assert summary["categories"] == {}
    assert summary["alerts"] == 0
    assert summary["skipped"] == 1


def test_publish_creates_the_history_directory(tmp_path, capsys):
    runs_db = _campaign(tmp_path, capsys)
    history = tmp_path / "nested" / "logs" / "ops.jsonl"
    payload = _publish(runs_db, history, capsys)
    assert payload["history"] == str(history)
    assert history.is_file()


def test_unwritable_history_is_an_input_error(tmp_path, capsys):
    runs_db = _campaign(tmp_path, capsys)
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    code = cli.main([
        "ops", "publish", "--runs-db", runs_db, "--tenant", "demo",
        "--workspace", "default", "--history", str(blocker / "ops.jsonl"),
        "--json"])
    assert code == 2
    assert _json(capsys)["ok"] is False


def test_missing_history_is_empty_not_an_error(tmp_path, capsys):
    assert cli.main(["ops", "history", "--history",
                     str(tmp_path / "none.jsonl"), "--json"]) == 0
    summary = _json(capsys)
    assert summary["ok"] is True
    assert summary["publishes"] == 0
    assert summary["last"] is None
    assert summary["last_recorded_at"] is None
