"""The adaptive experience cycle: ingest -> plan -> guarded update."""

import json
import os

from agentdescent import cli
from agentdescent.mcp import Tools


def _json(capsys):
    return json.loads(capsys.readouterr().out)


def _records(path):
    rows = [
        {"run_id": "r1", "features": [0.1, 0.2],
         "rewards": {"edit": 1.0, "retry": 0.0}, "accepted": True},
        {"run_id": "r2", "features": [0.2, 0.1],
         "rewards": {"edit": 0.5, "retry": 0.5}, "accepted": True},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n",
                    encoding="utf-8")
    return str(path)


def test_experience_cycle_runs_and_serializes(tmp_path, capsys):
    store = tmp_path / "experience.jsonl"
    records = _records(tmp_path / "records.jsonl")
    assert cli.main([
        "experience", "cycle", "--store", str(store), "--records", records,
        "--actions", "edit,retry", "--dim", "2", "--no-require-anchors",
        "--json"]) == 0
    payload = _json(capsys)
    assert payload["schema_version"] == 1
    assert payload["report"]["ingested"] == ["r1", "r2"]
    assert payload["report"]["plan"]["source"] == "self-validation"
    assert payload["report"]["update"]["accepted"] is True
    assert payload["report"]["update"]["rolled_back"] is False
    assert payload["summary"]["rows"] == 2


def test_experience_cycle_requires_records_or_validation(tmp_path, capsys):
    store = tmp_path / "experience.jsonl"
    assert cli.main([
        "experience", "cycle", "--store", str(store), "--actions", "edit",
        "--dim", "2", "--json"]) == 2
    payload = _json(capsys)
    assert payload["ok"] is False
    assert "records" in payload["error"]


def test_experience_cycle_human_output(tmp_path, capsys):
    store = tmp_path / "experience.jsonl"
    records = _records(tmp_path / "records.jsonl")
    assert cli.main([
        "experience", "cycle", "--store", str(store), "--records", records,
        "--actions", "edit,retry", "--dim", "2", "--no-require-anchors"]) == 0
    text = capsys.readouterr().out
    assert "ingested: 2" in text
    assert "source: self-validation" in text
    assert "accepted: True" in text


def test_mcp_experience_cycle_and_write_confinement(tmp_path):
    store = tmp_path / "experience.jsonl"
    records = [
        {"run_id": "r1", "features": [0.1, 0.2],
         "rewards": {"edit": 1.0, "retry": 0.0}, "accepted": True},
        {"run_id": "r2", "features": [0.2, 0.1],
         "rewards": {"edit": 0.5, "retry": 0.5}, "accepted": True},
    ]
    tools = Tools(surface_write_dirs=(str(tmp_path),))
    payload = tools.experience_cycle(
        str(store), records, None, None, "edit,retry", 2,
        require_anchors=False)
    assert payload["ok"] is True
    assert payload["report"]["update"]["accepted"] is True

    outside = str(tmp_path.parent / "experience-outside.jsonl")
    denied = tools.experience_cycle(
        outside, records, None, None, "edit,retry", 2,
        require_anchors=False)
    assert denied["ok"] is False
    assert "write allowlist" in denied["error"]
