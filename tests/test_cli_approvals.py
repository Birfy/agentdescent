"""``agentdescent governed-harness approvals|approve|reject``: the queue."""

import json

from agentdescent import cli
from agentdescent.approval import ApprovalStore


def _json(capsys):
    return json.loads(capsys.readouterr().out)


def _store(tmp_path):
    store = ApprovalStore(str(tmp_path / "approvals.jsonl"))
    store.request("key-1", "harness-1", "high-risk self-modification",
                  summary='{"target": "harness/f.py"}')
    store.request("key-2", "harness-2", "high-risk self-modification")
    return store.path


def test_approvals_list_defaults_to_pending(tmp_path, capsys):
    path = _store(tmp_path)
    assert cli.main(["governed-harness", "approvals", "--store", path,
                     "--json"]) == 0
    payload = _json(capsys)
    assert payload["schema_version"] == 1
    assert payload["ok"] is True
    assert [row["key"] for row in payload["approvals"]] == ["key-1", "key-2"]
    assert payload["counts"] == {
        "pending": 2, "approved": 0, "rejected": 0, "total": 2}


def test_approve_then_list_all(tmp_path, capsys):
    path = _store(tmp_path)
    assert cli.main(["governed-harness", "approve", "--store", path,
                     "--key", "key-1", "--note", "looks right", "--json"]) == 0
    approved = _json(capsys)
    assert approved["approval"]["status"] == "approved"
    assert approved["approval"]["note"] == "looks right"
    assert approved["counts"] == {
        "pending": 1, "approved": 1, "rejected": 0, "total": 2}

    assert cli.main(["governed-harness", "approvals", "--store", path,
                     "--status", "all", "--json"]) == 0
    rows = {row["key"]: row for row in _json(capsys)["approvals"]}
    assert rows["key-1"]["status"] == "approved"
    assert rows["key-2"]["status"] == "pending"

    assert cli.main(["governed-harness", "approvals", "--store", path,
                     "--json"]) == 0
    pending = _json(capsys)
    assert [row["key"] for row in pending["approvals"]] == ["key-2"]


def test_reject_marks_the_request(tmp_path, capsys):
    path = _store(tmp_path)
    assert cli.main(["governed-harness", "reject", "--store", path,
                     "--key", "key-2", "--note", "not safe", "--json"]) == 0
    rejected = _json(capsys)
    assert rejected["approval"]["status"] == "rejected"
    assert rejected["counts"]["rejected"] == 1

    assert cli.main(["governed-harness", "approvals", "--store", path,
                     "--status", "rejected", "--json"]) == 0
    rows = _json(capsys)["approvals"]
    assert [row["key"] for row in rows] == ["key-2"]
    assert rows[0]["decided_at"]


def test_unknown_key_is_an_input_error(tmp_path, capsys):
    path = _store(tmp_path)
    assert cli.main(["governed-harness", "approve", "--store", path,
                     "--key", "missing", "--json"]) == 2
    payload = _json(capsys)
    assert payload["ok"] is False
    assert "no approval request" in payload["error"]


def test_missing_store_lists_an_empty_queue_but_cannot_decide(
        tmp_path, capsys):
    path = str(tmp_path / "none.jsonl")
    assert cli.main(["governed-harness", "approvals", "--store", path,
                     "--json"]) == 0
    assert _json(capsys)["counts"]["total"] == 0

    assert cli.main(["governed-harness", "approve", "--store", path,
                     "--key", "x", "--json"]) == 2


def test_a_decided_request_cannot_be_flipped(tmp_path, capsys):
    path = _store(tmp_path)
    assert cli.main(["governed-harness", "reject", "--store", path,
                     "--key", "key-1", "--note", "unsafe", "--json"]) == 0
    capsys.readouterr()

    code = cli.main(["governed-harness", "approve", "--store", path,
                     "--key", "key-1", "--json"])
    assert code == 2
    payload = _json(capsys)
    assert payload["ok"] is False
    assert "already rejected" in payload["error"]
    assert "only pending" in payload["error"]
    assert ApprovalStore(path).status("key-1") == "rejected"

    assert cli.main(["governed-harness", "approvals", "--store", path,
                     "--status", "rejected", "--json"]) == 0
    assert _json(capsys)["counts"] == {
        "pending": 1, "approved": 0, "rejected": 1, "total": 2}


def test_approving_twice_is_refused(tmp_path, capsys):
    path = _store(tmp_path)
    assert cli.main(["governed-harness", "approve", "--store", path,
                     "--key", "key-1", "--json"]) == 0
    capsys.readouterr()
    assert cli.main(["governed-harness", "approve", "--store", path,
                     "--key", "key-1", "--json"]) == 2
    assert "already approved" in _json(capsys)["error"]
