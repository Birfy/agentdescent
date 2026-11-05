"""Layered trace digests: sourced summaries a reflector can actually use."""

import json

import pytest

from agentdescent import cli
from agentdescent.mcp import Tools
from agentdescent.trace_digest import (
    TraceDigestConfig,
    TraceDigestError,
    digest_traces,
    drilldown,
    load_trace_events,
)


def _events():
    return [
        {"run_id": "r1", "task_id": "t1", "step": 0, "kind": "prompt",
         "text": "solve 2+2"},
        {"run_id": "r1", "task_id": "t1", "step": 1, "kind": "tool",
         "name": "bash", "text": "timeout after 3s",
         "payload": {"status": "timeout"}},
        {"run_id": "r1", "task_id": "t1", "step": 2, "kind": "error",
         "text": "Traceback ValueError: 42"},
        {"run_id": "r1", "task_id": "t2", "step": 0, "kind": "tool",
         "name": "bash", "text": "timeout after 7s",
         "payload": {"status": "timeout"}},
        {"run_id": "r1", "task_id": "t2", "step": 1, "kind": "score",
         "text": "reward", "reward": 1.0},
        {"run_id": "r2", "task_id": "t3", "step": 0, "kind": "score",
         "text": "reward", "reward": 0.0},
        {"run_id": "r2", "task_id": "t3", "step": 1, "kind": "error",
         "text": "Traceback ValueError: 42"},
    ]


def _write_jsonl(tmp_path, rows):
    path = tmp_path / "traces.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return str(path)


def test_digest_summarises_runs_tasks_and_signatures():
    events = load_trace_events(_events())
    digest = digest_traces(events)

    assert digest.schema_version == 1
    assert digest.events_total == 7
    assert len(digest.runs) == 2
    assert len(digest.tasks) == 3
    assert digest.summary.startswith("2 run(s), 3 task(s), 7 event(s)")

    by_message = {row.message: row for row in digest.signatures}
    timeout = by_message["timeout after #s"]
    assert timeout.count == 2
    assert timeout.kind == "tool"
    assert timeout.name == "bash"
    assert len(timeout.event_ids) == 2
    assert set(timeout.run_ids) == {"r1"}
    assert set(timeout.task_ids) == {"t1", "t2"}

    value = by_message["traceback valueerror: #"]
    assert value.count == 2
    assert set(value.run_ids) == {"r1", "r2"}
    assert "timeout after #s" in digest.text
    assert "traceback valueerror: #" in digest.text

    runs = {row["run_id"]: row for row in digest.runs}
    assert runs["r1"]["reward"] == 1.0
    assert runs["r2"]["reward"] == 0.0
    tasks = {(row["run_id"], row["task_id"]): row for row in digest.tasks}
    assert tasks[("r1", "t2")]["reward"] == 1.0
    assert tasks[("r2", "t3")]["reward"] == 0.0
    assert tasks[("r1", "t1")]["outcome"] == "failed"
    assert tasks[("r1", "t2")]["outcome"] == "failed"
    assert tasks[("r2", "t3")]["outcome"] == "failed"


def test_digest_is_deterministic_and_sources_drill_down():
    first = digest_traces(load_trace_events(_events())).to_dict()
    second = digest_traces(load_trace_events(_events())).to_dict()
    assert first == second

    digest = digest_traces(load_trace_events(_events()))
    timeout = next(row for row in digest.signatures
                   if row.message == "timeout after #s")
    raw = drilldown(load_trace_events(_events()), signature=timeout.id)
    assert raw["count"] == 2
    assert {event["text"] for event in raw["events"]} == {
        "timeout after 3s", "timeout after 7s"}


def test_digest_accepts_jsonl_json_array_and_objects(tmp_path):
    rows = _events()
    path = _write_jsonl(tmp_path, rows)
    from_file = digest_traces(load_trace_events(path))
    from_list = digest_traces(load_trace_events(rows))
    array = tmp_path / "traces.json"
    array.write_text(json.dumps(rows), encoding="utf-8")
    from_array = digest_traces(load_trace_events(str(array)))
    from_object = digest_traces(load_trace_events({"events": rows}))
    for other in (from_list, from_array, from_object):
        assert other.to_dict() == from_file.to_dict()


def test_digest_rejects_bad_input(tmp_path):
    with pytest.raises(TraceDigestError):
        load_trace_events("")
    with pytest.raises(TraceDigestError):
        load_trace_events(str(tmp_path / "missing.jsonl"))
    bad = tmp_path / "bad.jsonl"
    bad.write_text("not json\n", encoding="utf-8")
    with pytest.raises(TraceDigestError):
        load_trace_events(str(bad))
    with pytest.raises(TraceDigestError):
        drilldown(load_trace_events(_events()), signature="sig_nope")
    with pytest.raises(TraceDigestError):
        digest_traces(load_trace_events(_events()),
                      config=TraceDigestConfig(max_digest_chars=0))


def test_digest_redacts_secret_values_and_keys():
    rows = [{"run_id": "r", "task_id": "t", "kind": "error",
             "text": "failed with ghp_abcdefghijklmnopqrstuvwxyz",
             "payload": {"token": "real-secret", "error": "boom"}},
            {"run_id": "r", "task_id": "t2", "kind": "tool",
             "payload": {"api_key": "abc123secret", "cmd": "ls"}}]
    digest = digest_traces(load_trace_events(rows))
    assert "ghp_abcdefghijklmnopqrstuvwxyz" not in digest.text
    assert "real-secret" not in digest.text
    assert "abc123secret" not in digest.text
    assert "[REDACTED]" in digest.text


def test_payload_value_is_not_a_reward_and_low_rewards_are_failures():
    rows = [
        {"run_id": "r", "task_id": "t1", "kind": "tool_result",
         "name": "calc", "payload": {"value": 0}},
        {"run_id": "r", "task_id": "t2", "kind": "score", "reward": 0.1},
    ]
    digest = digest_traces(load_trace_events(rows))
    tasks = {(row["run_id"], row["task_id"]): row for row in digest.tasks}
    assert tasks[("r", "t1")]["outcome"] == "unknown"
    assert tasks[("r", "t1")]["reward"] is None
    assert tasks[("r", "t2")]["outcome"] == "failed"
    assert all(row.name != "calc" for row in digest.signatures)


def test_drilldown_finds_a_signature_outside_the_digest_cap():
    full = digest_traces(load_trace_events(_events()))
    timeout = next(row for row in full.signatures
                   if row.message == "timeout after #s")
    capped = digest_traces(
        load_trace_events(_events()), config=TraceDigestConfig(max_signatures=1))
    assert len(capped.signatures) == 1
    assert timeout.id not in capped.sources
    raw = drilldown(load_trace_events(_events()), signature=timeout.id)
    assert raw["count"] == 2


def test_digest_bounds_the_prompt_text():
    rows = _events()
    digest = digest_traces(
        load_trace_events(rows), config=TraceDigestConfig(max_digest_chars=80))
    assert digest.truncated is True
    assert len(digest.text) <= 80


def test_digest_reports_signature_truncation():
    digest = digest_traces(
        load_trace_events(_events()),
        config=TraceDigestConfig(max_signatures=1))
    assert digest.truncated is True
    assert len(digest.signatures) == 1


def test_cli_traces_digest_and_show(tmp_path, capsys):
    path = _write_jsonl(tmp_path, _events())
    out = tmp_path / "digest.json"

    assert cli.main(["traces", "digest", "--events", path,
                     "--json", "--out", str(out)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == 1
    assert payload["events_total"] == 7
    assert json.loads(out.read_text())["events_total"] == 7

    signature = payload["signatures"][0]["id"]
    assert cli.main(["traces", "show", "--events", path,
                     "--signature", signature, "--json"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["signature"] == signature
    assert shown["count"] >= 1
    assert shown["events"]

    assert cli.main(["traces", "digest", "--events",
                     str(tmp_path / "missing.jsonl"), "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_mcp_traces_digest_is_read_only(tmp_path):
    path = _write_jsonl(tmp_path, _events())
    tools = Tools(surface_write_dirs=())
    payload = tools.traces_digest(path)
    assert payload["schema_version"] == 1
    assert payload["events_total"] == 7

    missing = tools.traces_digest(str(tmp_path / "missing.jsonl"))
    assert missing["ok"] is False
