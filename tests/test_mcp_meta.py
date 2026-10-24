"""The meta stack over MCP: `safety`, `portfolio`, `audit_trace` (#505).

A host agent asked "is this archive safe to seed from?" or "how should the next
budget window be split?" had only the CLI verbs, which the installed skill never
mentioned. The tools reuse the CLI's own payload functions, so these pin three
things: the tool returns exactly what `--json` prints, a missing archive is an
error that creates nothing, and no tool writes a file.
"""

import json
import os

import pytest

from agentdescent import cli
from agentdescent.mcp import Tools

from tests.test_meta_cli import _archive


def _tree(root):
    return sorted(
        (os.path.relpath(os.path.join(d, f), root),
         os.path.getmtime(os.path.join(d, f)))
        for d, _, files in os.walk(root) for f in files)


def _cli_json(capsys, argv):
    cli.main(["--json", *argv])
    return json.loads(capsys.readouterr().out)


def test_safety_matches_cli_json_and_writes_nothing(tmp_path, capsys):
    archive = _archive(tmp_path, n=2)
    before = _tree(tmp_path)
    got = Tools().safety(archive)
    assert _tree(tmp_path) == before
    assert got == _cli_json(capsys, ["safety", "--archive", archive])
    assert got["verdict"] in ("safe", "caution", "unsafe")
    assert got["archive"] == archive


def test_safety_min_exam_none_is_accepted(tmp_path):
    archive = _archive(tmp_path)
    assert "error" not in Tools().safety(archive, min_exam="none")
    assert "error" in Tools().safety(archive, min_exam="bogus")


def test_portfolio_matches_cli_json_and_writes_nothing(tmp_path, capsys):
    archive = _archive(tmp_path, n=2)
    before = _tree(tmp_path)
    got = Tools().portfolio(archive)
    assert _tree(tmp_path) == before
    assert got == _cli_json(capsys, ["portfolio", archive])
    assert set(got["reallocated"]) == {layer["name"] for layer in got["layers"]}


def test_portfolio_empty_archive_is_an_error(tmp_path):
    (tmp_path / "empty").mkdir()
    assert "no rule records" in Tools().portfolio(str(tmp_path / "empty"))["error"]


def test_audit_trace_matches_cli_json_and_writes_nothing(tmp_path, capsys):
    archive = _archive(tmp_path)
    before = _tree(tmp_path)
    got = Tools().audit_trace("run-0", archive=archive)
    assert _tree(tmp_path) == before
    assert got == _cli_json(capsys, ["audit", "run-0", "--archive", archive])
    assert got["links"]


def test_audit_trace_unknown_run_is_an_error(tmp_path):
    archive = _archive(tmp_path)
    assert "error" in Tools().audit_trace("nope", archive=archive)


@pytest.mark.parametrize("call", [
    lambda t, p: t.safety(p),
    lambda t, p: t.portfolio(p),
    lambda t, p: t.audit_trace("run-0", archive=p),
])
def test_missing_archive_is_an_error_and_is_not_created(tmp_path, call):
    missing = tmp_path / "typo"
    got = call(Tools(), str(missing))
    assert got == {"error": f"archive {str(missing)!r} does not exist"}
    assert not missing.exists()


@pytest.mark.parametrize("kwarg", ["meta_audit", "inner_audit"])
def test_audit_trace_missing_store_is_an_error_and_is_not_created(tmp_path, kwarg):
    archive = _archive(tmp_path)
    missing = str(tmp_path / "nope.jsonl")
    got = Tools().audit_trace("run-0", archive=archive, **{kwarg: missing})
    assert "does not exist" in got["error"]
    assert not os.path.exists(missing)


def test_safety_defaults_to_agentdescent_home(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTDESCENT_HOME", str(tmp_path / "home"))
    _archive(tmp_path, at="home/archive")
    got = Tools().safety()
    assert got["archive"] == str(tmp_path / "home" / "archive")
