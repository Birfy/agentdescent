"""`agentdescent release` and the MCP `release` tool: one trusted seam."""

import json

import pytest

from agentdescent import cli
from agentdescent.mcp import Tools
from agentdescent.release_surface import (
    ReleaseSurfaceError,
    load_lineage,
    load_scores,
    release_payload,
    release_text,
)


def _scores(gain=0.2, n=50, base=0.5):
    return {f"t{i}": base + gain for i in range(n)}


def _write(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


@pytest.fixture
def release_files(tmp_path):
    before = _write(tmp_path, "before.json", {f"t{i}": 0.5 for i in range(50)})
    after = _write(tmp_path, "after.json", _scores())
    return before, after


def _argv(before, after, tmp_path, **extra):
    argv = [
        "release", "--before", before, "--after", after,
        "--artifact-id", "skill-1", "--artifact-hash", "sha256:artifact-1",
        "--key-id", "ci", "--clean", "--risk", "0.05",
        "--min-effect", "0.01", "--json",
        "--lineage", str(tmp_path / "lineage.json"),
    ]
    for key, value in extra.items():
        argv.extend([key, value])
    return argv


def test_release_cli_signs_and_persists_lineage(
        tmp_path, release_files, capsys, monkeypatch):
    monkeypatch.setenv("AGENTDESCENT_RELEASE_KEY", "secret")
    before, after = release_files
    out = tmp_path / "release.json"
    code = cli.main(_argv(before, after, tmp_path, **{"--out": str(out)}))

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == 1
    assert payload["ok"] is True
    assert payload["released"] is True
    assert payload["decision"]["action"] == "promote"
    assert payload["release"]["verification"]["valid"] is True
    assert payload["release"]["manifest"]["content_hash"] == "sha256:artifact-1"
    assert json.loads(out.read_text())["released"] is True

    lineage = json.loads((tmp_path / "lineage.json").read_text())
    assert [node["artifact_id"] for node in lineage["nodes"]] == ["skill-1"]
    assert lineage["schema_version"] == 1


def test_reject_is_exit_one_and_leaves_lineage_untouched(
        tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("AGENTDESCENT_RELEASE_KEY", "secret")
    before = _write(tmp_path, "before.json", {f"t{i}": 0.5 for i in range(50)})
    after = _write(tmp_path, "after.json", {f"t{i}": 0.5 for i in range(50)})
    code = cli.main(_argv(before, after, tmp_path))

    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["released"] is False
    assert payload["release"] is None
    assert not (tmp_path / "lineage.json").exists()


def test_missing_secret_is_a_json_input_error(
        tmp_path, release_files, capsys, monkeypatch):
    monkeypatch.delenv("AGENTDESCENT_RELEASE_KEY", raising=False)
    before, after = release_files
    code = cli.main(_argv(before, after, tmp_path))

    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "AGENTDESCENT_RELEASE_KEY" in payload["error"]
    assert not (tmp_path / "lineage.json").exists()


def test_empty_artifact_hash_is_refused_before_any_release(
        tmp_path, release_files, capsys, monkeypatch):
    monkeypatch.setenv("AGENTDESCENT_RELEASE_KEY", "secret")
    before, after = release_files
    argv = _argv(before, after, tmp_path)
    argv[argv.index("--artifact-hash") + 1] = "   "
    code = cli.main(argv)

    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "artifact_hash" in payload["error"]


def test_rules_file_can_raise_the_bar(
        tmp_path, release_files, capsys, monkeypatch):
    monkeypatch.setenv("AGENTDESCENT_RELEASE_KEY", "secret")
    before, after = release_files
    rules = _write(tmp_path, "rules.json", {"min_effect": 0.9})
    code = cli.main(_argv(before, after, tmp_path, **{"--rules": rules}))

    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["released"] is False
    assert payload["decision"]["action"] in ("reject", "defer")


def test_parents_link_two_releases(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("AGENTDESCENT_RELEASE_KEY", "secret")
    before = _write(tmp_path, "before.json", {f"t{i}": 0.5 for i in range(50)})
    after = _write(tmp_path, "after.json", _scores())
    lineage = str(tmp_path / "lineage.json")

    first = cli.main([
        "release", "--before", before, "--after", after,
        "--artifact-id", "skill-1", "--artifact-hash", "sha256:a",
        "--key-id", "ci", "--clean", "--risk", "0.05",
        "--min-effect", "0.01", "--lineage", lineage, "--json"])
    assert first == 0
    capsys.readouterr()

    second = cli.main([
        "release", "--before", before, "--after", after,
        "--artifact-id", "skill-2", "--artifact-hash", "sha256:b",
        "--key-id", "ci", "--clean", "--risk", "0.05",
        "--min-effect", "0.01", "--lineage", lineage,
        "--parents", "skill-1", "--json"])
    assert second == 0
    capsys.readouterr()

    graph = load_lineage(lineage)
    assert graph.parents("skill-2") == ("skill-1",)


def test_human_output_is_short_and_actionable(
        tmp_path, release_files, capsys, monkeypatch):
    monkeypatch.setenv("AGENTDESCENT_RELEASE_KEY", "secret")
    before, after = release_files
    argv = _argv(before, after, tmp_path)
    argv.remove("--json")
    assert cli.main(argv) == 0
    text = capsys.readouterr().out
    assert "released: yes" in text
    assert "manifest:" in text
    assert "next:" in text


def test_mcp_release_tool_uses_the_same_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTDESCENT_RELEASE_KEY", "secret")
    before = {f"t{i}": 0.5 for i in range(50)}
    after = _scores()
    tools = Tools(surface_write_dirs=(str(tmp_path),))
    payload = tools.release(
        before, after, artifact_id="skill-mcp",
        artifact_hash="sha256:mcp", key_id="ci",
        lineage=str(tmp_path / "lineage.json"), contamination="clean",
        risk=0.05, min_effect=0.01)
    assert payload["ok"] is True
    assert payload["schema_version"] == 1
    assert payload["released"] is True
    assert (tmp_path / "lineage.json").exists()


def test_mcp_release_tool_reports_bad_material(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENTDESCENT_RELEASE_KEY", raising=False)
    tools = Tools(surface_write_dirs=(str(tmp_path),))
    payload = tools.release(
        {"t": 0.5}, {"t": 0.6}, artifact_id="skill",
        artifact_hash="sha256:x", key_id="ci")
    assert payload["ok"] is False
    assert "secret" in payload["error"]


def test_surface_loaders_are_fail_closed(tmp_path):
    path = tmp_path / "scores.json"
    path.write_text(json.dumps({"t": 0.5}), encoding="utf-8")
    assert load_scores(str(path), name="before") == {"t": 0.5}
    assert load_scores({"t": 0.5}, name="before") == {"t": 0.5}
    with pytest.raises(ReleaseSurfaceError):
        load_scores({}, name="before")
    with pytest.raises(ReleaseSurfaceError):
        load_scores({"t": float("nan")}, name="before")
    with pytest.raises(ReleaseSurfaceError):
        load_scores(str(tmp_path / "missing.json"), name="before")

    malformed = tmp_path / "lineage.json"
    malformed.write_text("{not json", encoding="utf-8")
    with pytest.raises(ReleaseSurfaceError):
        load_lineage(str(malformed))


def test_a_conformal_reject_is_not_released_by_default(monkeypatch):
    monkeypatch.setenv("AGENTDESCENT_RELEASE_KEY", "secret")
    before = {f"t{i}": 1.0 for i in range(30)}
    after = {f"t{i}": (2.0 if i < 25 else 0.0) for i in range(30)}
    payload = release_payload(
        before, after, artifact_id="skill", artifact_hash="sha256:x",
        key_id="ci", secret=b"secret", contaminated=False, risk=0.0,
        min_effect=0.0)
    assert payload["released"] is False
    assert "conformal" in " ".join(payload["decision"]["reasons"])
    assert payload["release"] is None


def test_release_text_handles_reject(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTDESCENT_RELEASE_KEY", "secret")
    payload = release_payload(
        {"t0": 0.5, "t1": 0.5}, {"t0": 0.5, "t1": 0.5},
        artifact_id="skill", artifact_hash="sha256:x", key_id="ci",
        secret=b"secret", contaminated=False, risk=0.05,
        min_effect=0.9)
    text = release_text(payload)
    assert "released: no" in text
    assert "decision:" in text
    assert "next:" in text
