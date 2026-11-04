"""``agentdescent release verify``: check a receipt without trusting it."""

import json

from agentdescent import cli
from agentdescent.mcp import Tools
from agentdescent.release_surface import verify_release


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _release(tmp_path, monkeypatch, *, key="secret"):
    """Run one accepted release and return its receipt path."""
    monkeypatch.setenv("AGENTDESCENT_RELEASE_KEY", key)
    before = _write(tmp_path / "before.json", {f"t{i}": 0.5 for i in range(50)})
    after = _write(tmp_path / "after.json", {f"t{i}": 0.7 for i in range(50)})
    receipt = tmp_path / "release.json"
    code = cli.main([
        "release", "--before", before, "--after", after,
        "--artifact-id", "skill-1", "--artifact-hash", "sha256:artifact-1",
        "--key-id", "ci", "--clean", "--risk", "0.05",
        "--min-effect", "0.01", "--lineage", str(tmp_path / "lineage.json"),
        "--json", "--out", str(receipt)])
    assert code == 0
    return receipt


def test_verify_accepts_an_untouched_receipt(tmp_path, monkeypatch, capsys):
    receipt = _release(tmp_path, monkeypatch)
    capsys.readouterr()

    code = cli.main([
        "release", "verify", "--receipt", str(receipt),
        "--lineage", str(tmp_path / "lineage.json"), "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == 1
    assert payload["ok"] is True
    assert payload["valid"] is True
    assert payload["artifact_id"] == "skill-1"
    assert payload["lineage_checked"] is True
    assert payload["decision_hash"]


def test_verify_reports_an_edited_manifest_as_invalid(
        tmp_path, monkeypatch, capsys):
    receipt = _release(tmp_path, monkeypatch)
    capsys.readouterr()
    payload = json.loads(receipt.read_text())
    payload["release"]["envelope"]["manifest"]["content_hash"] = "sha256:evil"
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    code = cli.main(["release", "verify", "--receipt", str(receipt), "--json"])

    assert code == 1
    checked = json.loads(capsys.readouterr().out)
    assert checked["valid"] is False
    assert "signature" in checked["reason"]


def test_verify_reports_a_wrong_key_as_invalid(tmp_path, monkeypatch, capsys):
    receipt = _release(tmp_path, monkeypatch)
    monkeypatch.setenv("AGENTDESCENT_RELEASE_KEY", "wrong-key")
    capsys.readouterr()

    code = cli.main(["release", "verify", "--receipt", str(receipt), "--json"])

    assert code == 1
    checked = json.loads(capsys.readouterr().out)
    assert checked["valid"] is False
    assert "signature" in checked["reason"]


def test_verify_reports_a_lineage_mismatch(tmp_path, monkeypatch, capsys):
    receipt = _release(tmp_path, monkeypatch)
    lineage_path = tmp_path / "lineage.json"
    lineage = json.loads(lineage_path.read_text())
    lineage["nodes"][0]["metadata"]["manifest_hash"] = "0" * 64
    lineage_path.write_text(json.dumps(lineage), encoding="utf-8")
    capsys.readouterr()

    code = cli.main([
        "release", "verify", "--receipt", str(receipt),
        "--lineage", str(lineage_path), "--json"])

    assert code == 1
    checked = json.loads(capsys.readouterr().out)
    assert checked["valid"] is False
    assert "manifest_hash" in checked["reason"]
    assert checked["lineage_checked"] is False


def test_verify_exit_two_for_a_missing_or_bad_receipt(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AGENTDESCENT_RELEASE_KEY", "secret")
    code = cli.main([
        "release", "verify", "--receipt", str(tmp_path / "missing.json"),
        "--json"])
    assert code == 2
    assert "cannot be read" in json.loads(capsys.readouterr().out)["error"]

    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    code = cli.main(["release", "verify", "--receipt", str(bad), "--json"])
    assert code == 2
    assert "not valid JSON" in json.loads(capsys.readouterr().out)["error"]


def test_verify_refuses_an_unsigned_receipt(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTDESCENT_RELEASE_KEY", "secret")
    receipt = _write(tmp_path / "rejected.json",
                     {"schema_version": 1, "ok": True, "released": False,
                      "release": None})
    try:
        verify_release(receipt)
    except ValueError as error:
        assert "no signed release" in str(error)
    else:
        raise AssertionError("an unsigned receipt must be an input error")


def test_mcp_release_verify_reads_the_receipt(tmp_path, monkeypatch):
    receipt = _release(tmp_path, monkeypatch)
    tools = Tools()
    ok = tools.release_verify(str(receipt))
    assert ok["ok"] is True and ok["valid"] is True

    monkeypatch.setenv("AGENTDESCENT_RELEASE_KEY", "wrong-key")
    bad = tools.release_verify(str(receipt))
    assert bad["ok"] is True and bad["valid"] is False


def test_verify_names_a_pinned_key_mismatch(tmp_path, monkeypatch, capsys):
    receipt = _release(tmp_path, monkeypatch)
    capsys.readouterr()
    code = cli.main(["release", "verify", "--receipt", str(receipt),
                     "--key-id", "prod", "--json"])
    assert code == 1
    checked = json.loads(capsys.readouterr().out)
    assert checked["valid"] is False
    assert "signed by 'ci'" in checked["reason"]
    assert "expected 'prod'" in checked["reason"]


def test_verify_requires_a_receipt_and_refuses_release_flags(capsys):
    code = cli.main(["release", "verify", "--json"])
    assert code == 2
    assert "needs --receipt" in json.loads(capsys.readouterr().out)["error"]

    code = cli.main(["release", "verify", "--receipt", "x.json",
                     "--artifact-id", "a", "--json"])
    assert code == 2
    assert "does not take --artifact-id" in json.loads(
        capsys.readouterr().out)["error"]


def test_mcp_release_verify_has_no_secret_argument():
    import inspect

    parameters = inspect.signature(Tools.release_verify).parameters
    assert "secret_env" not in parameters
    assert "secret_file" not in parameters
