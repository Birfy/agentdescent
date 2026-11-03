"""The module audit is read-only, and its categories match real import edges."""

import json
from pathlib import Path

from tools import module_audit


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _package(tmp_path):
    root = tmp_path / "pkg"
    _write(root / "__init__.py",
           'from .exported import A\n__all__ = ["A"]\n')
    _write(root / "cli.py",
           "from . import core\n"
           "from .subpkg.used import U\n"
           "SPEC = \"pkg.dynamic:build\"\n"
           "sub = object()\n"
           "sub.add_parser(\"run\")\n")
    _write(root / "mcp.py",
           "from typing import Any, Dict\n"
           "TOOL_DESCRIPTIONS: Dict[str, Any] = {\"a\": 1, \"b\": 2}\n")
    _write(root / "core.py", "VALUE = 1\n")
    _write(root / "exported.py", "A = 1\n")
    _write(root / "orphan.py", "def unused():\n    return 1\n")
    _write(root / "dynamic.py", "D = 1\n")
    _write(root / "subpkg" / "__init__.py", "")
    _write(root / "subpkg" / "used.py", "U = 1\n")
    _write(root / "subpkg" / "orphan.py", "def unused():\n    return 1\n")
    return root


def test_categories_follow_import_edges(tmp_path):
    report = module_audit.analyze(_package(tmp_path), package_name="pkg")
    assert report["modules"] == 9
    assert report["top_level_exports"] == 1
    assert report["cli_commands"] == 1
    assert report["mcp_tools"] == 2
    assert sum(row["modules"] for row in report["categories"].values()) == 9

    isolated = {row["module"] for row in report["isolated"]}
    assert isolated == {"orphan", "subpkg.orphan"}
    exported = {row["module"] for row in report["exported_only"]}
    assert "exported" in exported
    reachable = {row["module"] for row in report["isolated"]}
    assert "core" not in reachable
    assert "subpkg.used" not in reachable
    assert "subpkg" not in reachable
    assert "dynamic" not in reachable

    subpackages = {row["name"]: row for row in report["subpackages"]}
    assert subpackages["subpkg"]["modules"] == 3


def test_json_output_and_fail_on_isolated(tmp_path, capsys):
    root = _package(tmp_path)
    assert module_audit.main([
        "--root", str(root), "--package", "pkg", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == 1
    assert payload["package"] == "pkg"
    assert len(payload["isolated"]) == 2

    assert module_audit.main([
        "--root", str(root), "--package", "pkg", "--json",
        "--fail-on-isolated"]) == 1


def test_real_package_is_measurable():
    repo = Path(__file__).resolve().parents[1]
    report = module_audit.analyze(repo / "agentdescent")
    assert report["modules"] > 100
    assert report["top_level_exports"] > 0
    assert report["cli_commands"] > 0
    assert report["mcp_tools"] > 0
    assert sum(row["modules"] for row in report["categories"].values()) == (
        report["modules"])
    # The largest category is the product path, not the orphan set.
    assert report["categories"]["cli/mcp"]["modules"] > (
        report["categories"]["isolated"]["modules"])


def test_missing_root_is_an_error(tmp_path, capsys):
    assert module_audit.main(["--root", str(tmp_path / "missing")]) == 2
    assert "not a directory" in capsys.readouterr().err
