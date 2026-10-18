"""``agentdescent --version`` and the install identity in ``doctor`` (#416).

A host starts its own ``agentdescent mcp`` -- often through the interpreter
``install`` pinned -- which can be a different install from the one in the user's
shell. Neither the CLI nor ``doctor`` could say which one answered.
"""

import json
import os
import subprocess
import sys

import pytest

import agentdescent
from agentdescent import cli
from agentdescent.mcp import Tools

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.parametrize("flag", ["--version", "-V"])
def test_version_flag_prints_the_version_and_exits_zero(flag, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main([flag])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"agentdescent {agentdescent.__version__}"


def test_version_flag_works_as_a_subprocess_without_a_verb():
    proc = subprocess.run([sys.executable, "-m", "agentdescent.cli", "--version"],
                          capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0
    assert proc.stdout.strip() == f"agentdescent {agentdescent.__version__}"


def test_a_missing_verb_is_still_an_error():
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code == 2


def test_doctor_report_names_the_version_and_the_interpreter():
    rep = cli.doctor_report()
    assert rep["version"] == agentdescent.__version__
    assert rep["executable"] == sys.executable


def test_doctor_json_and_text_carry_both_fields(capsys):
    assert cli.main(["--json", "doctor"]) in (0, 1)
    rep = json.loads(capsys.readouterr().out)
    assert rep["version"] == agentdescent.__version__ and rep["executable"] == sys.executable
    cli.main(["doctor"])
    out = capsys.readouterr().out
    assert f"version: {agentdescent.__version__}\n" in out
    assert f"executable: {sys.executable}\n" in out


def test_mcp_doctor_tool_returns_the_same_fields(tmp_path):
    rep = Tools(str(tmp_path / "runs")).doctor()
    assert rep["version"] == agentdescent.__version__
    assert rep["executable"] == sys.executable


def test_the_docs_tell_a_host_to_compare_the_two():
    from agentdescent.integrations import skill_text

    assert "`version`" in skill_text() and "`executable`" in skill_text()
    with open(os.path.join(ROOT, "docs", "testing-the-plugins.md"), encoding="utf-8") as fh:
        assert "`version` and `executable`" in fh.read()


def test_version_does_not_traceback_from_a_cwd_that_shadows_the_package(tmp_path):
    """A bare `agentdescent/` directory in the cwd makes the package a namespace
    package with no `__version__` (the #394 case); `cli` must still import and
    answer, so `mcp`'s one-line diagnosis is not preempted by an ImportError."""
    cwd = tmp_path / "host-cwd"
    (cwd / "agentdescent").mkdir(parents=True)
    env = {**os.environ, "PYTHONPATH": ""}
    proc = subprocess.run([sys.executable, "-m", "agentdescent.cli", "--version"],
                          cwd=cwd, capture_output=True, text=True, env=env)
    assert "Traceback" not in proc.stderr, proc.stderr
