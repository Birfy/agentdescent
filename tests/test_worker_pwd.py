"""A worker CLI must see the rollout workspace as ``$PWD``, not the caller's (#371).

``cwd=`` moves the child process but not ``$PWD``; OpenCode (Bun) resolves its
project directory from ``$PWD``, so an inherited one made every rollout run in the
directory ``agentdescent`` was launched from.
"""
import os
import sys

import pytest

from agentdescent.agents import cli_agent, worker_env
from agentdescent.runners import _child_env

_PRINT_PWD = [sys.executable, "-c", "import os,sys;sys.stdout.write(os.environ.get('PWD',''))"]


@pytest.fixture
def elsewhere(tmp_path, monkeypatch):
    """The caller was launched from a different directory, with PWD exported."""
    here = tmp_path / "launch"
    here.mkdir()
    monkeypatch.chdir(here)
    monkeypatch.setenv("PWD", str(here))
    return str(here)


@pytest.mark.parametrize("isolate", [True, False])
def test_worker_env_pwd_is_the_workspace(tmp_path, elsewhere, isolate):
    ws = str(tmp_path / "ws")
    assert worker_env(ws, isolate=isolate)["PWD"] == ws


def test_worker_env_extra_can_still_override_pwd(tmp_path, elsewhere):
    assert worker_env(str(tmp_path), {"PWD": "/x"})["PWD"] == "/x"


def test_worker_env_without_workspace_keeps_the_callers_pwd(elsewhere):
    assert worker_env(None)["PWD"] == elsewhere


@pytest.mark.parametrize("isolate", [True, False])
def test_cli_agent_child_sees_workspace_as_pwd(tmp_path, elsewhere, isolate):
    ws = tmp_path / "ws"
    ws.mkdir()
    out = cli_agent(_PRINT_PWD, workspace=str(ws), isolate=isolate, timeout=30)("ignored")
    assert out == str(ws)


def test_code_runner_child_env_sets_pwd(tmp_path, elsewhere):
    assert _child_env(str(tmp_path), None)["PWD"] == str(tmp_path)
