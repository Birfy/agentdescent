"""A worker CLI must not read the caller's stdin (#372).

`opencode run "<prompt>"` waits on an inherited stdin that never reaches EOF, so
under a pipe every call burned the full agent timeout. A prompt given as an
argument means stdin is nothing; `via_stdin=True` means stdin is the prompt.
"""
import os
import sys
import time

import pytest

from agentdescent.agents import AgentError, cli_agent
from agentdescent.runners import _sh

pytestmark = pytest.mark.skipif(os.name == "nt", reason="fd 0 swap is POSIX-only")

# Reads stdin to EOF before answering, like a CLI that waits on an inherited pipe.
_READS_STDIN = [sys.executable, "-c",
                "import sys; sys.stdin.read(); sys.stdout.write('ok')"]


@pytest.fixture
def open_pipe_stdin():
    """The test process's fd 0 is a pipe whose writer stays open: never EOF."""
    r, w = os.pipe()
    saved = os.dup(0)
    os.dup2(r, 0)
    try:
        yield
    finally:
        os.dup2(saved, 0)
        for fd in (r, w, saved):
            os.close(fd)


def test_cli_agent_does_not_inherit_an_open_stdin_pipe(open_pipe_stdin, tmp_path):
    t0 = time.time()
    out = cli_agent(_READS_STDIN, workspace=str(tmp_path), timeout=20)("prompt")
    assert out == "ok"
    assert time.time() - t0 < 15


def test_cli_agent_without_workspace_does_not_inherit_stdin_either(open_pipe_stdin):
    assert cli_agent(_READS_STDIN, timeout=20)("prompt") == "ok"


def test_via_stdin_still_delivers_the_prompt(open_pipe_stdin, tmp_path):
    echo = [sys.executable, "-c", "import sys; sys.stdout.write(sys.stdin.read().upper())"]
    out = cli_agent(echo, via_stdin=True, workspace=str(tmp_path), timeout=20)("hello")
    assert out == "HELLO"


def test_runner_sh_does_not_inherit_an_open_stdin_pipe(open_pipe_stdin, tmp_path):
    proc = _sh(_READS_STDIN, str(tmp_path), 20, dict(os.environ))
    assert proc.returncode == 0 and proc.stdout == "ok"
