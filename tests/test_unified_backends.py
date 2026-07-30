"""One contract for every backend.

Before this, the framework had two unrelated agent contracts: `Completion`
(prompt -> text) for API models, and `AgentBackend.answer(question, document,
skills)` for tool-using ones -- a signature with three *domain* concepts baked
into what should be the general interface. Now every backend, API or tool-using,
is a `Completion`; `WorkspaceAgent` adds the one extra capability an acting agent
needs (a directory), and `document_agent` is an explicit domain adapter on top.
"""

import os
import tempfile

import pytest

from agentdescent.agents import (
    AgentError,
    Completion,
    Usage,
    WorkspaceAgent,
    claude_code,
    cli_agent,
    codex,
    echo,
)
from agentdescent.backends import document_agent


def _echo_cli():
    """A stand-in CLI agent that echoes its prompt (prompt arrives on argv)."""
    return cli_agent(["sh", "-c", 'printf "%s" "$1"', "_"])


def test_cli_agent_is_a_completion():
    agent = _echo_cli()
    assert callable(agent)
    assert agent("hello") == "hello"


def test_cli_agent_accepts_the_prompt_on_stdin_too():
    agent = cli_agent(["sh", "-c", "cat"], via_stdin=True)
    assert agent("piped") == "piped"


def test_cli_agents_are_workspace_agents_and_plain_completions_are_not():
    assert isinstance(_echo_cli(), WorkspaceAgent)
    assert isinstance(claude_code(), WorkspaceAgent)
    assert isinstance(codex(), WorkspaceAgent)
    assert not isinstance(echo(), WorkspaceAgent)      # a plain API-style completion


def test_in_workspace_runs_in_that_directory():
    wk = tempfile.mkdtemp()
    agent = cli_agent(["sh", "-c", "pwd"], via_stdin=True)
    assert os.path.realpath(agent.in_workspace(wk)("x")) == os.path.realpath(wk)


def test_in_workspace_does_not_mutate_the_original():
    wk = tempfile.mkdtemp()
    agent = cli_agent(["sh", "-c", "pwd"], via_stdin=True)
    bound = agent.in_workspace(wk)
    assert bound is not agent
    assert agent.workspace is None


def test_missing_binary_reports_which_one():
    with pytest.raises(AgentError) as e:
        cli_agent(["definitely-not-installed-xyz"])("hi")
    assert "definitely-not-installed-xyz" in str(e.value)


def test_a_hanging_agent_times_out():
    with pytest.raises(AgentError) as e:
        cli_agent(["sh", "-c", "sleep 30"], via_stdin=True, timeout=0.5)("hi")
    assert "timeout" in str(e.value)


def test_failure_carries_the_agents_own_stderr():
    with pytest.raises(AgentError) as e:
        cli_agent(["sh", "-c", "echo the-real-reason >&2; exit 3"], via_stdin=True)("hi")
    assert "the-real-reason" in str(e.value)


def test_usage_counts_agent_calls_and_failures():
    u = Usage()
    cli_agent(["sh", "-c", "true"], via_stdin=True, usage=u)("x")
    with pytest.raises(AgentError):
        cli_agent(["sh", "-c", "exit 1"], via_stdin=True, usage=u)("x")
    assert u.calls == 2 and u.failures == 1


def test_document_agent_stages_the_file_for_a_workspace_agent():
    """A tool-using agent must get a real file it can grep, not an inline blob."""
    reader = cli_agent(["sh", "-c", "cat document.txt"], via_stdin=True)
    out = document_agent(reader).answer("q?", "UNIQUE-MARKER-42")
    assert "UNIQUE-MARKER-42" in out


def test_document_agent_inlines_the_document_for_a_plain_completion():
    seen = {}

    def capture(prompt):
        seen["prompt"] = prompt
        return "done"

    document_agent(capture).answer("q?", "UNIQUE-MARKER-42")
    assert "UNIQUE-MARKER-42" in seen["prompt"], "no workspace -> doc must be inline"
    assert "q?" in seen["prompt"]


def test_document_agent_truncates_a_huge_inline_document():
    seen = {}

    def capture(prompt):
        seen["n"] = len(prompt)
        return "done"

    document_agent(capture, inline_chars=1000).answer("q?", "x" * 500_000)
    assert seen["n"] < 5000, "an inline document must be bounded"


def test_document_agent_passes_skills_through():
    seen = {}

    def capture(prompt):
        seen["p"] = prompt
        return "done"

    document_agent(capture).answer("q?", "doc", skills="### skill: verify twice")
    assert "verify twice" in seen["p"]


def test_openhands_shares_the_same_contract():
    """Constructing it must not require the optional dependency (lazy import)."""
    from agentdescent.backends import openhands

    agent = openhands(model="openai/x")
    assert isinstance(agent, WorkspaceAgent)
    assert callable(agent.in_workspace("/tmp"))


def test_openhands_backend_is_now_just_a_composition():
    from agentdescent.backends import openhands_backend

    assert hasattr(openhands_backend(model="openai/x"), "answer")
