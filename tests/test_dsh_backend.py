"""Offline tests for `agentdescent.backends.dsh` -- DeepSeek Harness as a Completion.

The SDK is an optional dependency that ships a Node runtime binary, so none of
this can run against the real thing here. A fake `deepseek_harness` module is
installed into `sys.modules` instead, which is enough to pin the four properties
that decide whether a rollout measures anything:

* a fresh session per call (contamination looks exactly like learning);
* `finish_reason == 'error'` raises rather than returning the partial text;
* `max-tokens` does *not* raise (dropping truncated samples biases the run);
* the runtime subprocess is reaped -- always for a workspace-bound agent, at
  interpreter exit for a reused one.

The fake mirrors the real dataclass surface (`RunResult`, `DeepSeekHarness.run`,
`close`) rather than the wire, so a signature change upstream shows up as an
obviously wrong fake instead of a passing test.
"""

import sys
import types

import pytest

from agentdescent.agents import AgentError, Usage, WorkspaceAgent
from agentdescent.backends import _LIVE_DSH, dsh, dsh_backend


class FakeRunResult:
    def __init__(self, session_id, final_response, finish_reason):
        self.session_id = session_id
        self.final_response = final_response
        self.finish_reason = finish_reason
        self.events = []
        self.notifications = []
        self.session_root = None


class FakeHarness:
    """Records what it was constructed with and every session it minted."""

    instances = []
    #: Seeds `self.raises` on every *future* instance. A workspace-bound agent
    #: closes its runtime after each call, so a test that wants the next call to
    #: fail cannot reach the object that will serve it.
    next_raises = None

    def __init__(self, **options):
        self.options = options
        self.sessions = []
        self.prompts = []
        self.closed = 0
        self.reply = "42"
        self.finish_reason = "completed"
        self.raises = FakeHarness.next_raises
        FakeHarness.instances.append(self)

    def run(self, prompt, *, session_id=None, on_notification=None):
        if self.raises is not None:
            raise self.raises
        self.prompts.append(prompt)
        # The real `DeepSeekHarness.run` mints `session-<uuid4>` when the caller
        # names no session; this stands in for that.
        sid = session_id or "session-%d" % len(self.sessions)
        self.sessions.append(sid)
        return FakeRunResult(sid, self.reply, self.finish_reason)

    def close(self):
        self.closed += 1


@pytest.fixture
def fake_sdk(monkeypatch):
    FakeHarness.instances = []
    FakeHarness.next_raises = None
    module = types.ModuleType("deepseek_harness")
    module.DeepSeekHarness = FakeHarness
    monkeypatch.setitem(sys.modules, "deepseek_harness", module)
    yield FakeHarness
    for agent in list(_LIVE_DSH):
        agent.close()


# ---------------------------------------------------------------------------
# The contract every other agent in the package also holds
# ---------------------------------------------------------------------------


def test_it_is_a_workspace_agent(fake_sdk):
    agent = dsh()
    assert isinstance(agent, WorkspaceAgent)
    assert callable(agent) and callable(agent.in_workspace)


def test_a_missing_sdk_says_what_to_install(monkeypatch):
    """The optional dependency must name itself, not raise a bare ImportError."""
    monkeypatch.setitem(sys.modules, "deepseek_harness", None)
    with pytest.raises(AgentError) as e:
        dsh()("hello")
    assert "deepseek-harness-sdk" in str(e.value)


def test_credentials_are_never_arguments(fake_sdk):
    """The runtime inherits them from the environment, as `openai_compatible` does."""
    import inspect

    params = inspect.signature(dsh).parameters
    assert not {"api_key", "base_url", "api_key_env"} & set(params)


# ---------------------------------------------------------------------------
# Session isolation -- the property the measurement rests on
# ---------------------------------------------------------------------------


def test_every_call_gets_its_own_session(fake_sdk):
    agent = dsh()
    for prompt in ("task one", "task two", "task three"):
        agent(prompt)
    harness = fake_sdk.instances[0]
    assert len(set(harness.sessions)) == 3, (
        "a reused session would carry the previous task's context into the next "
        "rollout, and the contamination would read as improvement")


def test_one_runtime_serves_every_call_when_unbound(fake_sdk):
    agent = dsh()
    agent("a")
    agent("b")
    assert len(fake_sdk.instances) == 1
    assert fake_sdk.instances[0].closed == 0


# ---------------------------------------------------------------------------
# Turn outcomes
# ---------------------------------------------------------------------------


def test_a_failed_turn_raises_instead_of_returning_partial_text(fake_sdk):
    agent = dsh()
    agent("warm up")                       # construct the harness
    harness = fake_sdk.instances[0]
    harness.finish_reason = "error"
    harness.reply = "I started to answer and then"
    with pytest.raises(AgentError) as e:
        agent("q")
    assert "error" in str(e.value)


def test_a_truncated_turn_is_a_low_score_not_an_exception(fake_sdk):
    agent = dsh()
    agent("warm up")
    harness = fake_sdk.instances[0]
    harness.finish_reason = "max-tokens"
    harness.reply = "the answer is 4"
    assert agent("q") == "the answer is 4"


def test_empty_output_is_a_string(fake_sdk):
    """`Completion` is prompt -> str; None four frames up is an AttributeError."""
    agent = dsh()
    agent("warm up")
    fake_sdk.instances[0].reply = None
    assert agent("q") == ""


def test_a_transport_failure_becomes_an_agent_error(fake_sdk):
    agent = dsh()
    agent("warm up")
    fake_sdk.instances[0].raises = RuntimeError("runtime exited")
    with pytest.raises(AgentError) as e:
        agent("q")
    assert "runtime exited" in str(e.value)


# ---------------------------------------------------------------------------
# Workspace binding and process lifetime
# ---------------------------------------------------------------------------


def test_in_workspace_binds_cwd_and_carries_the_configuration(fake_sdk):
    agent = dsh(model="deepseek-v4-pro", session_root="/tmp/sessions")
    bound = agent.in_workspace("/tmp/rollout-7")
    bound("q")
    options = fake_sdk.instances[0].options
    assert options["cwd"] == "/tmp/rollout-7"
    assert options["model"] == "deepseek-v4-pro"
    assert options["session_root"] == "/tmp/sessions"


def test_a_workspace_bound_agent_reaps_its_runtime_every_call(fake_sdk):
    """One node process per rollout would otherwise be one leak per rollout.

    The SDK resolves `cwd` once at construction, so a workspace-bound runtime
    cannot be reused for the next candidate directory anyway.
    """
    bound = dsh().in_workspace("/tmp/rollout-1")
    bound("a")
    bound("b")
    assert len(fake_sdk.instances) == 2
    assert [h.closed for h in fake_sdk.instances] == [1, 1]
    assert bound not in _LIVE_DSH


def test_a_failed_call_still_reaps_a_bound_runtime(fake_sdk):
    bound = dsh().in_workspace("/tmp/rollout-2")
    bound("warm up")
    # The first call already closed its runtime, so the failing one is the next
    # runtime to be built -- seed the class, not the instance.
    fake_sdk.next_raises = RuntimeError("boom")
    with pytest.raises(AgentError):
        bound("q")
    assert len(fake_sdk.instances) == 2
    assert all(h.closed == 1 for h in fake_sdk.instances)


def test_reuse_overrides_both_defaults(fake_sdk):
    bound = dsh(reuse=True).in_workspace("/tmp/rollout-3")
    bound("a")
    bound("b")
    assert len(fake_sdk.instances) == 1 and fake_sdk.instances[0].closed == 0

    unbound = dsh(reuse=False)
    unbound("a")
    unbound("b")
    assert len(fake_sdk.instances) == 3


def test_close_is_idempotent_and_deregisters(fake_sdk):
    agent = dsh()
    agent("a")
    assert agent in _LIVE_DSH
    agent.close()
    agent.close()
    assert agent not in _LIVE_DSH
    assert fake_sdk.instances[0].closed == 1


def test_it_works_as_a_context_manager(fake_sdk):
    with dsh() as agent:
        agent("a")
    assert fake_sdk.instances[0].closed == 1


# ---------------------------------------------------------------------------
# Metering and the document shape
# ---------------------------------------------------------------------------


def test_usage_counts_calls_and_failures(fake_sdk):
    usage = Usage()
    agent = dsh(usage=usage)
    agent("a")
    fake_sdk.instances[0].finish_reason = "error"
    with pytest.raises(AgentError):
        agent("b")
    assert usage.calls == 2 and usage.failures == 1


def test_usage_survives_workspace_rebinding(fake_sdk):
    """`in_workspace` mints a new object; a dropped meter loses the whole run."""
    usage = Usage()
    dsh(usage=usage).in_workspace("/tmp/w")("q")
    assert usage.calls == 1


def test_dsh_backend_stages_the_document_and_answers(fake_sdk, tmp_path):
    backend = dsh_backend()
    answer = backend.answer("how much?", "a document with rows\n")
    assert answer == "42"
    # `document_agent` gives a workspace agent a scratch directory with the
    # document written into it, so the harness was bound to one.
    assert fake_sdk.instances[0].options["cwd"] is not None
