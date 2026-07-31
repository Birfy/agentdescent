"""Provider-agnostic inference -- connect any agent or LLM to the framework.

This is the general "talk to a model/agent" layer. It is deliberately kept
**out of** :mod:`agentdescent.evolution`: skill evolution is just *one* application
built on the framework, and how you reach a model has nothing to do with it.

The whole contract is one type:

    Completion = Callable[[str], str]      # prompt -> text

Anything that maps a prompt to text is a completion -- an LLM call, a
tool-using agent loop, a canned stub for tests. The adapters below build
completions for Claude, an arbitrary callable, or a deterministic echo, and
:func:`with_retries` wraps any of them with backoff. Higher layers
(the evolution engine's LLMAgent, or your own) turn a completion
into whatever task interface they need.
"""

from __future__ import annotations

import json
import os
import threading
import time
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Protocol, Sequence, runtime_checkable

Completion = Callable[[str], str]


@dataclass
class Usage:
    """What a run cost: calls, tokens, and wall-clock spent in the model.

    A ``Completion`` is ``prompt -> text``, so token counts would be discarded at
    the adapter boundary even though the providers return them. Pass one of these
    to :func:`claude` / :func:`openai_compatible` and they record the **real**
    counts from the API response; wrap anything else in :func:`metered` to at
    least count calls and time.

    Safe to share across worker threads.
    """

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    failures: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def record(self, *, prompt_tokens: int = 0, completion_tokens: int = 0,
               seconds: float = 0.0, failed: bool = False) -> None:
        with self._lock:
            self.calls += 1
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.seconds += seconds
            self.failures += 1 if failed else 0

    def estimated_cost(self, per_1m_prompt: float, per_1m_completion: float) -> float:
        """Cost at the given per-million-token prices (both provider-specific)."""
        return (self.prompt_tokens * per_1m_prompt
                + self.completion_tokens * per_1m_completion) / 1_000_000

    def summary(self) -> str:
        return (f"{self.calls} calls, {self.prompt_tokens:,} prompt + "
                f"{self.completion_tokens:,} completion tokens, "
                f"{self.seconds:.1f}s in the model"
                + (f", {self.failures} failed" if self.failures else ""))


def metered(completion: Completion, usage: Usage) -> Completion:
    """Count calls and model wall-clock for *any* completion.

    Token counts are unavailable here -- a plain ``Completion`` never exposes
    them -- so use the ``usage=`` argument of :func:`claude` /
    :func:`openai_compatible` when you need exact tokens."""
    def complete(prompt: str) -> str:
        t0 = time.time()
        try:
            out = completion(prompt)
        except Exception:
            usage.record(seconds=time.time() - t0, failed=True)
            raise
        usage.record(seconds=time.time() - t0)
        return out
    return complete


# ---------------------------------------------------------------------------
# Tool-using agents -- still just a Completion
# ---------------------------------------------------------------------------
#
# A coding agent (Claude Code, Codex, OpenHands, aider, ...) differs from an API
# model in that it *acts* -- runs commands, reads and edits files -- before it
# answers. But its call contract is the same one: text in, text out. Keeping it a
# ``Completion`` means every consumer (``LLMAgent``, ``evolve``, the examples)
# accepts all of them with no special-casing.


class AgentError(RuntimeError):
    """A tool-using agent failed; the message carries its stderr / exit status."""


@runtime_checkable
class WorkspaceAgent(Protocol):
    """A :data:`Completion` that can additionally be bound to a directory.

    ``Completion`` deliberately stays ``prompt -> text`` for everything. But an
    agent that *acts* often needs a place to act in, and a caller that stages
    files (a document to grep, a repo to patch) needs to say where. This is that
    one extra capability, kept optional so plain API models never implement it::

        agent = claude_code()
        answer = agent.in_workspace("/tmp/task-17")("summarise report.txt")

    Consumers should feature-detect it (``isinstance(x, WorkspaceAgent)``) and
    fall back to putting the material in the prompt.
    """

    def __call__(self, prompt: str) -> str: ...

    def in_workspace(self, path: str) -> Completion: ...


class _CliAgent:
    """A command-line agent: a Completion that can be rebound to a workspace."""

    def __init__(self, command, *, workspace=None, via_stdin=False,
                 timeout=600.0, env=None, usage=None) -> None:
        if not command:
            raise ValueError("cli_agent needs a non-empty command")
        self.command, self.workspace, self.via_stdin = list(command), workspace, via_stdin
        self.timeout, self.env, self.usage = timeout, env, usage

    def in_workspace(self, path: str) -> "Completion":
        return _CliAgent(self.command, workspace=path, via_stdin=self.via_stdin,
                         timeout=self.timeout, env=self.env, usage=self.usage)

    def __call__(self, prompt: str) -> str:
        argv = list(self.command) if self.via_stdin else [*self.command, prompt]
        t0 = time.time()
        try:
            proc = subprocess.run(
                argv, input=prompt if self.via_stdin else None,
                capture_output=True, text=True, timeout=self.timeout,
                cwd=self.workspace,
                env={**os.environ, **self.env} if self.env else None,
            )
        except FileNotFoundError as e:
            raise AgentError(
                f"{self.command[0]!r} is not installed or not on PATH") from e
        except subprocess.TimeoutExpired as e:
            if self.usage is not None:
                self.usage.record(seconds=time.time() - t0, failed=True)
            raise AgentError(f"{self.command[0]} exceeded timeout={self.timeout}s") from e
        if self.usage is not None:
            self.usage.record(seconds=time.time() - t0, failed=proc.returncode != 0)
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()[:400] or "no output"
            raise AgentError(f"{self.command[0]} exited {proc.returncode}: {detail}")
        return proc.stdout.strip()


def cli_agent(command: Sequence[str], *, workspace: Optional[str] = None,
              via_stdin: bool = False, timeout: float = 600.0,
              env: Optional[Dict[str, str]] = None,
              usage: Optional[Usage] = None) -> "WorkspaceAgent":
    """Run any **command-line** coding agent as a :data:`Completion`.

    ``command`` is the argv prefix; the prompt is appended as the final argument,
    or written to stdin when ``via_stdin`` is set. Whatever the agent writes to
    stdout is the answer.

    ::

        cli_agent(["claude", "-p"])                        # Claude Code, print mode
        cli_agent(["codex", "exec"])                       # Codex CLI
        cli_agent(["claude", "-p"]).in_workspace("/tmp/w") # act inside a directory

    ``timeout`` is not optional in spirit: an agent that hangs would otherwise
    stall the round it belongs to (see ``evolve(round_timeout=)``). Failures raise
    :class:`AgentError` carrying the agent's own stderr rather than a bare exit
    code.
    """
    return _CliAgent(command, workspace=workspace, via_stdin=via_stdin,
                     timeout=timeout, env=env, usage=usage)


def claude_code(*, workspace: Optional[str] = None, extra_args: Sequence[str] = (),
                **kwargs) -> Completion:
    """Claude Code in non-interactive print mode, as a :data:`Completion`."""
    return cli_agent(["claude", "-p", *extra_args], workspace=workspace, **kwargs)


def codex(*, workspace: Optional[str] = None, extra_args: Sequence[str] = (),
          **kwargs) -> Completion:
    """OpenAI Codex CLI in non-interactive exec mode, as a :data:`Completion`."""
    return cli_agent(["codex", "exec", *extra_args], workspace=workspace, **kwargs)


def from_callable(fn: Completion) -> Completion:
    """Identity adapter -- documents that any ``prompt -> text`` callable works."""
    return fn


def echo(transform: Optional[Callable[[str], str]] = None) -> Completion:
    """A deterministic, no-network completion for tests and dry runs.

    Returns the prompt unchanged, or ``transform(prompt)`` if given."""
    def complete(prompt: str) -> str:
        return transform(prompt) if transform else prompt
    return complete


def with_retries(completion: Completion, attempts: int = 3,
                 backoff: float = 0.5, sleep: Callable[[float], None] = time.sleep) -> Completion:
    """Wrap a completion with exponential-backoff retries on any exception."""
    def complete(prompt: str) -> str:
        last: Optional[Exception] = None
        for i in range(attempts):
            try:
                return completion(prompt)
            except Exception as e:  # noqa: BLE001 - provider-agnostic retry
                last = e
                if i < attempts - 1:
                    sleep(backoff * (2 ** i))
        raise last  # type: ignore[misc]
    return complete


def claude(model: str = "claude-opus-4-8", max_tokens: int = 4096,
           client: Optional[object] = None, usage: Optional[Usage] = None,
           retries: int = 3, **create_kwargs) -> Completion:
    """A Claude-backed completion (requires ``pip install anthropic`` + creds).

    Pass ``client`` to reuse an existing ``anthropic.Anthropic`` instance;
    otherwise a default one is constructed lazily (resolving credentials from
    the environment / an ``ant auth login`` profile). Use a cheaper ``model``
    (e.g. ``"claude-haiku-4-5"``) for call-heavy loops. Pass ``usage=Usage()`` to
    accumulate the exact token counts the API reports.

    ``max_tokens`` defaults high on purpose: a reasoning model spends the budget
    on internal reasoning first, so too small a cap returns **empty visible
    content** rather than a short answer. Measured on ``deepseek-v4-flash``, a
    1024 cap returned nothing at all for 4 of 8 reflection prompts. You are billed
    for tokens generated, not for the cap, so a generous limit costs nothing."""
    _client = client

    def complete(prompt: str) -> str:
        nonlocal _client
        if _client is None:
            from anthropic import Anthropic  # lazy, optional dependency
            _client = Anthropic()
        t0 = time.time()
        try:
            msg = _client.messages.create(
                model=model, max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}], **create_kwargs,
            )
        except Exception:
            if usage is not None:
                usage.record(seconds=time.time() - t0, failed=True)
            raise
        if usage is not None:
            u = getattr(msg, "usage", None)
            usage.record(prompt_tokens=getattr(u, "input_tokens", 0) or 0,
                         completion_tokens=getattr(u, "output_tokens", 0) or 0,
                         seconds=time.time() - t0)
        return "".join(b.text for b in msg.content if b.type == "text")

    # Retried by default: a transient socket error is ordinary, and it used to end
    # whatever was running. Measured on a real run, one `RemoteDisconnected`
    # during an example's final held-out evaluation -- a plain `completion(...)`
    # call outside anything the engine protects -- discarded the entire run.
    # `retries=0` opts out.
    return with_retries(complete, attempts=retries) if retries > 1 else complete


def openai_compatible(model: str, *, base_url_env: str = "OPENAI_BASE_URL",
                      api_key_env: str = "OPENAI_API_KEY",
                      default_base_url: str = "https://api.openai.com/v1",
                      max_tokens: int = 4096, timeout: float = 120.0,
                      usage: Optional[Usage] = None, retries: int = 3,
                      **create_kwargs) -> Completion:
    """A completion for any OpenAI-compatible chat endpoint (GLM/Zhipu, proxies,
    local servers, OpenAI itself).

    The base URL and API key are read from the environment at call time -- they
    never pass through code or arguments. Point it at GLM, for example, by
    setting ``OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4`` and
    ``OPENAI_API_KEY=<your key>`` in your shell, then use ``model="glm-4.6"``.

    ``max_tokens`` defaults high on purpose -- see :func:`claude`: a reasoning
    model starved of budget returns empty content, and at 1024 that happened for
    half of one measured batch of reflection prompts.

    Extra keyword arguments go into the request body, so ``temperature=0`` and any
    provider-specific field work the same way they do on :func:`claude`."""
    def complete(prompt: str) -> str:
        base = os.environ.get(base_url_env, default_base_url).rstrip("/")
        key = os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(f"set {api_key_env} (and {base_url_env}) in your environment")
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            **create_kwargs,
        }).encode()
        req = urllib.request.Request(
            f"{base}/chat/completions", data=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.load(resp)
        except urllib.error.HTTPError as e:
            # The body carries the only useful part -- "rate limit: retry in 12s",
            # "context length exceeded", "insufficient quota" -- and it lives on
            # e.read(), so re-raising bare collapses every 4xx to "HTTP Error 429:
            # Too Many Requests". `_git` and `_CliAgent` both surface the
            # underlying detail; this was the one provider path that did not.
            if usage is not None:
                usage.record(seconds=time.time() - t0, failed=True)
            try:
                detail = e.read().decode("utf-8", "replace").strip()[:400]
            except Exception:  # noqa: BLE001 - the body is best-effort
                detail = ""
            raise RuntimeError(
                f"{base} returned HTTP {e.code} for model {model!r}"
                + (f": {detail}" if detail else "")) from e
        except Exception:
            if usage is not None:
                usage.record(seconds=time.time() - t0, failed=True)
            raise
        if usage is not None:
            u = data.get("usage") or {}
            usage.record(prompt_tokens=u.get("prompt_tokens", 0) or 0,
                         completion_tokens=u.get("completion_tokens", 0) or 0,
                         seconds=time.time() - t0)
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            # Some proxies answer HTTP 200 with {"error": ...}; without this the
            # caller gets a bare KeyError naming nothing.
            raise RuntimeError(
                f"{base} returned a response with no choices for model {model!r}: "
                f"{json.dumps(data)[:300]}") from None
        # `content` is JSON null, not "", when a reasoning model spends its whole
        # budget on `reasoning_content` -- DeepSeek's reasoner and GLM's thinking
        # modes both do it. Returning None breaks the one contract in the package
        # (`Completion` is prompt -> str) and surfaces as
        # "'NoneType' object has no attribute 'strip'" from inside LLMAgent, which
        # the engine then retries as a *backend transient*. Normalising to "" lets
        # the empty-completion warning fire and say the true cause.
        return message.get("content") or ""

    return with_retries(complete, attempts=retries) if retries > 1 else complete
