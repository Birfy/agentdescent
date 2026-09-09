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
from typing import (Any, Callable, Dict, List, Mapping, Optional, Protocol,
                    Sequence, Tuple, runtime_checkable)

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
    #: Model seconds spent inside calls that ultimately **failed** -- retry
    #: waits, hung connections, timeouts. Kept apart from `seconds` because the
    #: two answer different questions: `seconds` is what the run cost, and
    #: `seconds - failure_seconds` is what the run cost *net of endpoint
    #: weather* -- the number a wall-clock comparison between two arms should
    #: quote when one arm was unlucky. Measured need: a serial arm hit a
    #: network outage mid-sweep and its 43-minute wall carried ~17 minutes of
    #: stall that had nothing to do with the architecture under test.
    failure_seconds: float = 0.0
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
            if failed:
                self.failures += 1
                self.failure_seconds += seconds

    def estimated_cost(self, per_1m_prompt: float, per_1m_completion: float) -> float:
        """Cost at the given per-million-token prices (both provider-specific)."""
        return (self.prompt_tokens * per_1m_prompt
                + self.completion_tokens * per_1m_completion) / 1_000_000

    def summary(self) -> str:
        return (f"{self.calls} calls, {self.prompt_tokens:,} prompt + "
                f"{self.completion_tokens:,} completion tokens, "
                f"{self.seconds:.1f}s in the model"
                + (f", {self.failures} failed ({self.failure_seconds:.1f}s lost)"
                   if self.failures else ""))


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


#: Environment variables a host agent sets in its own session and its children
#: inherit. A worker that sees them believes it is *inside* that session:
#: Claude Code refuses to start, DSH loads the parent's profile, and a worker
#: whose host runs this package's MCP server would call it recursively.
SESSION_MARKERS: Tuple[str, ...] = (
    "CLAUDECODE", "CLAUDE_CODE_", "CLAUDE_CONFIG_DIR", "CODEX_", "DSH_",
    "OPENAI_AGENT_", "MCP_",
    # OpenCode reads three of these, and redirecting only the directory is not
    # isolation: `OPENCODE_CONFIG` (a file) and `OPENCODE_CONFIG_CONTENT`
    # (inline JSON) both win over `OPENCODE_CONFIG_DIR` -- measured, a config
    # named by `OPENCODE_CONFIG` still supplied its MCP servers with the dir
    # pointed at an empty directory. They are dropped by name rather than by an
    # "OPENCODE_" prefix on purpose: `OPENCODE_API_KEY` is a provider
    # credential, and a worker needs its keys.
    "OPENCODE_CONFIG_DIR", "OPENCODE_CONFIG", "OPENCODE_CONFIG_CONTENT",
    # Dropped so the redirect below can take: `setdefault` cannot override a
    # value the parent already exported, and this one usually is exported.
    # It is broader than the others -- every XDG-respecting tool the worker
    # runs sees the workspace copy -- which is the point of an isolated worker.
    "XDG_CONFIG_HOME",
)

#: Set for every worker so a tool the worker reaches (this package's own MCP
#: server, when the plugin that hosts it is being evolved) can tell it is inside
#: a run and refuse to start another. See ``agentdescent.mcp``.
NESTED_MARKER = "AGENTDESCENT_NESTED"


#: Each host's config-directory variable, and where an isolated worker's copy
#: goes inside the rollout workspace. One mapping because there are two things
#: to do with it -- set it, and *create* it -- and they were separate lists that
#: drifted: OpenCode was added to the first and not the second, and a missing
#: `OPENCODE_CONFIG_DIR` does not fail, it silently falls back to the user's
#: real config, so the isolation read as working and was not.
WORKER_CONFIG_DIRS: Dict[str, str] = {
    "CLAUDE_CONFIG_DIR": "claude",
    "CODEX_HOME": "codex",
    "DSH_HOME": "dsh",
    # OpenCode needs both, and the second is the one that works. Measured
    # against opencode 1.18: `OPENCODE_CONFIG_DIR` supplies a config only when
    # the user has none -- with a real `~/.config/opencode/opencode.jsonc`
    # present, a worker pointed at another directory still saw the user's MCP
    # servers, config file or no config file in the redirected one. OpenCode
    # resolves its config under XDG, so `XDG_CONFIG_HOME` is what actually
    # moves it: with that redirected the same worker sees "No MCP servers
    # configured". `OPENCODE_CONFIG_DIR` stays for the case where XDG is
    # honoured differently by a future version.
    "OPENCODE_CONFIG_DIR": "opencode",
    "XDG_CONFIG_HOME": "xdg",
}


def worker_env(workspace: Optional[str], extra: Optional[Mapping[str, str]] = None,
               *, isolate: bool = True) -> Dict[str, str]:
    """The environment a worker agent CLI runs with.

    Starts from the caller's environment (a worker needs its provider keys and
    PATH), drops every :data:`SESSION_MARKERS` variable, marks the process as
    nested, and -- when there is a workspace -- points each host's config
    directory *inside* it (``CLAUDE_CONFIG_DIR``, ``CODEX_HOME``, ``DSH_HOME``,
    ``OPENCODE_CONFIG_DIR``),
    so the worker starts clean and cannot read the user's real plugins, memory or
    MCP servers. ``extra`` wins over all of it. ``isolate=False`` keeps only the
    nested marker, for callers who want the worker to see the user's setup.
    """
    env = {k: v for k, v in os.environ.items()
           if not (isolate and any(k == m.rstrip("_") or k.startswith(m)
                                   for m in SESSION_MARKERS))}
    env[NESTED_MARKER] = "1"
    if isolate and workspace:
        home = os.path.join(workspace, ".agentdescent-worker")
        for var, leaf in WORKER_CONFIG_DIRS.items():
            env.setdefault(var, os.path.join(home, leaf))
    if extra:
        env.update(extra)
    return env


class _CliAgent:
    """A command-line agent: a Completion that can be rebound to a workspace."""

    def __init__(self, command, *, workspace=None, via_stdin=False,
                 timeout=600.0, env=None, usage=None, isolate=True) -> None:
        if not command:
            raise ValueError("cli_agent needs a non-empty command")
        self.command, self.workspace, self.via_stdin = list(command), workspace, via_stdin
        self.timeout, self.env, self.usage, self.isolate = timeout, env, usage, isolate

    def in_workspace(self, path: str) -> "Completion":
        return _CliAgent(self.command, workspace=path, via_stdin=self.via_stdin,
                         timeout=self.timeout, env=self.env, usage=self.usage,
                         isolate=self.isolate)

    def __call__(self, prompt: str) -> str:
        argv = list(self.command) if self.via_stdin else [*self.command, prompt]
        t0 = time.time()
        env = worker_env(self.workspace, self.env, isolate=self.isolate)
        if self.workspace:
            # Created, not merely pointed at. `codex` refuses to start when
            # CODEX_HOME does not exist ("Error finding codex home"), and
            # `opencode` does something worse -- it falls back to the user's
            # real config, so isolation silently does not happen.
            for key in WORKER_CONFIG_DIRS:
                if env.get(key, "").startswith(self.workspace):
                    os.makedirs(env[key], exist_ok=True)
        try:
            proc = subprocess.run(
                argv, input=prompt if self.via_stdin else None,
                capture_output=True, text=True, timeout=self.timeout,
                cwd=self.workspace, env=env,
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
              usage: Optional[Usage] = None, isolate: bool = True) -> "WorkspaceAgent":
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

    The child runs with :func:`worker_env`: the host session's markers dropped,
    ``AGENTDESCENT_NESTED=1`` set, and each host's config directory pointed inside
    the workspace. ``isolate=False`` keeps the caller's environment as it is
    (bar the nested marker) for a worker that should see the user's own setup.
    """
    return _CliAgent(command, workspace=workspace, via_stdin=via_stdin,
                     timeout=timeout, env=env, usage=usage, isolate=isolate)


def claude_code(*, workspace: Optional[str] = None, extra_args: Sequence[str] = (),
                **kwargs) -> Completion:
    """Claude Code in non-interactive print mode, as a :data:`Completion`."""
    return cli_agent(["claude", "-p", *extra_args], workspace=workspace, **kwargs)


def codex(*, workspace: Optional[str] = None, extra_args: Sequence[str] = (),
          **kwargs) -> Completion:
    """OpenAI Codex CLI in non-interactive exec mode, as a :data:`Completion`."""
    return cli_agent(["codex", "exec", *extra_args], workspace=workspace, **kwargs)


def dsh(*, workspace: Optional[str] = None, extra_args: Sequence[str] = (),
        **kwargs) -> Completion:
    """DeepSeek Harness (``dsh``) headless profile, as a :data:`Completion`.

    ``dsh --profile headless "<task>"`` runs one persisted session and prints the
    last assistant message to stdout, which is exactly the shape
    :func:`cli_agent` wants. Pass ``extra_args`` for ``--patch`` overlays or a
    different profile.
    """
    return cli_agent(["dsh", "--profile", "headless", *extra_args],
                     workspace=workspace, **kwargs)


def opencode(*, workspace: Optional[str] = None, extra_args: Sequence[str] = (),
             **kwargs) -> Completion:
    """OpenCode's non-interactive ``run`` mode, as a :data:`Completion`.

    ``opencode run "<task>"`` answers one message and exits, using the working
    directory as the project -- which is what :func:`cli_agent`'s ``workspace``
    binding gives it. Verified against opencode 1.18.
    """
    return cli_agent(["opencode", "run", *extra_args], workspace=workspace, **kwargs)


def from_callable(fn: Completion) -> Completion:
    """Identity adapter -- documents that any ``prompt -> text`` callable works."""
    return fn


def echo(transform: Optional[Callable[[str], str]] = None) -> Completion:
    """A deterministic, no-network completion for tests and dry runs.

    Returns the prompt unchanged, or ``transform(prompt)`` if given."""
    def complete(prompt: str) -> str:
        return transform(prompt) if transform else prompt
    return complete


class RateLimited(RuntimeError):
    """The provider said "too many requests", and for how long if it bothered to.

    Worth a type of its own because it wants a different retry from every other
    failure: a transport error is worth retrying in half a second, and a rate
    limit retried in half a second is three attempts spent inside two seconds
    against a limiter measured in tens of them.
    """

    def __init__(self, message: str, retry_after: Optional[float] = None) -> None:
        super().__init__(message)
        #: Seconds the provider asked for, from `Retry-After`, when it sent one.
        self.retry_after = retry_after


def with_retries(completion: Completion, attempts: int = 3,
                 backoff: float = 0.5, sleep: Callable[[float], None] = time.sleep,
                 rate_limit_backoff: float = 5.0,
                 max_sleep: float = 60.0) -> Completion:
    """Wrap a completion with exponential-backoff retries on any exception.

    Rate limits back off on their own, much longer, schedule. Measured against a
    throttling endpoint: the generic 0.5s/1.0s pair spent all three attempts
    inside two seconds, every one of them refused, and the engine -- which
    retires a worker after three consecutive failures -- lost every worker in
    about a minute and ended the run with an empty tree. A `Retry-After` header
    wins over both schedules when the provider sends one, and everything is
    capped at ``max_sleep`` so a hostile or mistaken header cannot park a worker
    for an hour.
    """
    def complete(prompt: str) -> str:
        last: Optional[Exception] = None
        for i in range(attempts):
            try:
                return completion(prompt)
            except Exception as e:  # noqa: BLE001 - provider-agnostic retry
                last = e
                if i < attempts - 1:
                    delay = backoff * (2 ** i)
                    if isinstance(e, RateLimited):
                        delay = max(delay, rate_limit_backoff * (2 ** i),
                                    e.retry_after or 0.0)
                    sleep(min(delay, max_sleep))
        raise last  # type: ignore[misc]
    return complete


def claude(model: str = "claude-opus-4-8", max_tokens: int = 4096,
           client: Optional[object] = None, usage: Optional[Usage] = None,
           retries: int = 3, timeout: float = 120.0, **create_kwargs) -> Completion:
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
    for tokens generated, not for the cap, so a generous limit costs nothing.

    ``timeout`` bounds one request, and it is not optional in spirit. Every other
    blocking boundary in this package has one -- ``_git`` at 120s, ``_CliAgent``
    at 600s, ``runners._sh`` per call, :func:`openai_compatible` at 120s -- and
    this was the exception. Without it the SDK's own 600s default applies, the
    SDK retries it internally, and :func:`with_retries` retries *that*, so one
    logical call against a stalled endpoint can block for well over half an hour
    with nothing in the log.

    Measured: a GEPA run against a hosted endpoint sat for **51 minutes on a
    single round**, 1.07s of CPU across the whole time and one ESTABLISHED socket
    -- a run that reports nothing and cannot be told apart from a slow one.
    Raise it for an agentic backend that legitimately takes longer; the
    equivalent knob on :func:`openai_compatible` has always been here."""
    _client = client

    def complete(prompt: str) -> str:
        nonlocal _client
        if _client is None:
            from anthropic import Anthropic  # lazy, optional dependency
            # max_retries=0: retrying is this function's job (`with_retries`
            # below). Leaving the SDK's default 2 in place multiplies the two
            # layers -- attempts x (1 + max_retries) x timeout -- and one
            # logical call against a stalled endpoint blocked ~45 minutes.
            _client = Anthropic(max_retries=0)
        t0 = time.time()
        try:
            msg = _client.messages.create(
                model=model, max_tokens=max_tokens, timeout=timeout,
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


def _read_sse(response: Any) -> Dict[str, Any]:
    """Reassemble an OpenAI-shaped SSE stream into the response it stands for.

    Returns the same shape the non-streaming path parses -- ``choices[0].message``
    plus ``usage`` -- so nothing downstream has to know which transport was used.

    Deliberately forgiving about the stream and strict about nothing: a chunk
    that will not parse is skipped rather than raised on, because one malformed
    frame in a hundred is a provider quirk and not a reason to throw away a
    completed answer. `reasoning_content` deltas are read and dropped: they are
    what keeps the connection from going idle, and they are not the reply.
    """
    text: List[str] = []
    usage_payload: Dict[str, Any] = {}
    finish: Optional[str] = None
    for raw in response:
        line = raw.decode("utf-8", "replace").strip()
        if not line or not line.startswith("data:"):
            continue
        chunk_text = line[len("data:"):].strip()
        if chunk_text == "[DONE]":
            break
        try:
            chunk = json.loads(chunk_text)
        except ValueError:
            continue
        if isinstance(chunk.get("usage"), dict):
            usage_payload = chunk["usage"]
        for choice in chunk.get("choices") or ():
            delta = choice.get("delta") or {}
            piece = delta.get("content")
            if piece:
                text.append(piece)
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    return {
        "choices": [{"message": {"role": "assistant", "content": "".join(text)},
                     "finish_reason": finish}],
        "usage": usage_payload,
    }


def openai_compatible(model: str, *, base_url_env: str = "OPENAI_BASE_URL",
                      api_key_env: str = "OPENAI_API_KEY",
                      default_base_url: str = "https://api.openai.com/v1",
                      max_tokens: int = 4096, timeout: float = 120.0,
                      usage: Optional[Usage] = None, retries: int = 3,
                      stream: bool = False,
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

    ``stream=True`` sends the request as SSE and reassembles the text, which is
    a **reliability** knob rather than a latency one: a non-streaming request to
    a reasoning model sends no bytes at all while the model thinks, and any
    gateway with an idle timeout closes the connection underneath it. Measured
    against one such endpoint: a single request finished in 205 s, but four
    concurrent ones -- queued behind each other, so each took longer -- were all
    cut at 301 s with `RemoteDisconnected`, three of them within 0.2 s of each
    other. The same prompts streamed ran past ten minutes with bytes arriving
    continuously, because the reasoning deltas themselves keep the connection
    warm. Usage still comes back: `stream_options.include_usage` puts it in the
    final chunk, and an endpoint that ignores that field simply reports zero
    tokens rather than failing.

    Extra keyword arguments go into the request body, so ``temperature=0`` and any
    provider-specific field work the same way they do on :func:`claude`."""
    def complete(prompt: str) -> str:
        base = os.environ.get(base_url_env, default_base_url).rstrip("/")
        key = os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(f"set {api_key_env} (and {base_url_env}) in your environment")
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            **create_kwargs,
        }
        if stream:
            payload["stream"] = True
            payload.setdefault("stream_options", {"include_usage": True})
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{base}/chat/completions", data=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = _read_sse(resp) if stream else json.load(resp)
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
            message = (f"{base} returned HTTP {e.code} for model {model!r}"
                       + (f": {detail}" if detail else ""))
            if e.code in (429, 503):
                # Told apart from every other 4xx so `with_retries` can wait the
                # seconds a limiter wants instead of the half-second a transport
                # blip wants. `Retry-After` is seconds or an HTTP date; only the
                # numeric form is honoured, and a missing or unreadable one
                # falls back to the rate-limit schedule.
                header = ""
                try:
                    header = (e.headers.get("Retry-After") or "").strip()
                except Exception:  # noqa: BLE001 - headers are best-effort
                    header = ""
                try:
                    retry_after = float(header) if header else None
                except ValueError:
                    retry_after = None
                raise RateLimited(message, retry_after) from e
            raise RuntimeError(message) from e
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


def anthropic_compatible(model: str, *, base_url_env: str = "ANTHROPIC_BASE_URL",
                         api_key_env: str = "ANTHROPIC_API_KEY",
                         default_base_url: str = "https://api.anthropic.com",
                         version: str = "2023-06-01",
                         max_tokens: int = 4096, timeout: float = 120.0,
                         usage: Optional[Usage] = None, retries: int = 3,
                         **create_kwargs) -> Completion:
    """A completion for any **Anthropic-format** endpoint, with no SDK dependency.

    :func:`claude` speaks the same protocol through ``pip install anthropic``.
    This is the twin of :func:`openai_compatible` on the other wire format, and it
    exists for the same reason that one does: a gateway. Anthropic-format
    endpoints now serve models that are not Claude -- vendor gateways, cloud
    marketplaces, local servers -- and reaching them through the SDK means taking
    an optional dependency, and a `base_url` override on a client whose defaults
    (retries, timeouts) then have to be undone. `urllib` is already imported here.

    The base URL and API key are read from the environment **at call time**, so
    neither passes through code or arguments -- point it at a gateway with
    ``ANTHROPIC_BASE_URL=https://host/anthropic`` and ``ANTHROPIC_API_KEY=<key>``
    and pass that gateway's own ``model`` id.

    ``max_tokens`` defaults high for the reason :func:`claude` gives: a reasoning
    model spends its budget on internal reasoning first, and too small a cap
    returns **empty visible content** rather than a short answer.

    Only ``text`` blocks are returned. A reasoning model answers with
    ``thinking`` blocks first and the visible answer after, and concatenating all
    of them would put the reasoning into the artifact's output -- where a scorer
    would grade it, a judge would read it, and a diff might commit it.
    """
    def complete(prompt: str) -> str:
        base = os.environ.get(base_url_env, default_base_url).rstrip("/")
        key = os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(f"set {api_key_env} (and {base_url_env}) in your environment")
        payload: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            **create_kwargs,
        }
        req = urllib.request.Request(
            f"{base}/v1/messages", data=json.dumps(payload).encode(),
            headers={"x-api-key": key, "anthropic-version": version,
                     "content-type": "application/json"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.load(resp)
        except urllib.error.HTTPError as e:
            # Same reasoning as `openai_compatible`: the body carries the only
            # useful part of a 4xx and it lives on `e.read()`.
            if usage is not None:
                usage.record(seconds=time.time() - t0, failed=True)
            try:
                detail = e.read().decode("utf-8", "replace").strip()[:400]
            except Exception:  # noqa: BLE001 - the body is best-effort
                detail = ""
            message = (f"{base} returned HTTP {e.code} for model {model!r}"
                       + (f": {detail}" if detail else ""))
            if e.code in (429, 503):
                header = ""
                try:
                    header = (e.headers.get("Retry-After") or "").strip()
                except Exception:  # noqa: BLE001 - headers are best-effort
                    header = ""
                try:
                    retry_after = float(header) if header else None
                except ValueError:
                    retry_after = None
                raise RateLimited(message, retry_after) from e
            raise RuntimeError(message) from e
        except Exception:
            if usage is not None:
                usage.record(seconds=time.time() - t0, failed=True)
            raise
        if usage is not None:
            u = data.get("usage") or {}
            usage.record(prompt_tokens=u.get("input_tokens", 0) or 0,
                         completion_tokens=u.get("output_tokens", 0) or 0,
                         seconds=time.time() - t0)
        blocks = data.get("content")
        if not isinstance(blocks, list):
            raise RuntimeError(
                f"{base} returned a response with no content for model {model!r}: "
                f"{json.dumps(data)[:300]}")
        return "".join(b.get("text") or "" for b in blocks
                       if isinstance(b, dict) and b.get("type") == "text")

    return with_retries(complete, attempts=retries) if retries > 1 else complete
