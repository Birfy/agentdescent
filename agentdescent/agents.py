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
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Optional

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


def claude(model: str = "claude-opus-4-8", max_tokens: int = 1024,
           client: Optional[object] = None, usage: Optional[Usage] = None,
           **create_kwargs) -> Completion:
    """A Claude-backed completion (requires ``pip install anthropic`` + creds).

    Pass ``client`` to reuse an existing ``anthropic.Anthropic`` instance;
    otherwise a default one is constructed lazily (resolving credentials from
    the environment / an ``ant auth login`` profile). Use a cheaper ``model``
    (e.g. ``"claude-haiku-4-5"``) for call-heavy loops. Pass ``usage=Usage()`` to
    accumulate the exact token counts the API reports."""
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

    return complete


def openai_compatible(model: str, *, base_url_env: str = "OPENAI_BASE_URL",
                      api_key_env: str = "OPENAI_API_KEY",
                      default_base_url: str = "https://api.openai.com/v1",
                      max_tokens: int = 1024, timeout: float = 120.0,
                      usage: Optional[Usage] = None) -> Completion:
    """A completion for any OpenAI-compatible chat endpoint (GLM/Zhipu, proxies,
    local servers, OpenAI itself).

    The base URL and API key are read from the environment at call time -- they
    never pass through code or arguments. Point it at GLM, for example, by
    setting ``OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4`` and
    ``OPENAI_API_KEY=<your key>`` in your shell, then use ``model="glm-4.6"``."""
    def complete(prompt: str) -> str:
        base = os.environ.get(base_url_env, default_base_url).rstrip("/")
        key = os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(f"set {api_key_env} (and {base_url_env}) in your environment")
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }).encode()
        req = urllib.request.Request(
            f"{base}/chat/completions", data=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.load(resp)
        except Exception:
            if usage is not None:
                usage.record(seconds=time.time() - t0, failed=True)
            raise
        if usage is not None:
            u = data.get("usage") or {}
            usage.record(prompt_tokens=u.get("prompt_tokens", 0) or 0,
                         completion_tokens=u.get("completion_tokens", 0) or 0,
                         seconds=time.time() - t0)
        return data["choices"][0]["message"]["content"]

    return complete
