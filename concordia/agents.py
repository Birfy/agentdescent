"""Provider-agnostic inference -- connect any agent or LLM to the framework.

This is the general "talk to a model/agent" layer. It is deliberately kept
**out of** :mod:`concordia.evolution`: skill evolution is just *one* application
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
import time
import urllib.request
from typing import Callable, Optional

Completion = Callable[[str], str]


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
           client: Optional[object] = None, **create_kwargs) -> Completion:
    """A Claude-backed completion (requires ``pip install anthropic`` + creds).

    Pass ``client`` to reuse an existing ``anthropic.Anthropic`` instance;
    otherwise a default one is constructed lazily (resolving credentials from
    the environment / an ``ant auth login`` profile). Use a cheaper ``model``
    (e.g. ``"claude-haiku-4-5"``) for call-heavy loops."""
    _client = client

    def complete(prompt: str) -> str:
        nonlocal _client
        if _client is None:
            from anthropic import Anthropic  # lazy, optional dependency
            _client = Anthropic()
        msg = _client.messages.create(
            model=model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}], **create_kwargs,
        )
        return "".join(b.text for b in msg.content if b.type == "text")

    return complete


def openai_compatible(model: str, *, base_url_env: str = "OPENAI_BASE_URL",
                      api_key_env: str = "OPENAI_API_KEY",
                      default_base_url: str = "https://api.openai.com/v1",
                      max_tokens: int = 1024, timeout: float = 120.0) -> Completion:
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
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
        return data["choices"][0]["message"]["content"]

    return complete
