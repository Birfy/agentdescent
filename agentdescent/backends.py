"""Agentic backends -- a base agent that *navigates documents with tools*, not
just maps a prompt to text.

:data:`agentdescent.agents.Completion` is ``prompt -> text``. That is enough for
ACE/GEPA/SkillOpt (fixed prompt in, answer out), but not for EvoSkill's OfficeQA:
the answer is a figure buried in a 200 KB - 1.2 MB financial table that often has
to be **found by grep and then computed** (e.g. summing the monthly "national
defense" rows for a calendar year). A single completion cannot do that; an
*agent with tools* can.

This module adds that abstraction:

    AgentBackend.answer(question, document, skills="") -> str

with two implementations:

* :func:`openhands_backend` -- runs a **real OpenHands agent** (OpenHands SDK
  v1.x) with ``terminal`` + ``file_editor`` tools over the document, driven by any
  LiteLLM-supported model. Point it at DeepSeek with ``model="openai/deepseek-v4-pro"``
  and ``base_url="https://api.deepseek.com"``. Requires ``pip install openhands-ai``
  (Python >= 3.12). No Docker needed (local runtime).
* :func:`tool_loop_backend` -- a dependency-free ``grep``/``read`` ReAct loop over
  the document using any :data:`~agentdescent.agents.Completion`. A lighter local
  stand-in that mirrors what OpenHands does; runs anywhere.

Both return a plain answer string, so they slot in wherever a base agent is
needed (see ``examples/evoskill_skill_discovery.py --backend openhands``).
"""

from __future__ import annotations

import os
import re
import tempfile
import warnings
from typing import Callable, Protocol, runtime_checkable

from .agents import Completion, WorkspaceAgent


@runtime_checkable
class AgentBackend(Protocol):
    """A base agent that answers a question about a document, possibly using tools."""

    def answer(self, question: str, document: str, *, skills: str = "") -> str: ...


class _FnBackend:
    """Wrap an ``answer(question, document, *, skills)`` closure as an AgentBackend."""

    def __init__(self, fn: Callable[..., str]) -> None:
        self._fn = fn

    def answer(self, question: str, document: str, *, skills: str = "") -> str:
        return self._fn(question, document, skills=skills)


# ---------------------------------------------------------------------------
# OpenHands backend -- a real tool-using agent (Read/Grep/Bash over the doc)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The domain adapter: a document task, on top of ANY agent
# ---------------------------------------------------------------------------


_DOC_INSTR = (
    "{skills}The file {fname} in this directory is a document (often large "
    "financial tables). Use grep/find to locate the relevant rows and read them; "
    "if the answer spans several rows (e.g. monthly values for a calendar year), "
    "compute it. Reply with ONLY the final answer value.\n\nQuestion: {q}"
)
_DOC_INSTR_INLINE = (
    "{skills}Below is a document (often large financial tables). Find the relevant "
    "rows; if the answer spans several rows, compute it. Reply with ONLY the final "
    "answer value.\n\nDocument:\n{doc}\n\nQuestion: {q}"
)


def document_agent(completion: Completion, *, doc_filename: str = "document.txt",
                   inline_chars: int = 200_000) -> AgentBackend:
    """Turn **any** :data:`~agentdescent.agents.Completion` into an
    :class:`AgentBackend` for document questions.

    This is the *domain* shape -- ``answer(question, document, skills)`` -- kept
    deliberately separate from the general contract. Give it whatever agent you
    have and it does the right thing for that agent:

    * a :class:`~agentdescent.agents.WorkspaceAgent` (OpenHands, Claude Code,
      Codex, ...) gets a scratch directory with the document written into it, so
      it can genuinely ``grep`` a 1 MB table;
    * a plain completion (an API model) gets the document inline in the prompt,
      truncated to ``inline_chars``.

    ::

        document_agent(openhands(model="openai/deepseek-v4-flash"))
        document_agent(claude_code())          # same task, different agent
        document_agent(claude(model="claude-haiku-4-5"))   # no tools: inline
    """
    def answer(question: str, document: str, *, skills: str = "") -> str:
        skill_block = f"Learned skills you should apply:\n{skills}\n\n" if skills.strip() else ""
        if isinstance(completion, WorkspaceAgent):
            workdir = tempfile.mkdtemp(prefix="agentdescent-doc-")
            with open(os.path.join(workdir, doc_filename), "w", encoding="utf-8") as f:
                f.write(document)
            prompt = _DOC_INSTR.format(skills=skill_block, fname=doc_filename, q=question)
            return completion.in_workspace(workdir)(prompt).strip()
        if len(document) > inline_chars:
            # Never drop half a document in silence: the answer may be in the part
            # the agent never saw, and an empty or wrong reply then looks like a
            # model failure rather than a truncation. A workspace agent avoids this
            # entirely -- it reads the file itself.
            warnings.warn(
                f"document_agent: inlining only {inline_chars:,} of "
                f"{len(document):,} chars ({100 * inline_chars / len(document):.0f}%) "
                "because this agent has no workspace to read the file from; the "
                "answer may lie in the truncated part. Pass a WorkspaceAgent "
                "(openhands(), claude_code(), codex(), cli_agent(...)) to let the "
                "agent grep the whole document, or raise inline_chars.",
                RuntimeWarning, stacklevel=3)
        prompt = _DOC_INSTR_INLINE.format(
            skills=skill_block, doc=document[:inline_chars], q=question)
        return completion(prompt).strip()

    return _FnBackend(answer)


class _OpenHandsAgent:
    """A real OpenHands agent as a workspace-bindable :data:`Completion`."""

    def __init__(self, model, base_url, api_key_env, temperature, max_iterations,
                 workspace=None) -> None:
        self._cfg = (model, base_url, api_key_env, temperature, max_iterations)
        self.workspace = workspace
        self._sdk = None

    def in_workspace(self, path: str) -> Completion:
        return _OpenHandsAgent(*self._cfg, workspace=path)

    def _load(self):
        if self._sdk is not None:
            return self._sdk
        model, base_url, api_key_env, temperature, max_iterations = self._cfg
        try:
            from pydantic import SecretStr
            from openhands.sdk import LLM, Agent, Conversation, Tool
            from openhands.tools import register_default_tools
        except Exception as e:  # noqa: BLE001 - surface the optional dependency
            raise RuntimeError(
                "openhands() needs `pip install openhands-ai` on Python >= 3.12. "
                f"Import failed: {type(e).__name__}: {e}") from e
        os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")
        register_default_tools(enable_browser=False)   # terminal / file_editor
        key = os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(f"set {api_key_env} for openhands()")
        llm = LLM(model=model, base_url=base_url, api_key=SecretStr(key),
                  usage_id="backend", drop_params=True, temperature=temperature,
                  native_tool_calling=True)
        self._sdk = (llm, Agent, Conversation, Tool, max_iterations)
        return self._sdk

    def __call__(self, prompt: str) -> str:
        llm, Agent, Conversation, Tool, max_iterations = self._load()
        workdir = self.workspace or tempfile.mkdtemp(prefix="agentdescent-oh-")
        agent = Agent(llm=llm, tools=[Tool(name="terminal"), Tool(name="file_editor")])
        conv = Conversation(agent=agent, workspace=workdir,
                           max_iteration_per_run=max_iterations)
        conv.send_message(prompt)
        conv.run()
        texts = []
        for ev in conv.state.events:
            m = getattr(ev, "llm_message", None) or getattr(ev, "message", None)
            if m is not None and getattr(m, "role", "") == "assistant":
                for c in getattr(m, "content", []) or []:
                    t = getattr(c, "text", None)
                    if t:
                        texts.append(t)
        return texts[-1].strip() if texts else ""


def openhands(model: str = "openai/deepseek-v4-pro", *,
              base_url: str = "https://api.deepseek.com",
              api_key_env: str = "OPENAI_API_KEY", temperature: float = 0.0,
              max_iterations: int = 40) -> "_OpenHandsAgent":
    """A **real OpenHands agent** (SDK v1.x) as a workspace-bindable Completion.

    Same contract as every other agent -- prompt in, text out -- so it drops into
    ``LLMAgent`` / ``evolve`` like an API model, and ``.in_workspace(path)`` lets a
    caller stage files for it to work on. The LLM is any LiteLLM model;
    ``openai/<name>`` + ``base_url`` targets an OpenAI-compatible endpoint such as
    DeepSeek. Needs ``pip install openhands-ai`` (Python >= 3.12); the import is
    lazy, so the rest of the framework runs without it.
    """
    return _OpenHandsAgent(model, base_url, api_key_env, temperature, max_iterations)


def openhands_backend(model: str = "openai/deepseek-v4-pro", *,
                      base_url: str = "https://api.deepseek.com",
                      api_key_env: str = "OPENAI_API_KEY", temperature: float = 0.0,
                      max_iterations: int = 40,
                      doc_filename: str = "document.txt") -> AgentBackend:
    """``document_agent(openhands(...))`` -- the document task on OpenHands.

    Kept for the EvoSkill example and existing callers; new code can compose the
    two directly and swap in any other agent."""
    return document_agent(openhands(model, base_url=base_url, api_key_env=api_key_env,
                                    temperature=temperature,
                                    max_iterations=max_iterations),
                          doc_filename=doc_filename)


# ---------------------------------------------------------------------------
# Tool-loop backend -- a dependency-free grep/read ReAct loop (a local stand-in)
# ---------------------------------------------------------------------------

_TOOL_SYSTEM = (
    "You answer a question about a large document by SEARCHING it, not guessing. "
    "{skills}You can issue ONE action per turn:\n"
    "  GREP <terms>   -- search the document (returns matching lines + a window "
    "and the nearest table header)\n"
    "  ANSWER <value> -- give the final answer (only when you have the figure)\n"
    "If the answer spans multiple rows (e.g. monthly values), GREP for them, then "
    "sum in your head and ANSWER.\n\nQuestion: {q}\n\nObservations so far:\n{obs}\n\n"
    "Your next action (start with GREP or ANSWER):")


def _grep(lines, query: str, window: int = 3, max_hits: int = 8) -> str:
    terms = [t for t in re.findall(r"[a-z0-9.]+", query.lower()) if len(t) >= 2]
    if not terms:
        return "(no search terms)"
    need = max(1, len(terms) // 2)
    hits = [i for i, ln in enumerate(lines)
            if sum(1 for t in terms if t in ln.lower()) >= need]
    keep, seen = [], set()
    for i in hits[:max_hits]:
        header = None                       # nearest table-header row (>=2 year tokens)
        for j in range(i, max(0, i - 40), -1):
            if len(re.findall(r"\b(?:18|19|20)\d{2}\b", lines[j])) >= 2:
                header = j
                break
        region = set(range(max(0, i - window), min(len(lines), i + window + 1)))
        if header is not None:
            region.add(header)
        for j in sorted(region):
            if j not in seen and lines[j].strip():
                keep.append(f"{j}: {lines[j]}")
                seen.add(j)
    return "\n".join(keep[:120]) or "(no matches)"


def tool_loop_backend(complete: Completion, *, max_steps: int = 5,
                      window: int = 3) -> AgentBackend:
    """A dependency-free ``grep``/``read`` ReAct loop over the document.

    Mirrors what the OpenHands agent does -- iteratively search the document, read
    the matching table region (with headers), then answer/compute -- but using a
    plain :data:`~agentdescent.agents.Completion`, so it runs on any Python. A lighter
    local stand-in for :func:`openhands_backend`."""

    def answer(question: str, document: str, *, skills: str = "") -> str:
        lines = document.splitlines()
        obs = "(none yet)"
        skill_block = f"Apply these learned skills:\n{skills}\n\n" if skills.strip() else ""
        prompt = ""
        for _ in range(max_steps):
            prompt = _TOOL_SYSTEM.format(skills=skill_block, q=question, obs=obs)
            reply = complete(prompt).strip()
            m = re.match(r"\s*ANSWER[:\s]+(.+)", reply, re.IGNORECASE | re.DOTALL)
            if m:
                return m.group(1).strip()
            g = re.match(r"\s*GREP[:\s]+(.+)", reply, re.IGNORECASE)
            if g:
                found = _grep(lines, g.group(1).strip(), window=window)
                obs = (obs + "\n" if obs != "(none yet)" else "") + \
                    f"GREP {g.group(1).strip()!r}:\n{found}"
            else:
                return reply                # no valid action -> treat as the answer
        return complete(prompt + "\n\nNow output ANSWER: <value>").strip()

    return _FnBackend(answer)
