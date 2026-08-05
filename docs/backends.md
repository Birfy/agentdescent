# Backends — a tool-using agent over a document

*Module:* [`agentdescent.backends`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/backends.py)
· *API:* [`AgentBackend`, `document_agent`, `openhands`, `tool_loop_backend`, …](api.md#document-backends)

[`Completion`](agents.md) is `prompt -> text`, and that is enough for a task
whose input fits in a prompt. It is not enough for
[EvoSkill's OfficeQA](algo-evoskill.md): the answer is a figure buried in a
200 KB – 1.2 MB financial table that has to be **found by grep and then
computed**. A single completion cannot do that; an agent with tools can.

```python
AgentBackend.answer(question, document, *, skills="", skill_files=None) -> str
```

Every argument there is a *domain* concept, which is why the shape is kept
deliberately **separate** from the general contract: it is a domain adapter built
on `Completion`, not a competitor to it.

## `document_agent` — adapt whatever agent you have

```python
from agentdescent.agents import claude, claude_code
from agentdescent.backends import document_agent, openhands

document_agent(openhands(model="openai/deepseek-v4-flash"))   # real tool agent
document_agent(claude_code())                                  # same task, other agent
document_agent(claude(model="claude-haiku-4-5"))               # no tools -> inline
```

It does the right thing for what it is given:

* a [`WorkspaceAgent`](agents.md#giving-an-agent-somewhere-to-work-workspaceagent) gets a scratch directory with
  the document written into it, so it can genuinely `grep` a 1 MB table;
* a plain completion gets the document inline in the prompt, truncated at
  `inline_chars`.

!!! warning "The inline path is a fallback, not an equivalent"
    Measured on three real OfficeQA items (documents of 266–390 KB) with
    `document_agent(openai_compatible(model="deepseek-v4-flash"))`: **1 of 3
    correct**, because at `inline_chars=200_000` roughly half of each document
    never reached the model. Truncation is never silent — it warns, naming the
    fraction dropped — because an empty or wrong answer otherwise looks like a
    model failure and is not one.

## Skills as files, not as prompt text

```python
backend.answer(question, document,
               skill_files={"lookup/SKILL.md": "read the header row first"})
```

For a workspace agent the library is written to `.claude/skills/` in that same
scratch directory and the prompt carries a **pointer**, so the agent opens the one
skill it needs. That is the entire reason a skill *directory* is worth more than
a concatenated string: inlining the whole library in every question is what it
avoids.

A backend with no workspace has nowhere to put them, so it folds them back into
the inline `skills` block rather than dropping them in silence.

This is what [EvoSkill](algo-evoskill.md) uses now that its artifact is a
[`FileTree`](directory-evolution.md) — with `--backend claude-code` the evolving
skill library reaches the agent as files.

## The two implementations

### `openhands` — a real OpenHands agent

```python
from agentdescent.backends import openhands, openhands_backend

agent = openhands(model="openai/deepseek-v4-pro", base_url="https://api.deepseek.com")
backend = openhands_backend(model="openai/deepseek-v4-pro")   # == document_agent(openhands(...))
```

OpenHands SDK v1.x with `terminal` + `file_editor` tools, driven by any LiteLLM
model — `openai/<name>` plus `base_url` targets any OpenAI-compatible endpoint.
Needs `pip install openhands-ai` (Python ≥ 3.12); the import is lazy, so the rest
of the framework runs without it. No Docker (local runtime).

`openhands()` is itself a `WorkspaceAgent`, so it also works anywhere a plain
completion does — including [`tree_runner`](directory-evolution.md).

### `tool_loop_backend` — the dependency-free stand-in

```python
from agentdescent.backends import tool_loop_backend

backend = tool_loop_backend(openai_compatible(model="glm-4.6"), max_steps=5)
```

A `GREP` / `ANSWER` ReAct loop over the document using any completion: search,
read the matching table region *with its nearest header row*, then compute and
answer. It mirrors what the real tool agent does, runs on any Python, and needs
no extra install — the right choice for reproducing the example locally.

## Which one to use

| situation | backend |
|---|---|
| document fits comfortably in the prompt | `document_agent(<any completion>)` |
| large document, no extra dependencies | `tool_loop_backend(<completion>)` |
| large document, real tool use, Python ≥ 3.12 | `openhands_backend(...)` |
| you already run Claude Code or Codex | `document_agent(claude_code())` |

All four satisfy the same `AgentBackend` protocol, so
[`examples/evoskill/evoskill_skill_discovery.py --backend …`](algo-evoskill.md) switches
between them with a flag and no other change.
