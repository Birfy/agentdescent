# Connecting agents & LLMs

> **Plugs into [`evolve`](evolution.md) via** `agent=LLMAgent(<completion>)` (or
> `run=`/`propose=`). This page is the completion layer that `LLMAgent` wraps.

`concordia.agents` is the **general "talk to a model/agent" layer**. It is
deliberately separate from `concordia.evolution` — how you reach a model has
nothing to do with skill evolution, and any application built on the framework
can use it.

The whole contract is one type:

```python
Completion = Callable[[str], str]      # prompt -> text
```

Anything that maps a prompt to text is a completion — an LLM call, a tool-using
agent loop, a canned stub. Adapters build completions; higher layers turn a
completion into whatever task interface they need.

## Adapters

```python
from concordia.agents import claude, from_callable, echo, with_retries

# Claude (needs: pip install anthropic + credentials / `ant auth login`)
model = claude(model="claude-opus-4-8")           # or claude-haiku-4-5 for cheap runs
text = model("What is 2+2?")

# Any callable you already have
model = from_callable(lambda prompt: my_llm.generate(prompt))

# Deterministic, no-network stub for tests / dry runs
stub = echo(str.upper)          # returns transform(prompt), or the prompt itself

# Wrap any completion with exponential-backoff retries
robust = with_retries(claude(model="claude-haiku-4-5"), attempts=3)
```

`claude()` accepts a `client=` to reuse an existing `anthropic.Anthropic`
instance, and forwards extra kwargs to `messages.create`.

## Bringing another provider

There is no provider lock-in — any `prompt -> text` function is a completion:

```python
def openai_completion(prompt: str) -> str:
    resp = openai_client.chat.completions.create(
        model="gpt-...", messages=[{"role": "user", "content": prompt}])
    return resp.choices[0].message.content

from concordia.agents import from_callable
model = from_callable(openai_completion)
```

## Using it in skill evolution

`concordia.evolution` consumes a completion through `LLMAgent`:

```python
from concordia.agents import claude
from concordia.evolution import LLMAgent

agent = LLMAgent(claude(model="claude-haiku-4-5"))
```

`claude_agent(model=...)` in the evolution engine is just a convenience for
`LLMAgent(claude(model))` — the provider code lives here, in `concordia.agents`.

## Tool-using agent backends (`concordia.backends`)

A `Completion` maps a prompt to text — enough for most examples. But some tasks
need the base agent to **navigate documents with tools**, not consume a fixed
excerpt: [EvoSkill's OfficeQA](algo-evoskill.md) answer is a figure buried in a
1 MB financial table that must be found by `grep` and then *computed*.
[`concordia.backends`](https://github.com/Birfy/concordia/blob/main/concordia/backends.py)
adds that layer — one contract, `AgentBackend.answer(question, document, skills="")`:

| Backend | What it is | Runs where |
|---|---|---|
| **`openhands_backend(model, base_url, …)`** | a **real OpenHands agent** (SDK v1.x, `terminal` + `file_editor` tools) driven by any LiteLLM model | Python ≥ 3.12 + `pip install openhands-ai` |
| **`tool_loop_backend(complete, …)`** | a dependency-free **grep/read ReAct loop** over the document using any `Completion` | anywhere |

```python
from concordia.backends import openhands_backend, tool_loop_backend

# a real OpenHands agent on DeepSeek (openai/<model> + base_url routes via LiteLLM):
backend = openhands_backend(model="openai/deepseek-v4-pro",
                            base_url="https://api.deepseek.com")
# or a portable local stand-in:
backend = tool_loop_backend(claude(model="claude-haiku-4-5"))

answer = backend.answer(question, document_text, skills=learned_skills)
```

EvoSkill selects one with `--backend openhands|toolloop|retrieval`. Measured lift
with the OpenHands backend + DeepSeek on OfficeQA (66.7% → 79.7%), plus the
DeepSeek structured-output gotcha, are on the [EvoSkill page](algo-evoskill.md#empirical-results-real-openhands-agent-deepseek-on-officeqa).
