# Connecting agents & LLMs

> **Plugs into [`evolve`](evolution.md) via** `agent=LLMAgent(<completion>)` (or
> `run=`/`propose=`). This page is the completion layer that `LLMAgent` wraps.

`agentdescent.agents` is the **general "talk to a model/agent" layer**. It is
deliberately separate from `agentdescent.evolution` — how you reach a model has
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
from agentdescent.agents import claude, from_callable, echo, with_retries

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

## What did the run cost? — `Usage`

A `Completion` is `prompt -> text`, so the token counts the providers *do* return
would be thrown away at the adapter boundary. Pass a `Usage` and they are kept:

```python
from agentdescent.agents import Usage, openai_compatible, metered

usage = Usage()
model = openai_compatible(model="deepseek-v4-flash", usage=usage)   # or claude(usage=usage)

evolve(tasks, reward, agent=LLMAgent(model), rounds=10)

print(usage.summary())          # 412 calls, 1,204,881 prompt + 96,004 completion tokens, 903.2s in the model
print(usage.estimated_cost(per_1m_prompt=0.28, per_1m_completion=0.42))
```

| | |
|---|---|
| `calls`, `failures` | attempts made, and how many raised (retries count individually — they cost money) |
| `prompt_tokens`, `completion_tokens`, `total_tokens` | **real** counts from the API response |
| `seconds` | wall-clock spent inside the model |
| `estimated_cost(...)` | you supply the per-million prices; the library ships no price table it would have to keep current |

`Usage` is safe to share across worker threads. For a backend that is not one of
the built-in providers — a tool-using agent loop, say — wrap it in
`metered(completion, usage)`, which counts calls, failures and time (tokens are
not observable through a plain `Completion`).

## Bringing another provider

There is no provider lock-in — any `prompt -> text` function is a completion:

```python
def openai_completion(prompt: str) -> str:
    resp = openai_client.chat.completions.create(
        model="gpt-...", messages=[{"role": "user", "content": prompt}])
    return resp.choices[0].message.content

from agentdescent.agents import from_callable
model = from_callable(openai_completion)
```

## Using it in skill evolution

`agentdescent.evolution` consumes a completion through `LLMAgent`:

```python
from agentdescent.agents import claude
from agentdescent.evolution import LLMAgent

agent = LLMAgent(claude(model="claude-haiku-4-5"))
```

`claude_agent(model=...)` in the evolution engine is just a convenience for
`LLMAgent(claude(model))` — the provider code lives here, in `agentdescent.agents`.

## Tool-using agent backends (`agentdescent.backends`)

A `Completion` maps a prompt to text — enough for most examples. But some tasks
need the base agent to **navigate documents with tools**, not consume a fixed
excerpt: [EvoSkill's OfficeQA](algo-evoskill.md) answer is a figure buried in a
1 MB financial table that must be found by `grep` and then *computed*.
[`agentdescent.backends`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/backends.py)
adds that layer — one contract, `AgentBackend.answer(question, document, skills="")`:

| Backend | What it is | Runs where |
|---|---|---|
| **`openhands_backend(model, base_url, …)`** | a **real OpenHands agent** (SDK v1.x, `terminal` + `file_editor` tools) driven by any LiteLLM model | Python ≥ 3.12 + `pip install openhands-ai` |
| **`tool_loop_backend(complete, …)`** | a dependency-free **grep/read ReAct loop** over the document using any `Completion` | anywhere |

```python
from agentdescent.backends import openhands_backend, tool_loop_backend

# a real OpenHands agent on DeepSeek (openai/<model> + base_url routes via LiteLLM):
backend = openhands_backend(model="openai/deepseek-v4-pro",
                            base_url="https://api.deepseek.com")
# or a portable local stand-in:
backend = tool_loop_backend(claude(model="claude-haiku-4-5"))

answer = backend.answer(question, document_text, skills=learned_skills)
```

EvoSkill selects one with `--backend openhands|toolloop|retrieval`; the measured
gated lift with the OpenHands backend + DeepSeek on OfficeQA (**58.0% → 65.7%**) is
on the [EvoSkill page](algo-evoskill.md#empirical-results-real-openhands-agent-deepseek-on-officeqa).

### Running the OpenHands backend

* **Model / provider.** The LLM is any LiteLLM model. `openai/<name>` + `base_url`
  targets an OpenAI-compatible endpoint — e.g. DeepSeek with
  `model="openai/deepseek-v4-pro"` + `base_url="https://api.deepseek.com"` (the
  key comes from `OPENAI_API_KEY`). Native tool-calling drives the `terminal` /
  `file_editor` tools; the agent `grep`s the document, `view`s the right rows, and
  **computes** the answer.
* **Environment.** The real OpenHands SDK needs **Python ≥ 3.12** (a `uv`-managed
  venv works — no Docker, no admin): `pip install openhands-ai`.
* **Structured-output gotcha.** OpenHands has no native structured output, so it
  re-asks the model to reformat the answer as JSON. Its default uses OpenAI strict
  `response_format:{type:"json_schema"}`, which **DeepSeek rejects** (HTTP 400
  *"response_format type is unavailable"*) — use `{type:"json_object"}` with the
  schema in the prompt instead. (Providers with native structured output — Claude,
  OpenAI, Codex — need no shim.)
* **Concurrent eval.** Backends are plain callables, so the aggregator fans out
  its held-out eval concurrently (`eval_concurrency`) — this is where the
  parallelism pays, since each OfficeQA question is an OpenHands rollout of
  ~3–6 min. (When every candidate must be gated on the full held-out set,
  barrier-free async adds nothing — the run is val-bound — so EvoSkill uses the
  synchronous `evolve()` path; see the [EvoSkill results](algo-evoskill.md#empirical-results-real-openhands-agent-deepseek-on-officeqa).)
