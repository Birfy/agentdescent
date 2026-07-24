# Connecting agents & LLMs

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
