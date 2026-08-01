# Strategies — what evolves, and how a proposal becomes a diff

*Modules:* [`agentdescent.strategies`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/strategies.py)
(text) · [`agentdescent.treestrategy`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/treestrategy.py)
(a directory) — one module per strategy family, none of them inside the engine
· *API:* [`Strategy`, `SingleSlot`, `AppendRules`, `KeyedRules`](api.md#the-loop), [`FileTree`](api.md#the-file-tree-strategy)

A strategy answers two questions and nothing else:

1. **What is the artifact?** — its state, as a flat `{key: value}` dict.
2. **What does a proposal mean?** — how a string from the reflector becomes a
   [`Diff`](data-model.md#diff-the-gradient).

```python
class Strategy(Protocol):
    def initial(self) -> Dict[str, str]: ...
    def render(self, state: Dict[str, str]) -> str: ...
    def to_diff(self, state, proposal, author, base_version, target) -> Optional[Diff]: ...
```

Three methods, no base class. Everything else — merging, conflict resolution,
acceptance, versioning — is the [aggregator](aggregator.md)'s job and needs no
cooperation from you.

## The four built-ins

```python
from agentdescent import SingleSlot, AppendRules, KeyedRules, FileTree

evolve(tasks, reward, agent=agent, strategy=SingleSlot(initial_value="Answer concisely."))
```

| strategy | the artifact is | competing proposals |
|---|---|---|
| **`SingleSlot`** | one value — a system prompt, an instruction, one document | always contradict; the best replaces the incumbent |
| `AppendRules` | a deduped list of lessons, keyed by content hash | almost always **fuse**; identical ones collapse |
| `KeyedRules(categories)` | one entry per named category | contradict *within* a category, fuse across |
| [`FileTree(files)`](directory-evolution.md) | a **directory**, one key per file path | contradict per file, fuse across files |

### The key space is the design decision

The keys decide what can merge concurrently, and that is the whole choice:

```
AppendRules   keys = hash(proposal)   →  N workers, N distinct lessons, all fuse
SingleSlot    keys = {"value"}        →  N workers, N candidates, one survives
KeyedRules    keys = your categories  →  parallelism bounded by category count
FileTree      keys = file paths       →  parallelism bounded by file count
```

`SingleSlot` maximises *selection pressure*: every round is a tournament and only
the best-scoring rewrite survives. `AppendRules` maximises *accumulation*: almost
everything merges, and the artifact grows. Neither is better; they answer
different questions about what you are evolving.

!!! tip "Split the artifact to buy parallelism"
    Two complementary edits to the same key are a contradiction, and one of them
    is thrown away. If your workers keep colliding, the fix is usually to give
    the artifact more keys — categories for `KeyedRules`, or more files for
    `FileTree` (a small `SKILL.md` plus `references/*.md`, one concern per file).

### `SingleSlot`

```python
SingleSlot(initial_value="You are a helpful assistant.",
           key="value", min_chars=1, empty_render="(no instruction yet)")
```

The most common thing anyone evolves, and until it existed three of the shipped
[algorithm ports](self-evolution-examples.md) each rolled their own variant.
`min_chars` guards against a reflector that replies with a terse non-answer.

### `AppendRules`

```python
AppendRules(title="# Playbook")
```

Each proposal becomes a rule keyed by `rule_id(text)` — a content hash — so two
workers that independently learn the same lesson produce the *same* key with the
*same* value, which the aggregator collapses to one. Rules are rendered sorted
under the title.

### `KeyedRules`

```python
KeyedRules(categories=["routing", "formatting", "units"])
```

Proposals look like `"formatting: always answer in cents"`. A proposal for an
existing category **overwrites** it, so two workers disagreeing about formatting
produce a contradiction the aggregator resolves on held-out score. An
unrecognised category falls back to content-addressed append behaviour.

`keys()` declares the category list, which is what
[tensor parallelism](parallelism.md) partitions into sections.

### `FileTree`

The artifact is a directory; a key is a relative file path. It has its own page:
[evolving a directory](directory-evolution.md).

## Writing your own

```python
from agentdescent import Diff

class OneValue:                      # this is SingleSlot, longhand
    def initial(self):
        return {}

    def render(self, state):
        return state.get("v", "(none)")

    def to_diff(self, state, proposal, author, base_version, target):
        if state.get("v") == proposal:
            return None              # None -> propose nothing
        return Diff(diff_id=f"{author}:{base_version}", target=target,
                    ops={"v": proposal}, author=author)

evolve(tasks, reward, agent=agent, strategy=OneValue())
```

Return `None` from `to_diff` for any proposal you do not want — malformed, a
no-op, out of bounds. That is the normal path, not an error path: a reflector
that ignores your protocol is a quality problem the run should absorb and count,
not crash on.

Four things to get right:

1. **`render` must be deterministic**, because it is the evaluation-cache key.
   Two states that render identically cannot score differently, and two states
   that render *differently* must be genuinely different — see below.
2. **Keys are your op-space.** Pick them so that things which can be improved
   independently land on different keys.
3. **`to_diff` sees the current `state`.** Use it to drop no-ops and to enforce
   whatever bounds you want before the aggregator's trust region does.
4. **Declare `keys()`** if you know the key space up front. Without it,
   `evolve()` refuses to pair the strategy with `TensorParallel` rather than
   silently dropping most proposals.

### `render` is the artifact's serialisation, not necessarily its prompt

`render(state)` feeds two things: `run(rendered, task)`, and the evaluation cache
key (`EvolvingArtifact._signature`).

For a text artifact those wants coincide. For a structured one they do not — a
lossy pretty-printed render would make two different artifacts share a cached
score. The resolution is that **`run` is where an artifact becomes a prompt**:

```python
def run(rendered, task):
    state = parse(rendered)                 # back to structure
    return model(my_prompt_format(state, task))
```

That is exactly how [`FileTree`](directory-evolution.md) keeps a lossless JSON
render while [EvoSkill](algo-evoskill.md) still shows the model its own
`### skill: <name>` format — byte for byte what it showed before its artifact
became a directory.

!!! important "The framework never injects the artifact into your prompt — you do"
    `run(rendered, task)` hands you the artifact as text. Where it goes is
    entirely your call: a system prompt, a prefix, a few-shot block, a tool
    description, or a file on disk. So "what evolves" is set by two things
    together — `strategy=` fixes the artifact's *shape*, `run=` decides how that
    shape reaches the model.

## Strategies in the algorithm ports

Each is a real `Strategy` you can read and reuse:

| strategy | port | the artifact |
|---|---|---|
| `ACEPlaybook` | [ACE](algo-ace.md) | an itemised, incremental-delta context playbook |
| `InstructionSlot` | [GEPA](algo-gepa.md) | one instruction prompt each proposal replaces |
| `SkillLibraryTree` | [EvoSkill](algo-evoskill.md) | a **directory** of `SKILL.md` files (a `FileTree` subclass) |
| `SkillDocStrategy` | [SkillOpt](algo-skillopt.md) | one markdown doc under bounded edit operations |
| `AgentDesignStrategy` | [ADAS](algo-adas.md) | one agentic-system design |
| `HarnessStrategy` | [DGM](algo-dgm.md) | a coding agent's capability set |
