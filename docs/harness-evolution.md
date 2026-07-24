# Example: harness evolution (L1)

Another use of the same [engine](evolution.md) — but the evolving artifact is a
**harness**, not a skill. "What evolves" is chosen by registration and blast
radius, not hard-coded: a harness (context policy, tool router, request
pipeline, learned verifier) is registered at a higher `blast_radius` so it lives
in the **L1 governance layer**, where the aggregator treats it conservatively
(every merge is forced through the oracle, staleness tolerance widens).

This example also shows you **don't need an LLM** — you drive `evolve` with plain
`run` / `reward` / `propose` functions. Just write the rules of evolution.

Source:
[`examples/harness_evolution.py`](https://github.com/Birfy/concordia/blob/main/examples/harness_evolution.py).

```python
from concordia.evolution import evolve, KeyedRules

# the harness is a request-processing pipeline: route / normalize / trim
def run(rendered, task):      # apply the steps currently in the harness
    ...
def propose(rendered, task, output, score):   # suggest a missing step
    ...

result = evolve(
    tasks, reward, run=run, propose=propose,     # plain functions, no LLM
    strategy=KeyedRules(categories=["route", "normalize", "trim"]),
    blast_radius=0.6,                            # -> L1 harness governance
    artifact_id="harness", rounds=10, n_workers=3,
)
```

## Run it

```bash
python -m examples.harness_evolution      # deterministic, no API key
```

Output:

```
Artifact : request-processing harness (steps: route / normalize / trim)
Governance: L1 (blast_radius=0.6) -> L1_SLOW, every merge oracle-gated

round   0  reward=0.760  items=1  +1/-0
round   1  reward=1.000  items=3  +1/-0
...
=== evolved harness ===
# Config (by category)
## normalize
apply the normalize step
## route
apply the route step
## trim
apply the trim step

held-out score: 0.760 -> 1.000
steps learned : 3
```

Parallel workers surface different missing steps and the aggregator **fuses**
them into one harness — but because this is L1, each merge had to clear the
**ground-truth oracle** first, not just cheap approximate evaluation. Swapping
`blast_radius=0.6` for `0.2` would make the identical code evolve an L2 artifact
instead. See [governance](concepts.md#6-governance-blast-radius-decides-parallelism).
