# AFlow — Agentic workflow search

> **Workflow self-evolution.** Search the space of code-expressed agentic
> workflows, with soft mixed selection over the top-k scored ones. Runs through
> the shared [`MethodPolicy`](policies.md) runner. Example:
> [`examples/aflow/aflow_workflow_search.py`](https://github.com/Birfy/agentdescent/blob/main/examples/aflow/aflow_workflow_search.py).

| | |
|---|---|
| **Paper** | *AFlow: Automating Agentic Workflow Generation* — Zhang et al., ICLR 2025 ([arXiv:2410.10762](https://arxiv.org/abs/2410.10762)) |
| **Upstream code** | [FoundationAgents/AFlow@3f457218](https://github.com/FoundationAgents/AFlow/tree/3f457218fc716093fe53f6df8a5d5e6379d66346) |
| **Example** | [`examples/aflow/aflow_workflow_search.py`](https://github.com/Birfy/agentdescent/blob/main/examples/aflow/aflow_workflow_search.py) |
| **Domain** | **GSM8K** ([`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k), `main`), 64/64/64 splits |
| **Layer** | L1 (`blast_radius=0.6`, set by the shared runner) |
| **Fidelity** | `mechanism_microport` — [what the classes mean](port-fidelity.md) |

This port is measured in the [runtime matrix](matrix-overview.md): the mechanism
is preserved and measured under AgentDescent's runtimes; it is **not** a
paper-benchmark reproduction.

## The mechanism

AFlow searches the space of code-expressed agentic workflows. Its selection is
**not UCT**: at the pinned revision it draws from the top-k scored workflows
plus the always-included seed with mixed probability
`λ·uniform + (1−λ)·softmax(α·(s−s_max))`, and keeps no visit counts
(`scripts/optimizer_utils/data_utils.py`). Expansion asks an optimizer LLM to
rewrite the selected workflow, with the parent's **experience** — its prior
modifications and whether each helped — injected into the prompt.

## Where each piece lives

| Upstream mechanism | Where it lives here |
|---|---|
| Soft mixed selection over top-k + seed | `SoftMixed(SelectionPolicy)`, driven by the population aggregator over the archive of committed workflows |
| Per-father experience | a per-parent modification log injected into the expansion prompt |
| Workflow as code | two fixed model nodes (Solve → ReviewAndRevise) as `FieldSlots` keys |
| Convergence over 20 rounds | the candidate budget |

## Boundaries

- Fixed two-node topology instead of code-level graph rewrites.
- Paper hyper-parameters α=0.4, λ=0.2 (the pinned code itself ships 0.2/0.3).

## Measured results — GSM8K

Three seeds, `async_pipeline`, 80 rollouts each, 8 workers, `--staleness full`,
reflective merge on (this method's own declaration), `deepseek-v4-flash` at
temperature 0.7. Recorded in
[`bench/results/aflow-gsm8k.json`](https://github.com/Birfy/agentdescent/blob/main/bench/results/aflow-gsm8k.json).

| seed | test quality | validation | accepted | calls | wall |
|---|---|---|---|---|---|
| 0 | 0.609 → **0.984** | 0.609 → 0.969 | 1/80 | 2065 | 705 s |
| 1 | 0.438 → **0.969** | 0.438 → 0.906 | 5/80 | 2071 | 600 s |
| 2 | 0.484 → **0.953** | 0.469 → 0.969 | 4/80 | 2072 | 592 s |

Mean final **0.969**, mean gain **+0.458**, all three seeds moving.

64 items per split from GSM8K's own train and test splits. The baseline is
0.438–0.609 — the model's own accuracy — not the 0.000 the previous fixture
installed by construction; see
[Self-Refine](algo-self-refine.md#measured-results-gsm8k) for why that fixture
was replaced, and the caveat on
[PromptBreeder](algo-promptbreeder.md#measured-results-gsm8k) on one run per seed.

### Two nodes cost two calls

~2070 calls per seed against [Self-Refine](algo-self-refine.md)'s ~1050, for
0.969 against 0.943. AFlow's workflow is Solve then ReviewAndRevise, so every
rollout is two model calls where Self-Refine's fused critique-and-refine is one.

On the old hand-written fixture the two were 0.896 and 0.979 — the gap ran the
other way, and it was an artifact of that domain's one-bit output convention,
which a second reviewing node could fix and a single instruction had to hold
alongside the arithmetic. Against a benchmark with a real floor they land within
0.026 of each other, and what separates them is what each spends to get there.
That reversal is the clearest thing the move to GSM8K bought.

## Run it

```bash
python -m examples.aflow.aflow_workflow_search --dry-run

# one seed of the three above
python -m examples.aflow.aflow_workflow_search --yes --seed 0 \
    --provider openai --model deepseek-v4-flash \
    --async --workers 8 --budget-rollouts 80 --staleness full \
    --temperature 0.7 --max-seconds 3600
```

Flags: [the MethodPolicy command line](self-evolution-examples.md#the-methodpolicy-command-line).

Offline tests: `tests/test_aflow_upstream.py`, `tests/test_candidate_methods.py`.
