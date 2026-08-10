# AFlow — Agentic workflow search

**Fidelity class: `mechanism_microport`** — see [port fidelity](port-fidelity.md) for
what the classes mean. This port is measured in the runtime matrix: the mechanism is
preserved and measured under AgentDescent's runtimes; it is **not** a
paper-benchmark reproduction.

| | |
|---|---|
| Paper | "AFlow: Automating Agentic Workflow Generation", Zhang et al., ICLR 2025 ([arXiv:2410.10762](https://arxiv.org/abs/2410.10762)) |
| Upstream code (pinned) | [FoundationAgents/AFlow@3f457218](https://github.com/FoundationAgents/AFlow/tree/3f457218fc716093fe53f6df8a5d5e6379d66346) |
| Definition | [`examples/aflow/aflow_workflow_search.py`](https://github.com/Birfy/agentdescent/blob/main/examples/aflow/aflow_workflow_search.py) |
| Domain | deterministic integer-cents arithmetic (12 tasks, disjoint splits) |

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
`deepseek-v4-flash` at temperature 0.7. Recorded in
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
[PromptBreeder](algo-promptbreeder.md#measured-results) on one run per seed.

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

!!! danger "The selection rule was uniform, and said it was not"
    The port implemented `λ·uniform + (1−λ)·softmax(α·(s−s_max))` and dropped
    one line of upstream's `select_round`:

    ```python
    scores = [item["score"] * 100 for item in sorted_items]
    ```

    Upstream's α is 0.2 *against percentages*, so the effective temperature is
    20 against accuracies in `[0, 1]` — and this port's scores are accuracies in
    `[0, 1]` exactly as upstream's are, so the scaling is not a unit conversion.
    It **is** the temperature. Over a pool scoring 0.50 / 0.40 / 0.25 / 0.10:

    | | pick distribution |
    |---|---|
    | upstream | `[0.688, 0.158, 0.079, 0.075]` |
    | this port, before | `[0.265, 0.257, 0.245, 0.233]` |
    | uniform | `[0.250, 0.250, 0.250, 0.250]` |

    Uniform to three digits. The port ran, logged "soft mixed probability", and
    had no exploitation at all.

    Three smaller departures went with it. **α and λ were the paper's 0.4 / 0.2**
    rather than the pinned code's own `DEFAULT_ALPHA = 0.2` /
    `DEFAULT_LAMBDA = 0.3` — where released code and paper disagree about a
    constant, the code is what produced the published numbers. **The seed
    workflow was force-appended to the pool**; upstream's `get_top_rounds` moves
    round 1 to the front only when it already made the cut, and `select_round`
    re-sorts by score immediately after, so that move changes nothing and
    membership is the whole rule. And **experience carried neither the parent's
    score nor whether each past modification helped**, where upstream's
    `format_experience` reports both and `check_modification` regenerates rather
    than accept a repeat.

    An offline test had pinned the wrong behaviour:
    `test_soft_mixed_keeps_the_seed_and_favours_scores` asserted the seed was
    always reachable, which was true only because of the force-append. The test
    was protecting the bug.
