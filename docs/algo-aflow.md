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

## Measured results

Three seeds, `async_pipeline`, 80 rollouts each, 8 workers, `--staleness full`,
`--reflective-merge`, `deepseek-v4-flash` at temperature 0.7 with thinking
disabled. Recorded in
[`bench/results/aflow-upstream-selection.json`](https://github.com/Birfy/agentdescent/blob/main/bench/results/aflow-upstream-selection.json).

| seed | test quality | validation | accepted | calls | wall |
|---|---|---|---|---|---|
| 0 | 0.000 → **0.812** | 0.000 → 0.938 | 3/80 | 767 | 248 s |
| 1 | 0.000 → **0.938** | 0.000 → 0.938 | 2/80 | 728 | 227 s |
| 2 | 0.000 → **0.938** | 0.000 → 1.000 | 3/80 | 767 | 220 s |

Three of three seeds moved, mean 0.896, against a domain whose ceiling is 1.000.

### Why this clears the domain when PromptBreeder does not

[PromptBreeder](algo-promptbreeder.md) reaches 0.271 on the same 48 items, the
same model, the same budget, and its successful seeds stall just under the
*format-only* ceiling of 0.479: it discovers what the grader wants and not that
it may reason first. AFlow does both, and the reason is topology rather than
tuning. Its workflow is two nodes — **Solve**, then **ReviewAndRevise** — so the
arithmetic and the output convention have separate places to live. A single
instruction has to hold both at once, and the prompts that nail the format are
the ones that forbid the working.

That is the comparison the matrix exists for: same domain, same runtime, same
budget, and the mechanism is the variable.


!!! note "These cross-algorithm figures demonstrate that each port runs, not which is best"
    Every row is one run per seed, and the seed fixes the data splits and the
    method's own sampler -- **not the model**, which is sampled at temperature
    0.7. Re-running an identical command at an identical seed moves the number:
    PromptBreeder's seed 0 scored 0.438 on one run and 0.875 on the next, from
    the same script and the same code. Read the table as evidence the mechanism
    executes end to end and produces a plausible artifact; a ranking would need
    repeats per seed, which these runs do not have.
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
