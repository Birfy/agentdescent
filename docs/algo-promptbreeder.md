# PromptBreeder — Prompt self-evolution (genetic)

**Fidelity class: `mechanism_microport`** — see [port fidelity](port-fidelity.md) for
what the classes mean. This port is measured in the runtime matrix: the mechanism is
preserved and measured under AgentDescent's runtimes; it is **not** a
paper-benchmark reproduction.

| | |
|---|---|
| Paper | "Promptbreeder: Self-Referential Self-Improvement Via Prompt Evolution", Fernando et al., 2023 ([arXiv:2309.16797](https://arxiv.org/abs/2309.16797)) |
| Upstream code (pinned) | paper only — no official released code |
| Definition | [`examples/promptbreeder/promptbreeder_genetic_prompts.py`](https://github.com/Birfy/agentdescent/blob/main/examples/promptbreeder/promptbreeder_genetic_prompts.py) |
| Domain | deterministic integer-cents arithmetic (48 tasks, disjoint 16/16/16 splits) |

## The mechanism

PromptBreeder evolves a population of units — each a set of task-prompts plus a
**mutation-prompt** — with a binary tournament genetic algorithm: sample two
units, mutate the winner with one of nine operators (uniformly drawn), and
overwrite the loser. The self-referential move is **hyper-mutation**: the
mutation-prompt is itself rewritten by applying it to itself. Fitness is
measured on a 100-item training batch; the paper runs a population of 50 for
20–30 generations.

## Where each piece lives

| Upstream mechanism | Where it lives here |
|---|---|
| Binary tournament replication | `PromptBreederPopulation`, in the engine's `aggregator_factory` seam: two units sampled uniformly, both re-scored, winner committed as the head the next batch mutates |
| **Loser overwritten**, population fixed at N | the same class: the tournament records the loser's slot and the next committed child replaces it |
| Task/mutation-prompt unit | `FieldSlots` genome: two plain-text ledger keys that union-merge on disjoint edits and model-merge when contested |
| Nine mutation operators, uniformly drawn | `_promptbreeder_operators.py`; all nine, sampled uniformly per replication event, with the realised histogram reported |
| Population-conditioned operators (EDA, EDA-rank-and-index, lineage, crossover) | `PopulationView`, the handle the aggregator writes and `propose` reads |
| Fitness on a random training batch | `PopulationContext.fitness(state, batch)` — a resampled batch of the **train** split |
| N-unit initialisation from description x mutation-prompt x thinking-style | `seed_population`, billed to an `init:` phase rather than to the proposal budget |

!!! warning "Three of these were missing, and each changed the search rather than the bookkeeping"
    The port ran, returned a number, and was wrong in ways nothing asserted --
    an archive that only grows still returns a best unit, a tournament ranked on
    held-out still picks a winner, and three operators in rotation still mutate.

    | | was | is |
    |---|---|---|
    | replacement | archive grew forever, no unit ever died | `\|P\| = N`, the loser's slot is reused |
    | fitness | the held-out split — the acceptance gate's own signal, which makes the tournament a second gate rather than a fitness measure | a random batch of the train split, resampled per tournament |
    | operators | 3 of 9, in fixed rotation | 9 of 9, uniformly sampled |

    Rotation is the subtle one: it *guarantees* each operator's share where
    sampling does not, so a run's operator mix was an assumption rather than an
    observation.

    Algorithm 1's tournament cannot be a `SelectionPolicy`, which is why it moved:
    a selection policy receives candidates carrying cached scores and returns one,
    while the paper's tournament has to **evaluate** both sampled units and
    **replace** the loser.

## Boundaries

- Population 8 and fitness batch 4, against the paper's 50 and 100.
- The unit carries no few-shot context, so context shuffling is expressed as
  prompt crossover over the population rather than over exemplars.
- The domain was 12 items in 4/4/4 splits. `run_port` refuses a run whose train
  split is smaller than the worker count, so that capped this port — and the ten
  others sharing the domain — at four workers, with the acceptance gate resting
  on four items. It is now 48 items in 16/16/16.

## Measured results

Three seeds, `async_pipeline`, 80 rollouts each, 8 workers, `--staleness full`,
`--reflective-merge`, `deepseek-v4-flash` at temperature 0.7 with thinking
disabled. Recorded in
[`bench/results/promptbreeder-algorithm1.json`](https://github.com/Birfy/agentdescent/blob/main/bench/results/promptbreeder-algorithm1.json).

| seed | test quality | validation | accepted | calls | wall |
|---|---|---|---|---|---|
| 0 | 0.000 → **0.438** | 0.000 → 0.312 | 5/80 | 631 | 365 s |
| 1 | 0.000 → 0.000 | 0.000 → 0.000 | 1/80 | 638 | 594 s |
| 2 | 0.000 → **0.375** | 0.000 → 0.375 | 4/80 | 640 | 377 s |

**Two of three seeds moved; one found nothing.** The mean is 0.271 and the range
is the result, not noise around it — a single seed here would have supported
either "it works" or "it does not", depending which one was run.

### What the numbers mean against the domain's own ceilings

Measured directly, on all 48 items, at temperature 0.0:

| a prompt that… | scores |
|---|---|
| is the starting instruction | 0.000 |
| names the integer-cents convention | 0.479 |
| also tells the model to work the arithmetic out and end with a final answer line | **1.000** |

The two successful seeds land at 0.375 and 0.438 — just under the *format-only*
ceiling. So the search reliably discovers **what the grader wants** (an integer
number of cents, stated nowhere in the problems) and does not discover that it is
**allowed to think first**. The remaining headroom is a single idea, and no
operator found it in 80 rollouts.

!!! note "Four things had to be fixed before any of this measured the algorithm"
    Each produced a run that completed, reported a number, and was wrong in a way
    nothing asserted.

    **The port was not Algorithm 1.** Its archive only grew, so no unit ever
    died; its tournament ranked on the held-out split, which is the acceptance
    gate's own signal and makes the tournament a second gate rather than a
    fitness measure; and three of the nine operators ran in a fixed rotation,
    which *guarantees* each operator's share where sampling does not. See the
    table above.

    **Hypermutation stopped halfway.** The paper writes a new mutation-prompt and
    then *applies it to the task-prompt*. Step one alone leaves the task-prompt
    unchanged, so such a candidate cannot move the score by construction — one
    seed spent 29 of its 80 draws on exactly that and finished at 0.000.

    **The grader forbade a chain of thought.** `parse_integer_answer` matched the
    whole reply, so the output convention and the reasoning were mutually
    exclusive: every method evolved prompts saying "no explanation", and without
    the working the arithmetic collapsed. Same model, same convention, all 48
    items: no working scored 0.562, *with* working scored **0.000** because the
    working itself failed the match, and with working under a final-line grader
    scored 0.979. A correct answer scored zero for having shown its arithmetic.

    **The feedback then described the old grader.** After the fix the reply's
    body was free, and the feedback still implied otherwise — so the search was
    being optimised against a false statement about its own channel.

    Two hypotheses were tested and **refuted** on the way, which is why they are
    named here rather than in a fix: that `--reflective-merge` was blending the
    eight workers' discoveries into mush (a no-merge control scored the same
    0.000), and that `finalize()` was picking the wrong unit off a 4-item batch
    (the whole population scored 0.000 — the maximum was a tie among zeros).

### Cost

Roughly 630 model calls per seed. The budget check passes in every run
(`observed_proposal_calls <= expected`), with hypermutation's second step
declared: `proposal_calls_per_candidate = 2`.
