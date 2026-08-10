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

## Measured results — GSM8K

Three seeds, `async_pipeline`, 80 rollouts each, 8 workers, `--staleness full`,
`--reflective-merge`, `deepseek-v4-flash` at temperature 0.7. Recorded in
[`bench/results/promptbreeder-gsm8k.json`](https://github.com/Birfy/agentdescent/blob/main/bench/results/promptbreeder-gsm8k.json).

| seed | test quality | validation | accepted | calls | wall |
|---|---|---|---|---|---|
| 0 | 0.562 → **0.984** | 0.547 → 0.969 | 2/80 | 1480 | 589 s |
| 1 | 0.375 → **0.953** | 0.391 → 0.906 | 3/80 | 1486 | 816 s |
| 2 | 0.484 → **0.969** | 0.531 → 0.984 | 3/80 | 1367 | 699 s |

Mean final **0.969**, mean gain **+0.495** — the largest of the ported methods
on this benchmark, and five times the noise floor below.

64 items per split from GSM8K's own train and test splits; the baseline is the
model's own accuracy rather than the 0.000 the previous fixture installed by
construction. See
[Self-Refine](algo-self-refine.md#measured-results-gsm8k) for why that fixture
was replaced.

!!! warning "The noise floor *for this configuration* is ±0.09, and [Reflexion](algo-reflexion.md) measured it"
    Reflexion accepted **nothing** on any of its three GSM8K seeds, so its
    artifact never changed — which makes each row's "baseline" and "final" two
    evaluations of the *same* instruction. They differ by **+0.094, −0.062 and
    −0.078**.

    That is what re-scoring one instruction on 64 items at temperature 0.7
    costs, and it is the scale every gain **on this page** should be read
    against: PromptBreeder runs the same `solve_gsm8k` wrapper on the same
    benchmark, so the figure transfers. A gain of +0.05 here would be
    indistinguishable from having changed nothing.

    It is **not** a constant of the study. Re-scoring one instruction five times
    under the `policy_prompt` wrapper on GSM-Hard moved it by 0.02 (sd 0.021) —
    four times tighter. The wrapper that suppresses working also suppresses the
    variance in it, so a floor measured under one prompt shape does not license a
    verdict under another. [SICA](algo-sica.md#measured-results-gsm-hard) reports
    gains against the tighter figure because that is the one its own
    configuration produces.

## Run it

```bash
python -m examples.promptbreeder.promptbreeder_genetic_prompts --dry-run

# the table above, one seed of the three (0, 1, 2)
python -m examples.promptbreeder.promptbreeder_genetic_prompts --yes --seed 0 \
    --budget-rollouts 80 --workers 8 \
    --async --async-ratio 1 --max-seconds 3600 \
    --staleness full --temperature 0.7 --no-thinking \
    --provider claude --model deepseek-v4-flash
```

**`--async-ratio 1` is what this row ran at.** The flag was declared by the
shared parser and never passed to `run_port`, so the run took the runner's own
default of 1 while
[`bench/results/promptbreeder-gsm8k.json`](https://github.com/Birfy/agentdescent/blob/main/bench/results/promptbreeder-gsm8k.json)
recorded the 2 its command line had asked for. The flag is threaded now and that
file records 1 — see
[the MethodPolicy command line](self-evolution-examples.md#the-methodpolicy-command-line)
for why the default here is 1 rather than the shared 3.

No `--reflective-merge`: the method's own `reflective` declaration set the merge,
and that declaration is a fidelity statement rather than a knob. `--max-seconds`
is the one setting the results file does not record; any value comfortably above
the row's `engine_s` leaves `--budget-rollouts` as the binding stop.

Offline tests: `tests/test_promptbreeder_algorithm1.py`, `tests/test_candidate_methods.py`.
