# Reflexion — Verbal reinforcement / episodic memory

**Fidelity class: `mechanism_microport`** — see [port fidelity](port-fidelity.md) for
what the classes mean. This port is measured in the runtime matrix: the mechanism is
preserved and measured under AgentDescent's runtimes; it is **not** a
paper-benchmark reproduction.

| | |
|---|---|
| Paper | "Reflexion: Language Agents with Verbal Reinforcement Learning", Shinn et al., 2023 ([arXiv:2303.11366](https://arxiv.org/abs/2303.11366)) |
| Upstream code (pinned) | [noahshinn/reflexion@218cf0ef](https://github.com/noahshinn/reflexion/tree/218cf0ef1df84b05ce379dd4a8e47f17766733a0) |
| Definition | [`examples/reflexion/reflexion_episodic_memory.py`](https://github.com/Birfy/agentdescent/blob/main/examples/reflexion/reflexion_episodic_memory.py) |
| Domain | **GSM-Hard** ([`reasoning-machines/gsm-hard`](https://huggingface.co/datasets/reasoning-machines/gsm-hard)), 64/64/64 shuffled splits |

## The mechanism

After a failed attempt, Reflexion converts the trajectory and the external
evaluator's signal into a **verbal reflection**, appends it to an episodic
memory, and retries the same task with that memory in context. The memory is
append-only and bounded to the last Ω entries (Ω=1–3 in the paper;
`memory[-3:]` in the pinned alfworld runs).

## Where each piece lives

| Upstream mechanism | Where it lives here |
|---|---|
| Bounded append-only memory | `WindowedMemory`: commit-ordered content-addressed keys, rendered as the last 3 entries |
| Reflect on external feedback | the proposal call, fed the strict evaluator's feedback |
| Retry with memory | the engine's held-out rerun |
| Parallel reflections | appends never contradict, so they union-merge with no ranking evaluation |

## Boundaries

- Upstream retries the same failed instance; the held-out rerun is the analogue.
- The equal-budget design requests a reflection after every rollout, not only on failure.

## Measured results — GSM-Hard

Three seeds, `async_pipeline`, 80 rollouts each, 8 workers, `--staleness full`,
`--reflective-merge`, `deepseek-v4-flash` at temperature 0.7. Recorded in
[`bench/results/reflexion-gsmhard.json`](https://github.com/Birfy/agentdescent/blob/main/bench/results/reflexion-gsmhard.json).

| seed | test quality | validation | accepted | invalid | calls |
|---|---|---|---|---|---|
| 0 | 0.656 → **0.688** | 0.406 → 0.469 | 1/80 | 2 | 445 |
| 1 | 0.562 → **0.547** | 0.484 → 0.531 | 1/80 | 4 | 434 |
| 2 | 0.531 → **0.562** | 0.547 → 0.609 | 1/80 | 4 | 515 |

**Validation rises on every seed (+0.047 to +0.063) and test barely moves
(+0.016 mean, one seed negative).** That gap is the port's actual finding, and
it is the honest answer to the question this port asks: a reflection written
from one failure does raise the score on the split the gate judges, and
transfers to unseen questions weakly at best. Reflexion's memory is per
*instance* upstream and its whole move is retrying **that** instance, so a
shared memory asked to generalise is a question the paper does not ask.
`--per-instance` runs the faithful variant and is expected to accept nothing.

!!! danger "On GSM8K this row accepted 0 of 80, three seeds running"
    The earlier measurement, kept in
    [`bench/results/reflexion-gsm8k.json`](https://github.com/Birfy/agentdescent/blob/main/bench/results/reflexion-gsm8k.json),
    read 0.797 → 0.891, 0.750 → 0.688, 0.734 → 0.656 with `accepted=0/80` and
    `invalid=0` every time. Nothing was committed, so each pair is **two
    evaluations of one unchanged instruction** — a noise measurement, not a
    result.

    Two causes, both fixed here rather than argued away.

    **The domain produced almost no failures.** Held-out sat at 0.75–0.80, so
    four rollouts in five succeeded and wrote nothing — a 32-candidate probe
    yielded 7 reflections. Worse, the failures that did occur shared no cause;
    each was an idiosyncratic misreading, and a memory read against *other*
    questions had nothing to carry. GSM-Hard is the same 1319 questions with
    large numbers substituted, where held-out starts at 0.41–0.55 and failures
    concentrate on one cause — arithmetic done in the model's head — which is a
    failure mode a transferable rule can address.

    **Two reflections in seven were a bare number.** The probe's memory received
    the strings `624` and `48`. The prompt shows the failed question and ends in
    `New plan:`, so answering it is a live continuation however firmly the
    instructions forbid it, and `WindowedMemory` is bounded, so each such entry
    displaced a real plan. `WindowedMemory` now takes a `validator`;
    `is_a_plan` requires six alphabetic words, which is a **shape** floor and not
    a quality bar — whether a plan that clears it is any good is the held-out
    gate's question. The `invalid` column above is that check firing 10 times
    across three seeds.

!!! warning "Its baseline is not comparable with the other ports'"
    `WindowedMemory.render` emits `MEMORY_HEADER` **even when the memory is
    empty**, so this port's seed artifact reads:

    > Solve the grade-school math word problem. Return only the final answer.
    > \# Plans from past attempts. You have attempted problems like this before
    > and failed; these plans say how to avoid failing the same way. Use them to
    > improve your strategy (most recent last).
    > (empty)

    Every other port starts from the first line alone, and that header is itself
    an instruction to be careful. Reflexion did not start where they started, so
    its *gain* is not theirs to compare against — only its final score is.

!!! danger "Without the worked examples, the reflector answered the arithmetic"
    `_generate_reflection_query` prepends `FEW_SHOT_EXAMPLES` under *"Here are
    two examples:"* — two failed trajectories each followed by its `New plan:`.
    A first version of this port matched upstream's **wording** and dropped that
    **structure**, and the run scored 0.125 / 0.000 / 0.062.

    Measured over 40 reflections with no examples:

    | | |
    |---|---|
    | reflections generated | 40 |
    | that mentioned the output convention | **1** |
    | the first four, verbatim | `925`, `264`, `465`, `$32.15` |
    | entries in the final memory | **0** |

    The query ends in `New plan:` and contains a question and its answer, so with
    nothing to imitate the likelier continuation is simply to answer it. The
    entries were then useless, nothing cleared the acceptance gate, the memory
    stayed empty, and every rollout saw the bare seed instruction. Adding the
    examples took the mean from 0.062 to 0.354.

    The first draft of those examples used `m03` and `m11` — real domain items,
    which land in held-out or test depending on the seed, so every reflection
    prompt in the run handed over two graded answers.
    `test_the_examples_do_not_hand_over_a_held_out_answer` caught it; the items
    are invented now.

## Run it

```bash
python -m examples.reflexion.reflexion_episodic_memory --dry-run

# the table above, one seed of the three (0, 1, 2)
python -m examples.reflexion.reflexion_episodic_memory --yes --seed 0 \
    --budget-rollouts 80 --workers 8 \
    --async --async-ratio 1 --max-seconds 3600 \
    --staleness full --temperature 0.7 --no-thinking \
    --provider claude --model deepseek-v4-flash
```

**`--async-ratio 1` is what this row ran at.** The flag was declared by the
shared parser and never passed to `run_port`, so the run took the runner's own
default of 1 while
[`bench/results/reflexion-gsmhard.json`](https://github.com/Birfy/agentdescent/blob/main/bench/results/reflexion-gsmhard.json)
recorded the 2 its command line had asked for. The flag is threaded now and that
file records 1 — see
[the MethodPolicy command line](self-evolution-examples.md#the-methodpolicy-command-line)
for why the default here is 1 rather than the shared 3.

No `--reflective-merge`: the method's own `reflective` declaration set the merge,
and that declaration is a fidelity statement rather than a knob. `--max-seconds`
is the one setting the results file does not record; any value comfortably above
the row's `engine_s` leaves `--budget-rollouts` as the binding stop.

`--per-instance` is off, which is the measured arrangement: one shared memory
across the run, the departure from upstream this port's notes declare. Pass it
for the paper's per-instance variant.

Offline tests: `tests/test_reflexion_upstream.py`.
