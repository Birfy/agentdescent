# Self-Refine — Iterative feedback refinement

**Fidelity class: `mechanism_microport`** — see [port fidelity](port-fidelity.md) for
what the classes mean. This port is measured in the runtime matrix: the mechanism is
preserved and measured under AgentDescent's runtimes; it is **not** a
paper-benchmark reproduction.

| | |
|---|---|
| Paper | "Self-Refine: Iterative Refinement with Self-Feedback", Madaan et al., 2023 ([arXiv:2303.17651](https://arxiv.org/abs/2303.17651)) |
| Upstream code (pinned) | [madaan/self-refine@9a206d41](https://github.com/madaan/self-refine/tree/9a206d41e5d2d0c241bb441f41eeadb945afaa55) |
| Definition | [`examples/self_refine/self_refine_feedback_loop.py`](https://github.com/Birfy/agentdescent/blob/main/examples/self_refine/self_refine_feedback_loop.py) |
| Domain | deterministic integer-cents arithmetic (12 tasks, disjoint splits) |

## The mechanism

One model plays generator, feedback provider, and refiner: GENERATE an answer,
FEEDBACK on it (a separate call), REFINE from the critique — iterated on the
same instance up to four times, stopping early when the feedback contains the
stop signal (the pinned gsm runner checks for the literal "it is correct").
No training of any kind.

## Where each piece lives

| Upstream mechanism | Where it lives here |
|---|---|
| FEEDBACK and REFINE as separate calls | a two-call proposal (`proposal_calls_per_candidate=2`) |
| The stop signal | feedback containing "it is correct" ends the refinement — the reserved call budget becomes an upper bound |
| Same-instance iteration | rollout → proposal → held-out rerun |

## Boundaries

- Fundamental analogue: upstream refines the *answer* to one instance; this port refines the *instruction artifact*.

## Measured results

Three seeds, `async_pipeline`, 80 rollouts each, 8 workers, `--staleness full`,
`--reflective-merge`, `deepseek-v4-flash` at temperature 0.7. Recorded in
[`bench/results/self-refine-fused-call.json`](https://github.com/Birfy/agentdescent/blob/main/bench/results/self-refine-fused-call.json).

| seed | test quality | validation | accepted | calls |
|---|---|---|---|---|
| 0 | 0.000 → **1.000** | 0.000 → 1.000 | 2/80 | 395 |
| 1 | 0.000 → **0.938** | 0.000 → 1.000 | 3/80 | 400 |
| 2 | 0.000 → **1.000** | 0.000 → 1.000 | 2/80 | 416 |

Mean 0.979 against a ceiling of 1.000, and validation reaches 1.000 on all three
seeds. This row clears the domain.

### Against the other mechanisms

Same 48 items, same model, same 80-rollout budget:

| | mean test | calls per seed |
|---|---:|---:|
| **Self-Refine** | **0.979** | ~400 |
| [AFlow](algo-aflow.md) | 0.896 | ~750 |
| [Reflexion](algo-reflexion.md) | 0.354 | ~460 |
| [PromptBreeder](algo-promptbreeder.md) | 0.271 | ~630 |

The best row is also the cheapest, and the two facts have the same cause. Every
Self-Refine candidate costs **one** model call — the fused critique-and-refine
that the fidelity fix below restored — where AFlow spends two calls per *rollout*
on its two workflow nodes. Fixing the fidelity halved the cost and did not trade
anything for it.


!!! note "These cross-algorithm figures demonstrate that each port runs, not which is best"
    Every row is one run per seed, and the seed fixes the data splits and the
    method's own sampler -- **not the model**, which is sampled at temperature
    0.7. Re-running an identical command at an identical seed moves the number:
    PromptBreeder's seed 0 scored 0.438 on one run and 0.875 on the next, from
    the same script and the same code. Read the table as evidence the mechanism
    executes end to end and produces a plausible artifact; a ranking would need
    repeats per seed, which these runs do not have.
!!! danger "FEEDBACK and REFINE are one call in the task whose domain this is"
    `GSMFeedback.__call__` makes a **single** request and splits the completion
    on a marker: `entire_output.split("def solution():")`, prose before is the
    critique, code after is the improved solution. `iterative_gsm` then checks
    the **critique half** for the stop signal.

    Six of the repository's seven tasks *do* have a separate `task_iterate.py`
    REFINE module. `gsm` does not — and `gsm` is the arithmetic-word-problem task
    this port's domain matches, and the one the port takes its stop signal from.
    The port had it both ways: two calls, citing gsm.

    An offline test was pinning the old shape.
    `test_dry_run_counts_two_call_proposals` asserted
    `reserved proposal calls=24`, a number that is only reachable when
    `self_refine` declares two calls per candidate.

    The critique prompt now carries worked examples of the fused shape, as
    `data/prompt/gsm/feedback.txt` carries four. Without them the model does not
    emit the marker and the refinement half comes back empty — the same failure
    [Reflexion](algo-reflexion.md) had, where a reflector with nothing to imitate
    answered the arithmetic instead of planning.

!!! note "The stop signal is stronger here than upstream's"
    `iterative_gsm` breaks on `"it is correct" in feedback.lower()`, and that
    phrase appears **nowhere** in `data/prompt/gsm/`. Nothing teaches the model to
    emit it, so upstream's check never fires and its loop runs the full
    `max_attempts`.

    This port instructs the critique to say it when the attempt is already right,
    which makes the check real. That is the paper's stopping criterion working
    rather than upstream's vestigial check reproduced, and the notes say which —
    a run that stops early is spending less than a run that does not, and the
    reason belongs next to the number.
