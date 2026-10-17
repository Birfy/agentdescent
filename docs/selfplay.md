# Solver-challenger self-play

`curriculum_evolve()` co-evolves the solver and its own task distribution. A
second artifact -- the **challenger** -- owns a task-generation spec; the
solver must keep learning, and the challenger must keep proposing questions at
the solver's frontier: hard enough to be worth asking, solvable enough that
learning pays, never so degenerate that a valid answer does not exist.

This is the GoDoors / R-Zero idea -- a *learned* objective rather than a fixed
one -- made concrete for harness evolution, with the regularizers this project
insists on:

* **Two-phase learnability, measured online.** A generated task's reward is not
  "does the solver answer it" but "does the solver *learn* it": its first vs
  last solver score across attempts. A task the solver already passes is
  dropped on its first attempt (it was never on the frontier); one it never
  learns is dropped as stuck, not kept because it happened to be hard.
* **The challenger never touches the final evaluation.** Generated tasks enter
  the *train* pool through the sampler; `evolve`'s held-out split is computed
  from the caller's seed tasks and the challenger cannot add to it. What a run
  reports is always measured on tasks the challenger never generated.
* **A validity gate is mandatory.** `Challenger.validate` raises on any payload
  that is not a well-formed, answerable task; a generator that cannot produce a
  valid task is counted, never crashed over.
* **The default is the default.** `curriculum_evolve` with a challenger is
  strictly opt-in; the solver is a normal `evolve()` run driven by the engine,
  **one engine instance for the whole run**, so the aggregator's state
  (posteriors, archive) is never rebuilt per round.

## API

| Piece | What it is |
| --- | --- |
| `Challenger` | `generate(spec, seed) -> payload`, `validate(payload) -> Task` (the validity gate; raises on malformed tasks), `propose(spec, report) -> spec`. All plain callables, so offline fixtures are deterministic and live runs can use a model. |
| `FrontierSampler` | A `TaskSampler` that mixes seed ids with frontier `Task` objects (`frontier_share`), tracks each frontier task's `first`/`last` solver score, and self-regulates the pool. |
| `curriculum_evolve(...)` | Co-evolves the solver (normal `evolve` arguments) with a `challenger=`; runs the engine once; returns `EvolutionResult` with `.curriculum` set. |

The engine change behind it is small and general: a sampler may return a
`Task` object instead of an id, resolved without a pool lookup (synchronous and
barrier-free paths).

## How the round works

1. **Challenge** -- the challenger's current spec generates `generated_per_round`
   valid tasks (invalid payloads are counted, not crashed).
2. **Solve** -- the solver rolls out over seed tasks plus the frontier; the
   sampler records each frontier task's first and last score.
3. **Judge** -- after the round, tasks the solver now passes are dropped and
   counted *learned*; tasks that never learned within `max_attempts` are
   dropped as *stuck*; `r0 = mean(last - first)` is the learnability report.
4. **Tune** -- `Challenger.propose(spec, report)` may raise the difficulty
   (or keep it), and the next frontier is generated.

## Governance

Generated tasks only ever enter the train pool. `held_out_frac` is applied to
`seed_tasks` alone, so `final_reward` and every acceptance gate measure tasks
the challenger never generated -- the same final-evaluation isolation every
other development set in the framework gets.

## Example

```bash
python -m examples.selfplay.selfplay_demo --dry-run
```

Deterministic, offline, no key: the solver's skill carries a capability cap
(`cap:N`); the challenger's generated carts push the cap; the measured
learnability (`r0`) drives the challenger's difficulty. With a model:

```bash
python -m examples.selfplay.selfplay_demo --model deepseek-v4-flash
```
