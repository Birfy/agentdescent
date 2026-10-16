---
description: A runnable microport of Dynamic Cheatsheet (DC-Cu), test-time memory with a Generator and a label-free Curator, the baseline ACE improves on. Vendored offline Game-of-24 slice.
---

# Dynamic Cheatsheet (DC-Cu) — test-time memory, the ACE precursor

> **Memory self-evolution, no labels.** A black-box model keeps one persistent,
> self-curated *cheatsheet* of strategies and snippets across test queries,
> without weight updates and without ground-truth labels. This is the baseline
> [ACE](algo-ace.md) cites and improves on with an explicit Reflector and
> structured delta bullets. Runs through the shared [`MethodPolicy`](policies.md)
> runner. Example:
> [`examples/dynamic_cheatsheet/dynamic_cheatsheet_cumulative.py`](https://github.com/Birfy/agentdescent/blob/main/examples/dynamic_cheatsheet/dynamic_cheatsheet_cumulative.py).

| | |
|---|---|
| **Paper** | *Dynamic Cheatsheet: Test-Time Learning with Adaptive Memory* — Suzgun, Yuksekgonul, Bianchi, Jurafsky, Zou, 2025 ([arXiv:2504.07952](https://arxiv.org/abs/2504.07952)) |
| **Upstream code** | [suzgunmirac/dynamic-cheatsheet@5cfe3c3](https://github.com/suzgunmirac/dynamic-cheatsheet/tree/5cfe3c37e8e52b1d858d0f3df46e7f17c50991b9) — `DynamicCheatsheet_Cumulative` |
| **Example** | [`examples/dynamic_cheatsheet/dynamic_cheatsheet_cumulative.py`](https://github.com/Birfy/agentdescent/blob/main/examples/dynamic_cheatsheet/dynamic_cheatsheet_cumulative.py) |
| **Domain** | **Game of 24** — 36 vendored puzzles (four integers in 1–13, each solvable), 12 / 12 / 12 train / held-out / test, fully offline |
| **Layer** | L2 skill (`blast_radius=0.2`, declared on the `MethodPolicy`) |
| **Fidelity** | `mechanism_microport` — [what the classes mean](port-fidelity.md) |

This port is a **mechanism microport**, not a paper-benchmark reproduction: the
domain is a compact vendored Game-of-24 slice, not upstream's AIME / GPQA /
Game-of-24 sets.

## The mechanism

DC-Cu is two completions per test query, in order:

1. **Gen** — answer the query `x_i` with the current cheatsheet `M_i` in
   context.
2. **Cur** — a second completion reads `(M_i, x_i, y_i)` and writes the
   **whole** next cheatsheet `M_{i+1}`.

The curator sees the question and *the model's own answer* `y_i`. It never
sees a gold label and never sees the grader's verdict: judging whether the
answer was any good, and what is worth remembering, is the model's job. That is
what makes it *test-time learning* rather than supervised prompt tuning. The
cheatsheet starts as `(empty)` and is rewritten wholesale — there are no delta
bullets, which is exactly the gap ACE fills (and why DC can suffer the context
collapse ACE is built to avoid).

## Where each piece lives

| Upstream mechanism | Where it lives here |
|---|---|
| Generator prompt with `[[CHEATSHEET]]` | `GENERATOR_TEMPLATE`; `solve` always passes the current rendered cheatsheet |
| Curator prompt (`[[PREVIOUS_CHEATSHEET]]`, `[[QUESTION]]`, `[[MODEL_ANSWER]]`) | `CURATOR_TEMPLATE`; `propose` — **no slot for a gold answer or a reward** |
| `extract_cheatsheet` (`<cheatsheet>…</cheatsheet>`) | `extract_cheatsheet`, called from `CheatsheetSlot.to_diff` |
| `cheatsheet = new_cheatsheet` after every query | `CommitEveryUpdate` acceptance (the default) |
| `cheatsheet = "(empty)"` | `INITIAL_CHEATSHEET` |
| `eval_for_GameOf24` | `examples/dynamic_cheatsheet/game24.py` (exact `Fraction` AST evaluation, no `eval`) |

```python
evolve(tasks, reward, run=gen, propose=curator,
       strategy=CheatsheetSlot(initial_value="(empty)", key="cheatsheet"),
       blast_radius=0.2, policies=Policies(acceptance=CommitEveryUpdate()))
```

* `strategy=` — one plain-text value, replaced by each curator rewrite.
* `propose=` — the curator.
* `acceptance=` — **online** by default (every rewrite kept); `--gated` installs
  the framework's held-out gate instead.

## Online versus gated

Online test-time learning is not hill-climbing on a validation split, and this
is where the port diverges from `evolve()`'s usual strict gate:

| | `acceptance` | A rewrite that drops useful content |
|---|---|---|
| default (**online**, upstream) | `CommitEveryUpdate` | kept (see the warning below for the two vetoes that had to be closed) |
| `--gated` (offline, ACE-like) | the framework default | rejected: held-out reward falls |

The default is the faithful one. `--gated` is the ACE-like offline reading: the
cheatsheet is trained on one split and only kept if it helps the held-out one.

!!! warning "Why online mode needs three runner fields, not one"
    Online DC-Cu is only online if nothing between the curator and the ledger
    can veto a rewrite. The runtime has two vetoes that sit **in front of** the
    acceptance policy, and an accept-everything `AcceptancePolicy` reaches
    neither:

    1. **L1 commit audit by blast radius.** Above `governance.FAST_MAX` (0.3)
       every commit is audited against the full held-out set and returns
       `oracle-rejected` unless it improved. The shared runner's default of 0.6
       is therefore a gate whatever policy is installed. `MethodPolicy` gained a
       `blast_radius` field (default 0.6, so every other port is unchanged) and
       the cheatsheet — skill-layer context, as ACE's playbook — declares 0.2.
    2. **Low-trust audit.** The audit also opens at *any* blast radius once its
       trust in the cheap ranking layer drops below 0.75, and trust halves
       (1.0 → 0.5) the first time the cheap sample and the full held-out set
       disagree about a candidate's direction. A rewrite that helps the
       engine's 8-puzzle cheap sample and hurts the 12-puzzle split is then
       `oracle-rejected` before `CommitEveryUpdate` is consulted.
       `MethodPolicy.cheap_eval_tasks` (default `None`, the engine's own 8)
       closes it: online mode sets it to the whole held-out split, the two
       layers become one measurement (the evaluation cache returns identical
       numbers), they cannot disagree, and the audit never opens.
       `test_engine_default_cheap_layer_vetoes_an_online_rewrite_on_disagreement`
       pins the engine behaviour this works around, and
       `test_full_held_out_cheap_layer_keeps_the_rewrite_online` the fix.
    3. **The test split is frozen by default.** The shared runner scores the
       test split with a fixed artifact, so no test answer would be curated and
       task *i+1* would not see what task *i* taught — a frozen memory trained
       on another split, not the Gen → Cur protocol. `MethodPolicy` gained an
       opt-in `test_time_learning` (default `False`); online mode sets it, and
       the runner then scores the test split as an ordered stream: solve,
       score, `propose`, apply the diff to an in-memory copy, next task.

    `test_the_shared_runner_really_is_ungated_online_and_gated_with_the_flag`
    pins both acceptance modes in all three schedulers.

!!! note "What the test stream does and does not measure"
    The standard `baseline_quality` / `final_quality` (and therefore
    `quality_gain`) stay **frozen-artifact** scores, so they compare the same
    protocol as every other port. The streamed score lives only under
    `framework.test_stream.quality`, with `frozen_quality` beside it, so the
    gain from continued learning is a difference inside that block, not a claim
    about the engine. The stream starts from the cheatsheet
    the engine evolved on the train / held-out splits, **not** from `(empty)` as
    upstream's single test stream does: it is transfer followed by online
    learning. It is strictly sequential whatever `--workers` says, uses its own
    `stream:` phases (the proposal budget check is unchanged), never writes to
    the ledger, and counts its refused or unchanged updates separately from
    `invalid` candidates. `--gated` is the offline reading and scores a frozen
    artifact. Only strategies whose diffs are key → value replacements can
    stream.

## Boundaries

- **DC-RS / DC-CumulativeRetrieval are out of scope.** Retrieving the top-k past
  `(x, y)` pairs needs an embedding / retrieval seam; this port has **no
  embedding dependency** and is DC-Cu only. It is the natural follow-up.
- **No code execution.** Upstream lets the generator emit `EXECUTE CODE!` and
  runs it; its headline Game-of-24 effect is the model learning to *write a
  brute-force program and reuse it*. Here snippets are text in the cheatsheet
  and nothing runs them, so the mechanism is preserved and the mechanism's
  strongest payoff is not.
- **A missing `<cheatsheet>` block is refused, not replaced.** Upstream returns
  the *old* cheatsheet; the [porting policy](porting-methodpolicy.md) forbids
  fallback substitution, so the proposal costs its candidate and is counted
  `invalid`.
- **The test stream starts from the evolved cheatsheet, not `(empty)`**, and the
  engine's train / held-out phase runs first; see the note above.
- **Parallel curators are merged.** Upstream is one sequential stream. Under
  `evolve()` several workers rewrite one base at once; contradictions go
  through the shared reflective merge (one model call, tag-free shape check).
- The 36-puzzle slice is not upstream's Game-of-24 set, and a single model run
  on 12 test puzzles moves in steps of 0.083.
- ACE is the structured successor: [`algo-ace.md`](algo-ace.md).

## Measured results — Game of 24

**No live-model run is recorded for this port yet.** What exists is the
deterministic offline run the test-suite uses: a stub generator that solves a
puzzle only when its cheatsheet already holds the brute-force tip, and a stub
curator that writes it. It shows the plumbing, not a model's behaviour.

| mode | acceptance | test, before → after | validation | accepted | invalid | calls |
|---|---|---:|---:|---:|---:|---:|
| serial / sync / async | online (`CommitEveryUpdate`), streamed test | 0.000 → **1.000** | 0.000 → 1.000 | 1/2 | 0 | 88 |
| serial / sync / async | `--gated`, frozen test | 0.000 → **1.000** | 0.000 → 1.000 | 1/2 | 0 | 64 |

The two rows agree because the stub curator never writes a harmful rewrite; the
online row's 24 extra calls are the test stream (12 solves, 12 curations — the
stub rewrites the cheatsheet to the text it already holds, so all 12 updates are
"unchanged"). The tests that separate the modes feed it one that is
(`test_online_mode_keeps_every_rewrite_and_it_round_trips_the_ledger`,
`test_gated_mode_rejects_a_rewrite_that_empties_useful_content`).

## Run it

```bash
python -m examples.dynamic_cheatsheet.dynamic_cheatsheet_cumulative --dry-run

# online DC-Cu (every curator rewrite is kept)
python -m examples.dynamic_cheatsheet.dynamic_cheatsheet_cumulative --yes \
    --provider openai --model deepseek-v4-flash --workers 2 --candidates 8

# ACE-like offline mode: the held-out gate vets every rewrite
python -m examples.dynamic_cheatsheet.dynamic_cheatsheet_cumulative --yes --gated \
    --provider openai --model deepseek-v4-flash --workers 2 --candidates 8
```

`--dry-run` prints the plan and returns with **zero network access and no API
key**; the dataset is vendored, so `build` loads nothing from outside the
repository.

Offline tests: [`tests/test_dynamic_cheatsheet.py`](https://github.com/Birfy/agentdescent/blob/main/tests/test_dynamic_cheatsheet.py)
and the matrix test in `tests/test_candidate_methods.py`, which runs the method
in every scheduler without an API key.
