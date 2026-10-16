# Dynamic Cheatsheet (DC-Cu) — test-time Gen + Cur memory

Port onto the AgentDescent engine, as a declarative
[`MethodPolicy`](../_method_policy.py).

| | |
|---|---|
| Kind | Test-time memory, no labels (the ACE precursor baseline) |
| Fidelity class | `mechanism_microport` |
| Paper | "Dynamic Cheatsheet: Test-Time Learning with Adaptive Memory", Suzgun et al., 2025 ([arXiv:2504.07952](https://arxiv.org/abs/2504.07952)) |
| Upstream code | [suzgunmirac/dynamic-cheatsheet@5cfe3c3](https://github.com/suzgunmirac/dynamic-cheatsheet/tree/5cfe3c37e8e52b1d858d0f3df46e7f17c50991b9) |
| Domain (compact) | 36 vendored Game-of-24 puzzles (12 / 12 / 12), offline |
| Definition plug-ins | `CheatsheetSlot` strategy (whole-value rewrite); `CommitEveryUpdate` acceptance (online) or the held-out gate (`--gated`); `blast_radius=0.2`; online-only `cheap_eval_tasks` (whole held-out split) and `test_time_learning` (sequential Gen → Cur test stream) |

## Run

```bash
python -m examples.dynamic_cheatsheet.dynamic_cheatsheet_cumulative --dry-run
python -m examples.dynamic_cheatsheet.dynamic_cheatsheet_cumulative --provider openai --model glm-5.2
python -m examples.dynamic_cheatsheet.dynamic_cheatsheet_cumulative --gated   # ACE-like offline mode
```

`--dry-run` prints the configuration and returns with **zero network access
and no API key**.

## What is in here

- [`dynamic_cheatsheet_cumulative.py`](dynamic_cheatsheet_cumulative.py) — the `MethodPolicy` definition and runnable entry point
- [`game24.py`](game24.py) — the vendored puzzle slice and its exact grader
- Mechanism notes, upstream trace, and every recorded boundary: [`docs/algo-dynamic-cheatsheet.md`](../../docs/algo-dynamic-cheatsheet.md)
- Offline tests: [`tests/test_dynamic_cheatsheet.py`](../../tests/test_dynamic_cheatsheet.py)

Limits worth knowing (details in the docs page): online mode reports the *streamed* test
quality (the frozen score is recorded beside it) and the stream starts from the
engine-evolved cheatsheet rather than `(empty)`; `--gated` scores a frozen artifact; the
two runtime vetoes that would make "online" a gate (L1 audit, low-trust audit) are
closed by `blast_radius=0.2` and `cheap_eval_tasks`.

DC-RS / retrieval (embeddings) is out of scope; [ACE](../ace) is the structured successor.
