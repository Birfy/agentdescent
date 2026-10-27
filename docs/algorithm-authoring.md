---
description: Register one named Algorithm card in a single PR — the short path for plan / init / hosts, not a full research port.
---

# Register a named Algorithm card (one PR)

Usability pillar 2: **quickly implement a new algorithm**.

This page is the short path for someone who only wants to ship **one named
card** that `plan` / `init` / hosts can call — without reading the whole
[porting checklist](porting-checklist.md) (kept ≤20 lines on purpose) or
[MethodPolicy checklist](porting-methodpolicy.md).

| You want… | Use… |
|---|---|
| A name in `agentdescent algorithms list` / `"algorithm": "…"` | **This page** |
| A full `examples/<name>/` research port | [porting-checklist.md](porting-checklist.md) |
| A `MethodPolicy` microport in the matrix | [porting-methodpolicy.md](porting-methodpolicy.md) |

The catalog of shipped cards is generated from the registry:
[named algorithms](algorithms.md).

## What a card is

An `Algorithm` (`agentdescent.algorithms`) is a **name**, a one-line
summary, the `kind`s it applies to, a fidelity label, typed `Param`s, and a
`build(ctx, params) -> Overrides` function. Everything it contributes is a
*default*: an explicit `policies` / `agg_config` / `evolve` field in the same
spec still wins. Fidelity is a label, not a number promise — see
[port fidelity](port-fidelity.md):

* `preset` — an engine component under an algorithm name (no claim to be the
  paper's whole method). Prefer this for a first card.
* `mechanism_microport` — the published mechanism, wired into this engine.
* `benchmark_faithful` — reproduces the published setup and its numbers
  (almost always a full `examples/` port, not a one-file card).

## Checklist (one PR)

Copy an existing small card, register it, expose params / kinds / fidelity,
add a thin doc page, and a changelog fragment. That is the whole PR.

1. **Copy a pattern.** Smallest shipped preset:
   [`archive_selection` in `agentdescent/algorithms/builtin.py`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/algorithms/builtin.py)
   (parent sampling only). Slightly richer mechanism card:
   [`ace.py`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/algorithms/ace.py).
   A fill-in scaffold lives at
   [`agentdescent/algorithms/_TEMPLATE.py`](https://github.com/Birfy/agentdescent/blob/main/agentdescent/algorithms/_TEMPLATE.py)
   (not imported by the package — copy it, rename it, then wire the import).
2. **Implement `build`.** Return
   `Overrides(policies=…, knobs=…, agg_config=…, notes=…)`. Policy slot names
   match [`Policies`](policies.md) (`selection`, `acceptance`, `proposal`, …).
3. **`register(Algorithm(...))`** at module import time with a unique `name`,
   `kinds`, `fidelity`, `params`, `source`, `extra_calls`, and optional
   `doc="algo-<name>.md"`.
4. **Wire the import** so the card loads with the shipped registry:
   * In-package: add `from . import my_algo  # noqa: E402,F401` at the bottom of
     `agentdescent/algorithms/builtin.py` (same pattern as `ace` / `gepa`).
   * Third-party package: expose the `Algorithm` under the
     `agentdescent.algorithms` entry-point group (name must match
     `Algorithm.name`):

     ```toml
     [project.entry-points."agentdescent.algorithms"]
     my_method = "my_pkg:ALGORITHM"
     ```
5. **Algo page skeleton** (when `doc=` is set): add `docs/algo-<name>.md` with a
   short lead, Paper / Source / Spec / Fidelity table, what the card changes,
   and how to select it. Keep it thin — this is not a full port write-up.
6. **Changelog fragment:** `changelog.d/<issue>-<slug>.md` (never edit
   `CHANGELOG.md`). See `changelog.d/README.md` in the repository root.
7. **Regenerate the catalog** after the card registers:

   ```bash
   python -m tools.gen_algorithms_docs
   ```

   CI fails if `docs/algorithms.md` drifts from the registry.

## Worked example — `archive_selection`

This preset only swaps the **selection** policy. The rest of the kind's recipe
(proposals, merge, acceptance) stays the engine default. That is the right
shape for a first card.

```python
def _build_archive(ctx, p):
    from agentdescent.selection import Archive
    archive = Archive(sampling=p["sampling"], temperature=p["temperature"],
                      seed=p["seed"])
    return Overrides(
        policies={"selection": archive},
        notes=["algorithm archive_selection: only the parent-selection rule "
               "changes; proposals, the merge and the acceptance gate are "
               "the engine's."])

register(Algorithm(
    name="archive_selection",
    summary="Parent sampling from an archive of every accepted candidate, "
            "tempered by novelty (DGM's rule by default) instead of always "
            "extending the head.",
    kinds=("text", "skill_dir", "agent_dir", "agent_code", "plugin"),
    fidelity="preset",
    params=(
        Param("sampling", "str", "sigmoid_novelty", "...",
              choices=("sigmoid_novelty", "novelty", "performance",
                       "uniform", "best")),
        Param("temperature", "float", 1.0, "...", minimum=1e-6),
        Param("seed", "int", 0, "seed of the parent draw"),
    ),
    build=_build_archive,
    source="Darwin Gödel Machine (choose_selfimproves), ADAS archive, SICA",
    extra_calls="none"))
```

(Full source: `builtin.py`. Spec form: `"algorithm": "archive_selection"` or
`{"name": "archive_selection", "params": {"seed": 1}}`.)

## Verify before you open the PR

```bash
pip install -e ".[dev]"
agentdescent algorithms list
agentdescent algorithms show YOUR_NAME
agentdescent algorithms show YOUR_NAME --json
# point plan at a tiny text spec that names the algorithm:
agentdescent plan path/to/spec.json --algorithm YOUR_NAME
pytest -q tests/test_algorithms.py tests/test_algorithm_entry_points.py
python -m tools.gen_algorithms_docs --check
mkdocs build --strict   # if you touched docs/ or mkdocs.yml
```

Acceptance for this path: a newcomer can add a named algorithm and see it in
`agentdescent algorithms list` and in `plan` by following **only** this page.

## Out of scope here

* Growing the research catalog itself ([#384](https://github.com/Birfy/agentdescent/issues/384)).
* Full MethodPolicy / `examples/` microports.
* Engine changes (`evolve()`, aggregators, hosts).
