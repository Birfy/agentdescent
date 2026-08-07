# Candidate methods — results overview

*This page aggregates the live serial/sync/async matrix across all eleven
candidate ports. It is **pending the post-restructuring rerun** of
`python -m bench.candidate_methods`; every table below is populated from
`bench/results/candidate-methods-framework-final.json` by
`bench/candidate_methods_report.py`. The previous run's report (against the
pre-restructuring implementation) remains at
[Candidate-method runtime study](algo-candidate-methods.md) for provenance.*

## The eleven ports

| Method | Fidelity class | Mechanism seams |
|---|---|---|
| [PromptBreeder](algo-promptbreeder.md) | `mechanism_microport` | selection (binary tournament), FieldSlots genome |
| [AFlow](algo-aflow.md) | `mechanism_microport` | selection (soft mixed), per-parent experience |
| [Reflexion](algo-reflexion.md) | `mechanism_microport` | WindowedMemory (bounded append-only) |
| [Self-Refine](algo-self-refine.md) | `mechanism_microport` | two-call FEEDBACK→REFINE, stop signal |
| [Voyager](algo-voyager.md) | `environment_analogue` | SkillLibrary, DifficultyWeighted, self-verify critic |
| [SkillWeaver](algo-skillweaver.md) | `environment_analogue` | SkillLibrary, DifficultyWeighted, self-verify reward model |
| [Absolute Zero](algo-absolute-zero.md) | `inference_analogue` | frozen self-play evaluation, learnability signal |
| [R-Zero](algo-r-zero.md) | `inference_analogue` | AdvantageAcceptance (GRPO shape), DifficultyWeighted |
| [Agent0](algo-agent0.md) | `inference_analogue` | DifficultyWeighted, calculator stop-and-go |
| [SICA](algo-sica.md) | `self_edit_analogue` | AST gate, Archive selection |
| [Gödel Agent](algo-godel-agent.md) | `self_edit_analogue` | AST gate, optional gateless acceptance |

## Quality across the matrix

*TBD after rerun: per-method strict test reward before → after, per mode, with
per-mode target-reach rates and invalid-candidate counts. Baselines are
expected to be near zero by construction (the hidden output convention); with
fallback substitution removed, any gain is learned, not injected.*

| Method | Serial quality | Sync quality | Async quality | Invalid candidates |
|---|---:|---:|---:|---:|
| *TBD* | | | | |

## Merging behaviour

*TBD after rerun: fusion calls, union merges, conflict drops per mode — the
matrix now runs worker-sized merge batches, with reflective merge on
text-valued artifacts.*

See also: [parallel speedup](candidate-parallel-speedup.md) ·
[async behaviour](candidate-async.md)
