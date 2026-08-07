# Runtime matrix — live report

!!! warning "This report measures the pre-restructuring implementation"
    The numbers below were produced by the implementation as of source
    fingerprint `381b663…` — before the ports moved to their declarative
    `MethodPolicy` form. The mechanisms and budgets are the same by design,
    but the code has since changed and the matrix is pending a rerun; see the
    [overview](matrix-overview.md).

## Scope

This is a live glm-5.2 experiment over 11 methods, 3 execution modes, and 3 paired seeds (99 observations). It uses no response replay and no synthetic latency.

The comparison changes AgentDescent scheduling, not the candidate or proposal-call budget:

- `serial`: `evolve(max_concurrency=1)`.
- `sync_parallel`: `evolve(max_concurrency=workers)` with its round barrier.
- `async_pipeline`: `async_evolve(...)` with completion-order merge sweeps.

A value above 1.0x means the comparison mode is faster. Every speedup is paired by method and seed.

Port author: `cyanneko`.

## Main results

| Method | Fidelity | Serial quality | Sync quality | Async quality | Serial E2E | Sync E2E | Async E2E | Sync / serial | Async / sync |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PromptBreeder | `mechanism_microport` | 0.000 -> 0.750 | 0.000 -> 1.000 | 0.000 -> 0.750 | 133.66 | 98.33 | 97.31 | 1.41x (n=3) | 1.01x (n=3) |
| AFlow | `mechanism_microport` | 0.000 -> 1.000 | 0.000 -> 0.500 | 0.000 -> 0.750 | 243.25 | 171.15 | 175.03 | 1.39x (n=3) | 0.99x (n=3) |
| Reflexion | `mechanism_microport` | 0.000 -> 1.000 | 0.000 -> 1.000 | 0.000 -> 1.000 | 122.43 | 91.42 | 101.49 | 1.32x (n=3) | 0.90x (n=3) |
| Self-Refine | `mechanism_microport` | 0.000 -> 1.000 | 0.000 -> 1.000 | 0.000 -> 1.000 | 126.79 | 95.70 | 103.17 | 1.31x (n=3) | 0.92x (n=3) |
| Voyager | `environment_analogue` | 0.000 -> 1.000 | 0.000 -> 1.000 | 0.000 -> 1.000 | 150.94 | 102.79 | 121.18 | 1.47x (n=3) | 0.86x (n=3) |
| SkillWeaver | `environment_analogue` | 0.000 -> 1.000 | 0.000 -> 1.000 | 0.000 -> 1.000 | 169.38 | 118.66 | 118.88 | 1.42x (n=3) | 1.01x (n=3) |
| Absolute Zero | `inference_analogue` | 0.000 -> 0.000 | 0.000 -> 0.000 | 0.000 -> 0.000 | 280.26 | 192.94 | 185.27 | 1.45x (n=3) | 1.03x (n=3) |
| R-Zero | `inference_analogue` | 0.000 -> 0.000 | 0.000 -> 0.000 | 0.000 -> 0.000 | 307.18 | 215.80 | 224.76 | 1.38x (n=3) | 1.03x (n=3) |
| Agent0 | `inference_analogue` | 0.750 -> 0.750 | 0.250 -> 1.000 | 0.250 -> 0.750 | 315.84 | 262.86 | 265.90 | 1.20x (n=3) | 1.00x (n=3) |
| SICA | `self_edit_analogue` | 0.000 -> 0.500 | 0.000 -> 0.250 | 0.000 -> 0.250 | 114.74 | 93.07 | 97.34 | 1.23x (n=3) | 0.96x (n=3) |
| Godel Agent | `self_edit_analogue` | 0.000 -> 1.000 | 0.000 -> 1.000 | 0.000 -> 1.000 | 129.95 | 93.16 | 98.05 | 1.36x (n=3) | 1.01x (n=3) |

E2E columns are median seconds including disjoint baseline and final tests. Quality cells are median strict test reward before -> after.

## Framework timing

| Method | Serial engine | Sync engine | Async engine | Sync / serial | Async / sync |
|---|---:|---:|---:|---:|---:|
| PromptBreeder | 82.74 | 42.41 | 39.11 | 1.94x (n=3) | 0.94x (n=3) |
| AFlow | 138.43 | 73.41 | 66.27 | 1.89x (n=3) | 1.11x (n=3) |
| Reflexion | 72.74 | 39.73 | 45.88 | 1.81x (n=3) | 0.87x (n=3) |
| Self-Refine | 72.69 | 39.52 | 43.69 | 1.82x (n=3) | 0.99x (n=3) |
| Voyager | 85.61 | 41.54 | 53.20 | 2.02x (n=3) | 0.78x (n=3) |
| SkillWeaver | 108.73 | 54.03 | 57.38 | 1.96x (n=3) | 0.95x (n=3) |
| Absolute Zero | 124.10 | 67.86 | 66.54 | 1.91x (n=3) | 0.98x (n=3) |
| R-Zero | 179.88 | 89.23 | 90.63 | 1.92x (n=3) | 0.97x (n=3) |
| Agent0 | 117.10 | 86.33 | 86.54 | 1.80x (n=3) | 1.00x (n=3) |
| SICA | 62.42 | 37.46 | 36.75 | 1.66x (n=3) | 1.11x (n=3) |
| Godel Agent | 76.76 | 40.25 | 50.24 | 1.79x (n=3) | 0.88x (n=3) |

Engine columns isolate the framework evolution window; E2E remains the user-visible completion time.

## Aggregate timing

- Sync vs serial end-to-end: **1.36x (n=33)**.
- Async vs sync end-to-end: **0.99x (n=33)**.
- Sync vs serial engine window: **1.89x (n=33)**.
- Async vs sync engine window: **0.97x (n=33)**.
- Sync vs serial time-to-quality: **1.47x (n=18)**.
- Async vs sync time-to-quality: **1.10x (n=15)**.

Full min / median / max intervals are in the JSON rather than being hidden behind a point estimate.

## Interpretation

- Sync parallel has a median end-to-end win for **11/11 methods**; across all paired method/seeds it is 1.36x (n=33) faster end-to-end and 1.89x (n=33) faster inside the framework evolution window.
- Async is not a general full-return speedup: its aggregate end-to-end result is 0.99x (n=33) and its engine-window result is 0.97x (n=33) relative to sync parallel.

## Time to quality

| Method | Serial TTQ | Sync TTQ | Async TTQ | Async / sync TTQ |
|---|---:|---:|---:|---:|
| PromptBreeder | 126.01 (1/3) | 83.91 (2/3) | 114.15 (1/3) | 0.76x (n=1) |
| AFlow | 218.35 (2/3) | 155.21 (1/3) | 143.19 (1/3) | -- |
| Reflexion | 111.58 (3/3) | 78.66 (3/3) | 68.45 (3/3) | 1.12x (n=3) |
| Self-Refine | 112.99 (2/3) | 77.90 (2/3) | 63.73 (3/3) | 1.24x (n=2) |
| Voyager | 133.87 (3/3) | 87.41 (3/3) | 87.62 (3/3) | 1.01x (n=3) |
| SkillWeaver | 152.62 (3/3) | 100.79 (3/3) | 82.09 (3/3) | 1.25x (n=3) |
| Absolute Zero | -- (0/3) | -- (0/3) | -- (0/3) | -- |
| R-Zero | -- (0/3) | -- (0/3) | -- (0/3) | -- |
| Agent0 | 135.63 (3/3) | 215.50 (3/3) | -- (0/3) | -- |
| SICA | 126.50 (1/3) | -- (0/3) | 70.19 (1/3) | -- |
| Godel Agent | 118.58 (3/3) | 76.92 (3/3) | 75.08 (3/3) | 1.02x (n=3) |

Repeated-seed async TTQ gains above 1.05x: SkillWeaver (1.25x, n=3), Self-Refine (1.24x, n=2), Reflexion (1.12x, n=3).

No mode has a positive median independent-test quality gain for: Absolute Zero, R-Zero. Timing results for these methods measure fixed-budget execution efficiency only.

`--` is intentional: a method that did not cross the fixed internal target has no TTQ, even if its final independent test improved.

## Cost and integrity

- Provider calls: **3852**.
- Framework actor calls: **1558**.
- Algorithm proposal calls: **288**.
- Logical calls from event traces: **3852**.
- Tokens: **501288**.
- Provider failures: **0**.
- Candidate/proposal budget mismatches: **0**.

| Mode | Runs | Provider calls | Actor calls | Proposal calls | Rollouts | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| `serial` | 33 | 1260 | 492 | 96 | 66 | 163472 |
| `sync_parallel` | 33 | 1260 | 488 | 96 | 66 | 164999 |
| `async_pipeline` | 33 | 1332 | 578 | 96 | 106 | 172817 |

- Candidate-method source SHA-256: `381b663017555d2b052af12597adba26a9adfd9521b294e6b3d0b05e2641f37a`.
- Raw prompts, responses, generated tasks, learned instructions, and generated source are intentionally absent from the result file.

## Fidelity boundaries

| Method | Upstream reference | What this experiment preserves | Boundary |
|---|---|---|---|
| PromptBreeder | [paper (no official released code)](https://arxiv.org/abs/2309.16797) | task/mutation prompt co-evolution and fitness selection | compact population and local arithmetic domain |
| AFlow | [FoundationAgents/AFlow@3f457218](https://github.com/FoundationAgents/AFlow/tree/3f457218fc716093fe53f6df8a5d5e6379d66346) | execution-feedback graph expansion and workflow evaluation | one MCTS depth and two model nodes |
| Reflexion | [noahshinn/reflexion@218cf0ef](https://github.com/noahshinn/reflexion/tree/218cf0ef1df84b05ce379dd4a8e47f17766733a0) | attempt, external feedback, verbal reflection, retry | deterministic evaluator instead of HotpotQA/ALFWorld |
| Self-Refine | [madaan/self-refine@9a206d41](https://github.com/madaan/self-refine/tree/9a206d41e5d2d0c241bb441f41eeadb945afaa55) | GENERATE, FEEDBACK, REFINE | one refinement iteration on a compact rubric |
| Voyager | [MineDojo/Voyager@55e45a88](https://github.com/MineDojo/Voyager/tree/55e45a880755d0c8c66ca7fb5fe7962ac8974f89) | automatic curriculum, executable skills, repair, critic | deterministic crafting world instead of Minecraft |
| SkillWeaver | [OSU-NLP-Group/SkillWeaver@f2a63d65](https://github.com/OSU-NLP-Group/SkillWeaver/tree/f2a63d65d0f6ff46ac30e817cede8797f8f25b97) | Propose, Practice, Verify, Hone and reusable APIs | deterministic settings site instead of WebArena |
| Absolute Zero | [LeapLabTHU/Absolute-Zero-Reasoner@484afa48](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner/tree/484afa480c8f6fd77faa3d35451f24f287f58ee1) | proposer, solver, grounded verifier, self-play curriculum | verbal policy memory instead of PPO weight updates |
| R-Zero | [Chengsong-Huang/R-Zero@5699329d](https://github.com/Chengsong-Huang/R-Zero/tree/5699329d018d79535b7910abdedf5a6eebf355fd) | separate Challenger/Solver roles and separate updates | verbal role memories instead of two-model GRPO |
| Agent0 | [aiming-lab/Agent0@f775b510](https://github.com/aiming-lab/Agent0/tree/f775b5101e62fe92976831adf4a21a38fcc0a767) | curriculum/executor co-evolution and multi-turn tools | calculator environment and memory instead of RL post-training |
| SICA | [MaximeRobeyns/self_improving_coding_agent@ed8275dc](https://github.com/MaximeRobeyns/self_improving_coding_agent/tree/ed8275dca4d3c5dbf77229964351fe9b424797dc) | real Python self-edit and measured utility gate | one AST-gated policy function instead of SWE-bench Docker |
| Godel Agent | [Arvid-pku/Godel_Agent@bbb50879](https://github.com/Arvid-pku/Godel_Agent/tree/bbb508796be31c7140cdfc7106efd830a1324242) | artifact-owned solve and recursive self-improvement functions | AST-gated replacement instead of full monkey-patched scaffold |

These fidelity labels are part of the result, not a disclaimer added after seeing the scores. Environment and inference analogues must not be cited as reproductions of paper benchmark numbers.

## Data provenance

Only observations whose result records AgentDescent `evolve` or `async_evolve` are included. Earlier hand-scheduled pilots are excluded from every table and aggregate.

- `candidate-methods-framework-calibration-all.json (source run; not retained in the repository)`: seeds 0; 33 completed observations.
- `candidate-methods-framework-seeds-100-200.json (source run; not retained in the repository)`: seeds 100, 200; 66 completed observations.

## Reproduction

```bash
python -m bench.candidate_methods --provider openai --model glm-5.2 \
  --workers 2 --candidates 2 --repeats 1 --seed 0 \
  --modes serial sync_parallel async_pipeline \
  --thinking disabled --temperature 0.0 --max-tokens 1024 \
  --output candidate-methods-framework-calibration-all.json (source run; not retained in the repository) --yes
python -m bench.candidate_methods --provider openai --model glm-5.2 \
  --workers 2 --candidates 2 --repeats 2 --seed 100 \
  --modes serial sync_parallel async_pipeline \
  --thinking disabled --temperature 0.0 --max-tokens 1024 \
  --output candidate-methods-framework-seeds-100-200.json (source run; not retained in the repository) --yes
python -m bench.candidate_methods_merge --inputs candidate-methods-framework-calibration-all.json (source run; not retained in the repository) candidate-methods-framework-seeds-100-200.json (source run; not retained in the repository) \
  --expected-seeds 0 100 200 --output bench/results/candidate-methods-framework-final.json
python -m bench.candidate_methods_report --input bench/results/candidate-methods-framework-final.json
```

The runner reads `OPENAI_API_KEY` and `OPENAI_BASE_URL` from the environment, rotates mode order, writes each paid observation atomically, and stops if the implementation fingerprint changes.

## Limits

- 3 paired seeds show an observed spread, not a confidence interval or a paper-scale result.
- Temperature zero does not make a hosted model or network deterministic.
- Final quality differences between modes can arise from completion order, stale candidates, and model variance; they are not evidence that parallelism improves reasoning quality.
- The principal supported claim is timing: equal candidate/proposal work can overlap; framework gate calls are measured because async can perform a different number of merge sweeps.
- Async helps TTQ only when a useful completion can commit before a sync barrier; it need not improve full-return E2E time.

Machine-readable source: `bench/results/candidate-methods-framework-final.json`.
