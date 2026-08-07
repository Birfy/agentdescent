# SkillWeaver — Web agent API synthesis

**Fidelity class: `environment_analogue`** — see [port fidelity](port-fidelity.md) for
what the classes mean. This is a candidate-method port: the mechanism is
preserved and measured under AgentDescent's runtimes; it is **not** a
paper-benchmark reproduction.

| | |
|---|---|
| Paper | "SkillWeaver: Web Agents can Self-Improve by Discovering and Honing Skills", Zheng et al., 2025 ([arXiv:2504.07079](https://arxiv.org/abs/2504.07079)) |
| Upstream code (pinned) | [OSU-NLP-Group/SkillWeaver@f2a63d65](https://github.com/OSU-NLP-Group/SkillWeaver/tree/f2a63d65d0f6ff46ac30e817cede8797f8f25b97) |
| Definition | [`examples/skillweaver/skillweaver_web_apis.py`](https://github.com/Birfy/agentdescent/blob/main/examples/skillweaver/skillweaver_web_apis.py) |
| Domain | deterministic settings web service (12 form tasks, disjoint splits) |

## The mechanism

SkillWeaver's pipeline has **three stages**: Skill Proposal (an LLM curriculum
proposes skills to practice), Skill Synthesis (practice the task, judge success
with an LLM reward model, synthesize the trajectory into a tested Python API),
and Skill Honing (unit-test the API, generate test parameters, **patch it when
execution throws**). The product is a growing library of plug-and-play APIs.

## Where each piece lives

| Upstream mechanism | Where it lives here |
|---|---|
| Proposal stage | task pool + `DifficultyWeighted` sampling |
| Practice + reward model | the engine rollout + the `self_verify` re-roll graded by the deterministic site |
| Honing from execution failures | the site simulator reports the first failed call; the HONE prompt sees that, never a required trace |
| Growing API library | `SkillLibrary` per-page keys; the reusable placeholder API evolves under the generic key |

## Boundaries

- A deterministic settings service replaces Dockerized WebArena.
- Key-match retrieval replaces the paper's API-doc retrieval.

## Measured results

*Pending: this section is populated from the live matrix
(`bench/results/candidate-methods-framework-final.json`) after the
post-restructuring rerun. See the
[candidate-method overview](candidate-results-overview.md) for the matrix-wide
tables (quality, [parallel speedup](candidate-parallel-speedup.md), and
[async behaviour](candidate-async.md)).*

| Mode | Quality (test, before → after) | E2E seconds | Engine seconds | TTQ |
|---|---|---|---|---|
| serial | *TBD* | *TBD* | *TBD* | *TBD* |
| sync_parallel | *TBD* | *TBD* | *TBD* | *TBD* |
| async_pipeline | *TBD* | *TBD* | *TBD* | *TBD* |
