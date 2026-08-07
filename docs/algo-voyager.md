# Voyager — Embodied skill-library agent

**Fidelity class: `environment_analogue`** — see [port fidelity](port-fidelity.md) for
what the classes mean. This is a candidate-method port: the mechanism is
preserved and measured under AgentDescent's runtimes; it is **not** a
paper-benchmark reproduction.

| | |
|---|---|
| Paper | "Voyager: An Open-Ended Embodied Agent with Large Language Models", Wang et al., 2023 ([arXiv:2305.16291](https://arxiv.org/abs/2305.16291)) |
| Upstream code (pinned) | [MineDojo/Voyager@55e45a88](https://github.com/MineDojo/Voyager/tree/55e45a880755d0c8c66ca7fb5fe7962ac8974f89) |
| Definition | [`examples/voyager/voyager_skill_library.py`](https://github.com/Birfy/agentdescent/blob/main/examples/voyager/voyager_skill_library.py) |
| Domain | deterministic crafting world (12 recipe goals, disjoint splits) |

## The mechanism

Voyager grows an **add-only library** of executable skills (Chroma-indexed by
description embedding, name collisions versioned, top-5 retrieval), driven by a
generative curriculum that proposes novel frontier tasks from the agent's
state. Failed programs are repaired from **environment feedback, interpreter
errors, and a separate GPT-4 critic's critique** — up to four rounds, never
from a gold program.

## Where each piece lives

| Upstream mechanism | Where it lives here |
|---|---|
| Add-only skill library | `SkillLibrary`: per-goal keys that accumulate and union-merge; the reusable placeholder skill evolves under the generic key |
| Repair from environment errors | a deterministic crafting simulator reports the first failed step; that message is all the repair prompt sees |
| The critic | the engine's `self_verify` rollout: every proposal re-runs and is judged by the environment reward |
| Curriculum at the frontier | `DifficultyWeighted` sampling over the task pool |

## Boundaries

- A deterministic crafting world replaces Minecraft.
- Key-match retrieval replaces embedding retrieval.
- The task pool + difficulty sampling is an analogue of the generative curriculum, which proposes novel tasks.

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
