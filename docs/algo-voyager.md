# Voyager — Embodied skill-library agent

**Fidelity class: `environment_analogue`** — see [port fidelity](port-fidelity.md) for
what the classes mean. This port is measured in the runtime matrix: the mechanism is
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

Three seeds, `async_pipeline`, 80 rollouts each, 8 workers, `--staleness full`,
`self_verify` on (the critic), `deepseek-v4-flash` at temperature 0.7. Recorded
in
[`bench/results/voyager-skill-library.json`](https://github.com/Birfy/agentdescent/blob/main/bench/results/voyager-skill-library.json).

| seed | test quality | validation | accepted | calls |
|---|---|---|---|---|
| 0 | 0.000 → **1.000** | 0.000 → 1.000 | 2/80 | 1207 |
| 1 | 0.000 → **1.000** | 0.000 → 1.000 | 1/80 | 1207 |
| 2 | 0.000 → 0.000 | 0.000 → 0.000 | 0/80 | 1287 |

Two seeds clear the world outright and one finds nothing. `accepted` is 2, 1 and
0 of eighty, which is the shape of the mechanism rather than an accident: every
proposal is re-rolled by the critic before it reaches the gate, and one skill
that works is all the run needs.

See the caveat on [PromptBreeder](algo-promptbreeder.md#measured-results-gsm8k): one
run per seed does not pin a number here either.

!!! danger "Three seeds scored 0.000 against a world that could not be solved"
    The first runs reported `quality 0.000 -> 0.000`, `accepted=0/80`, and
    **`invalid=0`** on every seed. Well-formed skills, none of them accepted.
    Reading the trajectories rather than the summary showed the solver had
    already discovered all three missing steps, and was failing on things the
    world never said.

    **Two arguments the world never named.** The required sequence contains
    `combine:water+mint` and `sanitize:vessel`, matched by string equality. The
    `X+Y` syntax appears in no primitive list, no seed skill and no feedback
    message; neither does the word "vessel". The solver wrote
    `combine:water_clove`, `combine:berry:water`, `combine:thyme_water` — every
    plausible spelling but that one — and `sanitize:water`, `sanitize:berry`.
    Steps are matched on **verb plus content** now: an argument the world does
    not name is matched on the verb alone, and an argument that follows from the
    goal must still contain it.

    **A feedback message that was wrong, not merely terse.** With `sanitize`
    written after the two `collect`s, the world consumed the sanitize, looked for
    a collect among the actions *after* it, found none, and reported *"a
    'collect' step never happened"* — to an agent that had written two. An agent
    that believes it adds a third collect; it never reorders. The message now
    separates a step that is **absent** from one that is **late**, while still
    naming neither the step nor where it belongs.

    Not handing over the required program and withholding what the world observed
    are different things, and upstream conflates neither: Minecraft tells the
    agent what it is missing. What has to be discovered here is still the
    **sequence**, which is the skill the paper is about.

!!! note "The library overwrites; it is not add-only"
    This page used to claim "an add-only skill library ... matching upstream's
    versioned, never-overwritten library". `SkillManager.add_new_skill` prints
    *"Skill {name} already exists. Rewriting!"*, deletes the vector-store entry
    and reassigns `self.skills[program_name]`. The older code is dumped to disk
    as `{name}V2.js` and **retrieval never reads it** — `retrieve_skills` returns
    `self.skills[...]["code"]`. Versioned on disk and overwritten in memory are
    different libraries, and only the second one is the algorithm.

    The domain was also 12 goals in 4/4/4 splits, which `run_port` refuses at
    eight workers; it is 48 in 16/16/16 now.
