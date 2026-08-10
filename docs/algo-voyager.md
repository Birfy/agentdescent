# Voyager — Embodied skill-library agent

> **Skill-library self-evolution.** Grow a library of executable skills from a
> curriculum, repairing failed programs from environment feedback. Runs through
> the shared [`MethodPolicy`](policies.md) runner with `self_verify` on (the
> critic). Example:
> [`examples/voyager/voyager_skill_library.py`](https://github.com/Birfy/agentdescent/blob/main/examples/voyager/voyager_skill_library.py).

| | |
|---|---|
| **Paper** | *Voyager: An Open-Ended Embodied Agent with Large Language Models* — Wang et al., 2023 ([arXiv:2305.16291](https://arxiv.org/abs/2305.16291)) |
| **Upstream code** | [MineDojo/Voyager@55e45a88](https://github.com/MineDojo/Voyager/tree/55e45a880755d0c8c66ca7fb5fe7962ac8974f89) |
| **Example** | [`examples/voyager/voyager_skill_library.py`](https://github.com/Birfy/agentdescent/blob/main/examples/voyager/voyager_skill_library.py) |
| **Domain** | deterministic crafting world — 48 recipe goals, 16/16/16 splits |
| **Layer** | L1 (`blast_radius=0.6`, set by the shared runner) |
| **Fidelity** | `environment_analogue` — [what the classes mean](port-fidelity.md) |

This port is measured in the [runtime matrix](matrix-overview.md): the mechanism
is preserved and measured under AgentDescent's runtimes; it is **not** a
paper-benchmark reproduction.

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

## Measured results — crafting world

Three seeds, `async_pipeline`, 80 rollouts each, 8 workers, `--staleness full`,
reflective merge **off** and `self_verify` on (both this method's own
declaration — the library overwrites a key, so a synthesised merge would not be
the algorithm), `deepseek-v4-flash` at temperature 0.7. Recorded
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

!!! note "Upstream's library overwrites; it is not add-only"
    `SkillManager.add_new_skill` prints *"Skill {name} already exists.
    Rewriting!"*, deletes the vector-store entry and reassigns
    `self.skills[program_name]`. The older code is dumped to disk as
    `{name}V2.js` and **retrieval never reads it** — `retrieve_skills` returns
    `self.skills[...]["code"]`. Versioned on disk and overwritten in memory are
    different libraries, and only the second one is the algorithm — which is also
    why this port declares `reflective=False`.

!!! warning "Seed 2 finds nothing, and the world is the reason to check first"
    What has to be discovered here is the **sequence**, which is the skill the
    paper is about — but a world that names none of its own vocabulary makes that
    undiscoverable rather than hard. Steps are matched on **verb plus content**:
    an argument the world never names is matched on the verb alone, and one that
    follows from the goal must contain it. Feedback separates a step that is
    **absent** from one that is **late**, because an agent told a step is missing
    writes another one and never reorders. Both were needed before any seed
    solved this world at all; one still does not.

## Run it

```bash
python -m examples.voyager.voyager_skill_library --dry-run

# one seed of the three above
python -m examples.voyager.voyager_skill_library --yes --seed 0 \
    --provider openai --model deepseek-v4-flash \
    --async --async-ratio 1 --workers 8 --budget-rollouts 80 --staleness full \
    --temperature 0.7 --max-seconds 3600
```

`--async-ratio 1` is what this row ran at: the flag was dropped before it
reached the runtime, so the run took the runner's default whatever the
command line said. It is passed explicitly here because the value matters and
a default can move.

Flags: [the MethodPolicy command line](self-evolution-examples.md#the-methodpolicy-command-line).

Offline tests: `tests/test_voyager_upstream.py`,
`tests/test_candidate_methods.py`.
