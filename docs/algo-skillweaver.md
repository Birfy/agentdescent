# SkillWeaver — Web agent API synthesis

> **API-library self-evolution.** Propose, practise, verify and hone reusable
> web APIs into a growing library. Runs through the shared
> [`MethodPolicy`](policies.md) runner with `self_verify` on (the reward model).
> Example:
> [`examples/skillweaver/skillweaver_web_apis.py`](https://github.com/Birfy/agentdescent/blob/main/examples/skillweaver/skillweaver_web_apis.py).

| | |
|---|---|
| **Paper** | *SkillWeaver: Web Agents can Self-Improve by Discovering and Honing Skills* — Zheng et al., 2025 ([arXiv:2504.07079](https://arxiv.org/abs/2504.07079)) |
| **Upstream code** | [OSU-NLP-Group/SkillWeaver@f2a63d65](https://github.com/OSU-NLP-Group/SkillWeaver/tree/f2a63d65d0f6ff46ac30e817cede8797f8f25b97) |
| **Example** | [`examples/skillweaver/skillweaver_web_apis.py`](https://github.com/Birfy/agentdescent/blob/main/examples/skillweaver/skillweaver_web_apis.py) |
| **Domain** | deterministic settings web service — 48 form tasks, 16/16/16 splits |
| **Layer** | L1 (`blast_radius=0.6`, set by the shared runner) |
| **Fidelity** | `environment_analogue` — [what the classes mean](port-fidelity.md) |

This port is measured in the [runtime matrix](matrix-overview.md): the mechanism
is preserved and measured under AgentDescent's runtimes; it is **not** a
paper-benchmark reproduction.

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

## Measured results — settings site

Three seeds, `async_pipeline`, 80 rollouts each, 8 workers, `--staleness full`,
reflective merge **off** and `self_verify` on (both this method's own
declaration), `deepseek-v4-flash` at temperature 0.7.
Recorded in
[`bench/results/skillweaver-web-apis.json`](https://github.com/Birfy/agentdescent/blob/main/bench/results/skillweaver-web-apis.json).

| seed | test quality | validation | accepted | calls |
|---|---|---|---|---|
| 0 | 0.000 → **0.750** | 0.000 → 0.875 | 3/80 | 1137 |
| 1 | 0.000 → **0.688** | 0.000 → 0.938 | 2/80 | 1253 |
| 2 | 0.000 → **0.875** | 0.000 → 1.000 | 3/80 | 1121 |

Mean 0.771, all three seeds moved. Compare
[Voyager](algo-voyager.md#measured-results-crafting-world)'s 1.000 / 1.000 / 0.000
on the same runtime and budget: **this site names the concepts its API needs** —
"the page hydrates before accepting input and confirms with a toast" — where
Voyager's world names neither. The environment that says more is the one whose
seeds agree, which is worth knowing before reading either as a fact about the
algorithm.

See the caveat on [PromptBreeder](algo-promptbreeder.md#measured-results-gsm8k): one
run per seed does not pin a number here either.

!!! note "Two departures that are not just \"a deterministic service replaces WebArena\""
    **The success check is a model upstream and the environment here.**
    `check_success_simple` asks a separate LM (`success_check_lm`, gpt-4o) to
    judge the trajectory and a screenshot. A model critic errs in both
    directions; the deterministic site cannot. This port therefore has a
    *cleaner* reward than the paper, not merely a cheaper one.

    **Upstream separates exploring from testing on a schedule.**
    `_should_perform_test` alternates the two, and `update` shows the synthesis
    model only functions with `test_count > 0` (`is_tested`). Verification is a
    scheduled phase over the library there, and a per-proposal re-roll here.

## Run it

```bash
python -m examples.skillweaver.skillweaver_web_apis --dry-run

# one seed of the three above
python -m examples.skillweaver.skillweaver_web_apis --yes --seed 0 \
    --provider openai --model deepseek-v4-flash \
    --async --async-ratio 1 --workers 8 --budget-rollouts 80 --staleness full \
    --temperature 0.7 --max-seconds 3600
```

`--async-ratio 1` is what this row ran at: the flag was dropped before it
reached the runtime, so the run took the runner's default whatever the
command line said. It is passed explicitly here because the value matters and
a default can move.

Flags: [the MethodPolicy command line](self-evolution-examples.md#the-methodpolicy-command-line).

Offline tests: `tests/test_skillweaver_upstream.py`,
`tests/test_candidate_methods.py`.
