# SkillWeaver — Web agent API synthesis

**Fidelity class: `environment_analogue`** — see [port fidelity](port-fidelity.md) for
what the classes mean. This port is measured in the runtime matrix: the mechanism is
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

Three seeds, `async_pipeline`, 80 rollouts each, 8 workers, `--staleness full`,
`self_verify` on (the reward model), `deepseek-v4-flash` at temperature 0.7.
Recorded in
[`bench/results/skillweaver-web-apis.json`](https://github.com/Birfy/agentdescent/blob/main/bench/results/skillweaver-web-apis.json).

| seed | test quality | validation | accepted | calls |
|---|---|---|---|---|
| 0 | 0.000 → **0.750** | 0.000 → 0.875 | 3/80 | 1137 |
| 1 | 0.000 → **0.688** | 0.000 → 0.938 | 2/80 | 1253 |
| 2 | 0.000 → **0.875** | 0.000 → 1.000 | 3/80 | 1121 |

Mean 0.771, all three seeds moved. Compare
[Voyager](algo-voyager.md#measured-results)'s 1.000 / 1.000 / 0.000 on the same
runtime and budget: this site *names the concepts* its API needs — "the page
hydrates before accepting input and confirms with a toast" — where Voyager's
world named neither the vessel nor its `X+Y` syntax. Even after both were made
learnable, the site that says more is the one whose seeds agree.

See the caveat on [PromptBreeder](algo-promptbreeder.md#measured-results-gsm8k): one
run per seed does not pin a number here either.

!!! danger "The site hinted the concepts and still demanded the exact tokens"
    Its message says the page hydrates and confirms with a toast. Under string
    equality it required `wait:hydration-complete` and `assert:saved-toast`
    exactly, so every one of these did the right thing and was refused:

    | written | refused because |
    |---|---|
    | `wait:hydration`, `wait:page-hydrated` | not the exact token |
    | `assert:toast`, `assert:success-toast` | not the exact token |
    | `fill:timezone = UTC` | spaces around the equals sign |

    Steps are matched on **verb plus content** now, declared once in `_STEPS`: a
    verb whose argument the site only gestures at is matched on the verb, and a
    verb whose argument comes from the task — the page, the field, the value —
    must carry it. The wrong page, the wrong field and the wrong value all still
    fail.

!!! danger "Missing, misplaced and wrong-argument were one message"
    The site reported *"a 'fill' step never succeeded"* to an agent that had
    written a `fill` in the wrong place, and the same words to one whose `fill`
    carried the wrong value. Three failures, three repairs: an agent told a step
    is missing writes another one, and an agent told a wrong value is out of
    order reorders it. Neither ever fixes what is actually wrong — the loop that
    held Voyager at 0.000 across three seeds.

    The site now separates *absent* from *present but late* from *present with
    an argument the page did not act on*, and still names neither the expected
    argument nor the steps to come.

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

    The domain was also 12 tasks in 4/4/4 splits, which `run_port` refuses at
    eight workers; it is 48 in 16/16/16 now.

## Run it

```bash
python -m examples.skillweaver.skillweaver_web_apis --dry-run

# the table above, one seed of the three (0, 1, 2)
python -m examples.skillweaver.skillweaver_web_apis --yes --seed 0 \
    --budget-rollouts 80 --workers 8 \
    --async --async-ratio 1 --max-seconds 3600 \
    --staleness full --temperature 0.7 --no-thinking \
    --provider claude --model deepseek-v4-flash
```

**`--async-ratio 1` is what this row ran at.** The flag was declared by the
shared parser and never passed to `run_port`, so the run took the runner's own
default of 1 while
[`bench/results/skillweaver-web-apis.json`](https://github.com/Birfy/agentdescent/blob/main/bench/results/skillweaver-web-apis.json)
recorded the 2 its command line had asked for. The flag is threaded now and that
file records 1 — see
[the MethodPolicy command line](self-evolution-examples.md#the-methodpolicy-command-line)
for why the default here is 1 rather than the shared 3.

No `--reflective-merge`: the method's own `reflective` declaration set the merge,
and that declaration is a fidelity statement rather than a knob. `--max-seconds`
is the one setting the results file does not record; any value comfortably above
the row's `engine_s` leaves `--budget-rollouts` as the binding stop.

Offline tests: `tests/test_skillweaver_upstream.py`.
