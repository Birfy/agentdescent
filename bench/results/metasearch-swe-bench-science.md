# SWE-bench-Science as a domain: what running it actually measures

Stage 2 of the meta-evolution plan (`docs/design-meta-evolution.md` §4.3) is
validating an evolved search rule on a 2026 frontier benchmark. That stage was
recorded as blocked on "no Docker daemon". **It was not blocked; the daemon was
merely not started.** Starting it exposed the real obstacles, which were four
bugs in the adapter and one in the benchmark's own published images.

This page is what one real task and fifteen real baselines measure, and what a
search on this domain can and cannot be told apart from noise. The dataset is
[`OpenMOSS-Team/SWE-bench-Science`](https://huggingface.co/datasets/OpenMOSS-Team/SWE-bench-Science):
119 tasks over 98 scientific repositories, Claude Code + Opus 5 pass@1 under 50%.

## The path works, and it is cheap

`task_001` — *"repair an inconsistent transition-state rotor workflow"*,
`Auto-Mech/autochem`, computational reaction chemistry — end to end through the
task's own published verifier image:

| | |
|---|---|
| one verification | **10.8 s** |
| baseline result | `reward 0`, `public 1/1`, `private 1/3` |
| repeated 3x | byte-identical metrics |

`task.toml` says `[agent] timeout_sec = 5400.0`, which reads like 90 minutes per
expansion. That is the *agent's* ceiling, not the cost: the public reproduction
runs in 6 s and the whole verifier in 11 s.

## Four bugs between the adapter and the benchmark

Each was found by running the thing, and each would have produced a clean,
plausible, meaningless null.

**The verifier is a separate image.** `[verifier].environment_mode = "separate"`:
the grader and the private tests live in a second image with its own copy of the
source. Mounting the public `tests/` over `/tests`, as the runner did, hides the
real grader — the public `test.sh` is only the entrypoint stub Harbor requires.

**The patch was being thrown away.** That image's `test.sh` opens with
`git reset --hard; git clean -ffdqx` and *then* applies
`/logs/artifacts/model.patch`. The runner applied the candidate to the workdir
itself, so the reset discarded it: **every candidate would have scored the
baseline**, and the search would have read a perfectly flat curve for a reason
that has nothing to do with the search. Now confirmed by a control edit that
moves `public.passed` 1 → 0.

**`reward.json` is written with `indent=2`.** Parsing the last *line* of the
verifier's output gets `}`. That raised `JSONDecodeError` on every task.

**The one fine-grained signal was being dropped.** Harbor requires a top-level
`reward`, and this grader writes it as `int(public_ok and private_ok)` — binary,
the coarsest reward there is. Its own partial credit is one level down:
`{"private": {"passed": 1, "collected": 3}}`. Flattening nested payloads makes
`private.passed` a selectable metric, and the difference is the whole
experiment: a binary reward cannot rank two failing patches, and 3 of 3, 5 of 10
or 17 of 31 can.

## A bug in the benchmark's images: a third of the sample could not be solved

**6 of the 15 verifier images sampled run
`pytest /tests/private_tests/test_task_NNN.py`, while the private test they ship
is named for its subject** — `test_res_export.py`, `test_state_consistency.py`,
`test_proforma_ion_semantics.py`. pytest exits 4 with *"file or directory not
found"*, the grader reports `private` 0 of 0, and **`reward` is 0 for any patch
whatsoever**. From the outside that is indistinguishable from a hard task nobody
solved.

The dataset's own `tests/grader.py` is the fixed version — it runs the whole
directory, with a comment saying the filenames are intentionally unconstrained.
So the images are stale, not the dataset, and mounting the published grader over
the image's repairs them:

| | before | after |
|---|---:|---:|
| images carrying a stale grader | 6 of 15 | — |
| baselines reporting `private 0 of 0` | **6 of 15** | **0 of 15** |
| tasks with any usable graded signal | 9 of 15 | **15 of 15** |
| tasks whose root is *partially* correct | 7 of 15 | **10 of 15** |

`task_030` goes from `0/0` to `5/10`; `task_015` from `0/0` to `17/31`.

## The four properties, measured

The checklist is [`metasearch-domain-selection.md`](metasearch-domain-selection.md).
Every previous real-data domain failed at least one of these.

| property | SWE-bench-Science | how it compares |
|---|---|---|
| **1. the run is a function of the candidate** | **exact.** Repeated verifications are byte-identical | AlgoTune: sd 0.054, structural. LLM-SRBench: only after raising a wall-clock budget |
| **2. the reward is fine-grained** | **3 to 31 levels** (median 10) once nested metrics are read | GSM-Hard: steps of 0.125, archive tied permanently |
| **3. candidates are incomparable** | a patch is not a chain — this is a tree search, not `evolve()`'s archive | the `selection` slot was inert because `argmax(score) == head` in 63/63 |
| **4. the search improves at an affordable budget** | **no** — 0 of 14 beat the root, on a weak model *and* on a strong one | this is where hyp2f1 and `lsr_transform` died, and it is the *only* thing wrong here |

Property 1 is the notable one. **This is the first real-data domain in this line
of work where the evaluator is deterministic by construction** rather than by
tuning a timeout out of the way: the verifier is a container running a fixed
test suite, and no timing enters the score or the prompt. AlgoTune's
irreproducibility was structural — the measured milliseconds *are* the feedback.
Here there is nothing of the kind to remove.

### Reward granularity across the 15 baselines

Private tests collected: `3, 3, 4, 6, 8, 8, 9, 10, 10, 10, 11, 11, 12, 15, 31`.

| root state | count | meaning |
|---|---:|---|
| **partially correct** (`0 < passed < all`) | **10** | a candidate can be *more* right than its parent without being wholly right |
| zero root (`passed == 0`) | 5 | any progress is invisible until the first test flips |

Baseline pass rates on the ten partial roots: 0.09, 0.13, 0.18, 0.33, 0.33,
0.38, 0.40, 0.40, 0.50, 0.55 — spread across the range rather than bunched at
either end.

**A count is not a score, and selecting one would have undone all of this.**
`harbor_domain`'s reward clamps to `[0, 1]`, so `private.passed` of 1 and of 31
both arrive as 1.0: every candidate that passes a single test would tie every
candidate that passes all of them, and the curve could not rise. The metric to
select is `private.pass_rate`, which `flatten_metrics` now publishes wherever a
payload carries both a count and its total. On `task_001` that is the difference
between a reward of 1.0 and of 0.333.

The ten partially-correct roots are exactly the shape that made `lsr_synth`
measurable and `lsr_transform` not — a step function has no slope for a
selection rule to accelerate, and a partially-passing test suite does.

### Cost

One verification of the baseline, 15 tasks:

| | min | p25 | median | p75 | max |
|---|---:|---:|---:|---:|---:|
| seconds | 1.5 | 2.3 | **5.9** | 11.7 | 109.2 |

Sum over all 15: 207 s. At the median, a 60-expansion inner search costs about
12 minutes of verification — the same order as the LLM-SRBench runs already done
here, and two orders below what AlgoTune's noise would have demanded. Cost
selection matters: `task_002` alone is 109 s, 18x the median.

## Property 4: nothing beat the root

A selection rule can only be judged by how fast the best-so-far curve rises, so
the domain has to show *someone* beating the root at an affordable budget. That
needs candidate patches, and this is where the honest boundary of the setup sits.

**There is no workspace agent here.** SWE-bench-Science is agentic — an agent
explores the checkout, and Harbor turns what it leaves behind into a patch. That
is `harbor run --agent`, which this repository deliberately does not reimplement.
Standing in for it: the call-graph closure of the failing reproduction is put in
the prompt, and the model returns edits. That is a **different, easier
localization problem and a harder editing one**, and any number from it describes
that setup, not the benchmark's headline.

Two things had to change before the sampling measured the science at all:

* **Asking for a unified diff measures diff syntax.** 3 of 3 first-round patches
  were rejected by the verifier's own `git apply` before a single test ran — and
  a patch that fails to apply scores exactly like one that applies and fails
  everything. `DockerRunner.export_baseline` now copies the baseline checkout out
  of the image and `whole_file_patch` builds the diff with `git`, so it applies
  by construction. Edits arrive as SEARCH/REPLACE blocks.
* **The endpoint rate-limits the account, not the connection.** Two workers on a
  30 KB prompt returns `AccountRateLimitExceeded` within a few calls. This is the
  same class of contamination as the quota exhaustion recorded in
  [`metasearch-domain-selection.md`](metasearch-domain-selection.md) §4, and the
  same guard applies: a run whose model calls fail is not a negative result.

A third thing had to be fixed before the numbers meant anything, and it is the
one most worth carrying: **the workspace was left holding the previous
candidate's edits.** `whole_file_patch` reset on the way in but not on the way
out, so every attempt was built on the last one rather than on the root. It
showed up as the model apparently being bad at quoting long spans -- every
`SEARCH` block naming `_06heur.py` stopped matching after the first sample --
and it was not: the file no longer held what the prompt had shown it. That is a
different experiment from the one being run.

### The verdict: property 4 fails for this setup

14 independent attempts, `deepseek-v4-flash`, thinking disabled, temperature
0.9, on a workspace reset to the baseline between each:

| outcome | count |
|---|---:|
| reached the verifier | **13 of 14** |
| scored *exactly* the baseline, `1 of 3` | **8** |
| **worse** — `0 of 1`, the patch broke the test module's import | **5** |
| **above the baseline** | **0** |
| solved (`reward = 1`) | 0 |

**The best-so-far curve is flat at the root**, which is the same failure that
ended hyp2f1 and `lsr_transform`. And the shape is the informative part: the
outcomes are *bimodal* — a candidate is either exactly the baseline or broken,
with nothing in between. The reward discriminates downward perfectly well (0.000
against the root's 0.333); it is upward movement that never happens. A rule that
decides what to expand next can only be judged by how fast the curve rises, and
here there is no rise to accelerate.

**This measures the setup, not the benchmark.** One task, a small model, a
single shot, and no workspace agent — against a benchmark whose own headline is
Claude Code with Opus 5 under 50% pass@1. 0 of 14 is the expected number.

### It is not the model: a much stronger generator gets 0 of 14 too

The obvious next suspect is the candidate generator, so the same check was run
again on a substantially stronger model (`deepseek-v4-pro` against
`deepseek-v4-flash`):

| | weak | **strong** |
|---|---:|---:|
| reached the verifier | 13 of 14 | **14 of 14** |
| scored exactly the baseline `1 of 3` | 8 | 13 |
| worse | 5 — all `0 of 1`, the patch broke the test module outright | 1 — `0 of 3`, collection intact |
| **above the baseline** | **0** | **0** |

The candidates are visibly healthier: edit blocks almost always match, and its
one failure leaves the test suite collectable instead of destroying the import.
**And the curve is still flat at the root.**

That is worth more than another null, because it removes the cheap explanation.
The three properties that killed every previous real-data domain hold here; the
fourth is not waiting on model strength, and it is not a property of the domain
either. It is waiting on the thing the benchmark is built around and this
repository deliberately does not reimplement: an agent working in the workspace.

So the next move is not another search-rule experiment on this domain, and not a
bigger model. It is `harbor run --agent`, or nothing — until then the curve stays
flat for reasons that have nothing to say about selection rules.

## Reproducing

```bash
dockerd &                      # the "blocker" was this line
python - <<'EOF'
from examples.metasearch._harbor import load_task, DockerRunner, whole_file_patch
task = load_task("tasks/task_001")          # task.toml + instruction.md + tests/grader.py
runner = DockerRunner()
print(runner.verify(task, ""))              # the baseline, ~11s
ws = runner.export_baseline(task, "ws")     # the checkout, from the image
EOF
```

Every baseline, both sampling runs and the full per-task rows are in
[`metasearch-swe-bench-science.json`](metasearch-swe-bench-science.json).

Task directories come from the dataset
(`.../resolve/main/tasks/task_NNN/{task.toml,instruction.md,metadata.json,tests/grader.py}`);
images are pinned by digest in `task.toml`. Fetch `tests/grader.py` — it is what
repairs a stale image.
