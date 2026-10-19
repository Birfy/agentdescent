# Contributing to AgentDescent

Thanks for your interest in improving AgentDescent. This is a research reference
implementation; contributions that keep it **correct, faithful, and testable**
are very welcome.

## Development setup

```bash
git clone https://github.com/Birfy/agentdescent
cd agentdescent
pip install -e ".[dev]"     # installs NumPy plus the pytest/PyYAML/Markdown development tools
```

Python ≥ 3.9 is required.

## Shortest verification path

From a fresh checkout, the shortest environment and behavior check is:

```bash
git clone https://github.com/Birfy/agentdescent
cd agentdescent
pip install -e ".[dev]"
pytest -q
python -m examples.run_demo
```

Then temporarily change `rounds = 40` to `rounds = 8` in
`examples/run_demo.py`, run the demo again, observe the shorter curve, and
restore the line. That proves you are executing the checkout you are editing.

## Running the tests

The suite is **offline and deterministic** (no network, no model API):

```bash
pytest -q
```

CI runs the same suite on Python 3.9 / 3.10 / 3.11 / 3.12 / 3.13 for every push and PR. Please
add or update tests for any behavior change — the reference domain
(`agentdescent/domains/router.py`) and the example stubs let you test the full
parallel/async loop without an API key.

## Trying it end to end

```bash
python -m examples.run_demo                       # synchronous DP, no API
python -m examples.efficiency                     # parallel scaling + async tail-hiding
python -m examples.skill_evolution --dry-run      # the flagship LLM example, no API
```

Every faithful algorithm port has a zero-network `--dry-run` mode for inspecting
its CLI configuration without loading a dataset or calling a model.

## Building the docs

```bash
pip install -e ".[docs]"
mkdocs serve            # live preview at http://127.0.0.1:8000
mkdocs build --strict   # must pass with no warnings (CI enforces this)
```

## Guidelines

- **Keep the analogy honest.** The design mirrors the deep-learning training
  stack, but *gradients add and diffs do not* — new merge/acceptance logic must
  go through the aggregator's conflict-resolution + statistical-acceptance path,
  not silent averaging.
- **Faithful ports stay faithful.** The algorithm examples follow each source
  repo's *released code* (not just the paper). If code and paper disagree, follow
  the code and note it. Start from the [porting checklist](docs/porting-checklist.md)
  and `examples/_TEMPLATE.py`.
- **One algorithm, one folder.** Each faithful port lives in
  `examples/<algorithm>/` — entry point, `README.md`, and any helper only that
  port uses. The framework demos (`run_demo`, `efficiency`, `parallelism`, …)
  stay at the top of `examples/`, because they belong to no single algorithm.
- **No hidden network in tests.** Anything requiring an API or heavy infra
  (SWE-bench Docker, gated datasets) must be gated behind a flag and documented.
- Match the style and comment density of the surrounding code.

## Pull requests

1. Branch from `main` (agents: a fresh clone per task; nobody pushes to `main`
   or force-pushes).
2. `pytest -q` and, for docs changes, `mkdocs build --strict` both pass. A bug
   fix comes with a test that fails on `main`.
3. A user-visible change adds a changelog fragment: one new file
   `changelog.d/<number>-<slug>.md` whose first line is the category (`Added`,
   `Changed`, `Deprecated`, `Removed`, `Fixed`, `Security`) and whose remaining
   lines are the entry text. Do **not** edit `CHANGELOG.md` in a PR: a shared
   edit point makes every open PR conflict whenever one merges. Maintainers fold
   fragments into `CHANGELOG.md` at release time. See
   [`changelog.d/README.md`](changelog.d/README.md); `pytest -q` validates your
   fragment.
4. A public API change regenerates the reference: `python -m tools.gen_api_docs`.
5. Open the PR with the template: `Fixes #N`, what changed and why, and the
   `Tip SHA:` / `Ran:` / `Not run:` lines. Merges are **squash only**, and the
   squash commit is the PR title and body, so keep the body short.

## Maintainer / agent loop

The full contract is in [AGENTS.md](AGENTS.md); in short:

1. Read [LESSONS.md](LESSONS.md) and [CODE_STANDARDS.md](CODE_STANDARDS.md) first.
2. Issues carry exactly one type label and one or more `area:*` labels
   ([`.github/labels.json`](.github/labels.json)). Agents work only on
   `agent-ready` issues, which only the maintainer (or the maintenance routine
   acting as `Birfy`) sets.
3. Every PR gets a risk tier from its diff
   (`python3 .github/scripts/risk_gate.py --pr <n>`; the controller labels it
   `risk:T0`..`risk:T3`). T0/T1 auto-merge on green CI after a self-review PASS
   at the tip; T2/T3 wait, labeled `awaiting-approval`, for the maintainer's
   merge click. Nobody admin-merges.
4. A review is a PR comment in the Verifier format
   ([review-pr](.claude/skills/review-pr/SKILL.md)); the `reviewed tip` check
   (`python3 .github/scripts/review_gate.py --pr <n>`) says whether the head has
   a PASS that counts. A new commit voids it.
5. `AGENT_PAUSE=true` (repository variable) stops the controller and the agents.

### Verifier checklist (every status report)

- [ ] Tip SHA named
- [ ] Commands **run** at that tip, with pass/fail
- [ ] Material checks **not run** listed (docs build, wheel smoke, Ark real model, Docker)
- [ ] The new test fails on `main` and passes with the change
- [ ] The fix covers the class, not one sample
- [ ] T2/T3: a PASS at the same tip, then the maintainer's click

## Agent docs map

| Doc | Role |
| --- | --- |
| [AGENTS.md](AGENTS.md) | Pipeline contract, merge policy, labels, controller, hard rules |
| [LESSONS.md](LESSONS.md) | Dated lessons (`Seen N×`); promoted after ≥3 days |
| [CODE_STANDARDS.md](CODE_STANDARDS.md) | Acceptance rules: Verifier honesty, tests that fail on `main`, merge gate |
| [`.claude/skills/fix-issue`](.claude/skills/fix-issue/SKILL.md) | One issue end to end |
| [`.claude/skills/review-pr`](.claude/skills/review-pr/SKILL.md) | The PASS / FAIL / INC report |
