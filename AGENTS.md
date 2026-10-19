# AGENTS.md

Instructions for AI coding agents working on `Birfy/agentdescent`. Human
contributors: [CONTRIBUTING.md](CONTRIBUTING.md) has the same rules with more
background. The owner and only maintainer is Danyang (GitHub `Birfy`).

## Before you code

1. Read [LESSONS.md](LESSONS.md). Open lessons are constraints.
2. Read [CODE_STANDARDS.md](CODE_STANDARDS.md): Verifier honesty, tests that
   fail on `main`, offline tests, generated files, the merge gate.
3. Keep the **Hard rules** below.

## Pipeline contract

One issue → one fresh clone → one branch → one PR → green CI → review → merge by tier.

1. **Issue.** Work one `agent-ready` issue per session, in a **fresh clone**
   (`git clone https://github.com/Birfy/agentdescent <run-dir>/repo`), never in a
   shared or stale checkout.
2. **Branch** from the latest `origin/main`. Write a test that **fails on
   `main`**, then fix the class of bug, not the one sample.
3. **Verify locally** at the tip: `pip install -e ".[dev]"`, `pytest -q`, and
   `mkdocs build --strict` when `docs/` or `mkdocs.yml` changed (see Commands).
4. **PR** to `main`. Imperative English title (`fix(runstore): …`,
   `feat(cli): …`, as in `git log`), `Fixes #N` in the body, and the
   [PR template](.github/pull_request_template.md)'s `Tip SHA:` / `Ran:` /
   `Not run:` lines. A user-visible change adds one fragment
   `changelog.d/<issue>-<slug>.md` ([changelog.d/README.md](changelog.d/README.md));
   never edit `CHANGELOG.md` in a PR. Squash is the only merge method and the
   squash commit is the PR title + body, so keep the body short.
5. **CI.** The required checks are `build` (docs) and `test (3.9)` …
   `test (3.13)`. Never merge around a red or pending check. `risk tier` and
   `reviewed tip` are report-only and are not required.
6. **Review and merge** by tier; see [Merge policy](#merge-policy).
7. **Stop** and hand over to Danyang on a security-sensitive change, an
   unresolved design decision, or when you cannot get a green tip.

When a class of mistake repeats, date a lesson in LESSONS.md (`Seen N×`); after
**≥3 days** promote the durable rule into CODE_STANDARDS.md / CONTRIBUTING.md.

## Merge policy

The tier comes from the diff, never from the PR text:
`python3 .github/scripts/risk_gate.py --pr <n> --repo Birfy/agentdescent`, and the
controller sets the `risk:T*` label within ~30 minutes. Rules:
[`.github/agent/risk-rules.json`](.github/agent/risk-rules.json) (docs T0; tests,
examples and `changelog.d/` T1; other package code and unknown paths T2; the
core loop, model adapters, installers, release files, `.github/**` and these
process docs T3; a PR over its tier's size cap is T3).

- **T0 / T1:** once the required checks are green at the tip, post a
  self-review PASS at that tip ([review-pr](.claude/skills/review-pr/SKILL.md);
  `Birfy` may PASS its own PR), then enable auto-merge pinned to that tip:

  ```sh
  gh pr merge <n> -R Birfy/agentdescent --auto --squash --match-head-commit <sha>
  ```

- **T2 / T3:** green checks + a PASS at the tip (self-PASS counts), then add
  `awaiting-approval` and **stop**. Danyang merges with an explicit click in the
  GitHub UI. An agent never merges a T2/T3 PR, and an approval typed in chat is
  not a click.
- **Every tier:** never `--admin`, never use the ruleset's admin bypass, never
  push to `main`, never force-push, never delete a remote branch by hand
  (merged branches are deleted automatically), never edit protection or rulesets
  as part of a PR's work. Any new commit voids the PASS: re-review the new tip.
  `python3 .github/scripts/review_gate.py --pr <n>` tells you whether the current
  head carries a PASS that counts.

## Labels

Source of truth: [`.github/labels.json`](.github/labels.json), synced by the
`labels` workflow on `main` (create and update only, never delete).

- **Type**, exactly one: `bug`, `enhancement`, `documentation`, `chore`.
- **Area**, one or more: `area:evolve`, `area:meta`, `area:runstore`,
  `area:audit`, `area:agents`, `area:plugins`, `area:cli`, `area:ports`,
  `area:bench`, `area:docs`, `area:ci`.
- **Source:** `self-test` (filed by the self-test routine; issues only).
- **State:** `needs-info` (spec unclear), `needs-human` (needs Danyang's
  decision; the last comment says what), `agent-ready`, `agent-working`,
  `agent-pr`, `awaiting-approval` (green T2/T3 PR waiting for a PASS and
  Danyang's click), `hold` (automation leaves the item alone).
- **Risk:** `risk:T0` … `risk:T3`, set by the controller only, never by hand.

### Triage: when an issue is `agent-ready`

Only Danyang or the maintenance routine (acting as `Birfy`) sets `agent-ready`,
with a triage comment, and only when (1) every design conflict is resolved
(recorded owner decisions, other open issues and PRs), (2) the expected
behavior and scope are clear, and (3) it is not `hold` and not an umbrella or
tracking issue. Otherwise label `needs-info` or `needs-human` and name the
question in a comment. Agents develop **only** `agent-ready` issues.

### Claim, PR, release

- Before work: skip an issue someone else is assigned to. Otherwise assign
  `Birfy`, add `agent-working`, and comment naming the branch.
- Stopping without a PR: unassign, remove `agent-working`, say why.
- The PR says `Fixes #N`, is assigned to `Birfy`, labeled `agent-pr` plus its
  type and area. Agent branches start with `agent/`.

## Controller

[`.github/workflows/controller.yml`](.github/workflows/controller.yml) runs every
30 minutes with the built-in `GITHUB_TOKEN` only (no GitHub App, no secrets) and
changes labels and nothing else: it sets `risk:T*` and `agent-pr` on open PRs,
removes `agent-ready`/`agent-working` when a non-maintainer set or edited an
issue after it was approved, and expires a claim after 6 h with no linked push.
It never adds `agent-ready`, never merges, comments or closes.

| Repository variable | Effect |
| --- | --- |
| `CONTROLLER_LIVE` | exactly `true` → labels are applied; anything else → dry run (plan in the job summary) |
| `AGENT_PAUSE` | `true` → kill switch: the controller reads and changes nothing. Agents and routines also stop when it is `true` |

## Hard rules

1. **No secrets.** Never write, print or commit an API key or token. Real-model
   tests read the key from the environment only (`ARK_API_KEY` for Volcengine
   Ark, through `api_key_env=` / the shared `FORWARDED_ENV` list); never put a
   key in a spec, a test, a log or a PR.
2. **The suite stays offline and deterministic.** No network or model API in
   `pytest -q`; a real-model or Docker check is gated behind a flag or an env
   var and reported under `Not run:` when skipped.
3. **Faithful ports stay faithful.** An algorithm port under
   `examples/<algorithm>/` follows the source repo's released code; where code
   and paper disagree, follow the code and say so ([porting
   checklist](docs/porting-checklist.md)).
4. **Generated files are regenerated, never hand-edited.** `docs/api.md` comes
   from `python -m tools.gen_api_docs`; `CHANGELOG.md` is folded from
   `changelog.d/` at release time (`python -m tools.changelog_fragments release`),
   not in a PR. The version lives only in `agentdescent/__init__.py`.
5. **Git hygiene.** Fresh clone per task, branch from `origin/main`, never push
   `main`, never force-push, never delete a remote branch by hand. Clean up the
   run directory when done.

## Commands

```sh
pip install -e ".[dev,mcp]"          # what CI installs (mcp needs Python >= 3.10)
pytest -q                            # the offline suite CI runs on 3.9-3.13
python -m tools.gen_api_docs         # regenerate docs/api.md after a public-API change
python -m tools.gen_api_docs --check # what tests/test_api_reference.py runs
python -m tools.changelog_fragments check   # validate changelog.d/ fragments
pip install -e ".[docs]" && mkdocs build --strict   # the `build` check, for docs changes
python -m build --wheel && bash scripts/smoke-wheel.sh dist/*.whl   # the wheel gate in CI
```

Real model, only when the change needs it and the key is in your environment
(`export ARK_API_KEY=…` is done by the person, not written by you):
`openai_compatible("deepseek-v4-flash", api_key_env="ARK_API_KEY",
default_base_url="https://ark.cn-beijing.volces.com/api/coding/v3")`. It costs
tokens; say in the PR what ran and at which tip.

## Layout

- `agentdescent/`: the package. `evolution.py` / `async_evolve.py` are the
  loop; `aggregator.py`, `population.py`, `policies.py`, `strategies.py` the
  pieces it composes; `meta*.py` the meta-evolution stack; `runstore.py`,
  `ledger.py`, `checkpoint.py` persistence; `audit/` the audit and oracle
  calibration; `agents.py` the model adapters; `cli.py` and `mcp.py` the
  command line and MCP server; `integrations/` the host installers.
- `integrations/`: the host plugins (Claude Code, DSH). `examples/`: one folder
  per faithful algorithm port plus the framework demos. `bench/`: A/B and matrix
  runs. `tools/`: generators and checks. `tests/`: the offline suite.
- `docs/`: the MkDocs site (every page must be in `mkdocs.yml`'s nav).
- `.github/agent/`: risk rules, review policy, controller config.

## Style

- Follow the code around your change; match its comment density. Comments say
  why (a constraint, a measured trap, an issue number), not what.
- Commit messages in English, imperative subject, the reason in the body.
- Keep a change to one subject.
