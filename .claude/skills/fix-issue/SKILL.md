---
name: fix-issue
description: Fix one AgentDescent issue end to end - claim, fresh clone, a test that fails on main, gates, PR, self-review at the tip, merge by tier, cleanup.
---

# AgentDescent fix-issue

One `agent-ready` issue, one fresh clone, one branch, one PR. This skill grants
no merge authority beyond AGENTS.md › Merge policy.

## 0. Before anything

- `AGENT_PAUSE` is not `true` (`gh variable list -R Birfy/agentdescent`).
- The issue is `agent-ready`, not `hold`, and nobody else is assigned.
- Read AGENTS.md, CODE_STANDARDS.md and LESSONS.md from `origin/main`.
- Claim: assign `Birfy`, add `agent-working`, comment with the branch name.

## 1. Fresh clone

```sh
RUN=<run-root>/$(date +%Y%m%d-%H%M%S)-<issue>
git clone https://github.com/Birfy/agentdescent "$RUN/repo" && cd "$RUN/repo"
git checkout -b agent/<issue>-<slug> origin/main
python3 -m venv "$RUN/venv" && "$RUN/venv/bin/pip" install -e ".[dev,mcp]"
```

## 2. Red, then green

1. Write the test first and run it on the unchanged `main`: it must fail.
2. Fix the class (every host, every adapter, both `evolve()` and
   `async_evolve()` when the loop is involved), not the sample.
3. Public API changed → `python -m tools.gen_api_docs`. User-visible change →
   `changelog.d/<issue>-<slug>.md`.

## 3. Gates at the tip you will link

```sh
pytest -q
python -m tools.gen_api_docs --check
python -m tools.changelog_fragments check
mkdocs build --strict          # docs changes only
```

Real-model (Ark) checks only with `ARK_API_KEY` already in the environment;
never write the key anywhere.

## 4. PR

- Push the branch (never `main`, never `--force`), open the PR with the
  template: imperative title, `Fixes #N`, short body, `Tip SHA:` / `Ran:` /
  `Not run:`. Assign `Birfy`; labels `agent-pr` + type + area.
- Tier: `python3 .github/scripts/risk_gate.py --pr <n>`.

## 5. Review at the tip

Wait for the required checks (`build`, `test (3.9)`..`test (3.13)`) to be
green at the tip, then review with [review-pr](../review-pr/SKILL.md) in a
fresh session and post the report. A finding → fix, push, and review the new
tip from scratch (a new commit voids the PASS).

## 6. Merge by tier

- **T0/T1:** `gh pr merge <n> -R Birfy/agentdescent --auto --squash --match-head-commit <sha>`.
- **T2/T3:** add `awaiting-approval` and stop; Danyang clicks merge.
- Never `--admin`, never the ruleset bypass.

## 7. Cleanup

Remove the run directory. Do not delete remote branches by hand (merged
branches are deleted automatically). File follow-ups as issues, not TODOs.
Stopping without a PR: unassign, remove `agent-working`, comment why.
