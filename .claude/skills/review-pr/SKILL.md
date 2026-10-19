---
name: review-pr
description: Review an AgentDescent PR (or local change) at a tip SHA and write a Verifier report with Tip SHA / Ran / Not run / Verdict PASS|FAIL|INC that the review gate can read.
---

# AgentDescent review-pr

Produces a Verifier report for `Birfy/agentdescent`. The report is evidence,
not merge authority: what may happen after a PASS is AGENTS.md › Merge policy.

## Establish the tip

1. Work in a **fresh clone** (`git clone https://github.com/Birfy/agentdescent`),
   check out the PR head (`gh pr checkout <n>`), record `git rev-parse HEAD`. If
   the head moves during the review, start over.
2. Read the rules from `origin/main`, not a stale copy: AGENTS.md,
   CODE_STANDARDS.md, LESSONS.md, and this file
   (`git show origin/main:.claude/skills/review-pr/SKILL.md`).
3. Inventory the diff against the merge base (`git diff origin/main...HEAD --stat`)
   and get the tier: `python3 .github/scripts/risk_gate.py --pr <n>`. The PR
   description is narrative, not evidence.

## Re-run the gates yourself

```sh
pip install -e ".[dev,mcp]"
pytest -q                               # the whole suite, not only the new tests
python -m tools.gen_api_docs --check    # when agentdescent/__init__.py or a public signature changed
python -m tools.changelog_fragments check
mkdocs build --strict                   # when docs/ or mkdocs.yml changed (needs .[docs])
```

- **Fails on `main`:** copy only the new/changed test files onto the merge base
  and run them; they must fail there and pass at the tip.
- Ark real-model checks only when the change touches a model path and
  `ARK_API_KEY` is already in the environment; otherwise list them under
  `Not run:`.

## Instant FAIL

- A key, token or credential anywhere in the diff.
- A network or model call reachable from `pytest -q`.
- `docs/api.md` edited by hand or out of sync; `CHANGELOG.md` edited in a PR.
- A claimed fix with no test that fails on `main`.
- An unconditional `@pytest.mark.skip` / `xfail` hiding a failure.
- A port that departs from its source's released code without saying so.

## Verdicts

| Verdict | Meaning |
| --- | --- |
| **PASS** | No actionable findings; every gate you list under `Ran:` was run by you at this tip and is green |
| **FAIL** | An actionable defect (correctness, a hard rule, a missing fail-on-main test, a dishonest claim) |
| **INC** | Incomplete: a material check not run, the tip moved, scope unclear. INC is not PASS |

## Report shape (the review gate parses this)

Post it as one PR comment from the `Birfy` login (self-review counts). Keep the
block intact; the **last** `Tip SHA:` block in the comment is the one read, the
newest report at a tip wins, and a PASS without a `Ran:` entry or without a
`Not run:` line does not count.

```text
Self-review (Birfy)

Tip SHA: <full 40-hex head>
Ran:
  - pytest -q → pass (<N> passed, <M> skipped)
  - python -m tools.gen_api_docs --check → pass
  - CI at this tip → green (build, test 3.9-3.13)
Not run:
  - Ark real-model run — no model path changed
Findings:
  - none
Verdict: PASS
```

Then check it counts: `python3 .github/scripts/review_gate.py --pr <n>`.
