# LESSONS.md

Dated, short lessons from real AgentDescent work. Agents read this **before
coding**. Durable rules get promoted into [CODE_STANDARDS.md](CODE_STANDARDS.md)
(and a pointer in [CONTRIBUTING.md](CONTRIBUTING.md) when humans need it).

## How to read and write

1. **Before a change:** skim every open lesson. Treat them as hard constraints
   until promoted or retired.
2. **When something bites again:** add or bump a lesson the same day. Date it
   (`YYYY-MM-DD`, Asia/Shanghai). Keep it to trigger → wrong move → right move.
3. **Seen N×:** each recurrence increments `Seen` and names the issue/PR. The
   first write-up is `Seen 1×`.
4. **Promote after ≥3 days:** once a lesson has stayed true for at least three
   calendar days, move the durable rule into CODE_STANDARDS.md and leave a
   one-line `Promoted → CODE_STANDARDS.md#…` stub here (or delete the lesson once
   a test enforces it).
5. **Security lessons stay hot:** secrets, sandboxing and untrusted-rule themes
   stay here until a test enforces them, even after promotion.
6. Write AgentDescent facts only; do not paste another project's process text.

## Open lessons

### 2026-10-18 — Provider keys reach a host only through `FORWARDED_ENV`

- **Seen:** 3× (#404 Codex MCP server, #415 DSH install + native plugin, #423 doctor)
- **Trigger:** a new install path, host launcher or `doctor` check that needs the model key (`ARK_API_KEY` and friends).
- **Wrong:** a hand-written list of variable names per host, or checking only `OPENAI_API_KEY`; the run then starts without a key or `doctor` reports a missing key that is set.
- **Right:** read the shared `agentdescent.integrations.FORWARDED_ENV` list (and its provider-key subset); add a test that the new path forwards exactly that list. Keys come from the environment only, never from a file you write.
- **Promote path:** 2026-10-21 → CODE_STANDARDS.md (Secrets and real models).

### 2026-10-19 — Parallel PRs collide on the shared edit points

- **Seen:** 1× (on 2026-10-19 #431–#435 are all `CONFLICTING` with `main`; they and #436–#439 all edit `agentdescent/__init__.py` and `docs/api.md`, most also `docs/meta-evolution.md`)
- **Trigger:** several open PRs that each export a new symbol or extend the same doc page.
- **Wrong:** hand-merging `docs/api.md`, or leaving the PR to rot until someone else rebases it.
- **Right:** merge `origin/main`, resolve `__init__.py` and the prose by hand, take main's `docs/api.md` and rerun `python -m tools.gen_api_docs`, then the full `pytest -q` at the new tip (it voids any earlier PASS).
- **Promote path:** after ≥3 days → CODE_STANDARDS.md (Generated files).

### 2026-10-19 — A changelog fragment is named after its issue

- **Seen:** 2× (#428 renamed its fragment after the fact; #433 carries placeholder fragments `changelog.d/999-*.md`)
- **Trigger:** adding `changelog.d/<number>-<slug>.md` before the number is known.
- **Wrong:** guessing the next PR number, or reusing another PR's number.
- **Right:** use the issue number; with no issue, open the PR first and rename the fragment to the PR number in the same branch before review.
- **Promote path:** after ≥3 days → CODE_STANDARDS.md (Changelog).

### 2026-10-19 — Process files are T3; the merge is Danyang's click

- **Seen:** 1× (the FlyBot process port: labels, risk tiers, controller, review gate, these docs)
- **Trigger:** a PR that touches `.github/**`, `AGENTS.md`, `LESSONS.md`, `CONTRIBUTING.md`, `CODE_STANDARDS.md`, `.claude/**` or `pyproject.toml`.
- **Wrong:** enabling auto-merge because CI is green and the change "is only docs".
- **Right:** `risk_gate.py` says T3 → self-PASS at the tip, add `awaiting-approval`, stop. Danyang merges.
- **Promote path:** after ≥3 days → CODE_STANDARDS.md (Merge only a reviewed tip).
