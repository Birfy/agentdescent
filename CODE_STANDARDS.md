# AgentDescent code standards (acceptance rules)

Short acceptance rules for agents and reviewers. The workflow is in
[AGENTS.md](AGENTS.md), the human guide in [CONTRIBUTING.md](CONTRIBUTING.md),
and fresh lessons in [LESSONS.md](LESSONS.md); a lesson moves here after ≥3 days.
This file is not in the docs site on purpose (`docs/` pages must all be in the
nav); editing it is a T3 change.

## Verifier honesty

When you report the state of a change (PR body, review, chat):

1. **Tip SHA:** the exact commit (`git rev-parse HEAD`, or the PR head).
2. **Ran:** each command you ran at that tip, with pass/fail and real numbers
   (`pytest -q → <N> passed, <M> skipped`).
3. **Not run:** every material check you did not run (`mkdocs build --strict`,
   the wheel smoke, an Ark real-model run, Docker sandboxes) and why. Missing
   evidence is a gap, not a PASS.
4. **No borrowed credit:** the author's narrative and CI on another SHA are not
   your result.

## A test that fails on `main`

- A bug fix or behavior change brings a test that **fails on the unchanged
  `main`** and passes with the change. Keep both halves of the evidence (run it
  at the merge base first); a reviewer re-checks it.
- Prefer a minimal fixture that shows the class of bug over one sample that
  only matches one string.

## Fix the class, not the sample

- One escape path found → fix the shared helper and cover its siblings (every
  host installer, every adapter, both `evolve()` and `async_evolve()`).
- A same-class red tightens LESSONS.md / this file; do not special-case it away.

## Offline, deterministic tests

- `pytest -q` uses no network and no model API. The reference domain
  (`agentdescent/domains/router.py`) and the example stubs exercise the full
  loop without a key.
- Anything that needs an API key, Docker or a gated dataset is behind a flag or
  env var, documented, and listed under `Not run:` when skipped.
- `@pytest.mark.skip` / `xfail` without a condition is a review finding (and
  makes the PR T3); `importorskip` / `skipif` for an optional extra is fine.

## Secrets and real models

- No key, token or credential in code, specs, tests, logs, fixtures or PR text.
- A real-model check reads its key from the environment only: Volcengine Ark is
  `ARK_API_KEY` via `api_key_env=`, forwarded to hosts through the shared
  `agentdescent.integrations.FORWARDED_ENV` list (LESSONS 2026-10-18).

## Faithful ports

- `examples/<algorithm>/` follows the source repo's released code; if code and
  paper disagree, follow the code and note it. Start from
  [docs/porting-checklist.md](docs/porting-checklist.md) and `examples/_TEMPLATE.py`.
- Do not report upstream numbers as AgentDescent results; measured results
  carry their own run.

## Generated files

- `docs/api.md` is generated from `agentdescent.__all__`: after a public API
  change run `python -m tools.gen_api_docs` and commit both;
  `tests/test_api_reference.py` fails otherwise. On a conflict take `main`'s
  copy and regenerate; never hand-merge it.
- Docs pages live in `docs/` and are in `mkdocs.yml`'s nav;
  `mkdocs build --strict` must pass.

## Changelog

- A user-visible change adds one `changelog.d/<issue>-<slug>.md` fragment (first
  line `Added|Changed|Deprecated|Removed|Fixed|Security`). Never edit
  `CHANGELOG.md` in a PR; the release folds the fragments. Contributor-only
  changes (CI, process docs) need none.

## Merge only a reviewed tip

- **T0/T1:** required checks green at the tip + a self-review PASS at that tip,
  then `gh pr merge <n> --auto --squash --match-head-commit <sha>`.
- **T2/T3:** required checks green + a PASS at the tip (self-PASS counts) +
  Danyang's explicit merge click; the agent labels `awaiting-approval` and stops.
- Any new commit voids a PASS. Never `--admin`, never the ruleset bypass, never
  push `main`, never force-push.

## Tiny loops

- One issue (or one tightly coupled pair) per fresh clone and session.
- Small PRs: the size caps in `.github/agent/risk-rules.json` (T0 300 lines /
  15 files, T1 400 / 20, T2 250 / 12) are where a PR becomes T3.
