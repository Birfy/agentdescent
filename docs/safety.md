# Safety case — may this archive keep seeding?

`verified_rules` gates seeding, `meta_audit` measures verifier bias and
`meta_exam` stamps whether a rule transferred. Each answers one question. A
recursive system needs the combined one before it keeps learning from an
archive: **is this archive safe to seed the next session from?**

`safety_report(store)` is read-only and side-effect free. It checks:

- **exam coverage** — unexamined rules seed provisionally; flagged/demoted
  rules are counted and warned;
- **seedable best** — every slot's best eligible rule must not be flagged or
  demoted (a slot with no eligible rule is a hard failure); an unexamined best
  is a warning, not a pass;
- **audit bias** — when a `MetaAuditStore` is supplied, the worst recorded
  oracle-vs-verifier gain bias **among the seedable rules** must be within
  `bias_threshold`. A demoted rule's bias is not a reason to block a clean
  seedable set. A missing audit is a warning: no evidence is not good evidence.

Eligibility is the seeder's, not a second interpretation: `--min-exam` defaults
to `passed` (the `verified_rules` default), `flagged` includes flagged rules,
and `none` keeps every non-demoted rule. Config-matching rules rank first,
exactly as `_archived_rules` sorts them. Any `fail` means `unsafe`; any `warn`
means `caution`; all pass means `safe`.

The result is `safe` / `caution` / `unsafe`, a per-check list with the reason,
and the raw exam counts. No model calls and no writes, so it can run before
every resume or campaign.

```python
from agentdescent import safety_report

report = safety_report(store, audit=audit, bias_threshold=0.1)
print(report.verdict, report.reason)
for check in report.checks:
    print(check.status, check.name, check.detail)
```

```bash
agentdescent safety [--archive PATH] [--audit audit.jsonl] \
  [--bias-threshold 0.1] [--min-exam passed|flagged|none] [--json]
```

Exit codes: `0` safe, `1` caution, `3` unsafe, `2` usage / missing path. A
missing `--audit` file is an error, not an empty audit: a typo must not look
like a real store with no labels.

## What it does not claim

`safety_report` judges the *trust evidence* in the archive, not whether the
search is still improving and not whether a candidate would regress a
capability. It is the pre-flight check those separate read/acceptance models do
not replace.
