# Leakage guard — a self-modification may not memorise the answer key

A recursive system evolved against examples can learn the *answers* instead of
the rule. A skill line, prompt edit or code patch that contains a gold string or
a task id scores perfectly on the archive and generalises to nothing. `rrsi`
screens for this inside one algorithm; `LeakageGuard` makes it a general
**screen** so every adoption path shares the check before paying for evaluation.

```python
from agentdescent import LeakageGuard, Policies

protected = [str(row["gold"]) for row in cases] + [row["id"] for row in cases]
policies = Policies(screen=LeakageGuard(protected))
```

- `screen(card)` checks the diff's keys and values. A leaking card is settled
  before the held-out sweep, so its inflated score never reaches the merge path;
  the outcome is the existing `critic-rejected`, so no new merge category is
  needed.
- The acceptance-policy shape is a backstop for paths that do not run a screen.
  It refuses with `critic-rejected` and, when a base artifact is available, only
  refuses **newly introduced** matches so a protected string already in the
  seed cannot freeze every future candidate.
- Matching is case-insensitive and whitespace-collapsed.
- Strings shorter than `min_length` (default 3) are ignored: a one-character
  gold would match everything. Numeric golds below the threshold are therefore
  not screened; lower `min_length` deliberately if that is acceptable.
- `inner` is the acceptance policy that would have run without the guard; it is
  consulted only after no protected string is found. `inner=None` is the shipped
  `DefaultAcceptance`, configured through the forwarded hooks.
- `LeakageReport` / `find_leakage` expose the matching strings for a log without
  printing the artifact.

The RRSI critic still has its own exact-match scan with different semantics
(case-sensitive, no length floor); this module is the shared screen, not a
drop-in replacement for that algorithm's internal critic.

The guard is only as good as the protected list: include gold answers, stable
task ids, and any private corpus strings that must never appear in the artifact.
Never include the final holdout, because the point is to keep it out of the
artifact, not to teach the guard it exists.
