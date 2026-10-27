Added
`LeakageGuard`, a pre-evaluation screen and acceptance backstop that rejects a
candidate introducing a protected gold answer, task id or corpus string. It
checks diff keys and values before the held-out sweep and reports the existing
`critic-rejected` outcome, so a leak never crashes the aggregator or buys an
inflated score. The acceptance form only refuses newly introduced matches, so a
protected string already in the seed does not freeze the run. Matching is
case-insensitive and whitespace-collapsed, and short strings are ignored.
