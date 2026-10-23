Added
`recursive_self_improve`: the self-improvement loop now carries a
`ReallocationPolicy`, and each outer round evolves that rule with
`portfolio_evolve` on the archive the loop just wrote. The first round uses the same fixed rule as `self_improve` (config floor/min_samples),
each later round searches from and validates against the incumbent, and a rejected
policy round is reported with the previous rule kept; a cold archive is a reported rejection,
not a failure. `SelfImproveConfig.min_exam` also applies the verified-seeding bar
before an archived rule is used as a prior.
