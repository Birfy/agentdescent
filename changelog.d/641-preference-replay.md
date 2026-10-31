Added
`PreferenceReplay`: ranks stored `ExperienceRecord`s by a fitted
`PreferenceModel` and ingests the preferred slice into `ExperienceEngine`
without duplicating records. The preference model becomes the replay-selection
policy, so human pairwise feedback affects what the meta-policy learns from.
