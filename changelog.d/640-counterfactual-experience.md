Added
`CounterfactualExperience`: turns `ReplayLab.shapley` step credit into per-step
`ExperienceRecord` rewards and ingests it into `ExperienceEngine`. Credits below
`min_credit` are dropped and an empty reward map is refused, so a trace with no
attributable gain cannot silently become zero-information replay.
