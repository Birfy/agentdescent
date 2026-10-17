Fixed
`FrontierSampler.install` keeps the measured history of a task that is redrawn
(an equivalent task is deduplicated, a conflicting task under a known id is
refused and counted invalid), so redrawing no longer erases learning or resets
the stuck threshold; and the engine records only the designated solver's
rollouts in the sampler, so an extra artifact's score on the same task is not
read as the primary's first-to-last learning (#336).
