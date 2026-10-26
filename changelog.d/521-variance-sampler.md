Added
`VarianceSampler`, a `TaskSampler` that prefers tasks whose recent reward
variance is high (flip-flopping tasks) over a lifetime pass-rate average, with a
UCB exploration bonus and round-index tie rotation. It is a drop-in
`task_sampler=` argument, thread-safe like `DifficultyWeighted`.
