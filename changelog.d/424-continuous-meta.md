Added
Continuous meta-learning (`agentdescent/continuous_meta_run`): the archive
drives the next session. `ConsolidationRecord` gains an optional `problem`
descriptor (backwards compatible); `archive_rule` writes an adopted slot rule
back to the archive with its problem descriptor; `problem_from_archive`
rebuilds a runnable inner `Problem` from an archived descriptor (the caller
supplies the tasks via `tasks_for`, the archive supplies the slot and inner
settings -- the "link, don't copy" rule applied to a problem instead of a
trajectory); and `continuous_meta_run` closes the loop between sessions --
train and validation problems come from the archive, the search starts from
the best archived rule for the slot/spec/config, the gain is measured both
against that prior and against the fresh seed (the honest cross-session
number), and an adopted rule is archived so the next session starts from a
better one. An empty archive is an ordinary `meta_run`. A record archived by a
different spec is refused at rebuild, never misread.
