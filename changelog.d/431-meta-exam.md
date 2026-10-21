Added
Self-exam (`agentdescent/meta_exam.py`): adoptions are provisional until the
oracle says so. `exam_archive` walks the archive's rule records, re-validates
each against the problems its own `problem` descriptor describes (rebuilt by
`problem_from_archive`) using an independent oracle, and stamps the outcome on
the record via `ConsolidationStore.mark_exam`: `passed` when the oracle gain
clears `min_gain` (flags reset), `flagged` when it does not (flags + 1), and
`demoted` when a rule fails truth `max_flags` times in a row. A demoted rule is
excluded from future seeding — `_archived_rules` skips it, so
`continuous_meta_run` and population selection never start a search from a rule
truth has retired. `ConsolidationRecord` gains optional `exam` / `exam_flags` /
`exam_note` fields (backwards compatible: an archive written before the exam
read back as `None` and stays seedable until examined); `mark_exam` rewrites
atomically (tmp + rename). Adoptions are thus provisional, and the archive is
self-correcting: what the loop believed is checked with evidence, and what
fails the check is retired.
