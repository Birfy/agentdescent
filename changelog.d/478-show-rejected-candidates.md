Added
`show` (CLI and MCP) can now say what a run that learned nothing actually
proposed. The run store writes `candidates.jsonl` beside `rounds.jsonl` with the
last few diffs the merger measured and refused: touched files, a short diff, the
rationale, the gated probability and the outcome. `MergeReport.candidate_diff`
and `evolve(on_reports=...)` expose the record; `record_rollouts: false` still
writes neither diagnostic file. `show` returns a bounded `rejected` block, and
the shipped SKILL.md tells the host to read `samples`/`rejected` before
suggesting more rounds.
