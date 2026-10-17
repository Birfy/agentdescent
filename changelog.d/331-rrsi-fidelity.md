Changed
RRSI wiring fidelity closure (`examples/rrsi/rrsi_policies.py`): the wired
proposer/selector now consume the **pinned upstream fixtures**, not
approximations. Component attribution goes through `components.normalize` /
`classify_diff` (the upstream `K` / `K_STR` evidence rules — `skills/x/SKILL.md`
is a `skill` edit, `ToolRegistry(...)` is `client_tool`, pure prose is
`prompt`), replacing the first-path-segment heuristic. The per-edit ledger is
`history.History` (the pinned durable JSONL) when `build_rrsi_bundle(
history_path=...)` is given, with an in-memory twin for scratch runs pinned to
the same summaries by a parity test. `RRsiProposalPolicy` reads
`History.render` / `prune_set` (with the actionable accepted edits) /
`stall_flag`; `RRsiAcceptance` scores novelty over
`History.incumbent_component_counts`; the aggregator checkpoint now carries
only the search state (round, S*, incumbent cost, progress), while the
per-edit evidence is the History's own file — so a resumed run reads the same
evidence the loop conditioned on. Six fidelity tests pin the vocabulary, the
memory/file parity, durable resume and a genuinely novel structural component
carrying a within-band candidate.
