Fixed
The RRSI example bundle's default in-memory history now rides the checkpoint, so
a resumed bundle keeps accepted component counts, pruning and proposer evidence
(an already-accepted skill is no longer "novel" again; a durable `history_path`
still reads its own file). `upstream_components` now classifies each op through
a real changed-line diff, and slot-keyed ops (`SingleSlot` / rule ids) as
`prompt`, so prose such as "Do not mention ToolRegistry" no longer gets
structural novelty, and comment- or string-only edits stay `prompt` (#337).
