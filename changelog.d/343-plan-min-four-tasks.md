Fixed
`plan` / MCP `plan` returns `ok: false` when a spec has fewer than 4 tasks —
the same train/held-out rule `evolve()` already raises — instead of a soft note
that let agents start a run that then crashed. Soft "8–20 is a workable minimum"
notes remain for 4 ≤ n < 8 (#343).
