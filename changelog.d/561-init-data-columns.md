Fixed
`agentdescent init --data` now reads the cases file's first row and, when it has
no `prompt`/`gold` column, maps a single common alias (`input`/`question`/`query`/`problem`,
`expected`/`answer`/`output`/`target`) and says so, or warns with the real column
names. Spec loading now names the `data.prompt` field in a missing-column error,
and `plan`/`evolve`/MCP refuse a gold-comparing scorer (`contains`, `exact`,
`last_number`, `numeric_close`) when no row has the `data.gold` column instead of
silently scoring every answer 0.
