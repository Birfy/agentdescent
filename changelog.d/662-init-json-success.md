Fixed
`agentdescent init --json` now prints one `{"ok": true, "spec", "kind",
"providers_config", "warnings", "notes", "next"}` object on success instead
of the human prose lines; plain output is unchanged (#662).
