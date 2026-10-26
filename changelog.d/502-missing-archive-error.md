Fixed
`agentdescent portfolio`, `exam` and `audit` no longer create a missing archive
directory and report it as empty: like `safety`, they now fail with
`error: archive '<path>' does not exist` (exit 2). `audit` likewise rejects a
`--meta-audit` / `--inner-audit` file that does not exist (#502).
