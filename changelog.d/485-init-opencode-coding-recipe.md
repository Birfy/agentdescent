Fixed
`agentdescent init` with an `opencode` worker and both `--model` and
`--base-url` now writes the OpenCode Coding recipe into `agent` (`-m
<provider>/<model>` plus `env.OPENCODE_CONFIG` naming a provider-only
`<spec>.opencode-providers.json` beside the spec, key as `{env:VAR}`, never
inline) instead of a bare `{"ref": "opencode"}` that ignored both flags.
Without both flags the bare ref and its warning are unchanged (#485).
