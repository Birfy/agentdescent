Fixed
`init` and `doctor` now name the OpenCode Coding worker recipe when only a
gateway key such as `ARK_API_KEY` is set. `init` used to write a silent
`{"ref": "opencode"}` (every isolated call then failed on OpenCode's built-in
Volcengine default) and `doctor` only mentioned the `openai_compatible` reflect
route; both now point at `docs/plugins.md` (`-m ark/…` plus `env.OPENCODE_CONFIG`).
The `plan` half of this landed earlier in the same issue.
