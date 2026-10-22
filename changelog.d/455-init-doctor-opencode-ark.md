Fixed
`init` and `doctor` now name the OpenCode Coding worker recipe when only a
gateway key such as `ARK_API_KEY` is set. `init` used to write a silent
`{"ref": "opencode"}` (every isolated call then failed on OpenCode's built-in
Volcengine default) and `doctor` only mentioned the `openai_compatible` reflect
route; both now point at `docs/plugins.md` (`-m` plus `env.OPENCODE_CONFIG`).
The Ark-specific wording (Volcengine default, `ark/deepseek-v4-flash`) appears only
when `ARK_API_KEY` is set; any other gateway key gets a generic
`<provider>/<model>` recipe, and `init` and `doctor` share one predicate, so
they also agree when `OPENAI_API_KEY` is set alongside a gateway key.
The `plan` half of this landed earlier in the same issue.
