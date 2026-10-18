Fixed
`agentdescent apply` (CLI and MCP `apply`, both through `cli.apply_payload`) no
longer reports every file as written, and no longer leaves a `.bak-N`, for a run
that changed nothing (#396). The evolved tree is compared with the target file by
file: `written` lists only files whose content differs (new, changed, or a script
that lost its exec bit), identical ones are in the new `unchanged` list, and the
backup is made only when something is written or pruned. When nothing would
change the result is `{"written": [], "unchanged": [...], "backup": [], "noop":
true}` and the CLI prints `nothing to apply: <dest> already matches run <id>` on
stderr (stdout stays the JSON document), exit 0. A single-file (`text`) target
gets the same treatment. `show`'s `apply would:` and `apply --dry-run` use the
same judgement (`EvolutionResult.write_to` now returns `unchanged` and `noop`
too, and only rewrites the files that changed). The SKILL.md step and the MCP
`apply` description say what `noop` means.
Files are compared and written verbatim (CRLF included), a script counts as
unchanged only with both user and group exec bits, the `text` dry-run result also
carries `written`, and an artifact with no files is a no-op that creates nothing.
