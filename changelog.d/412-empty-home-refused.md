Fixed
`install`, `uninstall` and `host_status` refuse an explicitly empty `--home`
(`--home ""`, which is what `--home "$UNSET_VAR"` expands to) with a clear error
and exit code 2 instead of falling back to the real home and its
`CODEX_HOME` / `DSH_HOME` / `XDG_CONFIG_HOME`, so a probe with a typo can no
longer edit or remove your real install. Omitting `--home` is unchanged (#412).
