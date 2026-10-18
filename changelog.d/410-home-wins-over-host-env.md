Fixed
`install`, `uninstall` and `host_status` with an explicit `--home` no longer let
an exported `CODEX_HOME`, `DSH_HOME` or `XDG_CONFIG_HOME` redirect them to the
real configuration: `--home` is now the host's whole world (a probe cannot edit,
or remove, your real install). Without `--home` the variables apply as before
(#410).
