Fixed
`agentdescent campaign` results can now be installed: `agentdescent apply --from
<final_target> --to <target> [--dry-run]` applies an on-disk artifact with the
same `.bak-N` backup and no-op check as `apply <run_id>`, and the campaign's
output ends with the ready-to-copy diff / dry-run / apply / cleanup commands
(`next` in `--json`). `docs/campaign.md` no longer points at an `apply --to`
form that failed with "has no result to apply yet".
