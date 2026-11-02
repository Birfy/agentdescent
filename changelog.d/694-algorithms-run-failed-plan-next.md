Fixed
When the plan step of `agentdescent algorithms run` fails, `next` now says to
fix the spec and run `agentdescent plan <spec>` again. Before, `--json` said
`agentdescent evolve <spec>` on a spec that plan had just refused. Both
failure shapes now carry `spec_path` as well, and plain mode prints the same
`next:` line after the error (#694).
