Added
`agentdescent --version` (and `-V`) prints `agentdescent <version>` and exits 0.
`doctor` — the CLI text and `--json` output and the MCP `doctor` tool — now
reports `version` (`agentdescent.__version__`) and `executable`
(`sys.executable`), so a host-side `doctor` and a shell-side `doctor` can be
compared directly to tell whether the host's MCP server runs the same install.
The SKILL's `doctor` step and `docs/testing-the-plugins.md` say to compare them
when results differ (#416).
