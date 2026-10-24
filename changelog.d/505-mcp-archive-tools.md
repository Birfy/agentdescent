Added
MCP tools `safety`, `portfolio` and `audit_trace`: a host agent can now ask
whether an archive is safe to seed from, how its layers rank for the next
budget window, and where one adoption's evidence came from, without finding the
CLI verbs in the source. They return the same structure as the CLI's `--json`
(which now also carries `archive` for `safety`), write nothing, and treat a
missing archive as an error. The installed `SKILL.md` says when to use them, and
the `doctor` tool description now lists `opencode` (#505).
