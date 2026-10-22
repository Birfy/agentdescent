Added
The named-algorithm registry is now reachable from the plugin surface, not just
from a spec file: MCP gains `list_algorithms` / `describe_algorithm`, `init` and
`plan` accept `--algorithm NAME --param KEY=VALUE`, and `docs/algorithms.md` is
generated from the registry. The shared SKILL.md, the Claude Code `/evolve`
command and `docs/plugins.md` explain when to offer a method (only when the user
names one or the artifact clearly fits) and that the kind's default recipe is
used otherwise.
