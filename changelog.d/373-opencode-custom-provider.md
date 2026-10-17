Fixed
An isolated `opencode` worker that cannot see a custom provider now says so
(#373). A custom provider (an OpenAI-compatible endpoint such as Volcengine Ark,
a local vLLM, a gateway) lives in the user's `opencode.json`, which an isolated
worker cannot read, so `-m ark/model` failed every call with OpenCode's opaque
`UnknownError` while `plan` was `ok`. The `AgentError` of an isolated `opencode`
worker that exits with `UnknownError` / `ProviderModelNotFound` now carries a
hint naming the cause and the two fixes (`"isolate": false`, or
`"env": {"OPENCODE_CONFIG": "/abs/providers.json"}` with a provider-only file);
the hint is skipped when `OPENCODE_CONFIG` is already set in the worker's `env`
and for every other CLI. The `plan` warning for an isolated `opencode` that pins
`-m/--model provider/model` now names custom providers and the
`OPENCODE_CONFIG` recipe. `docs/plugins.md`, `docs/agents.md` and the skill no
longer say `extra_args` pins an OpenCode model "without giving up isolation";
they document the recipe, and that an `OPENCODE_CONFIG` exported in the
launching shell is dropped for an isolated worker, so it belongs in `env`.
Providers are not copied out of the user's `opencode.json` automatically: that
file also holds inline keys, MCP servers and plugins that isolation keeps out.
