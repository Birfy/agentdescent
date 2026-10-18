Fixed
A spec-driven run (`evolve`, `evolve --detach`, MCP `start`) now counts the
tokens its spec-built model adapters report: the run's `Usage` reaches every
`openai_compatible` / `claude` (or any factory with a `usage=` parameter) named
in `agent`, `reflect` or `policies`, as a token-only view
(`Usage.token_sink()`), so `status` / `rounds.jsonl` show real `tokens` and
`evolve.max_tokens` stops the run (`stop_reason="max_tokens"`) instead of being
accepted and never firing. `calls` still counts one per actor invocation, so
`max_calls` and `usd_per_call` budgets are unchanged. The Python API
(`openai_compatible(usage=u)` with `evolve(usage=u)`) behaves as before (#405).
