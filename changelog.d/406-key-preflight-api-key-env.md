Fixed
`plan`, `evolve` and MCP `start` now check the variable an `openai_compatible`
reflector/agent will actually read -- the block's own `api_key_env` (and name
`base_url_env` in the hint) instead of a hard-coded `OPENAI_API_KEY`. A spec
with `"api_key_env": "ARK_API_KEY"` no longer gets a false "needs
OPENAI_API_KEY" alarm when `ARK_API_KEY` is set, and a missing `ARK_API_KEY` is
no longer hidden by an unrelated `OPENAI_API_KEY`. The `claude` branch does the
same, and `claude()` now really accepts `api_key_env` (it used to forward it to
`messages.create` and fail on the first call). String refs and default names
warn exactly as before (#406).
