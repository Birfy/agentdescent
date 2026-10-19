Fixed
`openai_compatible` and `anthropic_compatible` no longer put a connection-shaped
keyword (`base_url`, `api_base`, `api_url`, `base`, `url`, `endpoint`, `api_key`,
`apikey`, `key`) into the request body. A spec that wrote `"base_url": "https://..."`
was accepted, `plan` said `ok`, and every call went to `api.openai.com` carrying the
other provider's key. It is now refused by `plan`, `evolve` and MCP `start`, naming
the spelling that exists (`default_base_url` / `base_url_env` for the endpoint,
`api_key_env` for the key); other body extras such as `temperature` are unaffected.
`plan` also warns when `openai_compatible` would send a non-OpenAI `api_key_env` to
`api.openai.com` because no endpoint is configured, and `docs/plugins.md` documents
the three connection parameters (#422).
