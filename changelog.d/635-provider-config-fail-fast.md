Fixed
`openai_compatible` and `anthropic_compatible` now raise `ProviderConfigError`
on a deterministic refusal (HTTP 401/403/404, or a body naming
`UnsupportedModel` / `model_not_found` / `invalid_api_key`) and the engines stop
the run on the first one, instead of retrying it and spending a full round of
identical failing calls. The error names what to fix (the `model` or base URL,
or which `api_key_env` to check); 429, 5xx and timeouts still retry as before.
`diagnose` no longer recommends `log.txt` for a run that has none (#635).
