Fixed
`openai_compatible` and `anthropic_compatible` now raise `QuotaExhausted`
(a `ProviderConfigError`) on a quota- or billing-exhausted 429 — Ark
`AccountQuotaExceeded`, OpenAI `insufficient_quota`, and bodies naming
`quota` / `billing` — so `with_retries` and the engines fail on the first
call instead of backing off for minutes as a rate limit. Real throttling
(`rate_limit_exceeded`, bare 429 + `Retry-After`) still retries.
`diagnose` suggests waiting for the provider reset or pointing
`api_key_env` / `model` at another plan, not `doctor` (#653).
