Fixed
A reflector call through `openai_compatible` or `anthropic_compatible` that runs
out of `timeout` no longer surfaces as a bare `TimeoutError: The read operation
timed out`. It now raises a `RuntimeError` naming the endpoint, the model and the
seconds, and how to fix it (`"stream": true`, a higher `"timeout"`, or turning the
provider's thinking off), so `status` / `propose_error` say what to change. Read
and connect timeouts are both covered; `with_retries` still retries them. The
`openai_compatible` docs and the host skill now say reasoning models need
`"stream": true`.
