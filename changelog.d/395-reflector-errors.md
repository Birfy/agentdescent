Fixed
A reflector that raises on every call no longer ends a run as `done` with nothing
recorded (#395). `propose` exceptions used to be folded into the worker failures
after that worker's rollout had already counted as a success, so `rounds.jsonl`
showed only `considered: 0` and the cause appeared nowhere. Each round now carries
`propose_errors` (calls that raised) and `propose_error` (the first, as
`Type: message`), `status` / `watch` / `show` and the MCP `status` / `show`
payloads show them, and `evolve()` warns when any occurred. When every proposal
raises in two consecutive rounds (`evolution.PROPOSE_FAIL_ROUNDS`) the run stops
as `state: failed` with `error: reflector failed on every call in rounds N-M:
<first exception>`, and `agentdescent evolve` exits non-zero; a reflector that
fails only some of the time still finishes, and rounds with no `propose` call
neither count nor reset the streak. `agentdescent init` now warns when the
starter `reflect` is `openai_compatible` and `OPENAI_API_KEY` is unset, and the
`plan` warning says what will happen. The barrier-free `asynchronous=True` path is unchanged.
