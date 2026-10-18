Fixed
`plan` no longer reports `ok: true` for a spec whose `evolve` block names an
object-valued `evolve()` parameter as a string (`strategy`, `aggregator_factory`,
`run`, `propose`, `staleness_policy`, …), which `evolve` then crashed on in round
setup with `'str' object has no attribute 'initial'`; an unknown `evolve` key is
refused the same way, naming the valid knobs. A spec holds JSON, so objects come
through `policies` / `agent` / `reflect`, not `evolve` (#384).
