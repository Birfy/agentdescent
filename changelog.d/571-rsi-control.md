Added
`RSIController`, a duck-typed control plane that turns each session result into
`accept / reject / defer / rollback / continue / stop` using target, cost,
patience, risk, pluggable gates and a budget policy. It wraps the existing
guarded/controller/portfolio pieces instead of replacing them.
