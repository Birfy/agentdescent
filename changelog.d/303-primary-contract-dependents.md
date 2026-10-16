Fixed
Multi-artifact contract dependencies now include the primary artifact (#303):
the reverse dependency graph is built from every registered artifact's
effective contract, so a primary whose `contract=` `depends_on` an extra is
required to land an adapter with a contract-breaking change to that extra
(`missing-adapters` otherwise) and has its cached evaluations invalidated on
the branch that changed, in both `evolve()` and `async_evolve()`. Runs without
primary dependencies are unchanged.
