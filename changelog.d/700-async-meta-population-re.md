Fixed
`tests/test_async_meta_population.py` now really runs its outer meta-proposer
(a missing module-level `import re` made it raise `NameError` every round, so
no island was ever promoted) and checks that the async run promotes the same
island as the sequential one, each from its own copy of the archive (#700).
