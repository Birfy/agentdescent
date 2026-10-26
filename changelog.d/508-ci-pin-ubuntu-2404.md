Changed
CI's test matrix runs on `ubuntu-24.04` instead of `ubuntu-latest`: the
Ubuntu 26.04 runner image has no Python 3.9 for `setup-python`, so the required
`test (3.9)` check could never pass, and the sandboxed AlgoTune timing test was
failing there too.
