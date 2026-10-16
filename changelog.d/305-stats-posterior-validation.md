Fixed
`prob_improvement` no longer hangs forever on a non-finite posterior (a damaged
or hand-edited `checkpoint.json` restored via `Aggregator.restore()`) or
crashes on negative counts / `samples < 1`. `BetaPosterior` refuses a
non-finite or negative count or update weight, `observe_delta(nan)` is ignored,
`prob_improvement` raises `ValueError` before sampling, `restore()` skips such
a posterior like any malformed entry, and `DefaultAcceptance` rejects
`accept_samples < 1` at construction / configure time (#305).
