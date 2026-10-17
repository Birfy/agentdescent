Fixed
`audit.gate.RectifiedAcceptance` now requires `inflate_when_stale` to be a
finite real number `>= 1.0` (`bool` refused). NaN and inf used to pass the old
`< 1.0` check and yield `var_after=nan` (or zero evidence) on the stale path
(#332).
