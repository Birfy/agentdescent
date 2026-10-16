Fixed
`audit.ppi.t_ppf` is now accurate for every `df > 0`. The Cornish-Fisher expansion
was 11% too narrow at df = 1 (36% at the 99% level) and badly off below 1, and
cluster-robust PPI intervals (`df = groups - 1`) routinely land there. The
expansion is kept for `df >= 30`; df 1 and 2 use closed forms and every other
`df < 30` inverts the exact Student-t CDF (regularized incomplete beta) with a
bracket that grows instead of saturating (#313).
