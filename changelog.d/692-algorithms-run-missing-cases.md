Fixed
`agentdescent algorithms run` now tips the optional `cases.jsonl` third
argument (or checks the path you passed) when plan fails because `data.path`
does not exist, instead of only naming the missing file (#692).
