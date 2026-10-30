Fixed
The key-preflight tests no longer fail on a machine that exports `ARK_API_KEY`
or another provider key: their fixture clears every name the preflight reads
(#632).
