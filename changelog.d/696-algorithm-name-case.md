Fixed
Algorithm names now ignore case and `-` vs `_`: `agentdescent algorithms show
GEPA`, `algorithms run archive-selection …`, `campaign` and an EvolveSpec
`algorithm` resolve to the one registered algorithm they match instead of
failing as unknown. `algorithms run` writes the registered name into the spec
and its default file name; an ambiguous name is still reported as unknown
(#696).
