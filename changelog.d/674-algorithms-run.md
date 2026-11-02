Fixed
`agentdescent algorithms run NAME PATH [cases.jsonl]` now matches the plugin
slash `algorithms … run` verb: it writes a starter spec with that algorithm
(default `./agentdescent.<name>.spec.json`) and prints a `plan` estimate, then
points at `agentdescent evolve <spec>` — without starting a run (#674).
