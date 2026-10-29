Fixed
The `evoskill` algorithm card now accepts `kind: skill_dir` and `plugin` as well
as `agent_dir`, so `agentdescent init <skill> --kind skill_dir --algorithm
evoskill` no longer exits 2; the module docstring, `docs/algo-evoskill.md` and
the registry now agree (#604).
