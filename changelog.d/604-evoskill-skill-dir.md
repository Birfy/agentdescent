Fixed
The `evoskill` algorithm card now accepts `kind: skill_dir` and `plugin` as well
as `agent_dir`, so `agentdescent init <skill> --kind skill_dir --algorithm
evoskill` no longer exits 2; the module docstring, `docs/algo-evoskill.md` and
the registry now agree (#604). Its `SkillLibraryTree` is now built over the
kind's own `FileTree` (new `Overrides.strategy_from` hook), so a plugin keeps its
host's frozen paths and its `.ts`/`.js` code, and a `skill_dir` keeps its
SKILL.md gate; on a `skill_dir` the induced rule goes to `references/<slug>.md`,
linked from a managed index block in the root `SKILL.md`, instead of a nested
skill the host would not load.
