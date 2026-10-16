Added
`examples/gskill`, a `mechanism_microport` of GEPA's gskill: `optimize_anything`
learning a repository-specific `SKILL.md` for a coding agent. The artifact is a
`FileTree` skill directory behind the `skill_dir` structural validator, scored
by a binary fail-to-pass / pass-to-pass fitness over a vendored twelve-bug
fixture repository (no SWE-smith, no Docker), accepted under a strict held-out
gate, and exported as a valid `.claude/skills/<name>/` tree with
`--export-skill`. `python -m examples.gskill --dry-run` needs no network,
Docker or API key. Real gskill (Jinja, Bleve) numbers are not claimed (#291).
