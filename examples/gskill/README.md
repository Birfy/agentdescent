# gskill — A repository-specific skill for a coding agent

Port onto the AgentDescent engine, as a declarative
[`MethodPolicy`](../_method_policy.py) over a `FileTree` skill directory.

| | |
|---|---|
| Kind | Skill-directory self-evolution |
| Fidelity class | `mechanism_microport` |
| Upstream | GEPA's gskill pipeline — [gepa-ai/gepa@f1cfdee](https://github.com/gepa-ai/gepa/tree/f1cfdee2856e9dc4b5b4dc5cc2e391d0aa20d819/src/gepa/gskill), [blog](https://gepa-ai.github.io/gepa/blog/2026/02/18/automatically-learning-skills-for-coding-agents/) |
| Domain (compact) | vendored miniature repository `kvtools`, 12 injected bugs, 4/4/4 splits — not SWE-smith, no Docker |
| Definition plug-ins | `FileTree` + `skill_dir` validator; `Policies(acceptance=StrictHeldOutGate)`; binary fail-to-pass / pass-to-pass fitness |

## Run

```bash
python -m examples.gskill --dry-run
python -m examples.gskill --yes --provider openai --model glm-5.2 --max-tokens 4096 --export-skill ./out
```

`--dry-run` prints the configuration and returns with **zero network access,
no Docker and no API key**. `--export-skill DIR` writes the evolved skill as
`DIR/.claude/skills/kvtools-fixer/` once it passes `validate_skill_tree`.

## What is in here

- [`gskill_repo_skill.py`](gskill_repo_skill.py) — the `MethodPolicy` definition and entry point
- [`_fixtures.py`](_fixtures.py) and [`fixtures/`](fixtures) — the vendored repo, its injected bugs and the verifier (the agent never sees the tests)
- Mechanism notes, upstream trace and every boundary: [`docs/algo-gskill.md`](../../docs/algo-gskill.md)
- Offline tests: [`tests/test_gskill.py`](../../tests/test_gskill.py)

Real gskill numbers (Jinja, Bleve) are not reproduced here.
