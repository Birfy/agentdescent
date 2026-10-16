---
description: A mechanism microport of gskill — GEPA's optimize_anything learning a repository-specific SKILL.md from coding-agent traces and test feedback. Offline fixture instead of SWE-smith and Docker; not yet measured live.
---

# gskill — A repository-specific skill for a coding agent

> **Skill-directory self-evolution.** Given a repository and verifiable
> bug-fixing tasks, evolve one `SKILL.md` tree so a coding agent resolves more of
> them — and deploy the best tree unchanged. Runs through the shared
> [`MethodPolicy`](policies.md) runner over a [`FileTree`](directory-evolution.md)
> strategy with the `skill_dir` structural validator.
> Example:
> [`examples/gskill/gskill_repo_skill.py`](https://github.com/Birfy/agentdescent/blob/main/examples/gskill/gskill_repo_skill.py).

| | |
|---|---|
| **Paper** | none — a GEPA application: the [blog post](https://gepa-ai.github.io/gepa/blog/2026/02/18/automatically-learning-skills-for-coding-agents/) and [guide](https://gepa-ai.github.io/gepa/guides/gskill/) (2026) |
| **Upstream code** | [gepa-ai/gepa@f1cfdee](https://github.com/gepa-ai/gepa/tree/f1cfdee2856e9dc4b5b4dc5cc2e391d0aa20d819/src/gepa/gskill), `src/gepa/gskill` |
| **Example** | [`examples/gskill/gskill_repo_skill.py`](https://github.com/Birfy/agentdescent/blob/main/examples/gskill/gskill_repo_skill.py) |
| **Domain** | a vendored miniature repository (`kvtools`) with 12 injected bugs, 4/4/4 splits — **not** SWE-smith |
| **Layer** | L2 skill (`blast_radius=0.6`, set by the shared runner) |
| **Fidelity** | `mechanism_microport` — [what the classes mean](port-fidelity.md) |
| **Port author** | `chendanyang` |

This is a mechanism microport, not a reproduction. It **does not claim** the
blog's Jinja (55% → 82%) or Bleve (24% → 93%) numbers, nor its Claude Code
transfer results: those need SWE-smith tasks, Docker and a real coding agent,
none of which this port has.

## The mechanism

gskill has three stages. SWE-smith injects bugs into a clean checkout of the
target repository, producing verifiable instances (problem statement,
`FAIL_TO_PASS` tests, `PASS_TO_PASS` tests). GEPA's `optimize_anything` then
evolves **one text artifact — the skill, which starts empty**: run the agent on a
batch, pass the problem, the agent's trace and the test output to a reflection
model ("the agent will not be evaluated on the same task"), and keep the proposal
if it scores better. The best skill is injected into the agent's prompt and
transfers to other agents without retraining.

Its fitness (`swe_fitness_fn.py`) is **binary with a regression check**: 1.0 only
if the `FAIL_TO_PASS` tests pass *and* the full suite still passes; no patch, an
f2p failure and a p2p regression each score 0 and are reported differently.

## Where each piece lives

| Upstream mechanism | Where it lives here |
|---|---|
| SWE-smith instances | [`fixtures/bugs.json`](https://github.com/Birfy/agentdescent/blob/main/examples/gskill/fixtures/bugs.json): a mutation over the clean `fixtures/kvtools`, a problem statement, `FAIL_TO_PASS` ids; `PASS_TO_PASS` is every other test |
| The fitness function | `_fixtures.judge`: binary, verdicts `no_patch` / `f2p_failed` / `p2p_regression` / `all_passed` |
| Mini-SWE-Agent rollout | `solve`: one model call that returns the repaired file; the skill is in its prompt |
| Reflection on traces + test output | `propose`: problem, attempt, verdict and test output → `<EDITS>` on the skill tree |
| Skills start empty | `INITIAL_SKILL`: a *valid* `SKILL.md` whose body says nothing yet |
| Deploy `best_skills` | `export_skill` / `--export-skill DIR` → `DIR/.claude/skills/kvtools-fixer/` |
| Structurally valid skill | `FileTree(validators=skill_tree_validator(...))` — see below |
| Train / val / test | `gskill_splits(seed)`, stratified by the repository's conventions |

**Structural rejection.** The artifact is a directory, so a candidate can fail in
ways no rollout would show: delete `SKILL.md`, break its frontmatter, rename the
skill. The `skill_dir` validator rejects these *before* they are measured, and
`FileTree` counts them by reason (`structural_rejections`), so a reflector that
keeps producing unloadable skills is visible rather than silently wasted. The
exported tree must also pass `validate_skill_tree` or `export_skill` refuses it.

**Stratified splits.** `kvtools` has four house conventions — half-open ranges,
`ConfigError` for invalid values, partial pages count as one, a missing value is
`None` — none of them stated in code. Each convention has three bugs, one per
split. A skill that learned a convention from a train bug is therefore measured
on a *different* bug of the same convention in held-out and test: skills are
learned on some instances of a repository and judged on others, as in gskill.

## Boundaries

- **No SWE-smith and no Docker.** Twelve vendored bugs in a ~60-line repository
  replace hundreds of mined instances. The verifier runs candidate code in a
  throw-away directory under a plain subprocess with a timeout and POSIX
  CPU/memory/file-size limits — containment against a runaway loop, **not** a
  sandbox for hostile code. Real gskill uses a container per rollout.
- **The agent is one model call**, not Mini-SWE-Agent with bash. It sees the
  skill, the problem statement and the bugged file, and returns the whole
  repaired file; it does not explore the repository. (Real runs can: point them
  at SWE-smith and Docker with upstream's own `python -m gepa.gskill.train_optimize_anything`.)
- **Edit protocol.** Upstream's proposer rewrites one skill blob in a single code
  fence; here proposals are the shared `<EDITS>` protocol over a file tree, which
  is what lets the structural gate see a deleted or malformed `SKILL.md`.
- **No Pareto parent selection.** `optimize_anything` samples parents from the
  per-instance Pareto frontier. The shared population layer carries no
  per-instance scores yet, so the head is mutated under a SkillOpt-style
  [strict held-out gate](acceptance-policies.md) (`StrictHeldOutGate`, ~15 lines at
  the `acceptance` seam): commit only a strictly higher held-out pass rate. With
  a binary fitness on a handful of tasks, a statistical gate would mostly measure
  noise.
- **One failed task per proposal**, like upstream's `--proposer loop` item calls;
  its follow-up merge call is not ported. `reflective=False`, because a
  model-synthesised merge reaches the ledger without passing `to_diff` and would
  bypass the frontmatter validator.
- **Out of scope**, as in the issue: shipping SWE-smith, evolving executable skill
  scripts (gskill's "what's next"), and the plugin installers.

## Measured results — kvtools fixture

**No live run is recorded.** The offline tests drive the real engine in every
scheduler with a scripted model that applies a convention only when the skill
states it; that checks the loop (propose → validate → gate → export), not the
quality of any model, so no number from it belongs in
[the results table](self-evolution-examples.md#measured-results-all-twenty-two). The
row will be added when a live run exists.

## Run it

```bash
python -m examples.gskill --dry-run           # no network, no Docker, no API key

# live: any OpenAI-compatible endpoint or Claude; write the best skill for a host
python -m examples.gskill --yes --provider openai --model deepseek-v4-flash \
    --workers 4 --candidates 8 --max-tokens 4096 \
    --export-skill ./out
# -> ./out/.claude/skills/kvtools-fixer/SKILL.md
```

`--max-tokens 4096` because a proposal carries the whole skill file inside JSON.
`--export-skill` writes the skill the run ended on, after checking it is a valid
Agent Skills tree.

Flags: [the MethodPolicy command line](self-evolution-examples.md#the-methodpolicy-command-line).

Offline tests: `tests/test_gskill.py`, `tests/test_candidate_methods.py`.
