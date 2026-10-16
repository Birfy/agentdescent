"""Mechanism microport: **gskill** (gepa-ai/gepa@f1cfdee, ``src/gepa/gskill``).

gskill learns a *repository-specific skill file* for a coding agent. SWE-smith
injects bugs into a clean checkout to make verifiable tasks; GEPA's
``optimize_anything`` evolves one text artifact -- the skill -- by running the
agent on a batch, handing the traces and test results to a reflection model, and
keeping what scores better. The skill is injected into the agent's prompt, and
transfers to other agents unchanged.

Preserved, from ``train_optimize_anything.py`` / ``swe_fitness_fn.py``:

* **The artifact starts empty.** "GEPA starts with empty skills" -- here a valid
  ``SKILL.md`` with frontmatter and a body that says nothing yet.
* **The fitness is binary and has a regression check.** 1.0 only if the
  ``FAIL_TO_PASS`` tests now pass *and* the ``PASS_TO_PASS`` tests still do; no
  patch, f2p failure and p2p regression each score 0 and are told apart in the
  feedback (``no_patch`` / ``f2p_failed`` / ``p2p_regression`` / ``all_passed``).
* **The reflector sees what ``side_info`` carries**: the problem statement, the
  agent's attempt, the verdict and the test output -- and is told the agent will
  *not* be evaluated on the same task, so the skill must generalise. The prompt
  is ``LoggingProposer.DEFAULT_PROMPT_TEMPLATE``'s, shortened.
* **The skill is deployed unchanged.** :func:`export_skill` writes the evolved
  tree as ``.claude/skills/<name>/`` (``--export-skill DIR``), after
  ``validate_skill_tree`` says it is a valid Agent Skills directory.

Departures, each declared rather than hidden (and repeated in ``notes``):

* **No SWE-smith, no Docker, no real repository.** The domain is a vendored
  twelve-bug miniature, ``fixtures/kvtools`` (see :mod:`._fixtures`), and the
  "agent" is a single model call that returns the repaired file -- not
  Mini-SWE-Agent running bash in a container. Real gskill numbers (Jinja,
  Bleve) are neither claimed nor reproduced.
* **The artifact is a directory.** Upstream's skill is one text blob and the
  proposer rewrites it in a single code fence. Here it is a ``FileTree`` with the
  ``skill_dir`` validator, and the proposal is the shared ``<EDITS>`` protocol,
  so a candidate that deletes ``SKILL.md`` or breaks its frontmatter is rejected
  before it is ever measured.
* **No Pareto parent selection.** ``optimize_anything`` samples parents from the
  per-instance Pareto frontier; the shared population layer carries no
  per-instance scores yet, so this port mutates the head under a SkillOpt-style
  *strict* held-out gate (:class:`StrictHeldOutGate`) instead.
* **One failed task per proposal**, as upstream's ``--proposer loop`` does for its
  per-item calls; its follow-up *merge* call is not ported -- two workers editing
  ``SKILL.md`` at once are resolved by the engine's ranking, not a model merge,
  so ``reflective=False`` (a synthesised merge would bypass the structural
  validator).
"""

from __future__ import annotations

import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from agentdescent.evolution import Task
from agentdescent.filetree import TreeError, materialize, parse_tree
from agentdescent.policies import Policies
from agentdescent.skillspec import (SKILL_FILE, skill_tree_validator,
                                    validate_skill_tree)
from agentdescent.treestrategy import EDIT_PROTOCOL, FileTree

from examples._measure import extract_python
from examples._method_policy import MethodPolicy
from examples._method_runner import standard_main

from . import _fixtures


FIDELITY = "mechanism_microport"

SKILL_NAME = "kvtools-fixer"

#: "GEPA starts with empty skills" -- as a *valid* skill, so the structural gate
#: has something to hold: frontmatter, a trigger description, and no knowledge.
INITIAL_SKILL = f"""---
name: {SKILL_NAME}
description: Repository-specific guidance for fixing bugs in the kvtools repository. Use when asked to repair a failing behaviour in kvtools.
---

# kvtools

(No repository-specific knowledge has been learned yet.)
"""

#: What the loop may write: the skill's entry point and its reference notes.
EDITABLE = (SKILL_FILE, "references/**")

MAX_SKILL_CHARS = 6_000
MAX_TRACE_CHARS = 2_500


class SkillTree(FileTree):
    """``FileTree`` plus the invalid-proposal count the matrix reports.

    ``FileTree.to_diff`` already counts structural rejections by reason; what it
    does not do is say how many proposals produced *no diff at all* -- no
    parseable edit, nothing in scope, too many files, or a validator's refusal.
    That total is ``invalid_proposals``, the number every other port reports.
    """

    invalid_proposals: int = 0

    def to_diff(self, state, proposal, author, base_version, target):
        diff = super().to_diff(state, proposal, author, base_version, target)
        if diff is None:
            with self._rejections_lock:
                self.invalid_proposals += 1
        return diff


def make_strategy() -> SkillTree:
    return SkillTree(
        initial_files={SKILL_FILE: INITIAL_SKILL},
        editable=EDITABLE,
        max_files_per_diff=2,
        max_file_bytes=MAX_SKILL_CHARS,
        # Frontmatter that parses, a name that matches the directory the skill is
        # installed under: the invariants a host needs to ever load it.
        validators=(skill_tree_validator(root_name=SKILL_NAME),),
    )


class StrictHeldOutGate:
    """Commit only a strict improvement of the **full** held-out pass rate.

    SkillOpt's gate, at the standard acceptance seam: no Beta draw, no annealing.
    With a binary fitness on a handful of tasks a statistical test would mostly
    measure noise, and a skill that does not *raise* the held-out pass rate is
    not worth the tokens it costs every future rollout -- the reason gskill's
    own proposer prompt says "adding skills blindly will not help the agent".
    """

    def accept(self, ctx):
        from agentdescent.policies import AcceptDecision, MergeContext

        base = MergeContext.rate(ctx.base_counts)
        cand = MergeContext.rate(ctx.cand_counts)
        improved = cand > base
        return AcceptDecision(
            accept=improved, category="" if improved else "strict-gate",
            detail=f"held-out pass rate {base:.3f} -> {cand:.3f}",
            observed_delta=cand - base)


# ---------------------------------------------------------------------------
# The domain: tasks, the surrogate agent, the verifier
# ---------------------------------------------------------------------------


def gskill_splits(seed: int) -> Tuple[List[Task], List[Task], List[Task]]:
    """Train / held-out / test, frozen per seed, stratified by convention.

    Every house convention of the repository has three bugs, and each lands in a
    different split -- so a skill that learned a convention from a *train* bug is
    measured on a **different** bug of the same convention in held-out and again
    in test. That is gskill's own setup in miniature: skills are learned on some
    instances of a repository and judged on others.
    """
    by_convention: Dict[str, List[_fixtures.Instance]] = {}
    for item in _fixtures.load_instances():
        by_convention.setdefault(item.convention, []).append(item)
    train: List[Task] = []
    held_out: List[Task] = []
    test: List[Task] = []
    for convention in sorted(by_convention):
        group = list(by_convention[convention])
        # A string seed is hashed with sha512, not `hash()`: reproducible across
        # processes and Python versions.
        random.Random(f"gskill:{seed}:{convention}").shuffle(group)
        for bucket, item in zip((train, held_out, test), group):
            bucket.append(Task(id=f"gskill:{item.id}", prompt=item.prompt,
                               meta={"fixture": item.id,
                                     "convention": item.convention}))
    return train, held_out, test


def skill_text(files: Dict[str, str]) -> str:
    """The skill as the agent receives it: ``SKILL.md`` first, then references."""
    order = sorted(files, key=lambda path: (path != SKILL_FILE, path))
    return "\n\n".join(f"--- {path} ---\n{files[path].rstrip()}" for path in order)


def _attempt_source(output: str) -> Optional[str]:
    """The agent's replacement file from its reply, or ``None`` (no patch).

    The reply must carry a fenced block: an answer with no code in it is an empty
    patch, which gskill scores 0 and labels ``no_patch``, not a file to guess at.
    """
    if not isinstance(output, str) or "```" not in output:
        return None
    code = extract_python(output)
    return code.rstrip() + "\n" if code.strip() else None


def reward(task: Task, output: str) -> float:
    item = _fixtures.instance(task.meta["fixture"])
    return _fixtures.judge(item, _attempt_source(output)).score


def export_skill(files: Dict[str, str], dest: str, *,
                 name: str = SKILL_NAME) -> List[str]:
    """Write a skill tree where Claude Code finds it: ``<dest>/.claude/skills/<name>/``.

    Refuses a tree that is not a valid Agent Skills directory, with the reason
    codes -- the evolved artifact is only worth shipping if a host can load it.
    The skill directory is **replaced**, not overlaid: the tree is written to a
    sibling staging directory and swapped in, so a file the evolved skill deleted
    or renamed does not linger from an earlier export and keep teaching the
    agent something the validated artifact no longer says. Returns the written
    paths, relative to ``dest``.
    """
    errors = validate_skill_tree(files, root_name=name)
    if errors:
        raise ValueError(f"not a valid skill tree: {', '.join(errors)}")
    skills = Path(dest) / ".claude" / "skills"
    skills.mkdir(parents=True, exist_ok=True)
    target = skills / name
    staging = Path(tempfile.mkdtemp(prefix=f".{name}.", dir=str(skills)))
    try:
        written = materialize(files, str(staging))
        staging.chmod(0o755)
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.exists():
            shutil.rmtree(target)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return [f".claude/skills/{name}/{path}" for path in written]


def build(seed: int, export_dir: str = "") -> MethodPolicy:
    strategy = make_strategy()
    #: The artifact the *last* ``solve`` call saw. The runner finishes with a test
    #: evaluation of the final artifact, so after a run this is the final skill --
    #: which lets ``--export-skill`` deploy it without the runner having to hand
    #: the artifact back. A run that never reached that evaluation exports nothing.
    seen: Dict[str, str] = {}

    def solve(llm, rendered: str, task: Task) -> str:
        seen["rendered"] = rendered
        try:
            files = parse_tree(rendered)
        except TreeError:
            files = {}
        item = _fixtures.instance(task.meta["fixture"])
        return llm(
            "You are gskill's mini-agent, a coding agent fixing a bug in the "
            f"`kvtools` repository. The repository skill below was learned from "
            "earlier tasks; apply what is useful.\n\n"
            f"=== SKILL ===\n{skill_text(files)}\n=== END SKILL ===\n\n"
            f"{task.prompt}\n\n"
            f"Reply with the COMPLETE corrected contents of `{item.file}` in one "
            "```python fenced block and nothing else.",
            unit=task.id)

    def propose(llm, rendered: str, task: Task, output: str,
                score: float) -> Optional[str]:
        try:
            files = parse_tree(rendered)
        except TreeError:
            return None
        item = _fixtures.instance(task.meta["fixture"])
        outcome = _fixtures.judge(item, _attempt_source(output))
        shown = {path: text for path, text in files.items()
                 if strategy.writable(path)}
        return llm(
            "You are gskill's skill optimizer, helping a coding agent learn "
            "skills for fixing bugs in the `kvtools` repository.\n\n"
            "## Current skill files\n"
            + "\n".join(f"--- {path} ---\n{text}" for path, text in sorted(shown.items()))
            + "\n\n## Evaluation result\n"
            f"The agent attempted this task and scored {score:.2f} "
            "(1.00 only if the FAIL_TO_PASS tests pass and nothing else breaks).\n\n"
            f"{task.prompt}\n\n"
            f"Agent's attempt:\n{(output or '(no reply)')[:MAX_TRACE_CHARS]}\n\n"
            f"Test results:\n{outcome.report()}\n\n"
            "## Your task\n"
            "Improve the skill so the agent does better on OTHER bugs in this "
            "repository -- it will not be evaluated on this one. Record "
            "repository-specific conventions and debugging strategies that the "
            "evidence supports, keep patterns that worked, and stay concise; "
            "adding knowledge blindly will not help. Keep `SKILL.md` a valid "
            "Agent Skill: YAML frontmatter with `name: "
            f"{SKILL_NAME}` and a non-empty `description`.\n\n"
            + EDIT_PROTOCOL.format(max_files=strategy.max_files_per_diff,
                                   editable=", ".join(strategy.editable),
                                   frozen="(nothing)"),
            unit=task.id)

    def report() -> str:
        lines = [f"structural rejections: {dict(strategy.structural_rejections)}"]
        if export_dir:
            try:
                files = parse_tree(seen["rendered"])
                written = export_skill(files, export_dir)
            except (KeyError, TreeError, ValueError, OSError) as error:
                lines.append(f"[export-skill] refused: {error}")
            else:
                lines.append(f"[export-skill] wrote {len(written)} file(s) under "
                             f"{export_dir}/.claude/skills/{SKILL_NAME}/")
        return "\n".join(lines)

    train, held_out, test = gskill_splits(seed)
    return MethodPolicy(
        name="gskill",
        fidelity=FIDELITY,
        notes=(
            "Traced gepa-ai/gepa@f1cfdee src/gepa/gskill: optimize_anything evolves one repository skill from agent traces and test feedback; skills start empty and are deployed unchanged.",
            "The fitness is binary with a regression check, as swe_fitness_fn.py: all FAIL_TO_PASS pass and no PASS_TO_PASS regresses, else 0; no_patch / f2p_failed / p2p_regression are told apart in the reflector's feedback.",
            "The artifact is a skill directory (FileTree + the skill_dir validator): a candidate that deletes SKILL.md or breaks its frontmatter is rejected before it is measured and counted.",
            "No SWE-smith, no Docker, no real repository: twelve vendored bugs in a miniature repo stand in for the task set, and the 'agent' is one model call returning the repaired file rather than Mini-SWE-Agent. Real gskill numbers are neither claimed nor reproduced.",
            "Parent selection is the head under a strict held-out gate, not optimize_anything's per-instance Pareto frontier (the shared population layer carries no per-instance scores).",
            "One failed task per proposal, as --proposer loop's per-item calls; the follow-up merge call is not ported, so reflective=False.",
            "Splits are stratified by the repository's conventions: a skill learned on a train bug is measured on a different bug of the same convention.",
        ),
        strategy=strategy,
        train_tasks=tuple(train),
        held_out_tasks=tuple(held_out),
        test_tasks=tuple(test),
        solve=solve,
        propose=propose,
        reward=reward,
        proposal_calls_per_candidate=1,
        engine=Policies(acceptance=StrictHeldOutGate()),
        # `ReflectiveFusion` writes its synthesis to the ledger without passing
        # `to_diff`, so a model-merged SKILL.md would bypass the frontmatter gate.
        reflective=False,
        report=report,
    )


def _export_flag(parser) -> None:
    parser.add_argument(
        "--export-skill", default="", metavar="DIR",
        help=("after the run, write the evolved skill as DIR/.claude/skills/"
              f"{SKILL_NAME}/ (refused unless it is a valid Agent Skills tree)"))


def main(argv=None) -> int:
    return standard_main(build, argv, extra_args=_export_flag,
                         build_kwargs=lambda args: {"export_dir": args.export_skill})


if __name__ == "__main__":
    raise SystemExit(main())
