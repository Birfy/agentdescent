"""M0: evolve a skill your **real** DeepSeek Harness already loads.

Every other example builds a toy skill in a temp directory. This one does not:
it resolves the skill the way `dsh` resolves it -- five roots, rank order,
nearest wins -- runs the rollouts through a real harness over the Python SDK,
and can write the result back where the harness will pick it up.

    # what would happen, without calling a model
    python -m examples.dsh.evolve_dsh_skill --skill sql --tasks q.jsonl --dry-run

    # A: you have gold answers
    python -m examples.dsh.evolve_dsh_skill --skill sql --tasks q.jsonl --score contains

    # B: you have an objective instead -- the command is the judge
    python -m examples.dsh.evolve_dsh_skill --skill sql --tasks q.jsonl \\
        --check "psql --quiet -f {output}"

    # install it where the harness reads from (backs up first)
    python -m examples.dsh.evolve_dsh_skill --skill sql --tasks q.jsonl --install

`--tasks` is JSONL -- ``{"prompt": ..., "gold": ...}`` per line, ``gold``
optional -- and a line that is not JSON is taken as a bare prompt, so a plain
list of questions works too.

Needs ``pip install deepseek-harness-sdk`` and ``DEEPSEEK_API_KEY`` for a real
run. ``--dry-run`` needs neither: it resolves the skill, loads the tasks, builds
the objective and prints the plan. That is the part worth checking before
spending tokens, and it is what the test for this example exercises.

**Nothing is written unless you pass ``--install``.** The skill being evolved is
one a real harness is serving; a run that silently rewrote it would be changing
the user's agent underneath them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from agentdescent import Task, evolve_skill_dir
from agentdescent import rewards
from agentdescent.backends import dsh
from agentdescent.dsh import find_skill, list_skills

#: The gold-answer scorers, by the name you pass to ``--score``.
SCORERS = {
    "exact": rewards.exact_match,
    "contains": rewards.contains,
    "last_number": rewards.last_number,
    "numeric_close": rewards.numeric_close,
}


def load_tasks(path: str) -> List[Task]:
    """JSONL of ``{prompt, gold?}``; a non-JSON line is a bare prompt."""
    tasks: List[Task] = []
    with open(path, "r", encoding="utf-8") as fh:
        for n, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                row = {"prompt": line}
            if not isinstance(row, dict) or not row.get("prompt"):
                raise ValueError(
                    f"{path}:{n + 1} has no 'prompt'. Each line is either JSON "
                    f"with a 'prompt' key, or a bare prompt.")
            meta = {k: v for k, v in row.items() if k != "prompt"}
            tasks.append(Task(id=row.get("id") or f"t{len(tasks)}",
                              prompt=str(row["prompt"]), meta=meta))
    if not tasks:
        raise ValueError(f"{path} holds no tasks.")
    return tasks


def build_objective(args, tasks: List[Task]):
    """The reward, and a line describing it. Rung B beats rung A when both are given.

    Refusing to guess is the point: a run whose objective is not what the user
    meant produces a confidently wrong artifact, and there is no later stage that
    catches it.
    """
    if args.check:
        return rewards.command(args.check, timeout=args.check_timeout), \
            f"command exit 0 -- {args.check!r}"
    missing = [t.id for t in tasks if "gold" not in (t.meta or {})]
    if missing:
        raise SystemExit(
            f"--score {args.score} needs a 'gold' on every task; {len(missing)} "
            f"lack one (first: {missing[0]}). Give the tasks gold answers, or "
            f"pass --check with a command that judges the output instead.")
    return SCORERS[args.score](), f"{args.score} against task gold"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Evolve a skill your real dsh loads.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skill", help="skill name, resolved the way dsh resolves it")
    ap.add_argument("--tasks", help="JSONL of {prompt, gold?}")
    ap.add_argument("--list", action="store_true",
                    help="list the skills this harness would serve here, and exit")
    ap.add_argument("--cwd", default=None,
                    help="resolve project roots from here (default: this directory)")
    ap.add_argument("--score", default="contains", choices=sorted(SCORERS),
                    help="rung A: score against each task's gold (default: contains)")
    ap.add_argument("--check", default=None,
                    help="rung B: a command that judges the output; {output} is a "
                         "path to it. Wins over --score when both are given.")
    ap.add_argument("--check-timeout", type=float, default=30.0)
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--install", action="store_true",
                    help="write the evolved skill back where dsh reads it "
                         "(backs the current one up first)")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve everything and print the plan; call no model")
    args = ap.parse_args(argv)

    if args.list:
        found = list_skills(args.cwd)
        if not found:
            print("No skills found. Roots searched (rank order):")
            from agentdescent.dsh import skill_roots
            for root in skill_roots(args.cwd):
                print(f"  {root.rank:>3} {root.source:<16} {root.path}"
                      + ("" if root.exists else "   (absent)"))
            return 1
        for skill in found:
            shape = "dir " if skill.is_directory else "flat"
            print(f"{skill.name:<28} {shape}  {skill.root.source:<16} {skill.path}")
        return 0

    if not args.skill or not args.tasks:
        ap.error("--skill and --tasks are required (or use --list)")

    skill = find_skill(args.skill, args.cwd)
    if skill is None:
        print(f"No skill named {args.skill!r}. Try --list.", file=sys.stderr)
        return 1
    if not skill.is_directory:
        # A flat skill is one Markdown file sitting directly in a root. Evolving
        # it as a "directory" would load every *other* skill in that root as part
        # of the artifact and write them all back.
        print(f"{skill.name!r} is a flat file ({skill.path}), not a skill "
              f"directory. Convert it to {args.skill}/SKILL.md first -- evolving "
              f"it in place would take its whole root along with it.",
              file=sys.stderr)
        return 1

    tasks = load_tasks(args.tasks)
    reward, objective = build_objective(args, tasks)

    print(f"skill     : {skill.name}  ({skill.root.source}, rank {skill.root.rank})")
    print(f"path      : {skill.path}")
    print(f"tasks     : {len(tasks)} from {args.tasks}")
    print(f"objective : {objective}")
    print(f"agent     : dsh({args.model})")
    print(f"install   : {'yes, after the run' if args.install else 'no (--install to write back)'}")

    if args.dry_run:
        print("\ndry run -- nothing was called, nothing was written.")
        return 0

    result = evolve_skill_dir(
        skill.path, tasks, agent=dsh(model=args.model), score=reward,
        layout="dsh_skill",          # where *this* harness looks; see LAYOUTS
        rounds=args.rounds, n_workers=args.workers, max_concurrency=args.workers,
        verbose=True)

    print(f"\nfinal reward : {result.final_reward:.3f}")
    print(f"stop reason  : {result.stop_reason}")
    print(f"outcomes     : {result.outcomes()}")
    print(f"error        : {result.error}")

    if args.install:
        plan = result.write_to(skill.path)
        backup = plan["backup"][0] if plan.get("backup") else None
        print(f"\ninstalled {len(plan['written'])} file(s) to {skill.path}"
              + (f" (backup: {backup})" if backup else ""))
    else:
        print("\nnot installed. Re-run with --install to write this back.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
