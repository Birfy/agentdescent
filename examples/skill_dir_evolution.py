"""Evolve a **skill directory** that a real agent reads off disk.

Every other example evolves text that ends up *in a prompt*. This one evolves a
directory: the candidate is materialised into a throwaway workspace at
``.claude/skills/<name>/`` and the agent has to open the files to know what to do.
That is the whole point -- the skill only helps if the agent reads it, which means
the run measures the directory, not the model's prior knowledge.

The task: a CSV is staged in the workspace and the agent must total one of its
columns. The skill says which one. It starts out **wrong** (``COLUMN: id``), so
the initial reward is 0 and the loop has somewhere to go.

    python -m examples.skill_dir_evolution                  # offline, no API key
    python -m examples.skill_dir_evolution --agent claude-code
    python -m examples.skill_dir_evolution --agent claude-code --model claude-haiku-4-5

The offline mode is not a mock of the framework: the agent really is a separate
process bound to a workspace (`cli_agent`), and the only stub is the *reflector*,
which returns a fixed edit instead of calling a model. Swap in `--agent
claude-code` and `--model` and the same code path drives the real thing.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
import tempfile

from agentdescent import (
    AggregatorConfig, FileTree, evolve, load_tree, scorer, tree_reflector, tree_runner,
)
from agentdescent.agents import claude_code, cli_agent, codex, openai_compatible
from agentdescent.governance import SKILL_BLAST_RADIUS

SKILL_NAME = "csv-total"

# ---------------------------------------------------------------------------
# The skill directory this example starts from
# ---------------------------------------------------------------------------

_SKILL_MD = """# csv-total

Total one column of `data.csv`, which is in the working directory.

1. Read `references/rules.md` to find out **which** column to total.
2. Sum that column across every data row (skip the header).
3. Reply with only the number.
"""

_RULES_MD = """COLUMN: id
"""      # <- wrong on purpose: `id` is a row number, not the quantity asked for


def make_skill_dir() -> str:
    root = tempfile.mkdtemp(prefix="agentdescent-example-")
    path = os.path.join(root, SKILL_NAME)
    os.makedirs(os.path.join(path, "references"))
    with open(os.path.join(path, "SKILL.md"), "w") as fh:
        fh.write(_SKILL_MD)
    with open(os.path.join(path, "references", "rules.md"), "w") as fh:
        fh.write(_RULES_MD)
    return path


# ---------------------------------------------------------------------------
# The dataset: one CSV per task, staged into the workspace as a fixture
# ---------------------------------------------------------------------------


def make_rows(n: int = 16, seed: int = 0):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        amounts = [rng.randint(1, 99) for _ in range(rng.randint(3, 6))]
        csv = "id,amount,note\n" + "\n".join(
            f"{j + 1},{a},row-{j}" for j, a in enumerate(amounts))
        rows.append({
            "prompt": "What is the total of the requested column in data.csv?",
            "gold": str(sum(amounts)),
            "fixtures": {"data.csv": csv},
        })
    return rows


def to_tasks(rows):
    from agentdescent import Task

    return [Task(id=f"t{i}", prompt=r["prompt"],
                 meta={"gold": r["gold"], "fixtures": r["fixtures"]})
            for i, r in enumerate(rows)]


# ---------------------------------------------------------------------------
# Offline stand-ins -- a real subprocess agent, and a fixed reflector
# ---------------------------------------------------------------------------

_OFFLINE_AGENT = f"""
import csv, os, re, sys
skill = os.path.join('.claude', 'skills', '{SKILL_NAME}')
rules = ''
p = os.path.join(skill, 'references', 'rules.md')
if os.path.exists(p):
    rules = open(p).read()
m = re.search(r'COLUMN:\\s*(\\w+)', rules)
column = m.group(1) if m else 'id'
rows = list(csv.DictReader(open('data.csv')))
total = sum(int(r[column]) for r in rows if r.get(column, '').strip().isdigit())
print(total)
"""


def offline_agent():
    """A genuine WorkspaceAgent: a separate process, bound to the workspace."""
    return cli_agent([sys.executable, "-c", _OFFLINE_AGENT])


def offline_reflector():
    """Returns the edit a model would return -- derived from what it is shown."""
    def complete(prompt: str) -> str:
        m = re.search(r"--- references/rules\.md ---\n(.*?)\n(?:---|TASK )",
                      prompt, re.DOTALL)
        body = (m.group(1) if m else "COLUMN: id").strip()
        fixed = re.sub(r"COLUMN:\s*\w+", "COLUMN: amount", body) or "COLUMN: amount"
        if fixed == body:
            fixed = "COLUMN: amount"
        return "<EDITS>" + json.dumps({
            "rationale": "the question asks for a quantity; `id` is a row number",
            "edits": [{"path": "references/rules.md", "content": fixed + "\n"}],
        }) + "</EDITS>"
    return complete


# ---------------------------------------------------------------------------
# The run: a directory handed to evolve()
# ---------------------------------------------------------------------------


def evolve_directory(path, tasks, *, agent, reflect_with, score="contains", **kw):
    """Everything a skill directory needs before ``evolve()`` -- spelled out.

    ``load_tree`` turns the directory into artifact state, ``FileTree`` makes
    file paths the keys the aggregator merges on, ``tree_runner`` materialises
    each candidate into a fresh workspace and runs the agent there, and
    ``tree_reflector`` asks the reflection model for multi-file edits. Two
    engine defaults are flipped because a rollout here is a real agent call:
    no self-verify re-run, and a four-task cheap layer for ranking.
    """
    name = os.path.basename(path)
    tree = load_tree(path)
    strategy = FileTree(tree, max_files_per_diff=2)
    run = tree_runner(agent, layout="claude_skill", name=name,
                      overlay=strategy.frozen_files(tree))
    for key, value in (("self_verify", False), ("cheap_eval_tasks", 4),
                       ("held_out_frac", 0.3), ("patience", 3), ("target_reward", 0.98),
                       ("agg_config", AggregatorConfig(batch_trigger=2, max_wait_rounds=1))):
        kw.setdefault(key, value)
    return evolve(tasks, scorer(score), run=run, strategy=strategy, artifact_id=name,
                  propose=tree_reflector(reflect_with, strategy=strategy),
                  blast_radius=SKILL_BLAST_RADIUS, **kw)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agent", default="offline",
                    choices=["offline", "claude-code", "codex"],
                    help="who performs the rollouts (default: a local subprocess)")
    ap.add_argument("--model", default=None,
                    help="reflector model on an OpenAI-compatible endpoint "
                         "(needs OPENAI_API_KEY / OPENAI_BASE_URL)")
    ap.add_argument("--tasks", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--install-to", default=None,
                    help="write the evolved skill here when the run improves it")
    args = ap.parse_args()

    path = make_skill_dir()
    if args.agent == "offline":
        agent = offline_agent()
    elif args.agent == "claude-code":
        agent = claude_code(extra_args=["--permission-mode", "acceptEdits"],
                            timeout=300.0)
    else:
        agent = codex(timeout=300.0)
    reflect = (openai_compatible(model=args.model) if args.model
               else (offline_reflector() if args.agent == "offline" else agent))

    print(f"Skill    : {path}")
    print(f"Agent    : {args.agent}")
    print(f"Reflector: {args.model or ('stub' if args.agent == 'offline' else args.agent)}")
    print(f"Start    : rules.md = {_RULES_MD.strip()!r}  (wrong column)\n")

    result = evolve_directory(
        path, to_tasks(make_rows(args.tasks)), agent=agent, reflect_with=reflect,
        rounds=args.rounds, n_workers=args.workers, max_concurrency=args.workers,
        verbose=True)

    print(f"\nfinal reward : {result.final_reward:.3f}")
    print(f"stop reason  : {result.stop_reason}")
    print(f"outcomes     : {result.outcomes()}")
    print(f"error        : {result.error}")
    print("\nevolved rules.md:")
    print(result.state.get("references/rules.md", "(unchanged)").strip())

    if args.install_to:
        plan = result.write_to(args.install_to)
        print(f"\ninstalled {len(plan['written'])} file(s) to {args.install_to}"
              + (f" (backup: {plan['backup'][0]})" if plan["backup"] else ""))
    shutil.rmtree(os.path.dirname(path), ignore_errors=True)


if __name__ == "__main__":
    main()
