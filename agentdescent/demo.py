"""A complete, working evolution you can run with no API key: ``agentdescent demo``.

Every other entry point needs three things a newcomer does not have yet -- a
worker agent on ``PATH``, a provider key, and a dataset with known answers --
so the first thing they see is an error about a missing file. This module is
the answer to *"show me it working"*: one command builds a real skill
directory, a real dataset, and runs the real loop against a subprocess agent
that reads the skill off disk.

Nothing here is a mock of the framework. The agent is a separate process bound
to a workspace, so staging, the layout, the ledger, the merge and the held-out
gate all run exactly as they do against Claude Code. Only the *model* is
replaced: the "agent" obeys the skill mechanically, and the "reflector" writes
the one edit a model would write. That is the smallest change that removes the
API key while leaving the machinery real.

The two callables are public and resolvable from a spec
(``{"ref": "agentdescent.demo:offline_agent"}``), so anyone can point their own
spec at them to rehearse a run before spending anything.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
from typing import Any, Dict, List

from .agents import Completion, cli_agent

__all__ = [
    "DEMO_SKILL",
    "build",
    "demo_spec",
    "offline_agent",
    "offline_reflector",
]

#: The demo's skill directory name, and the artifact id the run reports.
DEMO_SKILL = "csv-total"

#: The agent, as a program. It reads `references/rules.md` out of the workspace
#: to learn which column to total -- so a skill that names the wrong column
#: really does produce wrong answers, and the run really does have to fix the
#: file rather than the prompt.
_AGENT_SRC = (
    "import csv,os,re,sys;"
    f"p=os.path.join('.claude','skills','{DEMO_SKILL}','references','rules.md');"
    "t=open(p).read() if os.path.exists(p) else '';"
    "m=re.search(r'COLUMN:\\s*(\\w+)',t);"
    "col=m.group(1) if m else 'id';"
    "rows=list(csv.DictReader(open('data.csv')));"
    "print(sum(int(r[col]) for r in rows) if rows and col in rows[0] else 'unknown')"
)

_SKILL_MD = f"""# {DEMO_SKILL}

Total one column of `data.csv`, which is in the working directory.

1. Read `references/rules.md` to find out **which** column to total.
2. Sum that column over every data row.
3. Reply with only the number.
"""

#: Wrong on purpose: `id` is a row number, not the quantity anyone wants summed.
#: The run's whole job is to discover that and rewrite this file.
_RULES_MD = "COLUMN: id\n"


def offline_agent() -> Completion:
    """A real workspace agent that needs no API key.

    A subprocess bound to the rollout's workspace, exactly like
    ``claude_code()`` -- it opens the candidate skill and obeys it. What it
    does not have is a model, which is what makes it free and deterministic.
    """
    return cli_agent([sys.executable, "-c", _AGENT_SRC])


def offline_reflector(prompt: str) -> str:
    """Stand in for the reflecting model: propose the fix the failure implies.

    A real reflector reads the same prompt and decides what to change. This one
    reads the current ``rules.md`` out of that prompt and names the column the
    data actually has, in the multi-file edit protocol a
    :class:`~agentdescent.treestrategy.FileTree` run expects.
    """
    match = re.search(r"--- references/rules\.md ---\n(.*?)\n(?:---|TASK)", prompt, re.S)
    body = (match.group(1) if match else "").strip()
    fixed = re.sub(r"COLUMN:\s*\w+", "COLUMN: amount", body) if body else "COLUMN: amount"
    if "COLUMN:" not in fixed:
        fixed = "COLUMN: amount"
    return "<EDITS>" + json.dumps({
        "rationale": "`id` is a row number; the quantity asked for is `amount`.",
        "edits": [{"path": "references/rules.md", "content": fixed + "\n"}],
    }) + "</EDITS>"


def _rows(n: int = 12, seed: int = 0) -> List[Dict[str, Any]]:
    """One CSV per task, staged into the workspace beside the skill."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        amounts = [rng.randint(1, 99) for _ in range(rng.randint(3, 6))]
        csv = "id,amount\n" + "\n".join(
            f"{i + 1},{a}" for i, a in enumerate(amounts))
        out.append({"prompt": "What is the total?", "gold": str(sum(amounts)),
                    "fixtures": {"data.csv": csv}})
    return out


def demo_spec(root: str) -> Dict[str, Any]:
    """The spec the demo runs: an ordinary one, pointed at the offline pair."""
    return {
        "version": 1,
        "kind": "skill_dir",
        "target": os.path.join(root, DEMO_SKILL),
        "data": {"path": os.path.join(root, "cases.jsonl"),
                 "prompt": "prompt", "gold": "gold"},
        "score": "exact",
        "agent": {"ref": "agentdescent.demo:offline_agent"},
        "reflect": {"ref": "agentdescent.demo:offline_reflector", "call": False},
        # The stub agent echoes what it is asked, so the task reaches it as-is.
        "prompt_template": "{prompt}",
        "evolve": {"rounds": 4, "n_workers": 2, "seed": 0},
    }


def build(root: str) -> Dict[str, Any]:
    """Write the demo's skill, dataset and spec under ``root``. Returns the spec."""
    skill = os.path.join(root, DEMO_SKILL)
    os.makedirs(os.path.join(skill, "references"), exist_ok=True)
    with open(os.path.join(skill, "SKILL.md"), "w", encoding="utf-8") as fh:
        fh.write(_SKILL_MD)
    with open(os.path.join(skill, "references", "rules.md"), "w", encoding="utf-8") as fh:
        fh.write(_RULES_MD)
    with open(os.path.join(root, "cases.jsonl"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(json.dumps(r) for r in _rows()) + "\n")
    spec = demo_spec(root)
    with open(os.path.join(root, "spec.json"), "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2)
    return spec
