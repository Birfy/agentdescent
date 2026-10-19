"""Create or update the labels listed in ``.github/labels.json``.

    python3 .github/scripts/sync_labels.py --repo Birfy/agentdescent           # dry run: print the plan
    python3 .github/scripts/sync_labels.py --repo Birfy/agentdescent --apply   # create / update

The ``labels`` workflow runs it with ``--apply`` when ``labels.json`` changes on
``main`` (built-in ``GITHUB_TOKEN`` only); a maintainer may also run it by hand
with their own ``gh`` login. It never deletes a label and never touches a label
that ``labels.json`` does not list: a label attached to an open issue is
remapped by a person, not dropped by a script.

Standard library only, Python 3.9+, the ``gh`` CLI for the API.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional
from urllib.parse import quote

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = os.path.join(HERE, "..", "labels.json")
GROUPS = ("type", "area", "source", "state", "risk")
COLOR = re.compile(r"^[0-9a-fA-F]{6}$")


def load(path: str = LABELS) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def validate(wanted: List[Dict[str, str]]) -> List[str]:
    """Problems with a labels.json list; empty when it is fine."""
    problems = []
    seen = set()
    for i, w in enumerate(wanted):
        name = w.get("name", "")
        if not name or len(name) > 50:
            problems.append(f"entry {i}: name must be 1-50 characters")
        if name.lower() in seen:
            problems.append(f"{name}: listed twice")
        seen.add(name.lower())
        if not COLOR.match(w.get("color", "")):
            problems.append(f"{name}: color must be 6 hex digits without '#'")
        if len(w.get("description", "")) > 100:
            problems.append(f"{name}: GitHub caps descriptions at 100 characters")
        if w.get("group") not in GROUPS:
            problems.append(f"{name}: group must be one of {GROUPS}")
        if w.get("group") == "area" and not name.startswith("area:"):
            problems.append(f"{name}: area labels start with 'area:'")
        if w.get("group") == "risk" and not re.match(r"^risk:T[0-3]$", name):
            problems.append(f"{name}: risk labels are risk:T0..risk:T3")
    return problems


def plan(wanted: List[Dict[str, str]], existing: List[Dict[str, str]]) -> List[Dict[str, object]]:
    """One ``create`` / ``update`` / ``keep`` step per wanted label.

    Names compare case-insensitively, like GitHub. Nothing in ``existing`` that
    is not wanted appears in the plan, so nothing is ever deleted.
    """
    have = {e["name"].lower(): e for e in existing}
    steps = []
    for w in wanted:
        cur = have.get(w["name"].lower())
        if cur is None:
            steps.append({"action": "create", "label": w})
            continue
        same = (cur.get("color", "").lower() == w["color"].lower()
                and (cur.get("description") or "") == w["description"]
                and cur["name"] == w["name"])
        steps.append({"action": "keep" if same else "update", "label": w, "current": cur})
    return steps


def _gh(*args: str) -> str:
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout


def _existing(repo: str) -> List[Dict[str, str]]:
    out = _gh("api", "--paginate", "--jq", ".[]", f"repos/{repo}/labels?per_page=100")
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", default=os.environ.get("GH_REPO"), help="owner/name")
    ap.add_argument("--apply", action="store_true", help="make the changes (default: dry run)")
    args = ap.parse_args(argv)
    if not args.repo or not re.match(r"^[\w.-]+/[\w.-]+$", args.repo):
        ap.error("pass --repo owner/name (or set GH_REPO)")
    wanted = load()
    problems = validate(wanted)
    if problems:
        print("sync-labels: labels.json is invalid:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    for step in plan(wanted, _existing(args.repo)):
        label = step["label"]
        print(f"{step['action']:<6} {label['name']}")
        if not args.apply or step["action"] == "keep":
            continue
        fields = ["-f", f"color={label['color']}", "-f", f"description={label['description']}"]
        if step["action"] == "create":
            _gh("api", "-X", "POST", f"repos/{args.repo}/labels", "-f", f"name={label['name']}", *fields)
        else:
            current = step["current"]["name"]
            _gh("api", "-X", "PATCH", f"repos/{args.repo}/labels/{quote(current, safe='')}",
                "-f", f"new_name={label['name']}", *fields)
    if not args.apply:
        print("\nDry run. Add --apply to make these changes.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as exc:
        print(f"sync-labels: gh failed: {(exc.stderr or '').strip()[:500]}", file=sys.stderr)
        sys.exit(1)
