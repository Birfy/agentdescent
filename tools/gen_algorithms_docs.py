"""Generate ``docs/algorithms.md`` from the in-package algorithm registry (#384).

The catalog has to be the same list ``agentdescent algorithms list`` prints and
the same cards ``show`` prints, or a user reads about a method the plugin cannot
run. This renders the page from :mod:`agentdescent.algorithms`.

    python -m tools.gen_algorithms_docs            # rewrite docs/algorithms.md
    python -m tools.gen_algorithms_docs --check    # exit 1 if out of date (CI)
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentdescent import algorithms  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "docs", "algorithms.md")
MAX_DIFF_LINES = 60


def _cell(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text.replace("|", "\\|").replace("\n", " ")


def _card(algorithm: algorithms.Algorithm) -> List[str]:
    card: Dict[str, Any] = algorithm.describe()
    lines = [
        f"## `{card['name']}`",
        "",
        card["summary"],
        "",
        f"- **Kinds:** {', '.join(f'`{k}`' for k in card['kinds'])}",
        f"- **Fidelity:** `{card['fidelity']}`",
        f"- **Cost:** {card['extra_calls']}",
    ]
    if card.get("source"):
        lines.append(f"- **Source:** {card['source']}")
    if card.get("doc"):
        lines.append(f"- **Details:** [{card['doc']}]({card['doc']})")
    lines.append(f"- **Spec:** `\"algorithm\": \"{card['name']}\"`")
    if card["params"]:
        lines += ["", "| Parameter | Type | Default | Meaning |",
                  "|---|---|---|---|"]
        for param in card["params"]:
            lines.append(
                f"| `{param['name']}` | `{param['type']}` | "
                f"`{_cell(param['default'])}` | {_cell(param['help'])} |")
    lines.append("")
    return lines


def render() -> str:
    lines = [
        "# Named algorithms",
        "",
        "*Generated from `agentdescent.algorithms`; do not edit by hand.*",
        "",
        "These are the methods a spec (or `agentdescent init`) can ask for by",
        "name. An algorithm contributes **defaults** — policies, `evolve()`",
        "knobs, an `agg_config` — that the same spec can override field by field.",
        "The default recipe for a `kind` is used when no algorithm is named, and",
        "`plan` always echoes what was resolved.",
        "",
        "```bash",
        "agentdescent algorithms [--kind K] [--json]      # same as `algorithms list`",
        "agentdescent algorithms list [--kind K] [--json]",
        "agentdescent algorithms show NAME",
        "agentdescent init PATH --algorithm NAME --param KEY=VALUE",
        "```",
        "",
        "To **register one named card in a single PR**, follow",
        "[Register a named Algorithm card](algorithm-authoring.md).",
        "Full research ports still use the",
        "[porting checklist](porting-checklist.md); MethodPolicy microports use",
        "[porting-methodpolicy.md](porting-methodpolicy.md).",
        "",
    ]
    for algorithm in algorithms.listing(builtin_only=True):
        lines += _card(algorithm)
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="exit 1 if docs/algorithms.md is out of date")
    args = parser.parse_args()
    generated = render()
    if args.check:
        current = open(OUT, encoding="utf-8").read() if os.path.exists(OUT) else ""
        if current != generated:
            print("docs/algorithms.md is out of date; run: "
                  "python -m tools.gen_algorithms_docs", file=sys.stderr)
            diff = list(difflib.unified_diff(
                current.splitlines(), generated.splitlines(),
                "docs/algorithms.md", "generated", lineterm=""))
            head, rest = diff[:MAX_DIFF_LINES], len(diff) - MAX_DIFF_LINES
            print("\n".join(head), file=sys.stderr)
            if rest > 0:
                print(f"... and {rest} more diff lines", file=sys.stderr)
            return 1
        print("docs/algorithms.md is up to date")
        return 0
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(generated)
    print(f"wrote {OUT} ({len(generated.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
