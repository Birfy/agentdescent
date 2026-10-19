"""Risk tier (T0-T3) of a pull request, computed from its diff.

    PR=12 GH_REPO=owner/repo GH_TOKEN=... python3 risk_gate.py    # what the workflow runs
    python3 .github/scripts/risk_gate.py --pr 12 [--repo owner/repo]
    python3 .github/scripts/risk_gate.py --replay 30 [--repo owner/repo]   # last N merged PRs

The pull request is read only through the GitHub API (the changed files and
their patches); no PR code is checked out or run. The tier is the highest of:
the path of every changed file (and the old path of a rename), content hits in
the added lines of code files, and a size cap. It never comes from PR text.
Exceeding the size cap of the computed tier makes the PR T3 directly.

The script prints the tier and exits 0; it exits 1 only when it could not
compute one. Report only: the workflow never blocks, labels or merges (the
controller sets the ``risk:T*`` labels from the same rules).

Standard library only, Python 3.9+, the ``gh`` CLI for the API. Ported from
ThinkFlowLab/FlyBot ``.github/scripts/risk-gate.mjs``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Pattern

ORDER = ["T0", "T1", "T2", "T3"]
MAX_FILES = 3000  # the pull request files API stops here
HERE = os.path.dirname(os.path.abspath(__file__))
RULES = os.path.join(HERE, "..", "agent", "risk-rules.json")


def glob_to_re(glob: str) -> Pattern[str]:
    """``**/`` any directories (or none), ``**`` anything, ``*`` within one
    segment, ``?`` one character, ``{a,b}`` literal alternatives. Everything
    else is literal."""
    out = []
    i = 0
    while i < len(glob):
        c = glob[i]
        if glob.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif glob.startswith("**", i):
            out.append(".*")
            i += 2
        elif c == "*":
            out.append("[^/]*")
            i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        elif c == "{" and glob.find("}", i) > i:
            j = glob.find("}", i)
            out.append("(?:" + "|".join(re.escape(a) for a in glob[i + 1:j].split(",")) + ")")
            i = j + 1
        else:
            out.append(re.escape(c))
            i += 1
    return re.compile("^" + "".join(out) + "$")


class Gate:
    def __init__(self, rules: Dict[str, Any]):
        self.rules = rules
        t3 = rules["tiers"]["T3"]
        self.paths = {t: [glob_to_re(g) for g in d.get("paths", [])] for t, d in rules["tiers"].items()}
        self.generated = [glob_to_re(g) for g in rules.get("generatedPaths", [])]
        self.code_ext = re.compile(t3["codeExt"])
        self.patterns = [(p, re.compile(p, re.M)) for p in t3.get("diffPatterns", [])]
        # Code outside tests only: a test may legitimately spawn a process or read the env.
        self.prose_patterns = [(p, re.compile(p, re.M)) for p in t3.get("diffPatternsNotInTests", [])]
        self.tests = [glob_to_re(g) for g in t3.get("testPaths", [])]
        self.topics = list(t3.get("topics", []))
        self.unknown = rules.get("unknownPathTier", "T2")
        self.meaning = {t: d.get("meaning", "") for t, d in rules["tiers"].items()}

    def is_generated(self, name: str) -> bool:
        return any(r.match(name) for r in self.generated)

    def tier_of(self, name: str) -> str:
        for t in ("T3", "T2", "T1", "T0"):
            if any(r.match(name) for r in self.paths.get(t, [])):
                return t
        return self.unknown

    def judge(self, files: List[Dict[str, Any]], truncated: bool = False) -> Dict[str, Any]:
        """``files``: the array ``GET /pulls/{n}/files`` returns."""
        state = {"tier": "T0"}
        why: List[str] = []

        def bump(t: str, reason: Optional[str]) -> None:
            if ORDER.index(t) > ORDER.index(state["tier"]):
                state["tier"] = t
            if reason and reason not in why:
                why.append(reason)

        if truncated or len(files) >= MAX_FILES:
            bump("T3", f"file list may be incomplete ({len(files)} files)")
        for f in files:
            for name in (f.get("filename"), f.get("previous_filename")):
                if name:
                    t = self.tier_of(name)
                    bump(t, f"path {name}" if t == "T3" else None)
            name = f.get("filename", "")
            if self.is_generated(name):
                continue  # regenerated and checked by its own test, not read here
            if f.get("changes", 0) > 0 and f.get("patch") is None:
                bump("T3", f"diff not inspectable: {name}")
                continue
            if not self.code_ext.search(name):
                continue  # the patterns look at code, not prose
            added = "\n".join(line for line in (f.get("patch") or "").split("\n")
                              if line.startswith("+") and not line.startswith("+++"))
            is_test = any(r.match(name) for r in self.tests)
            active = self.patterns if is_test else self.patterns + self.prose_patterns
            for src, rx in active:
                if rx.search(added):
                    bump("T3", f"{name}: /{src[:60]}/")
            for word in self.topics:
                if word in added:
                    bump("T3", f"{name}: topic {word}")
        real = [f for f in files if not self.is_generated(f.get("filename", ""))]
        lines = sum(f.get("additions", 0) + f.get("deletions", 0) for f in real)
        lim = self.rules.get("limits", {}).get(state["tier"])
        if lim and (lines > lim["lines"] or len(real) > lim["files"]):
            bump("T3", f"size {lines} lines / {len(real)} files > {lim['lines']} / {lim['files']} "
                       f"allowed for {state['tier']}")
        return {"tier": state["tier"], "why": why, "lines": lines, "files": len(real),
                "generated": len(files) - len(real)}


def render(result: Dict[str, Any], meaning: Dict[str, str]) -> str:
    gen = result["generated"]
    out = [
        f"### Risk tier: {result['tier']}",
        "",
        meaning.get(result["tier"], ""),
        "",
        f"Size (generated files excluded): {result['lines']} changed lines in {result['files']} files"
        + (f"; {gen} generated file(s) not counted" if gen else "") + ".",
    ]
    if result["why"]:
        out += ["", "Why:"] + [f"- {w}" for w in result["why"][:20]]
        if len(result["why"]) > 20:
            out.append(f"- ... and {len(result['why']) - 20} more")
    out += ["", "_Report only: this check never blocks, labels or merges. Rules: "
                "`.github/agent/risk-rules.json`; merge policy: AGENTS.md › Merge policy._"]
    return "\n".join(out)


def load_rules(path: str = RULES) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _gh(*args: str) -> str:
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout


def gh_list(path: str) -> List[Dict[str, Any]]:
    out = _gh("api", "--paginate", "--jq", ".[]", path)
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def pr_files(repo: str, number: int) -> List[Dict[str, Any]]:
    return gh_list(f"repos/{repo}/pulls/{number}/files?per_page=100")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Risk tier of a pull request (report only).")
    ap.add_argument("--repo", default=os.environ.get("GH_REPO"))
    ap.add_argument("--pr", type=int, default=int(os.environ["PR"]) if os.environ.get("PR") else None)
    ap.add_argument("--replay", type=int, help="tiers of the last N merged pull requests")
    ap.add_argument("--rules", default=os.environ.get("RISK_RULES", RULES))
    args = ap.parse_args(argv)
    if not args.repo:
        ap.error("set GH_REPO or pass --repo owner/name")
    gate = Gate(load_rules(args.rules))

    if args.replay:
        prs = json.loads(_gh("pr", "list", "--repo", args.repo, "--state", "merged",
                             "--limit", str(args.replay), "--json", "number,title"))
        dist = {t: 0 for t in ORDER}
        for pr in sorted(prs, key=lambda p: p["number"]):
            r = gate.judge(pr_files(args.repo, pr["number"]))
            dist[r["tier"]] += 1
            print(f"#{pr['number']}\t{r['tier']}\t{r['lines']}L/{r['files']}F\t{pr['title'][:60]}\n"
                  f"\t{' | '.join(r['why'][:4])}")
        print(f"\nDistribution over {len(prs)} merged PRs: "
              + ", ".join(f"{t} {dist[t]}" for t in ORDER))
        return 0

    if not args.pr:
        ap.error("set PR or pass --pr N (or --replay N)")
    r = gate.judge(pr_files(args.repo, args.pr))
    text = render(r, gate.meaning)
    print(f"risk-gate (report only): PR #{args.pr} tier {r['tier']}"
          + (("; " + "; ".join(r["why"][:5])) if r["why"] else ""))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")
    else:
        print("\n" + text)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (subprocess.CalledProcessError, ValueError, KeyError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        print(f"risk-gate: {str(detail).strip()[:500]}", file=sys.stderr)
        sys.exit(1)
