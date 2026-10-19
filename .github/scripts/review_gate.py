"""Does a pull request's current head carry a reviewer PASS?

    PR=12 GH_REPO=owner/repo GH_TOKEN=... python3 review_gate.py   # what the workflow runs
    python3 .github/scripts/review_gate.py --pr 12 [--repo owner/repo] [--sha <head>]

``decide`` is the whole rule. A report counts when:
  - it is a PR comment or a review body by a login in
    ``.github/agent/review-policy.json``'s ``reviewers`` (the base branch's copy);
  - it is not the PR author's own report, unless the author is in ``selfReview``;
  - it carries the Verifier block (see .claude/skills/review-pr/SKILL.md) with a
    ``Tip SHA:`` that is the head being judged and a ``Verdict:``;
  - for a PASS: it lists at least one command it ``Ran:`` and has a
    ``Not run:`` line (a report that ran nothing is not evidence).
The newest qualifying report that names this tip decides, so a later FAIL at
the same tip vetoes an earlier PASS. A new commit moves the tip and every old
report stops counting.

The PR is read only through the GitHub API; no PR code is checked out or run.
Exits 0 on a PASS at the tip, 1 otherwise (the workflow step is neutral).

Standard library only, Python 3.9+, the ``gh`` CLI for the API. Ported from
ThinkFlowLab/FlyBot ``.github/scripts/review-gate.mjs``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

SHORT = 12
HERE = os.path.dirname(os.path.abspath(__file__))
POLICY = os.path.join(HERE, "..", "agent", "review-policy.json")


def parse_report(body: Optional[str]) -> Dict[str, Any]:
    """The LAST Verifier block in one body: a comment may quote an earlier
    review, and only the last block is the verdict the reviewer stands behind."""
    lines = re.sub(r"[*`]", "", str(body or "")).split("\n")
    start = -1
    for i, line in enumerate(lines):
        if re.match(r"^Tip SHA\s*:", line.strip(), re.I):
            start = i
    out: Dict[str, Any] = {"tip": None, "verdict": None, "ran": [], "hasNotRun": False}
    i = max(start, 0)
    while i < len(lines):
        line = lines[i].strip()
        m = re.match(r"^Tip SHA\s*:\s*(.+)$", line, re.I)
        if m:
            hexes = re.search(r"[0-9a-f]{7,40}", m.group(1), re.I)
            out["tip"] = hexes.group(0).lower() if hexes else None
        m = re.match(r"^Verdict\s*:\s*(PASS|FAIL|INC)\b", line, re.I)
        if m:
            out["verdict"] = m.group(1).upper()
        m = re.match(r"^Ran\s*:\s*(.*)$", line, re.I)
        if m:
            out["ran"] = []
            first = m.group(1).strip()
            if first and first != "-":
                out["ran"].append(first)
            for nxt in (l.strip() for l in lines[i + 1:]):
                if re.match(r"^[-\u2022]\s*\S", nxt):
                    out["ran"].append(nxt)
                elif nxt == "" and out["ran"]:
                    break
                elif nxt != "":
                    break
        if re.match(r"^Not run\s*:", line, re.I):
            out["hasNotRun"] = True
        i += 1
    return out


def _login(r: Dict[str, Any]) -> str:
    return str(r.get("login") or "").lower()


def _where(r: Dict[str, Any]) -> str:
    return f"{'review' if r.get('kind') == 'review' else 'comment'} by {r.get('login')}"


def _ts(at: Optional[str]) -> float:
    try:
        return datetime.fromisoformat(str(at).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def decide(head: str, author: str, reports: List[Dict[str, Any]], policy: Dict[str, Any]) -> Dict[str, Any]:
    """reports: [{login, body, at, kind}] from the comments and review bodies of one PR."""
    head_sha = str(head or "").lower()
    author_login = str(author or "").lower()
    reviewers = {str(l).lower() for l in policy.get("reviewers", [])}
    self_review = {str(l).lower() for l in policy.get("selfReview", [])}
    reasons: List[str] = []
    judged: List[Dict[str, Any]] = []

    for r in reports:
        parsed = parse_report(r.get("body"))
        if not parsed["tip"] and not parsed["verdict"]:
            continue  # not a Verifier report
        entry = {"login": r.get("login"), "verdict": parsed["verdict"], "tip": parsed["tip"], "at": r.get("at"),
                 "kind": r.get("kind"), "parsed": parsed, "counts": False}
        judged.append(entry)
        if not parsed["tip"]:
            reasons.append(f"{_where(r)}: no Tip SHA")
            continue
        if not head_sha.startswith(parsed["tip"]):
            reasons.append(f"{_where(r)}: names {parsed['tip'][:SHORT]}, not the tip {head_sha[:SHORT]}")
            continue
        if _login(r) == author_login and author_login not in self_review:
            reasons.append(f"{_where(r)}: the pull request's own author is not a reviewer (not in selfReview)")
            continue
        if _login(r) not in reviewers:
            reasons.append(f"{_where(r)}: not a reviewer listed in .github/agent/review-policy.json")
            continue
        entry["counts"] = True

    counting = [e for e in judged if e["counts"]]
    if not counting:
        if not reviewers:
            reason = "no reviewer is listed in .github/agent/review-policy.json"
        else:
            reason = "; ".join(reasons) if reasons else f"no Verifier report names the tip {head_sha[:SHORT]}"
        return {"ok": False, "verdict": None, "by": None, "tip": None, "reason": reason, "judged": judged}

    # Newest wins; equal timestamps keep the API order (oldest first), so the later comment wins.
    last = sorted(enumerate(counting), key=lambda p: (_ts(p[1]["at"]), p[0]))[-1][1]
    parsed = last["parsed"]
    base = {"verdict": last["verdict"], "by": last["login"], "tip": last["tip"], "judged": judged}
    if last["verdict"] != "PASS":
        return dict(base, ok=False, reason=f"{last['login']} says {last['verdict'] or 'nothing'} at this tip")
    if not parsed["ran"]:
        return dict(base, ok=False, reason=f"{last['login']}'s PASS lists no command it ran")
    if not parsed["hasNotRun"]:
        return dict(base, ok=False, reason=f"{last['login']}'s PASS has no \"Not run:\" line")
    return dict(base, ok=True, reason=f"{last['login']}'s PASS names this tip")


def render(result: Dict[str, Any], pr: Any, head: str, author: str) -> str:
    out = [
        f"### Reviewed tip: {'PASS at ' + head[:SHORT] if result['ok'] else 'not reviewed'}",
        "",
        f"{result['by']} passed this tip." if result["ok"] else f"No PASS names this tip: {result['reason']}.",
        "",
        f"Pull request #{pr}, author {author}, head `{head[:SHORT]}`.",
    ]
    seen = [e for e in result["judged"] if e["counts"] or e["verdict"]]
    if seen:
        out += ["", "Verifier reports read:"]
        out += [f"- {e['login']} ({e['kind']}): {e['verdict'] or 'no verdict'} at {(e['tip'] or 'no tip')[:SHORT]}"
                + ("" if e["counts"] else " (not counted)") for e in seen[:10]]
    out += ["", "_Report only: this check never blocks, labels, approves or merges. Rules: "
                "`.github/agent/review-policy.json`; merge policy: AGENTS.md › Merge policy._"]
    return "\n".join(out)


def _gh(*args: str) -> str:
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout


def _list(path: str) -> List[Dict[str, Any]]:
    out = _gh("api", "--paginate", "--jq", ".[]", path)
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Does the PR's tip carry a PASS? (report only)")
    ap.add_argument("--repo", default=os.environ.get("GH_REPO"))
    ap.add_argument("--pr", type=int, default=int(os.environ["PR"]) if os.environ.get("PR") else None)
    ap.add_argument("--sha", help="judge this tip instead of the current head")
    ap.add_argument("--policy", default=os.environ.get("REVIEW_POLICY", POLICY))
    args = ap.parse_args(argv)
    if not args.repo or not args.pr:
        ap.error("set GH_REPO and PR, or pass --repo owner/name --pr N")
    with open(args.policy, encoding="utf-8") as fh:
        policy = json.load(fh)
    pull = json.loads(_gh("api", f"repos/{args.repo}/pulls/{args.pr}"))
    head = args.sha or pull["head"]["sha"]
    author = (pull.get("user") or {}).get("login", "")
    reports = [{"kind": "comment", "login": (c.get("user") or {}).get("login"), "body": c.get("body"),
                "at": c.get("created_at")}
               for c in _list(f"repos/{args.repo}/issues/{args.pr}/comments?per_page=100")]
    reports += [{"kind": "review", "login": (v.get("user") or {}).get("login"), "body": v.get("body"),
                 "at": v.get("submitted_at")}
                for v in _list(f"repos/{args.repo}/pulls/{args.pr}/reviews?per_page=100")
                if (v.get("body") or "").strip()]
    result = decide(head, author, reports, policy)
    print(f"review-gate (report only): PR #{args.pr} tip {head[:SHORT]} - "
          + (f"PASS by {result['by']}" if result["ok"] else result["reason"]))
    text = render(result, args.pr, head, author)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")
    else:
        print("\n" + text)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (subprocess.CalledProcessError, ValueError, KeyError, OSError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        print(f"review-gate: {str(detail).strip()[:500]}", file=sys.stderr)
        sys.exit(1)
