"""Label-only controller (see AGENTS.md › Controller).

    python3 .github/scripts/controller.py plan     read the repo (GET only), print a plan, write it to $GITHUB_OUTPUT
    PLAN='{...}' python3 .github/scripts/controller.py apply   validate the plan, add/remove labels

It manages labels and nothing else. It never merges, approves, comments on or
closes anything, never edits an issue or PR, and never runs or checks out
PR/issue content: titles and bodies are data for one bounded regex
(``Fixes #N``); changed files are read only through the API and judged by the
risk gate's rules.

  - sets ``agent-pr`` on pull requests an agent opened, and ``risk:T0..T3``
    (the risk gate's tier) on every open PR not on ``hold``
  - removes ``agent-ready`` (and ``agent-working``) when someone who is not a
    maintainer labeled the issue, or edited its text after it was labeled
  - removes ``agent-working`` from a claim older than the TTL with no recent
    push on a linked PR

Mode: DRY RUN unless the repository variable ``CONTROLLER_LIVE`` is exactly
``true``. ``AGENT_PAUSE=true`` is the kill switch: nothing is read, proposed or
changed. A label that does not exist is skipped, never created.

Standard library only, Python 3.9+, the ``gh`` CLI for the API. Ported from
ThinkFlowLab/FlyBot ``.github/scripts/controller.mjs``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional
from urllib.parse import quote

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import risk_gate  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RISK = ["risk:T0", "risk:T1", "risk:T2", "risk:T3"]
MANAGED = {
    # label -> what the controller may do with it. It can never ADD agent-ready.
    "add": set(["agent-pr"] + RISK),
    "remove": set(["agent-ready", "agent-working"] + RISK),
}
MAX_OPS = 200
REASONS = {"labeled-by-non-maintainer", "edited-by-non-maintainer", "claim-expired", "risk-tier", "agent-pr"}


# ---- mode -------------------------------------------------------------------

def decide_mode(env: Dict[str, str]) -> str:
    """Live only on an exact ``true``; dry run is the default; a pause wins."""
    if env.get("AGENT_PAUSE") == "true":
        return "paused"
    return "live" if env.get("CONTROLLER_LIVE") == "true" else "dry-run"


# ---- pure planning ----------------------------------------------------------

def _same(a: Any, b: Any) -> bool:
    return str(a or "").lower() == str(b or "").lower()


def is_maintainer(config: Dict[str, Any], login: Optional[str]) -> bool:
    return bool(login) and any(_same(m, login) for m in config["maintainers"])


_CLOSING = re.compile(r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\b:?[ \t]+((?:[\w.-]+/[\w.-]+)?)#(\d{1,7})\b",
                      re.I)


def closing_refs(body: Optional[str], repo: str) -> List[int]:
    """``Fixes #12``, ``closes: #12``, ``Resolved owner/repo#12`` -> [12]."""
    out = set()
    for m in _CLOSING.finditer(str(body or "")[:20000]):
        if not m.group(1) or _same(m.group(1), repo):
            out.add(int(m.group(2)))
    return sorted(out)


def is_agent_pr(pr: Dict[str, Any], config: Dict[str, Any], repo: str, agent_issues: Iterable[int]) -> bool:
    """By author, or a branch of this repository with an agent prefix, or a
    branch of this repository that closes an agent-ready/agent-working issue.
    A fork PR counts only by author."""
    if any(_same(a, pr["author"]) for a in config["agentAuthors"]):
        return True
    if not pr["sameRepo"]:
        return False
    if any(p and pr["headRef"].startswith(p) for p in config["agentBranchPrefixes"]):
        return True
    agent_issues = set(agent_issues)
    return any(n in agent_issues for n in closing_refs(pr["body"], repo))


def _last(events: List[Dict[str, Any]], pred: Callable[[Dict[str, Any]], bool]) -> Optional[Dict[str, Any]]:
    hits = sorted((e for e in events if pred(e)), key=lambda e: e["at"])
    return hits[-1] if hits else None


def judge_ready(issue: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Optional[str]]:
    labeled = _last(issue["events"], lambda e: e["type"] == "labeled" and _same(e["label"], "agent-ready"))
    if not labeled:
        return {"demote": None, "note": "cannot find when agent-ready was set"}
    if not is_maintainer(config, labeled["actor"]):
        return {"demote": "labeled-by-non-maintainer", "note": None}
    edits = [(e["at"], e["editor"]) for e in issue["edits"]]
    edits += [(e["at"], e["actor"]) for e in issue["events"] if e["type"] == "renamed"]
    if any(at > labeled["at"] and not is_maintainer(config, who) for at, who in edits):
        return {"demote": "edited-by-non-maintainer", "note": None}
    return {"demote": None, "note": None}


def parse_time(ts: str) -> float:
    """GitHub ISO-8601 (``...Z``) -> epoch seconds. ``fromisoformat`` on 3.9
    does not take a trailing ``Z``."""
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()


def claim_expired(issue: Dict[str, Any], linked: List[Dict[str, Any]], config: Dict[str, Any], now: float) -> bool:
    """agent-working has been on longer than the TTL and no linked open PR has
    had a commit (or been opened) within it."""
    claimed = _last(issue["events"], lambda e: e["type"] == "labeled" and _same(e["label"], "agent-working"))
    if not claimed:
        return False
    ttl = config["claimTtlHours"] * 3600
    if now - parse_time(claimed["at"]) <= ttl:
        return False
    return not any(now - parse_time(p["lastPushAt"]) <= ttl for p in linked)


def _diff_labels(have: List[str], add: Iterable[str] = (), remove: Iterable[str] = ()) -> List[Dict[str, str]]:
    has = {h.lower() for h in have}
    ops = [{"op": "remove", "label": l} for l in remove if l in MANAGED["remove"] and l.lower() in has]
    ops += [{"op": "add", "label": l} for l in add if l in MANAGED["add"] and l.lower() not in has]
    return ops


def plan(state: Dict[str, Any]) -> Dict[str, List[Any]]:
    """state: repo, config, now (epoch s), repoLabels [names], issues, prs, linked {issue: [prs]}."""
    config = state["config"]
    exists = {l.lower() for l in state["repoLabels"]}
    ops: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    notes: List[str] = []

    def push(kind: str, number: int, found: List[Dict[str, str]], reason: str) -> None:
        for f in found:
            if f["op"] == "add" and f["label"].lower() not in exists:
                skipped.append({"number": number, "label": f["label"], "reason": "label-missing"})
            else:
                ops.append({"kind": kind, "number": number, "op": f["op"], "label": f["label"], "reason": reason})

    agent_issues = {i["number"] for i in state["issues"]}
    for issue in state["issues"]:
        labels = issue["labels"]
        if any(_same(l, "hold") for l in labels):
            skipped.append({"number": issue["number"], "label": "*", "reason": "hold"})
            continue
        ready = any(_same(l, "agent-ready") for l in labels)
        working = any(_same(l, "agent-working") for l in labels)
        # The approval is the last time a maintainer set agent-ready, even if it was
        # later swapped for agent-working: text changed by anyone else voids both.
        verdict = judge_ready(issue, config)
        if verdict["note"] and ready:
            notes.append(f"#{issue['number']}: {verdict['note']}")
        if verdict["demote"]:
            push("issue", issue["number"], _diff_labels(labels, remove=["agent-ready", "agent-working"]),
                 verdict["demote"])
            continue
        if working and claim_expired(issue, state["linked"].get(issue["number"], []), config, state["now"]):
            push("issue", issue["number"], _diff_labels(labels, remove=["agent-working"]), "claim-expired")

    for pr in state["prs"]:
        if any(_same(l, "hold") for l in pr["labels"]):
            skipped.append({"number": pr["number"], "label": "*", "reason": "hold"})
            continue
        if not pr["tier"]:
            continue  # the files could not be read; leave the labels as they are
        want = f"risk:{pr['tier']}"
        push("pr", pr["number"], _diff_labels(pr["labels"], add=[want], remove=[l for l in RISK if l != want]),
             "risk-tier")
        if is_agent_pr(pr, config, state["repo"], agent_issues):
            push("pr", pr["number"], _diff_labels(pr["labels"], add=["agent-pr"]), "agent-pr")
    if len(ops) > MAX_OPS:
        notes.append(f"{len(ops)} operations; keeping the first {MAX_OPS}")
        del ops[MAX_OPS:]
    return {"ops": ops, "skipped": skipped, "notes": notes}


# ---- plan <-> apply contract ------------------------------------------------

def validate_plan(p: Any) -> Dict[str, Any]:
    """What ``apply`` accepts. Anything else is refused, whoever wrote the plan."""
    if not isinstance(p, dict) or p.get("version") != 1 or not isinstance(p.get("ops"), list):
        raise ValueError("not a controller plan")
    if len(p["ops"]) > MAX_OPS:
        raise ValueError(f"too many operations ({len(p['ops'])})")
    for o in p["ops"]:
        if not isinstance(o, dict):
            raise ValueError("bad operation")
        n = o.get("number")
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise ValueError("bad issue number")
        if o.get("op") not in ("add", "remove"):
            raise ValueError(f"bad operation {o.get('op')!r}")
        if o.get("label") not in MANAGED[o["op"]]:
            verb = "added" if o["op"] == "add" else "removed"
            raise ValueError(f"label {o.get('label')!r} may not be {verb} by the controller")
        if o.get("reason") not in REASONS:
            raise ValueError("bad reason")
    return p


def apply(p: Dict[str, Any], api: Any, repo: str) -> Dict[str, List[Dict[str, Any]]]:
    """api: get(path) -> obj, post(path, body), delete(path). Only label endpoints are written."""
    validate_plan(p)
    done: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for o in p["ops"]:
        base = f"repos/{repo}/issues/{o['number']}"
        now = api.get(base)  # the state may have changed since the plan was made
        labels = [l if isinstance(l, str) else l.get("name", "") for l in now.get("labels", [])]
        has = any(_same(l, o["label"]) for l in labels)
        why = None
        if now.get("state") != "open":
            why = "closed"
        elif any(_same(l, "hold") for l in labels):
            why = "hold"
        elif (o["op"] == "add") == has:
            why = "already"
        if why:
            skipped.append(dict(o, why=why))
            continue
        try:
            if o["op"] == "add":
                api.post(f"{base}/labels", {"labels": [o["label"]]})
            else:
                api.delete(f"{base}/labels/{quote(o['label'], safe='')}")
            done.append(o)
        except Exception as exc:  # one failed label must not stop the others; reported below
            skipped.append(dict(o, why="label-missing" if re.search(r"404|Not Found", str(exc)) else "error"))
    return {"done": done, "skipped": skipped}


# ---- GitHub access (gh CLI; GITHUB_TOKEN from the environment) -------------

class GhApi:
    def _run(self, args: List[str], stdin: Optional[str] = None) -> str:
        proc = subprocess.run(["gh", *args], input=stdin, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or proc.stdout).strip()[:500])
        return proc.stdout

    def get(self, path: str) -> Any:
        return json.loads(self._run(["api", path]))

    def list(self, path: str) -> List[Any]:
        out = self._run(["api", "--paginate", "--jq", ".[]", path])
        return [json.loads(line) for line in out.splitlines() if line.strip()]

    def graphql(self, query: str, variables: Dict[str, Any]) -> Any:
        args = ["api", "graphql", "-f", f"query={query}"]
        for k, v in variables.items():
            args += ["-F", f"{k}={v}"]
        return json.loads(self._run(args))

    def post(self, path: str, body: Dict[str, Any]) -> str:
        return self._run(["api", "-X", "POST", path, "--input", "-"], json.dumps(body))

    def delete(self, path: str) -> str:
        return self._run(["api", "-X", "DELETE", path])


ISSUES_QUERY = """query($owner: String!, $name: String!, $label: String!, $n: Int!) {
  repository(owner: $owner, name: $name) {
    issues(first: $n, states: OPEN, labels: [$label], orderBy: {field: UPDATED_AT, direction: DESC}) {
      nodes {
        number
        labels(first: 50) { nodes { name } }
        userContentEdits(first: 50) { nodes { createdAt editor { login } } }
        timelineItems(last: 100, itemTypes: [LABELED_EVENT, UNLABELED_EVENT, RENAMED_TITLE_EVENT]) {
          nodes {
            __typename
            ... on LabeledEvent { createdAt actor { login } label { name } }
            ... on UnlabeledEvent { createdAt actor { login } label { name } }
            ... on RenamedTitleEvent { createdAt actor { login } }
          }
        }
      }
    }
  }
}"""
EVENT_TYPES = {"LabeledEvent": "labeled", "UnlabeledEvent": "unlabeled", "RenamedTitleEvent": "renamed"}


def _login(node: Optional[Dict[str, Any]]) -> Optional[str]:
    return (node or {}).get("login")


def read_issues(api: Any, repo: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    owner, name = repo.split("/")
    by_number: Dict[int, Dict[str, Any]] = {}
    for label in ("agent-ready", "agent-working"):
        data = api.graphql(ISSUES_QUERY, {"owner": owner, "name": name, "label": label,
                                          "n": config["maxIssuesPerRun"]})
        for node in data["data"]["repository"]["issues"]["nodes"]:
            by_number[node["number"]] = {
                "number": node["number"],
                "labels": [l["name"] for l in node["labels"]["nodes"]],
                "edits": [{"at": e["createdAt"], "editor": _login(e.get("editor"))}
                          for e in node["userContentEdits"]["nodes"]],
                "events": [{"type": EVENT_TYPES.get(e["__typename"]), "at": e["createdAt"],
                            "actor": _login(e.get("actor")), "label": (e.get("label") or {}).get("name")}
                           for e in node["timelineItems"]["nodes"]],
            }
    return list(by_number.values())


def read_prs(api: Any, repo: str, config: Dict[str, Any], gate: "risk_gate.Gate") -> List[Dict[str, Any]]:
    n = config["maxPrsPerRun"]
    prs = api.get(f"repos/{repo}/pulls?state=open&per_page={n}&sort=updated&direction=desc")
    out = []
    for pr in prs[:n]:
        tier = None
        try:
            tier = gate.judge(api.list(f"repos/{repo}/pulls/{pr['number']}/files?per_page=100"))["tier"]
        except Exception:  # unreadable diff: leave this PR's risk label as it is
            tier = None
        last_push = pr["created_at"]
        try:
            date = api.get(f"repos/{repo}/commits/{pr['head']['sha']}")["commit"]["committer"]["date"]
            last_push = max(last_push, date)
        except Exception:  # keep created_at as the last activity
            pass
        head_repo = ((pr.get("head") or {}).get("repo") or {}).get("full_name") or ""
        out.append({
            "number": pr["number"],
            "labels": [l["name"] for l in pr.get("labels", [])],
            "author": _login(pr.get("user")) or "",
            "headRef": pr["head"]["ref"],
            "sameRepo": head_repo.lower() == repo.lower(),
            "body": pr.get("body") or "",
            "tier": tier,
            "lastPushAt": last_push,
        })
    return out


def linked_prs(prs: List[Dict[str, Any]], repo: str) -> Dict[int, List[Dict[str, Any]]]:
    linked: Dict[int, List[Dict[str, Any]]] = {}
    for pr in prs:
        for n in closing_refs(pr["body"], repo):
            linked.setdefault(n, []).append({"lastPushAt": pr["lastPushAt"]})
    return linked


# ---- reporting --------------------------------------------------------------

def _safe(s: Any) -> str:
    return re.sub(r"::+", "?", re.sub(r"[^\w.:\- ]", "?", str(s)))


def render(result: Dict[str, Any], mode: str) -> str:
    lines = [f"### Controller (labels only): {mode}", ""]
    if mode == "paused":
        return "\n".join(lines + ["AGENT_PAUSE is `true`: nothing was read, proposed or changed."])
    if mode == "dry-run":
        lines += ["**Dry run.** Nothing was changed. Set the repository variable `CONTROLLER_LIVE` "
                  "to `true` to apply labels.", ""]
    rows = [f"| {'PR' if o['kind'] == 'pr' else 'issue'} #{o['number']} | {o['op']} | `{_safe(o['label'])}` "
            f"| {_safe(o['reason'])} |" for o in result["ops"]]
    lines.append("\n".join(["| Target | Operation | Label | Why |", "| --- | --- | --- | --- |"] + rows)
                 if rows else "No label changes to make.")
    if result["skipped"]:
        lines += ["", "Skipped:"] + [f"- #{s['number']} `{_safe(s['label'])}`: {_safe(s.get('reason') or s.get('why'))}"
                                     for s in result["skipped"][:30]]
    if result.get("notes"):
        lines += ["", "Notes:"] + [f"- {_safe(n)}" for n in result["notes"][:30]]
    missing = sorted({s["label"] for s in result["skipped"] if s.get("reason") == "label-missing"})
    if missing:
        lines += ["", "Labels missing in this repository (not created): "
                  + ", ".join(f"`{_safe(l)}`" for l in missing)
                  + ". Sync them with `python3 .github/scripts/sync_labels.py --repo OWNER/REPO --apply`."]
    return "\n".join(lines)


def _out(text: str) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")
    else:
        print(text)


def _set_output(name: str, value: str) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"{name}={value}\n")


def _read_json(rel: str) -> Any:
    with open(os.path.join(HERE, rel), encoding="utf-8") as fh:
        return json.load(fh)


def main(command: Optional[str], env: Dict[str, str], api: Any = None) -> None:
    repo = env.get("GH_REPO", "")
    if not re.match(r"^[\w.-]+/[\w.-]+$", repo):
        raise ValueError("set GH_REPO to owner/name")
    if command == "plan":
        mode = decide_mode(env)
        if mode == "paused":
            empty = {"version": 1, "mode": mode, "ops": [], "skipped": [], "notes": []}
            _out(render(empty, mode))
            _set_output("live", "false")
            _set_output("plan", json.dumps(empty))
            return
        api = api or GhApi()
        config = _read_json("../agent/controller.json")
        gate = risk_gate.Gate(risk_gate.load_rules())
        repo_labels = [l["name"] for l in api.list(f"repos/{repo}/labels?per_page=100")]
        issues = read_issues(api, repo, config)
        prs = read_prs(api, repo, config, gate)
        result = plan({"repo": repo, "config": config, "now": time.time(), "repoLabels": repo_labels,
                       "issues": issues, "prs": prs, "linked": linked_prs(prs, repo)})
        full = dict(result, version=1, mode=mode)
        _out(render(full, mode))
        print(f"controller: {mode}, {len(result['ops'])} operation(s), {len(result['skipped'])} skipped")
        _set_output("live", "true" if mode == "live" and result["ops"] else "false")
        _set_output("plan", json.dumps({"version": 1, "mode": mode, "ops": result["ops"]}))
    elif command == "apply":
        if env.get("CONTROLLER_LIVE") != "true" or env.get("AGENT_PAUSE") == "true":
            raise ValueError("refusing to apply: not live (CONTROLLER_LIVE must be true and AGENT_PAUSE must not)")
        p = json.loads(env.get("PLAN") or "null")
        if not isinstance(p, dict) or p.get("mode") != "live":
            raise ValueError("refusing to apply a plan that was not made in live mode")
        result = apply(p, api or GhApi(), repo)
        _out(f"### Controller applied\n\n{len(result['done'])} label change(s) made, "
             f"{len(result['skipped'])} skipped.\n\n"
             + "\n".join(f"- {o['op']} `{_safe(o['label'])}` on #{o['number']} ({_safe(o['reason'])})"
                         for o in result["done"]))
        print(f"controller: applied {len(result['done'])}, skipped {len(result['skipped'])}")
    else:
        raise ValueError("usage: controller.py plan | apply")


if __name__ == "__main__":
    try:
        main(sys.argv[1] if len(sys.argv) > 1 else None, dict(os.environ))
    except Exception as exc:  # any failure is one line in the log and a red run
        print(f"controller: {_safe(exc)}", file=sys.stderr)
        sys.exit(1)
