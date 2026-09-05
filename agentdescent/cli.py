"""``agentdescent`` on the command line: the verbs the MCP server also exposes.

One surface, two transports. Everything a host agent can do through the MCP
server a person can do by hand here, in the same words, so a run started from
Claude Code or DeepSeek Harness can be inspected from a shell and a run started
from a shell can be picked up by an agent. The skill file falls back to these
commands on a host with no MCP.

    agentdescent init    <path> [--kind ...] [--data cases.jsonl]   write a starter spec
    agentdescent plan    <spec.json>                                validate + cost, no run
    agentdescent evolve  <spec.json> [--detach] [--budget USD]      start a run
    agentdescent status  [<run_id>] [--brief]                       one run or all
    agentdescent watch   <run_id>                                   follow rounds
    agentdescent show    <run_id> [--diff]                          the evolved tree
    agentdescent apply   <run_id> [--to PATH] [--dry-run]           install it, backed up
    agentdescent cancel  <run_id>
    agentdescent resume  <run_id>
    agentdescent doctor                                             what is installed
    agentdescent install <dsh|claude-code|codex|opencode>           wire a host
    agentdescent mcp                                                serve over stdio
    agentdescent serve   [--port N]                                 read-only run panel

``argparse`` only: the CLI is in the core and the core has no dependencies.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

from . import runstore
from .evolvespec import EvolveSpec, SpecError, compose, estimate, load_spec

__all__ = ["main", "doctor_report", "starter_spec"]

#: Set in every worker's environment by the plugin runner and the MCP server so a
#: host that is itself being evolved cannot start a nested run. See docs/plugins.md.
NESTED_ENV = "AGENTDESCENT_NESTED"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _out(payload: Any, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    elif isinstance(payload, str):
        print(payload)
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def _status_line(st: runstore.RunStatus) -> str:
    reward = "" if st.best_reward is None else f" best={st.best_reward:.3f}"
    rounds = f"{st.round}/{st.rounds}" if st.rounds else f"{st.round}"
    usd = "" if st.usd is None else f" ${st.usd:.2f}"
    tail = f" [{st.stop_reason}]" if st.stop_reason else ""
    err = f" error: {st.error}" if st.error else ""
    return (f"{st.run_id}  {st.state:<9} round {rounds:<6} calls={st.calls}{reward}{usd}"
            f"  {st.kind or '?'}:{st.target or '?'}{tail}{err}")


def starter_spec(path: str, *, kind: Optional[str] = None, data: Optional[str] = None,
                 agent: str = "claude_code") -> Dict[str, Any]:
    """A spec to edit, with ``kind`` guessed from what ``path`` is."""
    p = os.path.expanduser(path)
    if kind is None:
        if os.path.isfile(p):
            kind = "text"
        elif os.path.isdir(p) and os.path.exists(os.path.join(p, "tests")) and any(
                f.endswith(".py") for f in os.listdir(p)):
            kind = "agent_code"
        elif os.path.isdir(p) and (os.path.exists(os.path.join(p, ".claude-plugin"))
                                   or os.path.exists(os.path.join(p, "cordis.patch.yml"))):
            kind = "plugin"
        elif os.path.basename(os.path.dirname(p.rstrip(os.sep))) == "agents":
            kind = "agent_dir"
        else:
            kind = "skill_dir"
    spec: Dict[str, Any] = {
        "version": 1, "kind": kind, "target": path,
        "data": {"path": data or "eval/cases.jsonl", "prompt": "prompt", "gold": "gold"},
        "score": "contains",
        "agent": {"ref": agent},
        "evolve": {"rounds": 6, "n_workers": 4},
    }
    if kind == "text":
        spec["agent"] = {"ref": "openai_compatible", "model": "deepseek-v4-flash"}
        spec["evolve"] = {"rounds": 8, "n_workers": 8}
    if kind == "skill_dir":
        spec["reflect"] = {"ref": "openai_compatible", "model": "deepseek-v4-flash"}
        if agent == "claude_code":
            spec["agent"] = {"ref": "claude_code",
                             "extra_args": ["--permission-mode", "acceptEdits"]}
    if kind == "agent_code":
        spec["entrypoint"] = ["python", "main.py"]
        spec["reflect"] = {"ref": "openai_compatible", "model": "deepseek-v4-flash"}
    if kind == "plugin":
        spec["host"] = ("claude_code" if os.path.exists(os.path.join(p, ".claude-plugin"))
                        else "dsh")
        spec["reflect"] = {"ref": "openai_compatible", "model": "deepseek-v4-flash"}
    return spec


def doctor_report() -> Dict[str, Any]:
    """What this machine can run: agent CLIs, provider keys, optional pieces."""
    clis = {name: shutil.which(name)
            for name in ("claude", "codex", "dsh", "opencode", "git", "node", "pnpm")}
    keys = {name: bool(os.environ.get(name)) for name in (
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL", "DEEPSEEK_API_KEY")}
    optional: Dict[str, bool] = {}
    for mod in ("anthropic", "mcp"):
        try:
            __import__(mod)
            optional[mod] = True
        except ImportError:
            optional[mod] = False
    try:
        from .sandbox_container import detect_engine
        container = detect_engine()
    except Exception:  # noqa: BLE001 - detection must never fail the report
        container = None
    problems: List[str] = []
    if not clis["git"]:
        problems.append("git is not on PATH; the ledger needs it")
    if not any(clis[c] for c in ("claude", "codex", "dsh", "opencode")):
        problems.append("no worker agent CLI on PATH (claude / codex / dsh / opencode); "
                        "directory kinds need one")
    if not (keys["ANTHROPIC_API_KEY"] or keys["OPENAI_API_KEY"] or keys["DEEPSEEK_API_KEY"]):
        problems.append("no provider key in the environment; a reflector needs one "
                        "(under dsh, forward keys in the mcp-client env block)")
    if not optional["mcp"]:
        problems.append("the mcp package is missing: pip install 'agentdescent[mcp]' "
                        "to serve tools (the CLI works without it)")
    return {
        "python": sys.version.split()[0],
        "agent_clis": clis, "provider_keys_present": keys,
        "optional": optional, "container_engine": container,
        "nested": bool(os.environ.get(NESTED_ENV)),
        "run_store": runstore.root(),
        "problems": problems,
    }


def _tree_diff(before: Dict[str, str], after: Dict[str, str]) -> str:
    out: List[str] = []
    for path in sorted(set(before) | set(after)):
        a, b = before.get(path), after.get(path)
        if a == b:
            continue
        out.extend(difflib.unified_diff(
            (a or "").splitlines(keepends=True), (b or "").splitlines(keepends=True),
            fromfile=f"a/{path}" if a is not None else "/dev/null",
            tofile=f"b/{path}" if b is not None else "/dev/null"))
    return "".join(out)


# ---------------------------------------------------------------------------
# verbs
# ---------------------------------------------------------------------------


def cmd_init(a: argparse.Namespace) -> int:
    spec = starter_spec(a.path, kind=a.kind, data=a.data, agent=a.agent)
    out = a.out or os.path.join(".agentdescent", f"{os.path.basename(a.path.rstrip('/')) or 'spec'}.evolve.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2)
    print(f"wrote {out}")
    print("edit data.path (your cases), score, and agent; then: agentdescent plan " + out)
    return 0


def plan_payload(spec: EvolveSpec, *, usd_per_call: Optional[float] = None) -> Dict[str, Any]:
    comp = compose(spec)
    return {"ok": True, "spec": spec.to_dict(), "tasks": len(comp.tasks),
            "artifact_id": spec.artifact_id(),
            "evolve_kwargs": {k: (v if isinstance(v, (int, float, str, bool, type(None)))
                                  else type(v).__name__)
                              for k, v in comp.kwargs.items()},
            "estimate": estimate(comp, usd_per_call=usd_per_call), "notes": comp.notes}


def cmd_plan(a: argparse.Namespace) -> int:
    spec = load_spec(a.spec)
    try:
        payload = plan_payload(spec, usd_per_call=a.usd_per_call)
    except SpecError as e:
        _out({"ok": False, "error": str(e)}, as_json=a.json)
        return 2
    _out(payload, as_json=a.json)
    return 0


def cmd_evolve(a: argparse.Namespace) -> int:
    spec = load_spec(a.spec)
    try:
        compose(spec)                        # fail here, not in the detached child
    except SpecError as e:
        print(f"spec error: {e}", file=sys.stderr)
        return 2
    rd = runstore.create(spec.to_dict(), store=a.store)
    if a.detach:
        st = runstore.launch(rd, budget_usd=a.budget, usd_per_call=a.usd_per_call)
        _out({"run_id": rd.run_id, "state": st.state, "pid": st.pid, "dir": rd.path}
             if a.json else f"{rd.run_id}  started (pid {st.pid})  {rd.path}", as_json=a.json)
        return 0
    result = runstore.execute(rd, budget_usd=a.budget, usd_per_call=a.usd_per_call)
    st = rd.status()
    _out({"run_id": rd.run_id, "state": st.state, "final_reward": result.final_reward,
          "stop_reason": result.stop_reason, "outcomes": result.outcomes(),
          "error": result.error, "dir": rd.path} if a.json else
         f"{_status_line(st)}\nfinal reward: {result.final_reward:.3f}  "
         f"outcomes: {result.outcomes()}\n{rd.path}", as_json=a.json)
    return 0 if result.error is None else 1


def cmd_run(a: argparse.Namespace) -> int:
    """The body of a detached run; not for people."""
    rd = runstore.RunDir(a.run_dir)
    result = runstore.execute(rd, budget_usd=a.budget, usd_per_call=a.usd_per_call)
    return 0 if result.error is None else 1


def status_payload(run_id: Optional[str], *, store: Optional[str] = None,
                   recent_rounds: int = 3) -> Any:
    if run_id:
        rd = runstore.get(run_id, store=store)
        st = rd.status().to_dict()
        st["recent_rounds"] = rd.rounds()[-recent_rounds:]
        return st
    return [st.to_dict() for st in runstore.list_runs(store=store)]


def cmd_status(a: argparse.Namespace) -> int:
    if a.run_id:
        rd = runstore.get(a.run_id, store=a.store)
        st = rd.status()
        if a.json:
            _out(status_payload(a.run_id, store=a.store), as_json=True)
        else:
            print(_status_line(st))
            if not a.brief:
                for r in rd.rounds()[-3:]:
                    print(f"  round {r['round']:>3}  reward={r['held_out_reward']:.3f}  "
                          f"+{r['committed']}/-{r['rejected']}  {r.get('reasons', {})}")
        return 0
    runs = runstore.list_runs(store=a.store)
    if a.brief:
        live = [s for s in runs if s.state == "running"]
        if not live:
            return 0
        for s in live:
            print(_status_line(s))
        return 0
    if a.json:
        _out([s.to_dict() for s in runs], as_json=True)
    elif not runs:
        print(f"no runs under {a.store or runstore.root()}")
    else:
        for s in runs:
            print(_status_line(s))
    return 0


def cmd_watch(a: argparse.Namespace) -> int:
    rd = runstore.get(a.run_id, store=a.store)
    seen = 0
    while True:
        rounds = rd.rounds()
        for r in rounds[seen:]:
            print(f"round {r['round']:>3}  reward={r['held_out_reward']:.3f}  "
                  f"+{r['committed']}/-{r['rejected']}  {r.get('reasons', {})}", flush=True)
        seen = len(rounds)
        st = rd.status()
        if st.state != "running":
            print(_status_line(st))
            return 0 if st.state == "done" else 1
        time.sleep(a.interval)


def show_payload(run_id: str, *, store: Optional[str] = None, diff: bool = True,
                 max_chars: int = 40_000) -> Dict[str, Any]:
    rd = runstore.get(run_id, store=store)
    st = rd.status()
    result = rd.result()
    payload: Dict[str, Any] = {"run_id": run_id, "state": st.state,
                               "stop_reason": st.stop_reason, "error": st.error}
    if result is None:
        payload["note"] = "no result yet"
        return payload
    payload.update(final_reward=result.final_reward, outcomes=result.outcomes(),
                   rounds=len(result.history))
    spec = EvolveSpec.from_dict(rd.spec_dict())
    if spec.kind == "text":
        payload["rendered"] = result.rendered[:max_chars]
        return payload
    payload["files"] = sorted(result.state)
    if diff:
        from .filetree import load_tree
        try:
            before = load_tree(spec.target)
        except Exception as e:  # noqa: BLE001 - the original may have moved
            before, payload["diff_note"] = {}, f"could not read original {spec.target}: {e}"
        payload["diff"] = _tree_diff(before, result.state)[:max_chars]
        try:
            payload["apply_plan"] = result.write_to(os.path.expanduser(spec.target), dry_run=True)
        except Exception as e:  # noqa: BLE001
            payload["apply_plan"] = {"error": str(e)}
    return payload


def cmd_show(a: argparse.Namespace) -> int:
    payload = show_payload(a.run_id, store=a.store, diff=a.diff)
    if a.json:
        _out(payload, as_json=True)
        return 0
    for k in ("run_id", "state", "stop_reason", "error", "final_reward", "outcomes", "note"):
        if payload.get(k) is not None:
            print(f"{k}: {payload[k]}")
    if "rendered" in payload:
        print("\n" + payload["rendered"])
    if payload.get("diff"):
        print("\n" + payload["diff"])
    elif "files" in payload:
        print("files:", ", ".join(payload["files"]))
    if payload.get("apply_plan"):
        print("apply would:", json.dumps(payload["apply_plan"]))
    return 0


def apply_payload(run_id: str, *, to: Optional[str] = None, store: Optional[str] = None,
                  dry_run: bool = False, backup: bool = True) -> Dict[str, Any]:
    rd = runstore.get(run_id, store=store)
    result = rd.result()
    if result is None:
        raise runstore.RunStoreError(f"{run_id} has no result to apply yet")
    spec = EvolveSpec.from_dict(rd.spec_dict())
    dest = os.path.expanduser(to or spec.target)
    if spec.kind == "text":
        if dry_run:
            return {"would_write": dest, "chars": len(result.rendered)}
        if backup and os.path.exists(dest):
            shutil.copy2(dest, dest + ".bak")
        with open(dest, "w", encoding="utf-8") as fh:
            fh.write(result.rendered)
        return {"written": [dest], "backup": [dest + ".bak"] if backup else []}
    plan = result.write_to(dest, backup=backup, dry_run=dry_run)
    plan["dest"] = dest
    return plan


def cmd_apply(a: argparse.Namespace) -> int:
    payload = apply_payload(a.run_id, to=a.to, store=a.store, dry_run=a.dry_run,
                            backup=not a.no_backup)
    _out(payload, as_json=True)
    return 0


def cmd_cancel(a: argparse.Namespace) -> int:
    st = runstore.cancel(a.run_id, store=a.store)
    print(_status_line(st))
    return 0


def cmd_resume(a: argparse.Namespace) -> int:
    st = runstore.resume(a.run_id, store=a.store, budget_usd=a.budget,
                         usd_per_call=a.usd_per_call)
    print(_status_line(st))
    return 0


def cmd_doctor(a: argparse.Namespace) -> int:
    rep = doctor_report()
    if a.json:
        _out(rep, as_json=True)
    else:
        for k, v in rep.items():
            if k != "problems":
                print(f"{k}: {v}")
        for p in rep["problems"]:
            print(f"! {p}")
        if not rep["problems"]:
            print("ok: nothing missing")
    return 0 if not rep["problems"] else 1


def cmd_install(a: argparse.Namespace) -> int:
    from .integrations import install

    written = install(a.host, dry_run=a.dry_run, home=a.home)
    for line in written:
        print(line)
    return 0


def cmd_serve(a: argparse.Namespace) -> int:
    print(f"agentdescent: run panel on http://{a.host}:{a.port}/  (read-only; Ctrl-C to stop)",
          file=sys.stderr)
    runstore.serve_http(host=a.host, port=a.port, store=a.store)
    return 0


def cmd_mcp(a: argparse.Namespace) -> int:
    # A host starts this as a subprocess and shows the user nothing but
    # "CONNECTION_CLOSED" when it dies, so the one thing that can go wrong
    # before the protocol starts -- the SDK not being installed -- has to say so
    # in one line on stderr rather than as a traceback nobody will see.
    try:
        from .mcp import serve

        serve(store=a.store)
    except ImportError as e:
        # The instruction is spelled out here rather than taken from the
        # exception, because which import failed decides what `e` says and the
        # user needs the same one line either way.
        print('agentdescent mcp: cannot start the MCP server -- '
              'pip install "agentdescent[mcp]". '
              f'(The CLI works without it.) Underlying error: {e}', file=sys.stderr)
        return 3
    return 0


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agentdescent",
                                description="Evolve skills, agents, prompts, code and plugins "
                                            "with a parallel, merge-based optimiser.")
    p.add_argument("--store", help="run store directory (default ~/.agentdescent/runs "
                                    "or $AGENTDESCENT_HOME/runs)")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="write a starter spec for a path")
    s.add_argument("path")
    s.add_argument("--kind", choices=("text", "skill_dir", "agent_dir", "agent_code", "plugin"))
    s.add_argument("--data", help="cases file to point the spec at")
    s.add_argument("--agent", default="claude_code", help="worker agent short name")
    s.add_argument("--out", help="where to write the spec")
    s.set_defaults(fn=cmd_init)

    s = sub.add_parser("plan", help="validate a spec and estimate its cost; runs nothing")
    s.add_argument("spec")
    s.add_argument("--usd-per-call", type=float, dest="usd_per_call")
    s.set_defaults(fn=cmd_plan)

    s = sub.add_parser("evolve", help="run a spec")
    s.add_argument("spec")
    s.add_argument("--detach", action="store_true", help="return at once; poll with status")
    s.add_argument("--budget", type=float, help="stop after this many dollars (needs --usd-per-call)")
    s.add_argument("--usd-per-call", type=float, dest="usd_per_call")
    s.set_defaults(fn=cmd_evolve)

    s = sub.add_parser("run", help=argparse.SUPPRESS)
    s.add_argument("--run-dir", required=True, dest="run_dir")
    s.add_argument("--budget", type=float)
    s.add_argument("--usd-per-call", type=float, dest="usd_per_call")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("status", help="one run, or all of them")
    s.add_argument("run_id", nargs="?")
    s.add_argument("--brief", action="store_true", help="only running runs, one line each")
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("watch", help="follow a run's rounds until it ends")
    s.add_argument("run_id")
    s.add_argument("--interval", type=float, default=5.0)
    s.set_defaults(fn=cmd_watch)

    s = sub.add_parser("show", help="the evolved artifact, and the diff against the original")
    s.add_argument("run_id")
    s.add_argument("--diff", action="store_true", default=True)
    s.add_argument("--no-diff", action="store_false", dest="diff")
    s.set_defaults(fn=cmd_show)

    s = sub.add_parser("apply", help="write the evolved artifact back, backing up first")
    s.add_argument("run_id")
    s.add_argument("--to", help="a different destination than the spec's target")
    s.add_argument("--dry-run", action="store_true", dest="dry_run")
    s.add_argument("--no-backup", action="store_true", dest="no_backup")
    s.set_defaults(fn=cmd_apply)

    s = sub.add_parser("cancel", help="stop a run and every worker it started")
    s.add_argument("run_id")
    s.set_defaults(fn=cmd_cancel)

    s = sub.add_parser("resume", help="continue a stopped run on its ledger")
    s.add_argument("run_id")
    s.add_argument("--budget", type=float)
    s.add_argument("--usd-per-call", type=float, dest="usd_per_call")
    s.set_defaults(fn=cmd_resume)

    s = sub.add_parser("doctor", help="which agents, keys and optional pieces are available")
    s.set_defaults(fn=cmd_doctor)

    s = sub.add_parser("install", help="wire the skill and MCP server into a host")
    s.add_argument("host", choices=("dsh", "claude-code", "codex", "opencode"))
    s.add_argument("--dry-run", action="store_true", dest="dry_run")
    s.add_argument("--home", help="the host's home directory (default: the real one)")
    s.set_defaults(fn=cmd_install)

    s = sub.add_parser("serve", help="serve a read-only run panel on loopback, for a host UI")
    s.add_argument("--host", default=runstore.DEFAULT_HTTP_HOST)
    s.add_argument("--port", type=int, default=runstore.DEFAULT_HTTP_PORT)
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("mcp", help="serve the tools over stdio (needs agentdescent[mcp])")
    s.set_defaults(fn=cmd_mcp)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return a.fn(a)
    except (SpecError, runstore.RunStoreError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
