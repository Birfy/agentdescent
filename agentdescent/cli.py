"""``agentdescent`` on the command line: the verbs the MCP server also exposes.

One surface, two transports. Everything a host agent can do through the MCP
server a person can do by hand here, in the same words, so a run started from
Claude Code or DeepSeek Harness can be inspected from a shell and a run started
from a shell can be picked up by an agent. The skill file falls back to these
commands on a host with no MCP.

    agentdescent demo                                               run one, offline, no key
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
    agentdescent --version                                          which AgentDescent this is
    agentdescent install <dsh|claude-code|codex|opencode>           wire a host
    agentdescent uninstall <dsh|claude-code|codex|opencode>         undo `install`
    agentdescent algorithms list [--kind K] | show <name>           named algorithms for a spec
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
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from . import runstore
from .evolvespec import (KINDS, EvolveSpec, SpecError, compose, estimate, load_spec,
                         _POLICY_SLOTS)

__all__ = ["main", "doctor_report", "starter_spec"]


def _version() -> str:
    """``agentdescent.__version__``, resolved lazily.

    Not a module-level import: with an ``agentdescent/`` directory in the cwd
    shadowing the package (the case #394 diagnoses) the name resolves to a
    namespace package that never ran ``__init__``, and an import-time failure
    here would be a raw traceback ahead of ``cmd_mcp``'s one-line explanation.
    """
    try:
        from . import __version__
        return __version__
    except ImportError:          # shadowed / namespace-package import
        try:
            from importlib.metadata import version
            return version("agentdescent")
        except Exception:  # noqa: BLE001 - reporting a version must never raise
            return "unknown"

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
    tok = f" tok={st.tokens:,}" if st.tokens else ""
    # Show in-flight work so a long first propose is not mistaken for a hang
    # at calls=0 (#358). Hidden when nothing is pending.
    pending = ""
    if st.pending_calls:
        ph = f" phase={st.phase}" if st.phase else ""
        pending = f" pending={st.pending_calls}{ph}"
    tail = f" [{st.stop_reason}]" if st.stop_reason else ""
    err = f" error: {st.error}" if st.error else ""
    # Reflector exceptions, cumulative (#395): a run can finish `done` with only
    # these to say that it never learned anything.
    perr = ""
    if st.propose_errors:
        perr = f" propose_errors={st.propose_errors}"
        if st.propose_error and not st.error:
            perr += f" (first: {st.propose_error})"
    return (f"{st.run_id}  {st.state:<9} round {rounds:<6} "
            f"calls={st.calls}{pending}{tok}{perr}{reward}{usd}"
            f"  {st.kind or '?'}:{st.target or '?'}{tail}{err}")


#: Which worker CLI ``init`` picks when several are on PATH (#361): the first
#: present one wins. Same refs as ``_CLI_BINARIES``; ``claude_code`` stays the
#: placeholder when none is installed.
_DEFAULT_AGENT_ORDER = ("claude_code", "codex", "dsh", "opencode")
_PLACEHOLDER_AGENT = "claude_code"


def default_agent_ref() -> Tuple[str, bool]:
    """``(ref, found)``: the worker CLI ``init`` writes when ``--agent`` is omitted.

    Uses :func:`doctor_report` (the same view ``plan`` gates on, #346), so a
    machine with only ``opencode`` on PATH gets ``opencode`` rather than a
    ``claude_code`` spec that ``plan`` immediately refuses (#361). ``found`` is
    False when no worker CLI is on PATH; the ref is then the ``claude_code``
    placeholder and the caller should warn.
    """
    clis = doctor_report()["agent_clis"]
    for ref in _DEFAULT_AGENT_ORDER:
        if clis.get(_CLI_BINARIES[ref]):
            return ref, True
    return _PLACEHOLDER_AGENT, False


def starter_spec(path: str, *, kind: Optional[str] = None, data: Optional[str] = None,
                 agent: Optional[str] = None) -> Dict[str, Any]:
    """A spec to edit, with ``kind`` guessed from what ``path`` is.

    ``agent=None`` picks the worker from the CLIs on PATH (:func:`default_agent_ref`).
    """
    if agent is None:
        agent = default_agent_ref()[0]
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
    if kind == "policy_slot":
        # `target` is a slot, not a path, and `data` holds refs rather than rows
        # -- an inner problem is a callable and no row format can express one.
        spec["target"] = path if path in _POLICY_SLOTS else "selection"
        spec["data"] = {"problems": data or "mypkg.problems:build", "seeds": [0]}
        spec["score"] = "auc"
        spec["agent"] = {"ref": "openai_compatible", "model": "deepseek-v4-flash"}
        # A round is `n_workers` whole inner searches, so ask for few and budget
        # in wall clock: the recorded run asked for 8 rounds and got 2 in 90 min.
        spec["evolve"] = {"rounds": 4, "n_workers": 2, "max_seconds": 5400}
        spec["allow"] = ["mypkg"]
    return spec


#: Every published `mcp` requires Python >= 3.10, and this project supports 3.9.
#: On 3.9 the extra installs nothing (see the marker in pyproject.toml), so
#: "pip install agentdescent[mcp]" is not advice, it is a dead end -- the only
#: true instruction is a newer interpreter.
MCP_MIN_PYTHON = (3, 10)


def mcp_unavailable() -> Optional[str]:
    """Why the MCP server cannot run here, or None if it can."""
    if sys.version_info < MCP_MIN_PYTHON:
        return ("the MCP SDK needs Python >= %d.%d and this is %s, so "
                "`agentdescent[mcp]` installs nothing here -- the CLI verbs are "
                "the whole surface on this interpreter, and the shipped SKILL.md "
                "falls back to them" % (MCP_MIN_PYTHON[0], MCP_MIN_PYTHON[1],
                                        ".".join(map(str, sys.version_info[:3]))))
    # Delegated so there is one answer to "is the SDK here", and so a caller
    # that stubs it (the install-warning test does) steers every message.
    from .integrations import mcp_sdk_missing

    if mcp_sdk_missing():
        return ('the mcp package is missing: pip install "agentdescent[mcp]" '
                "to serve tools (the CLI works without it)")
    return None


def doctor_report() -> Dict[str, Any]:
    """What this machine can run: agent CLIs, provider keys, optional pieces."""
    clis = {name: shutil.which(name)
            for name in ("claude", "codex", "dsh", "opencode", "git", "node", "pnpm")}
    keys = {name: bool(os.environ.get(name)) for name in (
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL", "DEEPSEEK_API_KEY")}
    # The *value* of the base URL, not just its presence: it is a URL and not a
    # secret, and without it a caller writing a spec cannot tell that
    # `OPENAI_API_KEY` points at some other provider entirely. Measured: an
    # agent given only "OPENAI_API_KEY: true" wrote `"model": "gpt-4o-mini"`
    # against an endpoint that serves nothing of the sort.
    base_url = os.environ.get("OPENAI_BASE_URL")
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
    if base_url:
        problems.append(
            f"OPENAI_BASE_URL is {base_url} -- an OpenAI-compatible endpoint that "
            "is not OpenAI, so its model names are its own; ask the user which "
            "model to name in a spec rather than assuming an OpenAI one")
    if not (keys["ANTHROPIC_API_KEY"] or keys["OPENAI_API_KEY"] or keys["DEEPSEEK_API_KEY"]):
        problems.append("no provider key in the environment; a reflector needs one "
                        "(under dsh, forward keys in the mcp-client env block)")
    why = mcp_unavailable()
    if why:
        problems.append(why)
    hosts, hints = _host_integrations(clis)
    for name, info in hosts.items():
        if info["state"] in ("stale", "partial"):
            problems.append(
                f"{name}: the installed integration is {info['state']} "
                f"(skill {info['skill']}, mcp {info['mcp']}); run `agentdescent install {name}`")
    return {
        # Which install answered: the version and the interpreter running it. A host
        # starts its own `agentdescent mcp` (often through the interpreter `install`
        # pinned, #394), which can be a different install from the one in the user's
        # shell; with these two fields a host-side `doctor` and a shell-side `doctor`
        # compare directly (#416).
        "version": _version(),
        "executable": sys.executable,
        "python": sys.version.split()[0],
        "agent_clis": clis, "provider_keys_present": keys,
        "openai_base_url": base_url,
        "optional": optional, "container_engine": container,
        "hosts": hosts, "hints": hints,
        "nested": bool(os.environ.get(NESTED_ENV)),
        "run_store": runstore.root(),
        "problems": problems,
    }


def _host_integrations(clis: Dict[str, Any]):
    """Per-host install state for :func:`doctor_report`, plus the hints.

    A host whose CLI is on ``PATH`` but has no integration gets a *hint*, not a
    problem: ``pip install`` does not wire anything, and nobody wants every
    host on the machine touched, so the exit code stays about what is broken.
    A stale or half-written integration is a problem, because it is installed
    and teaching the host a procedure the package no longer has."""
    from .integrations import HOST_CLI, host_status

    hosts: Dict[str, Dict[str, Any]] = {}
    hints: List[str] = []
    for name, binary in HOST_CLI.items():
        try:
            info: Dict[str, Any] = dict(host_status(name))
        except Exception:  # noqa: BLE001 - reading a config must never fail doctor
            info = {"state": "unknown", "skill": "unknown", "mcp": "unknown", "path": ""}
        info["cli"] = clis.get(binary)
        hosts[name] = info
        if info["cli"] and info["state"] == "missing":
            # Installed through the host's own marketplace leaves no files here.
            extra = {"codex": " (or `codex plugin add agentdescent@agentdescent`)",
                     "claude-code": " (or `/plugin install agentdescent@agentdescent`)",
                     }.get(name, "")
            hints.append(f"{binary} is on PATH but agentdescent is not wired into it: "
                         f"`agentdescent install {name}`{extra}")
    return hosts, hints


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


def cmd_demo(a: argparse.Namespace) -> int:
    """Build a complete example and run it, so the first thing seen works."""
    import tempfile

    from . import demo as demo_mod

    root = os.path.abspath(os.path.expanduser(a.dir)) if a.dir else tempfile.mkdtemp(
        prefix="agentdescent-demo-")
    os.makedirs(root, exist_ok=True)
    spec_dict = demo_mod.build(root)
    skill = os.path.join(root, demo_mod.DEMO_SKILL)
    rules = os.path.join(skill, "references", "rules.md")

    print(f"A skill that totals a column of a CSV, in {root}")
    print(f"  {demo_mod.DEMO_SKILL}/references/rules.md  ->  "
          f"{open(rules).read().strip()!r}   (wrong: `id` is a row number)")
    print("  cases.jsonl                        ->  12 CSVs with known totals")
    print("  the agent is a local program, so this costs nothing and needs no key\n")

    spec = EvolveSpec.from_dict(spec_dict)
    comp = compose(spec)
    est = estimate(comp)
    print(f"plan: {len(comp.tasks)} tasks, "
          f"{comp.kwargs['rounds']} rounds x {comp.kwargs['n_workers']} workers, "
          f"up to {est['agent_calls_upper_bound']} agent calls\n")

    rd = runstore.create(spec.to_dict(), store=a.store)
    print(f"running ({rd.run_id})...")
    result = runstore.execute(rd)
    for info in result.history:
        print(f"  round {info.round:>2}  reward={info.held_out_reward:.3f}  "
              f"+{info.committed}/-{info.rejected}")

    print(f"\nheld-out reward: {result.final_reward:.3f}   outcomes: {result.outcomes()}")
    print(f"what it learned:  rules.md -> "
          f"{result.state.get('references/rules.md', '').strip()!r}")
    if result.final_reward < 1.0:
        print("\n(that is lower than expected for the demo; the run store has the detail:")
        print(f" {rd.path})")
        return 1
    print(f"""
The skill on disk is untouched -- an evolved artifact only lands when you say so:

  agentdescent show  {rd.run_id}          the diff
  agentdescent apply {rd.run_id} --dry-run  what it would write

Your own run is the same spec with three things changed -- your directory, your
cases.jsonl, and a real agent instead of the offline one:

  agentdescent init <your-skill-dir> --data <your-cases.jsonl>
  agentdescent doctor        # what is missing before a real run
  agentdescent plan spec.json""")
    return 0


def cmd_init(a: argparse.Namespace) -> int:
    agent, found = (a.agent, True) if a.agent else default_agent_ref()
    spec = starter_spec(a.path, kind=a.kind, data=a.data, agent=agent)
    out = a.out or os.path.join(".agentdescent", f"{os.path.basename(a.path.rstrip('/')) or 'spec'}.evolve.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2)
    print(f"wrote {out}")
    if not found and spec["kind"] not in ("text", "policy_slot"):
        print(f"warning: no worker CLI (claude, codex, dsh, opencode) is on PATH; "
              f"wrote the placeholder agent `{agent}`. Run `agentdescent doctor`, "
              "install a CLI, or re-run with `--agent <ref>` / edit `agent.ref` "
              "before `plan`")
    elif a.agent is None and spec["kind"] not in ("text", "policy_slot"):
        print(f"agent: `{agent}` (the worker CLI found on PATH; "
              "override with `--agent <ref>`)")
    reflect = spec.get("reflect")
    if (isinstance(reflect, dict) and reflect.get("ref") == "openai_compatible"
            and not os.environ.get("OPENAI_API_KEY")):
        # The starter spec's reflector is a plain OpenAI-compatible call whatever
        # worker CLI was found, so a machine with only `opencode` / `claude`
        # gets a reflector it cannot call. Run, it used to finish `done` with
        # nothing learned (#395); say so here, where it is cheapest to fix.
        print("warning: `reflect` is `openai_compatible` and OPENAI_API_KEY is not set, "
              "so every proposal would fail and the run would learn nothing. Export "
              "OPENAI_API_KEY (and OPENAI_BASE_URL for a non-OpenAI endpoint), or "
              f"change `reflect` in {out} before `plan`")
    host = spec.get("host") if spec["kind"] == "plugin" else None
    if host in _CLI_BINARIES and not doctor_report()["agent_clis"].get(_CLI_BINARIES[host]):
        # The plugin's format fixes its host; the worker found on PATH does
        # not run it, so say so here rather than only at `plan` (#368).
        print(f"warning: this plugin's host `{host}` needs `{_CLI_BINARIES[host]}` "
              f"on PATH to load it and run rollouts, and it is not there; "
              f"`plan` will refuse until it is installed")
    if spec["kind"] == "policy_slot":
        problems = spec["data"]["problems"]
        print(f"next: provide {problems} -- a zero-argument builder "
              "returning a mapping of inner problems")
        print("      make its module importable and include its package in the spec's allow list")
    else:
        data = spec["data"]["path"]
        if not os.path.exists(os.path.expanduser(data)):
            print(f"next: create {data} -- one JSON object per line, "
                  '{"prompt": "...", "gold": "..."}')
            print("      (`agentdescent demo` runs a complete example, offline, to copy from)")
    print(f"then: agentdescent plan {out}")
    return 0


#: What an isolated worker of each CLI can fall back on, once it has been
#: pointed away from the config directory its interactive login lives in.
#:
#: Only `dsh` is confirmed: its own error names the variable ("or export
#: DEEPSEEK_API_KEY in the launching environment"). `codex` is confirmed to be
#: the opposite -- measured against codex-cli 0.153, with OPENAI_API_KEY and
#: OPENAI_BASE_URL both set it ignored the base URL, went to api.openai.com and
#: sent no credentials at all ("Missing bearer or basic authentication in
#: header"). The other two are unverified, so they get the mechanism and not a
#: promise: claiming a key would fix it is the false assurance this warning
#: exists to remove.
_CLI_ENV_FALLBACK = {"dsh": "DEEPSEEK_API_KEY"}


def _opencode_model_arg(block: Any) -> Optional[str]:
    """The ``provider/model`` an ``opencode`` agent block pins with ``-m/--model``.

    A static look at ``extra_args`` -- no network, no OpenCode. ``None`` when the
    block pins nothing or pins a bare model name with no provider part.
    """
    args = block.get("extra_args") if isinstance(block, dict) else None
    if not isinstance(args, (list, tuple)):
        return None
    args = [a for a in args if isinstance(a, str)]
    for i, a in enumerate(args):
        value = None
        if a in ("-m", "--model") and i + 1 < len(args):
            value = args[i + 1]
        elif a.startswith("--model="):
            value = a[len("--model="):]
        if value and "/" in value:
            return value
    return None


def _opencode_provider_remedy(block: Any) -> Optional[str]:
    """The plan warning's remedy for an isolated ``opencode`` pinning ``-m p/m``.

    ``None`` -- keep the generic remedy -- unless a model with a provider part is
    pinned and the caller has not already pointed ``OPENCODE_CONFIG`` at a file
    through the block's ``env``.
    """
    model = _opencode_model_arg(block)
    env = block.get("env") if isinstance(block, dict) else None
    if not model or (isinstance(env, dict)
                     and ("OPENCODE_CONFIG" in env or "OPENCODE_CONFIG_CONTENT" in env)):
        return None
    return (f"`-m {model}` works only if that provider is built into OpenCode: a "
            "custom provider (an OpenAI-compatible endpoint, a local server, a "
            "gateway) lives in your opencode.json, which an isolated worker cannot "
            "read, and every call then fails with an opaque `UnknownError`. To keep "
            "isolation instead, point the agent at a provider-only config file: "
            '"env": {"OPENCODE_CONFIG": "/abs/path/providers.json"} (keys via '
            "{env:VAR}, never inline; an OPENCODE_CONFIG exported in your shell is "
            "dropped for an isolated worker)")


_CLI_ENV_USELESS = {
    "codex": ("an environment key is not a substitute here: measured, `codex` "
              "ignored OPENAI_API_KEY and OPENAI_BASE_URL and called "
              "api.openai.com unauthenticated"),
}

#: Short ref -> binary on PATH. Missing binary is a hard plan/start error (#346);
#: isolation/login when the binary *is* present stays a soft warning.
_CLI_BINARIES = {
    "claude_code": "claude",
    "codex": "codex",
    "dsh": "dsh",
    "opencode": "opencode",
}


def _missing_cli_messages(spec: EvolveSpec,
                          fields: Optional[Sequence[str]] = None) -> List[str]:
    """Hard-fail messages when a CLI the run will execute is absent from PATH.

    ``plan`` exists so a host does not start a doomed run; soft ``warnings`` were
    easy for MCP hosts to ignore while ``ok`` stayed true (#346).

    Checks every role separately: ``agent`` and ``reflect``, and for
    ``kind="plugin"`` the ``host`` CLI that actually loads the plugin and runs
    each rollout (``plugin_runner(spec.host)``). An installed agent must not
    make an absent host pass (#368). Spec-driven runs execute the host on this
    machine's PATH, so that is the context checked here.

    ``fields`` limits the check to those roles (``"agent"``, ``"reflect"``,
    ``"host"``); ``None`` checks all of them.
    """
    report = doctor_report()
    clis = report["agent_clis"]
    present = [b for b in ("claude", "codex", "dsh", "opencode") if clis.get(b)]
    bin_to_ref = {v: k for k, v in _CLI_BINARIES.items()}
    roles: List[Tuple[str, Optional[str]]] = []
    for field in ("agent", "reflect"):
        block = getattr(spec, field, None)
        if isinstance(block, dict):
            ref = block.get("ref")
        elif isinstance(block, str):
            ref = block
        else:
            ref = None
        roles.append((field, ref))
    if spec.kind == "plugin":
        roles.append(("host", spec.host))
    if fields is not None:
        roles = [(f, r) for f, r in roles if f in fields]
    out: List[str] = []
    for field, ref in roles:
        if not ref or ref not in _CLI_BINARIES:
            continue
        binary = _CLI_BINARIES[ref]
        if clis.get(binary):
            continue
        if field == "host":
            # The host is fixed by the plugin's format (a `.claude-plugin/`
            # tree is a Claude Code plugin); switching it to whatever CLI is
            # installed would load the plugin into a host that cannot read it.
            out.append(f"host: `{ref}` loads the plugin and runs every rollout, "
                       f"and needs `{binary}` on PATH; install `{binary}` (the "
                       f"host must match the plugin's format, so a different "
                       f"`agent.ref` does not help); see `agentdescent doctor`")
            continue
        if present:
            # Map binary -> short ref for the switch hint.
            alts = ", ".join(f"`{bin_to_ref[b]}`" for b in present)
            remedy = (f"install `{binary}`, or switch `{field}.ref` to a CLI that "
                      f"is on PATH ({alts}), or use a non-CLI agent "
                      f"(e.g. `openai_compatible`); see `agentdescent doctor`")
        else:
            remedy = (f"install `{binary}`, or set `{field}.ref` to a non-CLI agent "
                      f"(e.g. `openai_compatible`); see `agentdescent doctor`")
        out.append(f"{field}: `{ref}` needs `{binary}` on PATH; {remedy}")
    return out


def require_cli_on_path(spec: EvolveSpec) -> None:
    """Refuse evolve/MCP start when a CLI binary the run needs is missing
    (#346; the plugin ``host`` too, #368).

    Shared with :func:`plan_payload` so plan and start fail closed the same way,
    before ``runstore.create`` / ``launch``.
    """
    msgs = _missing_cli_messages(spec)
    if msgs:
        raise SpecError("; ".join(msgs))


def require_resume_host_on_path(run_id: str, *, store: Optional[str] = None) -> None:
    """Refuse to relaunch a stopped plugin run whose ``host`` CLI is not on PATH (#370).

    The environment that resumes is often not the one that started: a run begun
    where ``claude`` was installed and resumed where it is not would otherwise
    reach :func:`runstore.launch`, write a fresh ``running`` status and only fail
    inside the detached child. Only the plugin ``host`` is gated here -- the
    agent/reflect roles keep their soft resume warnings -- and the check reads
    the *stored* spec, never switching the host to whatever CLI is installed.

    A still-live run is left to :func:`runstore.resume`'s own rejection, and a
    stored spec that no longer parses keeps the old behaviour (no gate).
    """
    rd = runstore.get(run_id, store=store)
    st = rd.status()
    if st.state == "running" and st.alive():
        return
    try:
        spec = EvolveSpec.from_dict(rd.spec_dict())
    except Exception:  # noqa: BLE001 - an unreadable spec fails in the child as before
        return
    if spec.kind != "plugin":
        return
    msgs = _missing_cli_messages(spec, fields=("host",))
    if msgs:
        raise SpecError(f"cannot resume {run_id}: " + "; ".join(msgs))


def _key_env(block: Any, default: str, *, field: str = "api_key_env") -> str:
    """The variable the adapter will read: the block's own ``api_key_env`` /
    ``base_url_env`` when the spec sets one, else the adapter's default. A string
    ref (``"reflect": "openai_compatible"``) has no kwargs, so it is the default."""
    name = block.get(field) if isinstance(block, dict) else None
    return name if isinstance(name, str) and name else default


_OPENAI_HOST = "api.openai.com"


def _key_sent_to_openai(block: Any, key_env: str, url_env: str) -> Optional[str]:
    """Say so when ``openai_compatible`` would send a non-OpenAI key to OpenAI (#422).

    The adapter's base URL is ``$base_url_env`` when that variable is set, else the
    block's ``default_base_url``, else ``https://api.openai.com/v1``; the key is
    sent as ``Authorization: Bearer`` to whatever that resolves to. A key variable
    that is not OpenAI's (``ARK_API_KEY``) with no endpoint configured therefore
    ships a third-party secret to OpenAI, and the run fails with a 401 from a host
    the user never named. This is the part of "will run here" that a shape check
    cannot see. Only a key that is *set* is warned about -- an unset one has its
    own warning -- and a name containing ``OPENAI`` (``MY_OPENAI_KEY``) is taken
    to be an OpenAI key."""
    if key_env == "OPENAI_API_KEY" or "OPENAI" in key_env.upper():
        return None
    default = block.get("default_base_url") if isinstance(block, dict) else None
    if not (isinstance(default, str) and default):
        default = f"https://{_OPENAI_HOST}/v1"
    base = os.environ.get(url_env, default)
    try:
        host = urlparse(base).hostname
    except ValueError:
        return None
    if host != _OPENAI_HOST:
        return None
    return (f"`openai_compatible` would send {key_env} to {base} ({_OPENAI_HOST}): no "
            f"other endpoint is configured ({url_env} is unset and `default_base_url` "
            f"does not name another host), and {key_env} is not an OpenAI key. Set "
            f"`default_base_url` (the endpoint URL) or export {url_env}")


def _unusable_refs(spec: EvolveSpec) -> List[str]:
    """Warnings for agents this machine cannot actually run.

    `plan` is the step that exists so nobody spends a run finding out, and it
    was only checking the spec's *shape*: a spec naming `claude` -- the Anthropic
    SDK completion, not the CLI -- passed and was priced at 72 calls on a machine
    with neither the `anthropic` package nor a key, and would have failed on the
    first one. Shape is not the same as "will run here".
    """
    report = doctor_report()
    keys, clis = report["provider_keys_present"], report["agent_clis"]
    out: List[str] = []
    for field in ("agent", "reflect"):
        block = getattr(spec, field, None)
        if isinstance(block, dict):
            ref = block.get("ref")
        elif isinstance(block, str):
            ref = block
        else:
            ref = None
        if not ref:
            continue
        if ref == "host_model":
            from .host_sampling import HOST_CLI_ENV, SAMPLING_URL_ENV

            if not (os.environ.get(SAMPLING_URL_ENV) or os.environ.get(HOST_CLI_ENV)):
                # Measured: a spec with `reflect: host_model`, resumed from a
                # shell, did four rounds with `considered: 0` and finished at
                # reward 0.0 -- every proposal raised and the run reported
                # nothing but silence. `host_model` means "the host that started
                # this run", and a shell is not one.
                out.append(
                    f"{field}: `host_model` borrows the model of the agent session that "
                    "started the run, and there is no host in this environment -- from a "
                    "shell it raises on every call and the run proposes nothing. Start it "
                    "from the agent, or name a model here")
        elif ref == "claude":
            missing = []
            if not report["optional"].get("anthropic"):
                missing.append("the `anthropic` package")
            key_env = _key_env(block, "ANTHROPIC_API_KEY")
            if not os.environ.get(key_env):
                missing.append(key_env)
            if missing:
                out.append(f"{field}: `claude` is the Anthropic SDK completion (the "
                           f"CLI is `claude_code`) and needs {' and '.join(missing)}")
        elif ref in ("openai_compatible", "agentdescent.agents:openai_compatible"):
            # The short name and the qualified one resolve to the same adapter, so
            # both get the preflight (a qualified ref used to skip it).
            key_env = _key_env(block, "OPENAI_API_KEY")
            url_env = _key_env(block, "OPENAI_BASE_URL", field="base_url_env")
            if not os.environ.get(key_env):
                out.append(f"{field}: `openai_compatible` needs {key_env} (and "
                           f"{url_env} for a non-OpenAI endpoint); without it "
                           + ("every proposal raises and the run fails after two rounds "
                              "with nothing learned" if field == "reflect"
                              else "every call raises"))
            else:
                leak = _key_sent_to_openai(block, key_env, url_env)
                if leak:
                    out.append(f"{field}: {leak}")
        elif ref in _CLI_BINARIES:
            binary = _CLI_BINARIES[ref]
            # Missing binary is a hard error via _missing_cli_messages / plan ok
            # and require_cli_on_path (#346); only soft-warn about isolation here.
            isolate = block.get("isolate") if isinstance(block, dict) else None
            if clis.get(binary) and isolate is not False and not keys.get(
                    _CLI_ENV_FALLBACK.get(ref, ""), False):
                # The trap that ate a 60-round run: `codex` was on PATH, so
                # nothing complained, but a worker runs with the host's config
                # directory redirected -- an interactive login does not carry
                # over. Every rollout failed. Being present is not being usable.
                remedy = _CLI_ENV_USELESS.get(ref) or (
                    ref == "opencode" and _opencode_provider_remedy(block)) or (
                    f"or export {_CLI_ENV_FALLBACK[ref]}"
                    if ref in _CLI_ENV_FALLBACK else
                    "whether a provider key in the environment is enough instead "
                    "is untested for this CLI")
                out.append(
                    f"{field}: `{ref}` is on PATH but a worker runs isolated, so an "
                    f'interactive login does not carry over. Set "isolate": false '
                    f"to use the signed-in CLI -- {remedy}")
    return out


def _selection_not_merging(comp) -> Optional[str]:
    """Say so when `n_workers` buys selection rather than the merge it looks like.

    The reflective pair is the default now, so this fires on the one case that
    is left: a spec whose only model is a file-editing CLI agent, where paying a
    whole agent session per merge is not something to switch on unasked. Without
    a fusion policy a one-key artifact -- `SingleSlot`, every prompt -- has its
    worker proposals contradict by construction and collapse to one candidate,
    so no fusion is built. Asked against the *composed* bundle rather than the
    spec's `policies` block, because the default never appears in the latter.
    """
    installed = comp.kwargs.get("policies")
    if installed is not None and (installed.fusion is not None
                                  or installed.conflict is not None):
        return None
    strategy = comp.kwargs.get("strategy")
    workers = comp.kwargs.get("n_workers") or 1
    try:
        keys = list(strategy.keys())
    except Exception:  # noqa: BLE001 - a strategy that cannot enumerate: say nothing
        return None
    if workers > 1 and len(keys) == 1:
        return (f"n_workers={workers} on a one-key artifact ({type(strategy).__name__}) "
                "is best-of-N selection, not merging: worker proposals contradict by "
                "construction and collapse to one candidate, so no fusion is built. "
                "Name a cheap model in `reflect` and it merges them instead.")
    return None


#: Same rule as ``evolve()`` / ``_build_engine`` — keep the strings aligned (#343).
_MIN_TASKS_FOR_SPLIT = 4
_MIN_TASKS_MSG = "need at least 4 tasks to split train/held-out"


def plan_payload(spec: EvolveSpec, *, usd_per_call: Optional[float] = None) -> Dict[str, Any]:
    comp = compose(spec)
    warnings = _unusable_refs(spec)
    selection = _selection_not_merging(comp)
    if selection:
        warnings.append(selection)
    audit = None
    if comp.audit is not None:
        audit = {"store": comp.audit.store.path,
                 "enabled": comp.audit.enabled,
                 "verifier_version": comp.audit.verifier_version,
                 "oracle": type(comp.audit.reward.oracle).__name__,
                 "sample_rate": comp.audit.reward.sample_rate,
                 "draw_by": comp.audit.reward.draw_by}
    n_tasks = len(comp.tasks)
    # Fail closed here: plan exists to catch what evolve would raise, before a
    # background run is started. Soft "8-20" notes stay for 4 <= n < 8 (#343).
    # Missing CLI binaries are hard errors, not warnings (#346).
    too_few = n_tasks < _MIN_TASKS_FOR_SPLIT
    hard_errors: List[str] = []
    if too_few:
        hard_errors.append(
            f"{_MIN_TASKS_MSG} (plan saw {n_tasks}; add cases until you have at "
            f"least {_MIN_TASKS_FOR_SPLIT}, then re-run plan)"
        )
    hard_errors.extend(_missing_cli_messages(spec))
    payload = {"ok": not hard_errors, "spec": spec.to_dict(), "tasks": n_tasks,
            "warnings": warnings,
            "artifact_id": spec.artifact_id(),
            **({"audit": audit} if audit else {}),
            "evolve_kwargs": {k: (v if isinstance(v, (int, float, str, bool, type(None)))
                                  else type(v).__name__)
                              for k, v in comp.kwargs.items()},
            **({"algorithm": comp.algorithm} if comp.algorithm else {}),
            "estimate": estimate(comp, usd_per_call=usd_per_call), "notes": comp.notes}
    if hard_errors:
        payload["error"] = "; ".join(hard_errors)
    return payload


def cmd_plan(a: argparse.Namespace) -> int:
    spec = load_spec(a.spec)
    try:
        payload = plan_payload(spec, usd_per_call=a.usd_per_call)
    except SpecError as e:
        _out({"ok": False, "error": str(e)}, as_json=a.json)
        return 2
    _out(payload, as_json=a.json)
    return 0 if payload.get("ok", True) else 2


def _warn_unusable(spec: EvolveSpec) -> None:
    """Print `plan`'s warnings on the verbs that actually start a run.

    They were only ever shown by `plan`, which is the one verb that does not
    spend anything -- so a spec that could not work reached `evolve` and
    `resume` in silence.
    """
    for line in _unusable_refs(spec):
        print(f"warning: {line}", file=sys.stderr)


def cmd_evolve(a: argparse.Namespace) -> int:
    spec = load_spec(a.spec)
    try:
        comp = compose(spec)                 # fail here, not in the detached child
    except SpecError as e:
        print(f"spec error: {e}", file=sys.stderr)
        return 2
    # Before create/launch: a missing --usd-per-call must not leave an orphan run
    # whose status later reads as "process ended without reporting" (#342).
    runstore.require_budget_pair(a.budget, a.usd_per_call)
    # Same gate as plan / MCP start: missing CLI binary must not report started (#346).
    require_cli_on_path(spec)
    _warn_unusable(spec)
    rd = runstore.create(spec.to_dict(), store=a.store)
    if a.detach:
        # The child composes again and rewrites `notes`, but only once it is up;
        # recording them here means `status` has them from the first poll (#414).
        rd.update_status(notes=comp.notes)
        st = runstore.launch(rd, budget_usd=a.budget, usd_per_call=a.usd_per_call)
        if not a.json:
            for line in comp.notes:
                print(f"note: {line}", file=sys.stderr)
        _out({"run_id": rd.run_id, "state": st.state, "pid": st.pid, "dir": rd.path,
              "notes": comp.notes}
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
        if os.path.exists(rd.audit_path):
            # Only once it exists: a path to a file that was never written
            # invites `audit_status` to report an empty store as an answer.
            st["audit_store"] = rd.audit_path
        st["checkpoint"] = _checkpoint_summary(rd)
        return st
    return [st.to_dict() for st in runstore.list_runs(store=store)]


def _checkpoint_summary(rd: "runstore.RunDir") -> Optional[dict]:
    """The latest checkpoint of this run's ledger, for the status line.

    ``None`` when the run has none (no checkpointing aggregator, or never
    got through a round). A present checkpoint is what makes a resume
    continue the *search* rather than just the artifact — worth showing,
    because "resumed" has meant two very different things since checkpoints
    landed and only one of them keeps the archive.
    """
    from .checkpoint import load_checkpoint
    if not os.path.isdir(rd.ledger_path):
        return None
    payload = load_checkpoint(rd.ledger_path)
    if payload is None:
        return None
    return {
        "round": payload.get("round"),
        "aggregator": payload.get("aggregator_type"),
        "early_stop": payload.get("early_stop"),
    }


def _fmt_reward(value: Any) -> str:
    """A ``rounds.jsonl`` reward for a text line.

    ``append_round`` stores a non-finite reward as ``null`` (#271), so a row can
    carry ``None``; ``:.3f`` on that raised ``TypeError`` and took ``status`` /
    ``watch`` down with it. Anything else that is not a number prints as-is.
    """
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _round_errors(r: Dict[str, Any]) -> str:
    """``  propose_errors=N: <first>`` for a ``rounds.jsonl`` row that has any (#395)."""
    n = r.get("propose_errors") or 0
    return f"  propose_errors={n}: {r.get('propose_error')}" if n else ""


def cmd_status(a: argparse.Namespace) -> int:
    if a.run_id:
        rd = runstore.get(a.run_id, store=a.store)
        st = rd.status()
        if a.json:
            _out(status_payload(a.run_id, store=a.store), as_json=True)
        else:
            print(_status_line(st))
            if not a.brief:
                for note in st.notes:
                    print(f"  note: {note}")
                for r in rd.rounds()[-3:]:
                    print(f"  round {r['round']:>3}  reward={_fmt_reward(r.get('held_out_reward'))}  "
                          f"+{r['committed']}/-{r['rejected']}  {r.get('reasons', {})}"
                          f"{_round_errors(r)}")
                cp = _checkpoint_summary(rd)
                if cp:
                    es = cp.get("early_stop") or {}
                    stalled = es.get("stalled", "?")
                    print(f"  checkpoint: round {cp['round']} "
                          f"({cp['aggregator']}) — a resume continues the "
                          f"search, stall counter at {stalled}")
                else:
                    print("  checkpoint: none — a resume restarts the search "
                          "from the ledger head")
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
        print(f"no runs under {runstore.store_path(a.store)}")
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
            print(f"round {r['round']:>3}  reward={_fmt_reward(r.get('held_out_reward'))}  "
                  f"+{r['committed']}/-{r['rejected']}  {r.get('reasons', {})}"
                  f"{_round_errors(r)}", flush=True)
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
    # Reflector exceptions (#395). Only when there were some, so a healthy run's
    # payload is unchanged.
    perr = sum(h.propose_errors for h in result.history)
    if perr:
        payload["propose_errors"] = perr
        payload["propose_error"] = next(
            (h.propose_error for h in result.history if h.propose_error), None)
    spec = EvolveSpec.from_dict(rd.spec_dict())
    # What `apply` would overwrite, on every kind. Without it the caller can see
    # the evolved artifact and still not know which file it belongs to -- and an
    # agent that cannot name the file it is about to overwrite is right to
    # refuse, which is what happened.
    payload.update(kind=spec.kind, target=spec.target)
    if spec.kind == "policy_slot":
        # The artifact is source text, like `text` -- but `target` is a *slot*,
        # so there is no file it came from and none to diff against. Listing it
        # as a tree gave a file called `value` and an `apply_plan` holding an
        # error, which is worse than saying plainly where it can go.
        payload["rendered"] = result.rendered[:max_chars]
        payload["apply_plan"] = {
            "needs": "--to PATH",
            "why": f"the target is the {spec.target!r} policy slot, not a file; "
                   "an evolved rule is pasted into a spec or passed to "
                   "evolve(policies=...), so apply must be told where to write it",
            "chars": len(result.rendered)}
        return payload
    if spec.kind == "text":
        payload["rendered"] = result.rendered[:max_chars]
        payload["apply_plan"] = _text_plan(os.path.expanduser(spec.target), result.rendered)
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
    for k in ("run_id", "state", "stop_reason", "error", "final_reward", "outcomes",
              "propose_errors", "propose_error", "note"):
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


def _text_plan(dest: str, rendered: str) -> Dict[str, Any]:
    """What applying a single-file artifact to ``dest`` would do (#396).

    One judgement for ``show``, ``apply --dry-run`` and ``apply``: a file that
    already holds the evolved text is ``unchanged`` -- nothing is written and no
    backup is made.
    """
    try:
        with open(dest, encoding="utf-8", newline="") as fh:     # verbatim, CRLF included
            same = fh.read() == rendered
    except (OSError, UnicodeDecodeError):
        same = False
    if same:
        return {"would_write": None, "written": [], "unchanged": [dest],
                "chars": len(rendered), "noop": True}
    return {"would_write": dest, "written": [dest], "unchanged": [],
            "chars": len(rendered), "noop": False}


def apply_payload(run_id: str, *, to: Optional[str] = None, store: Optional[str] = None,
                  dry_run: bool = False, backup: bool = True) -> Dict[str, Any]:
    rd = runstore.get(run_id, store=store)
    result = rd.result()
    if result is None:
        raise runstore.RunStoreError(f"{run_id} has no result to apply yet")
    spec = EvolveSpec.from_dict(rd.spec_dict())
    if spec.kind == "policy_slot" and not to:
        # Without this the slot name is treated as a path and `write_to` raises
        # a TreeError that `main()` does not catch, so the command traces back.
        raise runstore.RunStoreError(
            f"{run_id} evolved the {spec.target!r} policy slot, which is not a file: "
            "pass --to PATH to write the rule, or read it with `show`")
    dest = os.path.expanduser(to or spec.target)
    if spec.kind in ("text", "policy_slot"):
        plan = _text_plan(dest, result.rendered)
        if dry_run:
            return {**plan, "dest": dest}
        if plan["noop"]:
            return {"written": [], "unchanged": [dest], "backup": [], "noop": True,
                    "dest": dest}
        saved = []
        if backup and os.path.exists(dest):
            # `.bak-N`, never a fixed `.bak`: the directory path a few lines down
            # has always numbered its backups, and this one overwrote the single
            # slot on every apply. Apply, hand-edit, apply again and the second
            # backup was the hand-edit -- the original the first apply had saved
            # was gone, from the same command that promises to keep one.
            n = 0
            while os.path.exists(f"{dest}.bak-{n}"):
                n += 1
            shutil.copy2(dest, f"{dest}.bak-{n}")
            saved = [f"{dest}.bak-{n}"]
        with open(dest, "w", encoding="utf-8", newline="") as fh:
            fh.write(result.rendered)
        return {"written": [dest], "unchanged": [], "backup": saved, "noop": False,
                "dest": dest}
    plan = result.write_to(dest, backup=backup, dry_run=dry_run)
    plan["dest"] = dest
    return plan


def cmd_apply(a: argparse.Namespace) -> int:
    payload = apply_payload(a.run_id, to=a.to, store=a.store, dry_run=a.dry_run,
                            backup=not a.no_backup)
    _out(payload, as_json=True)
    if payload.get("noop"):
        # After the JSON, on stderr: stdout stays one machine-readable document.
        print(f"nothing to apply: {payload.get('dest') or '(target)'} already matches "
              f"run {a.run_id}", file=sys.stderr)
    return 0


def cmd_cancel(a: argparse.Namespace) -> int:
    st = runstore.cancel(a.run_id, store=a.store)
    print(_status_line(st))
    return 0


def cmd_resume(a: argparse.Namespace) -> int:
    # The environment that resumes is rarely the one that started: a run begun
    # from an agent session and resumed from a shell has lost whatever
    # `host_model` was borrowing.
    runstore.require_budget_pair(a.budget, a.usd_per_call)
    # Hard gate, unlike the warnings below: a missing plugin host cannot run a
    # single rollout, so do not relaunch the detached process at all (#370).
    require_resume_host_on_path(a.run_id, store=a.store)
    try:
        _warn_unusable(EvolveSpec.from_dict(runstore.get(a.run_id, store=a.store).spec_dict()))
    except Exception:  # noqa: BLE001 - a warning must never block a resume
        pass
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
            if k not in ("problems", "hosts", "hints"):
                print(f"{k}: {v}")
        for name, info in rep["hosts"].items():
            print(f"host {name}: {info['state']}"
                  + (f" (cli {info['cli']})" if info["cli"] else " (cli not on PATH)"))
        for h in rep["hints"]:
            print(f"- {h}")
        for p in rep["problems"]:
            print(f"! {p}")
        if not rep["problems"]:
            print("ok: nothing missing")
    return 0 if not rep["problems"] else 1


def _algorithm_card_text(card: Dict[str, Any]) -> str:
    lines = [f"{card['name']}  [{card['fidelity']}]", f"  {card['summary']}",
             f"  kinds:       {', '.join(card['kinds'])}",
             f"  extra calls: {card['extra_calls']}"]
    if card["source"]:
        lines.append(f"  source:      {card['source']}")
    if card["doc"]:
        lines.append(f"  docs:        docs/{card['doc']}")
    lines.append("  params (spec: \"algorithm\": {\"name\": \"%s\", \"params\": {...}}):"
                 % card["name"])
    if not card["params"]:
        lines.append("    (none)")
    for p in card["params"]:
        bounds = (f" choices={p['choices']}" if "choices" in p else "") + (
            f" min={p['minimum']}" if "minimum" in p else "") + (
            f" max={p['maximum']}" if "maximum" in p else "")
        lines.append(f"    {p['name']}: {p['type']} = {json.dumps(p['default'])}{bounds}")
        if p["help"]:
            lines.append(f"        {p['help']}")
    return "\n".join(lines)


def cmd_algorithms(a: argparse.Namespace) -> int:
    from . import algorithms

    as_json = a.json
    if a.action == "list":
        if a.kind is not None and a.kind not in KINDS:
            raise SpecError(f"--kind must be one of {list(KINDS)}, got {a.kind!r}")
        found = algorithms.listing(a.kind)
        if as_json:
            _out([x.describe() for x in found], as_json=True)
        elif not found:
            print(f"no algorithms for kind {a.kind!r}")
        else:
            width = max(len(x.name) for x in found)
            for x in found:
                print(f"{x.name:<{width}}  {x.fidelity:<19}  {','.join(x.kinds):<42}  {x.summary}")
        return 0
    try:
        card = algorithms.get(a.name, where="show").describe()
    except algorithms.AlgorithmError as e:
        raise SpecError(str(e)) from None
    _out(card if as_json else _algorithm_card_text(card), as_json=as_json)
    return 0


def cmd_install(a: argparse.Namespace) -> int:
    from .integrations import install

    try:
        written = install(a.host, dry_run=a.dry_run, home=a.home)
    except ValueError as e:        # e.g. an empty --home (#412)
        raise SpecError(f"install: {e}") from None
    for line in written:
        print(line)
    return 0


def cmd_uninstall(a: argparse.Namespace) -> int:
    from .integrations import uninstall

    try:
        lines = uninstall(a.host, dry_run=a.dry_run, home=a.home)
    except ValueError as e:        # e.g. an empty --home (#412)
        raise SpecError(f"uninstall: {e}") from None
    for line in lines:
        print(line)
    # 1 when something was kept and so is still wired: a script or an agent can
    # tell "fully removed" from "needs a manual step". Nothing installed is 0.
    return 1 if any(line.startswith("kept ") for line in lines) else 0


def cmd_serve(a: argparse.Namespace) -> int:
    print(f"agentdescent: run panel on http://{a.host}:{a.port}/  (read-only; Ctrl-C to stop)",
          file=sys.stderr)
    runstore.serve_http(host=a.host, port=a.port, store=a.store)
    return 0


def _shadowing_dirs() -> Optional[List[str]]:
    """Where ``agentdescent`` resolved to, when that is not the installed package.

    ``python -m`` puts the host's current directory first on ``sys.path``, so a
    directory called ``agentdescent/`` there (a source checkout's parent, say)
    can win the import as a namespace package: no ``__file__``, none of our
    modules. Returns its path entries, or None when the package is the real one.
    """
    pkg = sys.modules.get("agentdescent")
    if pkg is None or getattr(pkg, "__file__", None) is not None:
        return None
    return [str(p) for p in getattr(pkg, "__path__", [])] or ["<unknown location>"]


def _mcp_import_failure(e: BaseException) -> str:
    """One line for why ``agentdescent mcp`` could not import its server.

    A host shows the user nothing but "Connection closed" when this subprocess
    dies, so this must be a single actionable line and must not itself depend on
    the modules that just failed to import (a shadowed package breaks
    ``mcp_unavailable``'s own ``from .integrations import ...``).
    """
    shadow = _shadowing_dirs()
    if shadow:
        return ("`agentdescent` was resolved to %s, not the installed package: a directory "
                "named agentdescent/ in the current directory (%s) shadows it. Start the "
                "server from another directory, or re-run `agentdescent install <host>` "
                "(the launcher it writes ignores the current directory). Underlying "
                "error: %s" % (", ".join(shadow), os.getcwd(), e))
    try:
        why = mcp_unavailable()
    except Exception:  # noqa: BLE001 - the reason must never become a traceback
        why = None
    # The underlying ImportError says "pip install agentdescent[mcp]", which on
    # 3.9 is the one thing that will not help -- so it is printed only when we
    # have nothing better, never after a reason that contradicts it.
    return why or ('pip install "agentdescent[mcp]" (the CLI works without it). '
                   f'Underlying error: {e}')


def cmd_mcp(a: argparse.Namespace) -> int:
    # A host starts this as a subprocess and shows the user nothing but
    # "CONNECTION_CLOSED" when it dies, so the one thing that can go wrong
    # before the protocol starts -- the SDK not being installed -- has to say so
    # in one line on stderr rather than as a traceback nobody will see.
    try:
        from .mcp import serve

        serve(store=a.store)
    except ImportError as e:
        print(f"agentdescent mcp: cannot start the MCP server -- {_mcp_import_failure(e)}",
              file=sys.stderr)
        return 3
    return 0


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


# Sub-commands that exist for AgentDescent's own use and are kept out of
# `--help` and the usage line.
_INTERNAL_COMMANDS = frozenset({"run"})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agentdescent",
                                description="Evolve skills, agents, prompts, code and plugins "
                                            "with a parallel, merge-based optimiser.")
    p.add_argument("-V", "--version", action="version",
                   version=f"%(prog)s {_version()}",
                   help="print the AgentDescent version and exit")
    p.add_argument("--store", help="run store directory (default ~/.agentdescent/runs "
                                    "or $AGENTDESCENT_HOME/runs)")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("demo", help="build a complete example and run it, offline")
    s.add_argument("--dir", help="where to build it (default: a temp directory)")
    s.set_defaults(fn=cmd_demo)

    s = sub.add_parser("init", help="write a starter spec for a path")
    s.add_argument("path")
    s.add_argument("--kind", choices=("text", "skill_dir", "agent_dir", "agent_code",
                                  "plugin", "policy_slot"))
    s.add_argument("--data", help="cases file, or a module:attribute problems ref for policy_slot")
    s.add_argument("--agent", default=None,
                   help="worker agent short name (default: the first of claude_code, "
                        "codex, dsh, opencode whose CLI is on PATH; claude_code if none)")
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

    # The detached worker `evolve --detach` spawns (runstore: `python -m
    # agentdescent.cli run --run-dir <dir>`); not a user verb.  argparse does
    # not honour help=SUPPRESS on a sub-parser (it prints "==SUPPRESS==" and
    # keeps the name in the {...} list), so register it with no help at all
    # -- no help line -- and leave it out of the metavar below (#420).
    s = sub.add_parser("run", description="internal: the worker `agentdescent evolve "
                                          "--detach` starts; use `evolve` instead")
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

    s = sub.add_parser("uninstall", help="undo `install` for a host")
    s.add_argument("host", choices=("dsh", "claude-code", "codex", "opencode"))
    s.add_argument("--dry-run", action="store_true", dest="dry_run")
    s.add_argument("--home", help="the host's home directory (default: the real one)")
    s.set_defaults(fn=cmd_uninstall)

    s = sub.add_parser("algorithms", help="the named algorithms a spec can ask for")
    algo_sub = s.add_subparsers(dest="action", required=True)
    s2 = algo_sub.add_parser("list", help="every algorithm, or those that apply to a kind")
    s2.add_argument("--kind", help="only algorithms for this spec kind")
    s2.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                    help="machine-readable output")
    s2.set_defaults(fn=cmd_algorithms)
    s2 = algo_sub.add_parser("show", help="one algorithm: what it does, its parameters")
    s2.add_argument("name")
    s2.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                    help="machine-readable output")
    s2.set_defaults(fn=cmd_algorithms)

    s = sub.add_parser("serve", help="serve a read-only run panel on loopback, for a host UI")
    s.add_argument("--host", default=runstore.DEFAULT_HTTP_HOST)
    s.add_argument("--port", type=int, default=runstore.DEFAULT_HTTP_PORT)
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("mcp", help="serve the tools over stdio (needs agentdescent[mcp])")
    s.set_defaults(fn=cmd_mcp)

    sub.metavar = "{" + ",".join(n for n in sub.choices if n not in _INTERNAL_COMMANDS) + "}"
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
