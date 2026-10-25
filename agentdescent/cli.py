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
    agentdescent portfolio [archive]                          rank layers, reallocate budget
    agentdescent exam     [archive] --domain module:builder    re-judge adoptions with an oracle
    agentdescent audit    <run_id> --archive PATH              trace an adoption's evidence rings
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
from collections import deque
import difflib
import json
import os
import shutil
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
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


#: What ``init`` writes where only the user knows the value (#451): the model a
#: reflector names and, for a non-OpenAI key, its endpoint. The shipped SKILL.md
#: says "ask the user for the endpoint and the model; never guess either", and a
#: guessed pair (``deepseek-v4-flash`` against api.openai.com) passed ``plan``
#: and failed every call. ``plan`` refuses a spec that still holds one.
PLACEHOLDER_MODEL = "<your-model>"
PLACEHOLDER_URL = "<your-endpoint-url>"


def starter_model_block(*, model: Optional[str] = None, base_url: Optional[str] = None,
                        key_env: Optional[str] = None,
                        keys: Optional[Dict[str, bool]] = None) -> Dict[str, Any]:
    """The ``openai_compatible`` block ``init`` writes, matched to the keys set here.

    The same reading of the environment ``doctor`` gives (#451): with
    ``OPENAI_API_KEY`` the adapter's defaults apply; with only another provider
    key (``ARK_API_KEY``, ``DEEPSEEK_API_KEY``, ...) the block names it in
    ``api_key_env`` and needs ``default_base_url`` -- otherwise that key would be
    sent to api.openai.com. The model, and that endpoint, are the user's
    (``--model`` / ``--base-url``) or a placeholder ``plan`` refuses; never a guess.
    """
    if keys is None:
        keys = doctor_report()["provider_keys_present"]
    if key_env is None and not keys.get("OPENAI_API_KEY"):
        from .integrations import provider_key_names

        other = [n for n in provider_key_names()
                 if keys.get(n) and n not in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")]
        if other:
            key_env = other[0]
    block: Dict[str, Any] = {"ref": "openai_compatible", "model": model or PLACEHOLDER_MODEL}
    if key_env and key_env != "OPENAI_API_KEY":
        block["api_key_env"] = key_env
        if not base_url and "OPENAI" not in key_env.upper():
            base_url = PLACEHOLDER_URL
    if base_url:
        block["default_base_url"] = base_url
    return block


def _opencode_provider_id(base_url: str) -> str:
    """The provider id an ``init``-written OpenCode config uses for ``base_url``.

    ``ark`` for Volcengine Ark, matching docs/plugins.md; ``custom`` otherwise.
    It is only a local name inside the file ``init`` writes, so nothing about
    the endpoint is guessed from it.
    """
    from urllib.parse import urlparse

    host = (urlparse(base_url).hostname or "").lower()
    return "ark" if host.endswith("volces.com") else "custom"


def opencode_coding_recipe(*, model: str, base_url: str, key_env: str,
                           config_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """The ``agent`` block and provider-only config for an isolated OpenCode worker (#485).

    The documented Coding recipe (docs/plugins.md): ``-m <provider>/<model>`` plus
    ``env.OPENCODE_CONFIG`` naming a file that holds only a ``provider`` block,
    with the key as ``{env:VAR}`` -- never inline. Returns ``(agent, config)``;
    the caller writes ``config`` to ``config_path`` (an absolute path).
    """
    provider = _opencode_provider_id(base_url)
    agent = {"ref": "opencode", "extra_args": ["-m", f"{provider}/{model}"],
             "env": {"OPENCODE_CONFIG": config_path}}
    config = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {provider: {
            "npm": "@ai-sdk/openai-compatible",
            "name": "Ark" if provider == "ark" else "Custom",
            "options": {"baseURL": base_url, "apiKey": "{env:%s}" % key_env},
            "models": {model: {"name": model}},
        }},
    }
    return agent, config


def parse_cli_params(pairs: Optional[Sequence[str]]) -> Dict[str, Any]:
    """``--param k=v`` -> a JSON-typed dict (bare strings when JSON cannot parse).

    Repeated keys are an error: silently keeping the last one makes a typo or a
    copied flag look applied.
    """
    out: Dict[str, Any] = {}
    for item in pairs or ():
        if "=" not in item:
            raise ValueError(f"--param {item!r} must be KEY=VALUE")
        key, raw = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--param {item!r} has an empty name")
        if key in out:
            raise ValueError(f"--param {key!r} was given more than once")
        try:
            out[key] = json.loads(raw)
        except ValueError:
            out[key] = raw
    return out


def algorithm_override(name: Optional[str], params: Optional[Mapping[str, Any]],
                       kind: str) -> Optional[Any]:
    """Validate ``--algorithm``/``--param`` for ``kind`` and return the spec value.

    Returns ``None`` when no algorithm was named, the name itself when it takes
    no parameters, and ``{"name": ..., "params": {...}}`` otherwise. Raises
    :class:`~agentdescent.algorithms.AlgorithmError` for an unknown name, an
    unsupported kind or a bad parameter -- the same errors ``plan`` would give,
    but before a spec is written.
    """
    if not name:
        if params:
            raise ValueError("--param needs --algorithm")
        return None
    from . import algorithms

    alg = algorithms.get(name, where="--algorithm")
    if kind not in alg.kinds:
        raise algorithms.AlgorithmError(
            f"--algorithm {name!r} applies to kind {list(alg.kinds)}, not {kind!r}")
    algorithms.resolve_params(alg, params)      # validates types/unknowns/choices
    return name if not params else {"name": name, "params": dict(params)}


def starter_spec(path: str, *, kind: Optional[str] = None, data: Optional[str] = None,
                 agent: Optional[str] = None,
                 agent_explicit: bool = False, model: Optional[str] = None,
                 base_url: Optional[str] = None,
                 key_env: Optional[str] = None,
                 algorithm: Optional[str] = None,
                 algorithm_params: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """A spec to edit, with ``kind`` guessed from what ``path`` is.

    ``agent=None`` picks the worker from the CLIs on PATH (:func:`default_agent_ref`).
    ``agent_explicit`` says the user named ``agent`` (``init --agent``): it is
    then kept for every kind instead of being replaced by the kind's default
    model (#456).
    ``model`` / ``base_url`` / ``key_env`` fill the ``openai_compatible`` blocks
    (:func:`starter_model_block`); left out, they are placeholders or follow the
    keys in the environment.
    """
    if agent is None:
        agent = default_agent_ref()[0]
    llm = starter_model_block(model=model, base_url=base_url, key_env=key_env)
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
        if not agent_explicit or agent == "openai_compatible":
            spec["agent"] = dict(llm)
        spec["evolve"] = {"rounds": 8, "n_workers": 8}
    if kind == "skill_dir":
        spec["reflect"] = dict(llm)
        if agent == "claude_code":
            spec["agent"] = {"ref": "claude_code",
                             "extra_args": ["--permission-mode", "acceptEdits"]}
    if kind == "agent_code":
        spec["entrypoint"] = ["python", "main.py"]
        spec["reflect"] = dict(llm)
    if kind == "plugin":
        spec["host"] = ("claude_code" if os.path.exists(os.path.join(p, ".claude-plugin"))
                        else "dsh")
        spec["reflect"] = dict(llm)
    if kind == "policy_slot":
        # `target` is a slot, not a path, and `data` holds refs rather than rows
        # -- an inner problem is a callable and no row format can express one.
        spec["target"] = path if path in _POLICY_SLOTS else "selection"
        spec["data"] = {"problems": data or "mypkg.problems:build", "seeds": [0]}
        spec["score"] = "auc"
        if not agent_explicit or agent == "openai_compatible":
            spec["agent"] = dict(llm)
        # A round is `n_workers` whole inner searches, so ask for few and budget
        # in wall clock: the recorded run asked for 8 rounds and got 2 in 90 min.
        spec["evolve"] = {"rounds": 4, "n_workers": 2, "max_seconds": 5400}
        spec["allow"] = ["mypkg"]
    chosen = algorithm_override(algorithm, algorithm_params, spec["kind"])
    if chosen is not None:
        spec["algorithm"] = chosen
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
    from .integrations import provider_key_names

    # The four names this dict has always had come first, so the output keeps its
    # order; every other key `FORWARDED_ENV` carries (``ARK_API_KEY``, ...) follows.
    # One list, not two: it used to be a hard-coded three, and a key every
    # installer forwards read as "no provider key" (#423).
    key_names = provider_key_names()
    keys = {name: bool(os.environ.get(name)) for name in (
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL", "DEEPSEEK_API_KEY",
        *key_names)}
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
    present_keys = [n for n in key_names if keys[n]]
    if not present_keys:
        problems.append("no provider key in the environment; a reflector needs one "
                        "(under dsh, forward keys in the mcp-client env block)")
    why = mcp_unavailable()
    if why:
        problems.append(why)
    hosts, hints = _host_integrations(clis)
    other = [n for n in present_keys if n not in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")]
    if other and len(other) == len(present_keys):
        # `openai_compatible` reads OPENAI_API_KEY and talks to api.openai.com unless
        # told otherwise, so a key like ARK_API_KEY works only when the spec says so.
        hints.append(
            f"only {', '.join(other)} is set: use it with `openai_compatible` and "
            f'`"api_key_env": "{other[0]}"` plus `default_base_url` (or `base_url_env`) '
            "for its endpoint; the endpoint and the model name come from the user, "
            "never guess them")
    # Reflect is not the only role: an isolated `opencode` worker with a gateway
    # key still needs the Coding recipe (#455). Same predicate as `init`
    # (`_opencode_gateway_keys`): warn whenever a gateway key is present, even
    # alongside OPENAI_API_KEY, so the two commands never disagree.
    gateway = _opencode_gateway_keys(keys)
    if gateway and clis.get("opencode"):
        hints.append(
            "opencode is on PATH and %s is set: for an isolated worker %s"
            % (", ".join(gateway), _opencode_coding_recipe(gateway)))
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

  agentdescent init <your-skill-dir> --data <your-cases.jsonl> --out spec.json
  agentdescent doctor        # what is missing before a real run
  agentdescent plan spec.json""")
    return 0


def _starter_model_warnings(spec: Dict[str, Any], out: str) -> List[str]:
    """What ``init`` says about the blocks it just wrote.

    One answer with ``doctor`` (#451): for ``openai_compatible`` it names the key
    the block will read -- the one that is set here, not always OPENAI_API_KEY --
    and every value left for the user to fill in, which ``plan`` refuses until
    they do. For a bare ``opencode`` worker with only a gateway key (#455), it
    points at the documented ``OPENCODE_CONFIG`` Coding recipe so ``init`` does
    not leave a silent ``{"ref": "opencode"}`` that dies on OpenCode's built-in
    Volcengine default under isolation.
    """
    lines: List[str] = []
    for field in ("agent", "reflect"):
        block = spec.get(field)
        if not (isinstance(block, dict) and block.get("ref") == "openai_compatible"):
            continue
        key_env = _key_env(block, "OPENAI_API_KEY")
        todo = [f"`{field}.{k}`" for k in ("model", "default_base_url")
                if _is_placeholder(block.get(k))]
        if todo:
            via = (f" It reads {key_env}, the provider key set here." if key_env != "OPENAI_API_KEY"
                   and os.environ.get(key_env) else "")
            lines.append(
                f"{' and '.join(todo)} in {out} {'is a placeholder' if len(todo) == 1 else 'are placeholders'}"
                f": the model (and the endpoint for a non-OpenAI key) come from you, never "
                f"a guess.{via} Fill {'it' if len(todo) == 1 else 'them'} in, or re-run with "
                "`--model` / `--base-url`; `plan` refuses the spec until then")
        if not os.environ.get(key_env):
            # The starter reflector is a model call whatever worker CLI was found,
            # so a machine with only `opencode` / `claude` gets one it cannot call.
            # Run, it used to finish `done` with nothing learned (#395).
            lines.append(
                f"`{field}` is `openai_compatible` and {key_env} is not set, so every "
                "call would fail" + (" and the run would learn nothing" if field == "reflect" else "")
                + ". Export a provider key (`agentdescent doctor` lists the ones it "
                "knows), or pass `--key-env NAME` for one that is set")
    # Bare isolated OpenCode + gateway key: same trap `plan` already names (#455).
    # Warn at init so the written `{"ref": "opencode"}` is not silent.
    agent = spec.get("agent")
    if isinstance(agent, dict) and agent.get("ref") == "opencode" and not _opencode_configured(agent):
        keys = doctor_report()["provider_keys_present"]
        remedy = _opencode_provider_remedy(agent, keys)
        if remedy and not _opencode_model_arg(agent):
            lines.append(
                f"`agent` in {out} is bare `opencode`: {remedy}. "
                "See docs/plugins.md (OpenCode Coding / OPENCODE_CONFIG)")
    return lines


#: Built-in short refs by what they resolve to; keep in step with
#: ``evolvespec.SHORT_REFS`` / ``WORKSPACE_AGENT_REFS``. A full
#: ``module:attribute`` ref outside these is not checked here (``plan`` does).
_WORKSPACE_REFS = ("claude_code", "codex", "dsh", "opencode", "cli_agent")
_PLAIN_REFS = ("openai_compatible", "host_model", "claude", "echo")


def init_agent_mismatch(kind: str, ref: str) -> Optional[str]:
    """Why ``plan`` would refuse agent ``ref`` for ``kind``, or None (#456).

    The same two rules ``compose`` enforces, checked before ``init`` writes a
    spec that is doomed at ``plan``: a directory kind needs a WorkspaceAgent,
    and kind=text needs a plain Completion.
    """
    short = ref.split(":", 1)[1] if ref.startswith("agentdescent.agents:") else ref
    if short == "agentdescent.host_sampling:host_model":
        short = "host_model"
    if kind in ("skill_dir", "agent_dir") and short in _PLAIN_REFS:
        return (f"kind={kind!r} needs a WorkspaceAgent "
                f"({' / '.join(_WORKSPACE_REFS)}) so the directory is put in front "
                f"of it; `{ref}` is a plain model call")
    if kind == "text" and short in _WORKSPACE_REFS:
        return (f"kind='text' needs a plain Completion ({' / '.join(_PLAIN_REFS[:3])}), "
                f"not a WorkspaceAgent; `{ref}` is a file-editing agent and would "
                "cost a whole agent session per case")
    return None


def cmd_init(a: argparse.Namespace) -> int:
    agent, found = (a.agent, True) if a.agent else default_agent_ref()
    try:
        params = parse_cli_params(getattr(a, "param", None))
        spec = starter_spec(a.path, kind=a.kind, data=a.data, agent=agent,
                            agent_explicit=bool(a.agent), model=a.model,
                            base_url=a.base_url, key_env=a.key_env,
                            algorithm=getattr(a, "algorithm", None),
                            algorithm_params=params)
    except ValueError as e:      # AlgorithmError is a ValueError
        print(f"error: {e}", file=sys.stderr)
        return 2
    if a.agent:
        why = init_agent_mismatch(spec["kind"], a.agent)
        if why:
            # Refuse before writing: `plan` would reject this spec anyway, and a
            # spec that silently swaps the user's --agent is worse (#456).
            print(f"error: --agent {a.agent}: {why}; pick a matching ref "
                  "(see `agentdescent doctor`) or omit --agent for the default",
                  file=sys.stderr)
            return 2
    out = a.out or os.path.join(".agentdescent", f"{os.path.basename(a.path.rstrip('/')) or 'spec'}.evolve.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    providers_out = None
    if spec.get("agent") == {"ref": "opencode"} and a.model and a.base_url:
        # The user named the model and endpoint, so wire them into the worker
        # too, not only the reflector: a bare `opencode` would ignore both and
        # fall back on OpenCode's own default (#485). Without both flags the
        # bare ref and the #455 warning stay -- nothing is guessed.
        llm = next((b for b in (spec.get("reflect"), spec.get("agent"))
                    if isinstance(b, dict) and b.get("ref") == "openai_compatible"), {})
        key_env = llm.get("api_key_env") or a.key_env or "OPENAI_API_KEY"
        stem = os.path.splitext(os.path.basename(out))[0]
        providers_out = os.path.join(os.path.dirname(os.path.abspath(out)),
                                     f"{stem}.opencode-providers.json")
        agent_block, providers = opencode_coding_recipe(
            model=a.model, base_url=a.base_url, key_env=key_env,
            config_path=providers_out)
        spec["agent"] = agent_block
        with open(providers_out, "w", encoding="utf-8") as fh:
            json.dump(providers, fh, indent=2)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2)
    print(f"wrote {out}")
    if providers_out:
        print(f"wrote {providers_out} (OpenCode provider-only config for the worker; "
              f"key read from ${key_env}, never inline)")
    if not found and spec["kind"] not in ("text", "policy_slot"):
        print(f"warning: no worker CLI (claude, codex, dsh, opencode) is on PATH; "
              f"wrote the placeholder agent `{agent}`. Run `agentdescent doctor`, "
              "install a CLI, or re-run with `--agent <ref>` / edit `agent.ref` "
              "before `plan`")
    elif a.agent is None and spec["kind"] not in ("text", "policy_slot"):
        print(f"agent: `{agent}` (the worker CLI found on PATH; "
              "override with `--agent <ref>`)")
    for line in _starter_model_warnings(spec, out):
        print(f"warning: {line}")
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


def _opencode_has_config(block: Any) -> bool:
    """The block hands an isolated ``opencode`` its own provider config file."""
    env = block.get("env") if isinstance(block, dict) else None
    return isinstance(env, dict) and (
        "OPENCODE_CONFIG" in env or "OPENCODE_CONFIG_CONTENT" in env)


def _opencode_configured(block: Any) -> bool:
    """An isolated ``opencode`` that needs no login: ``-m p/m`` plus a config (#455).

    The documented recipe (``docs/plugins.md``): a pinned ``provider/model`` and
    ``env.OPENCODE_CONFIG`` naming a provider-only file whose key comes from
    ``{env:VAR}``. Measured with Ark Coding under isolation, it answers in ~7 s
    and a 1-round evolve made 4 calls with 0 failures, so "a login does not carry
    over" is no warning for it.
    """
    return bool(_opencode_model_arg(block)) and _opencode_has_config(block)


def _opencode_gateway_keys(keys: Optional[Dict[str, bool]]) -> List[str]:
    """Present gateway keys: set ``*_API_KEY`` other than OpenAI / Anthropic.

    OpenAI / Anthropic accounts serve their provider's default model; a gateway
    key such as ARK_API_KEY is where OpenCode's default was unusable (#455).
    One predicate for ``init``, ``plan`` and ``doctor``.
    """
    return [n for n, v in (keys or {}).items()
            if v and n.endswith("_API_KEY")
            and n not in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")]


def _opencode_coding_recipe(gateway: List[str]) -> str:
    """The OpenCode Coding worker recipe, concrete only where it was measured.

    Ark wording (Volcengine default, ``ark/deepseek-v4-flash``) only when
    ARK_API_KEY is among the present keys; any other gateway gets the generic
    recipe whose provider and model come from the user, never guessed.
    """
    if "ARK_API_KEY" in gateway:
        return ("use the OpenCode Coding recipe in docs/plugins.md: a bare "
                "`ref: opencode` falls back to OpenCode's built-in Volcengine "
                "default, which a Coding-only account may not have activated "
                "(measured with only ARK_API_KEY set: every call failed). Pin the "
                'model and hand the worker a provider-only config: '
                '"extra_args": ["-m", "ark/deepseek-v4-flash"], '
                '"env": {"OPENCODE_CONFIG": "/abs/path/providers.json"} '
                "(the file as in docs/plugins.md, keys via {env:ARK_API_KEY}, "
                "never inline)")
    return ("use the OpenCode Coding recipe in docs/plugins.md: pin "
            '"extra_args": ["-m", "<provider>/<model>"] and set '
            '"env": {"OPENCODE_CONFIG": "/abs/path/providers.json"} with the key '
            "as {env:%s} (never inline); the provider and model come from the "
            "user, never guess them" % gateway[0])


def _opencode_provider_remedy(block: Any, keys: Optional[Dict[str, bool]] = None
                              ) -> Optional[str]:
    """The plan warning's remedy for an isolated ``opencode``.

    With ``-m p/m`` pinned and no ``OPENCODE_CONFIG`` in the block's ``env``: the
    custom-provider trap (#373). With no pin while a provider key is set: the
    default-model trap (#455) -- OpenCode then picks its own default model for
    that provider, which the account may not have activated. ``None`` keeps the
    generic remedy.
    """
    model = _opencode_model_arg(block)
    if not model:
        present = _opencode_gateway_keys(keys)
        if not present:
            return None
        return (f"a provider key ({', '.join(present)}) is not enough on its own: "
                "with no `-m`, OpenCode picks its own default model for that "
                "provider, which the account may not have activated. "
                + _opencode_coding_recipe(present))
    if _opencode_has_config(block):
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
    msgs = _missing_cli_messages(spec) + _placeholder_messages(spec)
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


def _is_placeholder(value: Any) -> bool:
    """A value ``init`` left for the user (``<your-model>``), not a real one."""
    return isinstance(value, str) and value.startswith("<") and value.endswith(">")


def _placeholder_messages(spec: EvolveSpec) -> List[str]:
    """Hard-fail messages for a starter-spec placeholder that was never filled in (#451).

    ``openai_compatible(model="<your-model>")`` composes fine and fails on every
    call, so the shape check alone would let it through to a run."""
    msgs: List[str] = []
    for field in ("agent", "reflect"):
        block = getattr(spec, field, None)
        if not isinstance(block, dict):
            continue
        for key, value in block.items():
            if _is_placeholder(value):
                msgs.append(f"{field}.{key} is still the placeholder {value!r} that `init` "
                            "wrote; put the real value there (it comes from the user, "
                            "never guess it)")
    return msgs


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
                # Name what *is* set, as `doctor` does: "needs OPENAI_API_KEY" on a
                # machine with ARK_API_KEY sent users to the wrong fix (#451).
                # Only for the adapter's default key: a spec that named its own
                # `api_key_env` chose it, and other keys are not the fix there.
                present = [n for n, v in keys.items() if v and n.endswith("_KEY")
                           and n not in (key_env, "ANTHROPIC_API_KEY")
                           ] if key_env == "OPENAI_API_KEY" else []
                route = (f"; {', '.join(present)} {'is' if len(present) == 1 else 'are'} "
                         f'set -- name one with "api_key_env" plus `default_base_url` '
                         "for its endpoint" if present else "")
                out.append(f"{field}: `openai_compatible` needs {key_env} (and "
                           f"{url_env} for a non-OpenAI endpoint){route}; without it "
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
                    _CLI_ENV_FALLBACK.get(ref, ""), False) and not (
                    ref == "opencode" and _opencode_configured(block)):
                # The trap that ate a 60-round run: `codex` was on PATH, so
                # nothing complained, but a worker runs with the host's config
                # directory redirected -- an interactive login does not carry
                # over. Every rollout failed. Being present is not being usable.
                remedy = _CLI_ENV_USELESS.get(ref) or (
                    ref == "opencode" and _opencode_provider_remedy(block, keys)) or (
                    f"or export {_CLI_ENV_FALLBACK[ref]}"
                    if ref in _CLI_ENV_FALLBACK else
                    "whether a provider key in the environment is enough instead "
                    "is untested for this CLI")
                out.append(
                    f"{field}: `{ref}` is on PATH but a worker runs isolated, so an "
                    f'interactive login does not carry over. Set "isolate": false '
                    f"to use the signed-in CLI -- {remedy}")
    return out


def _merge_policies_replaced(comp) -> Optional[str]:
    """Say so when the spec's merge policies will be ignored by a custom aggregator.

    An ``aggregator_factory`` -- the one ``gepa``, ``skillopt`` and ``dgm`` bring --
    replaces the shipped merge pipeline, so ``conflict``/``fusion`` the spec
    names never run. ``evolve()`` warns about it at run time; ``plan`` is where
    that belongs, before a detached run buries it in ``log.txt`` (#504).
    """
    if comp.kwargs.get("aggregator_factory") is None:
        return None
    installed = comp.kwargs.get("policies")
    dropped = sorted(name for name in ("conflict", "fusion")
                     if installed is not None and getattr(installed, name, None) is not None)
    if not dropped:
        return None
    who = (f"algorithm {comp.algorithm['name']!r}" if comp.algorithm
           else "aggregator_factory")
    return (f"policies {dropped} are replaced by {who}'s own aggregator and will not "
            "run; drop them from `policies` (or the algorithm) to silence evolve's "
            "RuntimeWarning.")


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
    replaced = _merge_policies_replaced(comp)
    if replaced:
        warnings.append(replaced)
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
    hard_errors.extend(_placeholder_messages(spec))
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


def _apply_algorithm_flag(spec: EvolveSpec, a: argparse.Namespace) -> None:
    """`plan/evolve --algorithm ... --param ...` override the spec in memory.

    A same-name override **merges** params (the CLI wins per key), so
    `--param b_min=2` on a spec that already set `b_max: 3` keeps `b_max`; a
    different name replaces the method and its params. `--param` without
    `--algorithm` tweaks the method the spec already names.
    """
    name = getattr(a, "algorithm", None)
    params = parse_cli_params(getattr(a, "param", None))
    current = spec.algorithm
    current_name = (current.get("name") if isinstance(current, dict) else current)
    current_params = (dict(current.get("params") or {})
                      if isinstance(current, dict) else {})
    if not name:
        if not params:
            return
        if not isinstance(current_name, str):
            raise ValueError("--param needs --algorithm (this spec names none)")
        name = current_name
    merged = ({**current_params, **params} if name == current_name else dict(params))
    spec.algorithm = algorithm_override(name, merged, spec.kind)


def cmd_plan(a: argparse.Namespace) -> int:
    spec = load_spec(a.spec)
    try:
        _apply_algorithm_flag(spec, a)
        payload = plan_payload(spec, usd_per_call=a.usd_per_call)
    except (SpecError, ValueError) as e:
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
        _apply_algorithm_flag(spec, a)
        comp = compose(spec)                 # fail here, not in the detached child
    except (SpecError, ValueError) as e:
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


#: How many failing rollouts ``show`` lists, how much of each answer it keeps,
#: and the hard ceiling on the whole ``samples`` block in bytes of JSON, so an
#: MCP ``show`` stays small however long the agent's answers were (#452).
SHOW_SAMPLES = 5
SHOW_SAMPLE_CHARS = 300
SHOW_SAMPLES_MAX_BYTES = 4096


def _head(value: Any, limit: int = SHOW_SAMPLE_CHARS) -> Optional[str]:
    if value is None:
        return None
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[:limit] + "..."


def show_samples(rd: "runstore.RunDir", spec: EvolveSpec, rendered: Optional[str], *,
                 n: int = SHOW_SAMPLES,
                 max_bytes: int = SHOW_SAMPLES_MAX_BYTES) -> List[Dict[str, Any]]:
    """The lowest-reward rollouts of the run's final artifact, next to their gold.

    A run that learned nothing used to leave nothing to say why: the reward
    curve shows *that* the agent missed, never *what* it answered (#452). The
    engine already records each rollout's boundary (task id, output, reward);
    this reads those records back. Rollouts of the final artifact come first
    (its rendering is the record's identity, so the match is exact); if none
    were recorded against it, every recorded rollout is a candidate. One
    sample per task, worst first. Diagnostics: any failure yields ``[]``.
    """
    try:
        from .trajectory import TrajectoryStore
        records = list(TrajectoryStore(rd.trajectories_path).iter_records())
    except Exception:  # noqa: BLE001 - diagnostics must never break `show`
        return []
    if not records:
        return []
    own = [r for r in records if rendered is not None and r.rendered == rendered]
    pool = own or records
    try:
        from .evolvespec import build_tasks
        gold = {t.id: t.meta.get("gold") for t in build_tasks(spec)}
    except Exception:  # noqa: BLE001 - the dataset may have moved since the run
        gold = {}
    pool.sort(key=lambda r: (r.reward, r.task_id, r.purpose, r.ts))
    out: List[Dict[str, Any]] = []
    seen = set()
    for r in pool:
        if r.task_id in seen:
            continue
        seen.add(r.task_id)
        out.append({"task_id": r.task_id, "reward": r.reward,
                    "output": _head(r.output), "gold": _head(gold.get(r.task_id)),
                    "purpose": r.purpose,
                    "of": "final" if own else "any"})
        if len(out) >= n:
            break
    while out and len(json.dumps(out, default=str).encode("utf-8")) > max_bytes:
        out.pop()
    return out


#: How many rejected candidates ``show`` lists, and the byte ceiling on the
#: block -- both files are diagnostic, so an MCP payload must stay small (#452).
SHOW_REJECTED = 5
SHOW_REJECTED_MAX_BYTES = 4096


def show_rejected(rd: "runstore.RunDir", *, n: int = SHOW_REJECTED,
                  max_bytes: int = SHOW_REJECTED_MAX_BYTES) -> List[Dict[str, Any]]:
    """The last few diffs the merger measured and did not adopt (#452).

    ``runstore.execute`` writes one line per rejected candidate: touched files,
    a short diff, the rationale, the gated probability and the outcome. That is
    the other half of "why did this run learn nothing" -- the rollouts say what
    the agent answered, these say what the reflector tried to change. Read from
    the tail so a long run stays cheap; any failure yields ``[]``.
    """
    try:
        with open(rd.candidates_path, encoding="utf-8", errors="replace") as fh:
            lines = deque(fh, maxlen=200)
    except (OSError, AttributeError):
        return []
    out: List[Dict[str, Any]] = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(record, dict):
            out.append(record)
        if len(out) >= n:
            break
    out.reverse()
    while out and len(json.dumps(out, default=str).encode("utf-8")) > max_bytes:
        out.pop(0)
    return out


#: How many rounds a resultless ``show`` lists (the latest ones).
SHOW_HISTORY = 20


def _partial_show(rd: "runstore.RunDir", st: Any) -> Dict[str, Any]:
    """What ``show`` can say about a run that has no result (#466)."""
    out: Dict[str, Any] = {}
    try:
        rounds = rd.rounds()
    except Exception:  # noqa: BLE001 - diagnostics must never break `show`
        rounds = []
    if rounds:
        out["rounds"] = len(rounds)
        keep = ("round", "held_out_reward", "committed", "rejected", "reasons",
                "propose_errors", "propose_error")
        out["history"] = [{k: r[k] for k in keep if r.get(k) not in (None, {})}
                          for r in rounds[-SHOW_HISTORY:]]
    if getattr(st, "best_reward", None) is not None:
        out["best_reward"] = st.best_reward
    perr = getattr(st, "propose_errors", 0) or sum(
        int(r.get("propose_errors") or 0) for r in rounds)
    if perr:
        out["propose_errors"] = perr
        out["propose_error"] = getattr(st, "propose_error", None) or next(
            (r.get("propose_error") for r in rounds if r.get("propose_error")), None)
    try:
        spec = EvolveSpec.from_dict(rd.spec_dict())
        samples = show_samples(rd, spec, None)
    except Exception:  # noqa: BLE001
        samples = []
    if samples:
        out["samples"] = samples
    rejected = show_rejected(rd)
    if rejected:
        out["rejected"] = rejected
    return out


def show_payload(run_id: str, *, store: Optional[str] = None, diff: bool = True,
                 max_chars: int = 40_000) -> Dict[str, Any]:
    rd = runstore.get(run_id, store=store)
    st = rd.status()
    result = rd.result()
    payload: Dict[str, Any] = {"run_id": run_id, "state": st.state,
                               "stop_reason": st.stop_reason, "error": st.error}
    if result is None:
        # A failed, cancelled or still-running run has no artifact and no diff,
        # but what it did is on disk: the rounds that finished, the reflector
        # error, the recorded rollouts. Those are the runs most in need of a
        # diagnosis, so say what is known instead of only "no result yet"
        # (#466). A run with nothing recorded is unchanged.
        payload.update(_partial_show(rd, st))
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
    # Why the reward is what it is (#452): the worst rollouts of the final
    # artifact, output vs gold. Only when some were recorded, so a run with
    # `record_rollouts: false` (or one from before #452) is unchanged.
    samples = show_samples(rd, spec, getattr(result, "rendered", None))
    if samples:
        payload["samples"] = samples
    rejected = show_rejected(rd)
    if rejected:
        payload["rejected"] = rejected
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
              "best_reward", "propose_errors", "propose_error", "note"):
        if payload.get(k) is not None:
            print(f"{k}: {payload[k]}")
    if payload.get("history"):
        print("rounds so far:")
        for h in payload["history"]:
            line = f"  round {h.get('round')}"
            if h.get("held_out_reward") is not None:
                line += f"  held_out_reward={h['held_out_reward']:.3f}"
            for k in ("committed", "rejected", "propose_errors"):
                if h.get(k):
                    line += f"  {k}={h[k]}"
            if h.get("propose_error"):
                line += f"  propose_error={h['propose_error']!r}"
            print(line)
    if "rendered" in payload:
        print("\n" + payload["rendered"])
    if payload.get("diff"):
        print("\n" + payload["diff"])
    elif "files" in payload:
        print("files:", ", ".join(payload["files"]))
    if payload.get("apply_plan"):
        print("apply would:", json.dumps(payload["apply_plan"]))
    if payload.get("samples"):
        print("\nlowest-reward rollouts (output vs gold):")
        for smp in payload["samples"]:
            print(f"  {smp['task_id']}  reward={smp['reward']:.3f}  "
                  f"output={smp['output']!r}  gold={smp['gold']!r}")
    if payload.get("rejected"):
        print("\nrejected candidates (what the reflector tried):")
        for rec in payload["rejected"]:
            head = f"  round {rec.get('round', '?')}  {rec.get('outcome', 'rejected')}"
            if rec.get("gated_reward") is not None:
                head += f"  p_improve={rec['gated_reward']:.3f}"
            print(head)
            if rec.get("rationale"):
                print(f"    rationale: {rec['rationale']}")
            if rec.get("files"):
                print(f"    files: {', '.join(rec['files'])}")
            if rec.get("diff"):
                for line in str(rec["diff"]).splitlines()[:8]:
                    print(f"      {line}")
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


# ---------------------------------------------------------------------------
# The meta stack on the command line: portfolio, exam, audit
# ---------------------------------------------------------------------------


def _archive_root(archive: Optional[str]) -> str:
    """The consolidation archive a meta command reads: ``--archive``, else
    ``$AGENTDESCENT_HOME/archive``, else ``~/.agentdescent/archive``."""
    if archive:
        return os.path.abspath(os.path.expanduser(archive))
    home = os.environ.get("AGENTDESCENT_HOME") or "~/.agentdescent"
    return os.path.join(os.path.abspath(os.path.expanduser(home)), "archive")


def _missing_archive(root: str) -> bool:
    """Report a missing archive the way ``safety`` does and return True.

    The read-only archive commands must not create ``root`` as a side effect
    of opening it: ``ConsolidationStore`` makes its directory, so a typo'd path
    used to come back as an empty-but-valid archive (#502). The message names
    the only writer (#512)."""
    if os.path.isdir(root):
        return False
    from .consolidation import missing_archive_message
    print(f"error: {missing_archive_message(root)}", file=sys.stderr)
    return True


def _load_ref(ref: str) -> Any:
    """Resolve ``module:attribute`` to the object it names."""
    if ":" not in ref:
        raise SpecError(f"expected 'module:attribute', got {ref!r}")
    module_name, attr = ref.rsplit(":", 1)
    try:
        module = __import__(module_name, fromlist=[attr])
        return getattr(module, attr)
    except (ImportError, AttributeError) as error:
        raise SpecError(f"cannot resolve {ref!r}: {error}") from error


class MetaCommandError(Exception):
    """A read-only meta command (portfolio / safety / audit trace) cannot answer.

    ``code`` is the CLI exit status: 2 for a missing archive or store, an
    unknown run or a bad argument; 1 when the archive exists but holds nothing
    to report on. The MCP tools return the message as ``{"error": ...}``.
    """

    def __init__(self, message: str, code: int = 2) -> None:
        super().__init__(message)
        self.code = code


def _require_archive(root: str) -> None:
    """Refuse a missing archive *before* anything opens it (#502 / #512).

    ``ConsolidationStore`` makes its directory by default, so a typo'd path
    would come back as an empty-but-valid archive. The message names the only
    writer so a plugin host does not invent a nonexistent next step."""
    if not os.path.isdir(root):
        from .consolidation import missing_archive_message
        raise MetaCommandError(missing_archive_message(root))


def _require_file(flag: str, path: str) -> None:
    if not os.path.isfile(os.path.expanduser(path)):
        raise MetaCommandError(f"{flag} store {path!r} does not exist")


def _meta_error(error: MetaCommandError) -> int:
    print(f"error: {error}" if error.code == 2 else str(error), file=sys.stderr)
    return error.code


def portfolio_payload(archive: Optional[str] = None, *, floor: float = 0.05,
                      min_samples: int = 3, grow: float = 0.5,
                      shrink: float = 0.5,
                      cap: Optional[float] = None) -> Dict[str, Any]:
    """What ``agentdescent portfolio --json`` prints, for the CLI and the MCP
    ``portfolio`` tool alike. Reports a reallocation; writes nothing."""
    from .consolidation import ConsolidationStore
    from .portfolio import layers_from_archive, portfolio_report

    root = _archive_root(archive)
    _require_archive(root)
    layers = layers_from_archive(ConsolidationStore(root, create=False))
    if not layers:
        raise MetaCommandError(
            f"no rule records under {root} to build layers from; "
            f"run continuous_meta_run (or a meta session) first", code=1)
    report = portfolio_report(layers, floor=floor, min_samples=min_samples,
                              grow=grow, shrink=shrink, cap=cap)
    return {
        "archive": root,
        "total_budget": report.total_budget,
        "layers": [{"name": l.name, "budget": l.budget, "spent": l.spent,
                    "return_": l.return_, "samples": l.samples,
                    "return_sd": l.return_sd,
                    "diminishing": l.diminishing} for l in report.layers],
        "reallocated": report.reallocated,
        "reasons": report.reasons,
    }


def cmd_portfolio(a: argparse.Namespace) -> int:
    """``agentdescent portfolio [archive]`` -- rank the archive's layers and
    reallocate the next budget window."""
    try:
        payload = portfolio_payload(
            getattr(a, "archive", None), floor=a.floor,
            min_samples=a.min_samples, grow=a.grow, shrink=a.shrink, cap=a.cap)
    except MetaCommandError as error:
        return _meta_error(error)
    if a.json:
        _out(payload, as_json=True)
        return 0
    for layer in payload["layers"]:
        spent = layer["spent"]
        rpc = (layer["return_"] / spent) if spent > 0 else 0.0
        new = payload["reallocated"][layer["name"]]
        print(f"{layer['name']}: return-per-cost {rpc:.3f} "
              f"(return {layer['return_']:.3f} / spend {spent:.1f}, "
              f"{layer['samples']} measurements) -> budget {new:.1f}")
        print(f"    {payload['reasons'][layer['name']]}")
    print(f"archive: {payload['archive']}  "
          f"(total budget {payload['total_budget']:.1f})")
    return 0


def cmd_exam(a: argparse.Namespace) -> int:
    """``agentdescent exam [archive] --domain module:builder`` -- re-judge the
    archive's adopted rules with an independent oracle."""
    from .consolidation import ConsolidationStore
    from .meta_exam import exam_archive

    root = _archive_root(getattr(a, "archive", None))
    if _missing_archive(root):
        return 2
    domain = _load_ref(a.domain)
    built = domain() if callable(domain) and not isinstance(domain, dict) else domain
    if not isinstance(built, Mapping):
        raise SpecError("--domain must resolve to a builder that returns a dict "
                        "with keys: oracle, tasks_for, run, reward, propose, "
                        "strategy (and optionally slot, spec)")
    store = ConsolidationStore(root, create=False)
    config = None
    if getattr(a, "config", None) is not None:
        try:
            config = json.loads(a.config)
        except json.JSONDecodeError as error:
            raise SpecError(f"--config must be a JSON dict, got {a.config!r}: {error}")
        if not isinstance(config, dict):
            raise SpecError(f"--config must be a JSON dict, got "
                            f"{type(config).__name__}")
    report = exam_archive(
        store, slot=a.slot,
        spec=built.get("spec") or _slot_spec(a.slot),
        oracle=built["oracle"], tasks_for=built["tasks_for"],
        run=built["run"], reward=built["reward"], propose=built["propose"],
        strategy=built["strategy"], config=config,
        min_gain=a.min_gain, max_flags=a.max_flags)
    if a.json:
        _out({"archive": root, "slot": a.slot, "passed": report.passed,
              "flagged": report.flagged, "demoted": report.demoted,
              "examined": [{**({"run_id": e.run_id, "oracle_gain": e.oracle_gain,
                                "verdict": e.verdict, "reason": e.reason})}
                           for e in report.examined]}, as_json=True)
        return 0
    print(f"{report.reason}  (archive: {root})")
    for e in report.examined:
        print(f"  {e.run_id}: {e.verdict}  (oracle gain {e.oracle_gain:+.3f}) "
              f"{e.reason}")
    return 0


def _slot_spec(slot: str) -> Any:
    from .meta import policy_source, priority_selection
    from .meta import SLOTS
    if slot not in SLOTS:
        raise SpecError(f"--slot must be an evolvable slot, one of {SLOTS}; "
                        f"got {slot!r}")
    if slot == "selection":
        return priority_selection()
    return policy_source(slot)


def audit_trace_payload(run_id: str, *, archive: Optional[str] = None,
                        meta_audit: Optional[str] = None,
                        inner_audit: Optional[str] = None,
                        harness: Any = None,
                        bias_threshold: float = 0.1) -> Dict[str, Any]:
    """What ``agentdescent audit <run_id> --json`` prints, for the CLI and the
    MCP ``audit_trace`` tool alike. ``harness`` is an already-resolved
    callable: resolving a ``module:attr`` runs code, so only the CLI does it."""
    from .chain_audit import chain_audit
    from .consolidation import ConsolidationStore
    from .meta_audit import MetaAuditStore

    root = _archive_root(archive)
    _require_archive(root)
    if meta_audit:
        _require_file("--meta-audit", meta_audit)
    if inner_audit:
        _require_file("--inner-audit", inner_audit)
    store = ConsolidationStore(root, create=False)
    meta = MetaAuditStore(meta_audit) if meta_audit else None
    inner = None
    if inner_audit:
        from .audit import AuditStore
        inner = AuditStore(inner_audit)
    try:
        report = chain_audit(store, run_id, meta_audit=meta, inner_audit=inner,
                             harness_evidence=harness,
                             bias_threshold=bias_threshold)
    except KeyError as error:
        # str(KeyError) wraps the message in quotes; report the message
        # itself, as an unknown-run error like every other (#481).
        raise MetaCommandError(
            str(error.args[0] if error.args else error)) from None
    return {"run_id": run_id, "verified": report.verified, "reason": report.reason,
            "links": [{"layer": l.layer, "status": l.status,
                       "evidence": l.evidence, "reason": l.reason}
                      for l in report.links]}


def cmd_audit(a: argparse.Namespace) -> int:
    """``agentdescent audit <run_id> --archive PATH`` -- trace one adoption
    through its evidence rings."""
    try:
        root = _archive_root(a.archive)
        _require_archive(root)
        harness = None
        if a.harness:
            fn = _load_ref(a.harness)
            if not callable(fn):
                raise SpecError(f"--harness {a.harness!r} does not resolve to a "
                                f"callable ({type(fn).__name__})")
            harness = fn
        payload = audit_trace_payload(
            a.run_id, archive=a.archive, meta_audit=a.meta_audit,
            inner_audit=a.inner_audit, harness=harness,
            bias_threshold=a.bias_threshold)
    except MetaCommandError as error:
        return _meta_error(error)
    if a.json:
        _out(payload, as_json=True)
        return 0
    print(f"run {payload['run_id']}: "
          f"{'VERIFIED' if payload['verified'] else 'NOT VERIFIED'}")
    print(f"  {payload['reason']}")
    for link in payload["links"]:
        print(f"  {link['layer']}: {link['status']}  {link['reason']}")
    return 0


def cmd_algorithms(a: argparse.Namespace) -> int:
    from . import algorithms

    for name, error in sorted(algorithms.external_errors().items()):
        print(f"warning: algorithm plugin {name!r} failed to load: {error}",
              file=sys.stderr)
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


def safety_payload(archive: Optional[str] = None, *,
                   audit: Optional[str] = None, bias_threshold: float = 0.1,
                   min_exam: Optional[str] = "passed") -> Dict[str, Any]:
    """What ``agentdescent safety --json`` prints, for the CLI and the MCP
    ``safety`` tool alike. ``min_exam`` is ``passed``, ``flagged``, or
    ``none`` / None for every non-demoted rule. Reads only."""
    from .meta_audit import MetaAuditStore
    from .safety import safety_report

    root = _archive_root(archive)
    store = None
    if audit:
        if not os.path.isfile(os.path.expanduser(audit)):
            raise MetaCommandError(f"audit store {audit!r} does not exist")
        store = MetaAuditStore(audit)
    try:
        report = safety_report(
            root, audit=store, bias_threshold=bias_threshold,
            min_exam=None if min_exam in (None, "none") else min_exam)
    except (FileNotFoundError, ValueError) as error:
        raise MetaCommandError(str(error)) from None
    return {"archive": root, **report.to_dict()}


def cmd_safety(a: argparse.Namespace) -> int:
    """``agentdescent safety`` -- is this archive safe to keep seeding from?"""
    try:
        payload = safety_payload(
            getattr(a, "archive", None), audit=getattr(a, "audit", None),
            bias_threshold=getattr(a, "bias_threshold", 0.1),
            min_exam=getattr(a, "min_exam", "passed"))
    except MetaCommandError as error:
        return _meta_error(error)
    if a.json:
        _out(payload, as_json=True)
    else:
        print(f"verdict: {payload['verdict']}  ({payload['reason']})")
        print(f"archive: {payload['archive']}  rules: {payload['rules']}  "
              f"unexamined: {payload['unexamined']}  "
              f"flagged: {payload['flagged']}  demoted: {payload['demoted']}")
        for check in payload["checks"]:
            print(f"  [{check['status']:<4}] {check['name']:<14} {check['detail']}")
    return {"safe": 0, "caution": 1, "unsafe": 3}.get(payload["verdict"], 1)


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
    s.add_argument("--model", default=None,
                   help="model name for the openai_compatible reflector/agent; with "
                        "--base-url and an `opencode` worker it is also pinned into "
                        "the worker (-m plus a provider-only OPENCODE_CONFIG) "
                        "(default: a placeholder plan refuses -- never guessed)")
    s.add_argument("--base-url", dest="base_url", default=None,
                   help="endpoint URL for that model (written as default_base_url)")
    s.add_argument("--key-env", dest="key_env", default=None,
                   help="environment variable holding its key (default: OPENAI_API_KEY, "
                        "or the one other provider key that is set)")
    s.add_argument("--algorithm", default=None,
                   help="named method from `agentdescent algorithms list` "
                        "(e.g. rrsi); other spec fields keep their kind "
                        "defaults and can override the method's")
    s.add_argument("--param", action="append", default=[], metavar="KEY=VALUE",
                   help="algorithm parameter, repeatable; values are JSON "
                        "when they parse, otherwise strings")
    s.add_argument("--out", help="where to write the spec")
    s.set_defaults(fn=cmd_init)

    s = sub.add_parser("plan", help="validate a spec and estimate its cost; runs nothing")
    s.add_argument("spec")
    s.add_argument("--algorithm", default=None,
                   help="override the spec's algorithm by name (the same name "
                        "merges --param with the spec's params)")
    s.add_argument("--param", action="append", default=[], metavar="KEY=VALUE",
                   help="algorithm parameter for --algorithm, repeatable")
    s.add_argument("--usd-per-call", type=float, dest="usd_per_call")
    s.set_defaults(fn=cmd_plan)

    s = sub.add_parser("evolve", help="run a spec")
    s.add_argument("spec")
    s.add_argument("--algorithm", default=None,
                   help="override the spec's algorithm by name (the same name "
                        "merges --param with the spec's params)")
    s.add_argument("--param", action="append", default=[], metavar="KEY=VALUE",
                   help="algorithm parameter for --algorithm, repeatable")
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

    s = sub.add_parser("portfolio",
                       help="rank the archive's layers and reallocate the next budget window")
    s.add_argument("archive", nargs="?", default=None,
                   help="consolidation archive directory (default $AGENTDESCENT_HOME/archive)")
    s.add_argument("--floor", type=float, default=0.05)
    s.add_argument("--min-samples", type=int, default=3, dest="min_samples")
    s.add_argument("--grow", type=float, default=0.5)
    s.add_argument("--shrink", type=float, default=0.5)
    s.add_argument("--cap", type=float, default=None)
    s.set_defaults(fn=cmd_portfolio)

    s = sub.add_parser("exam",
                       help="re-judge the archive's adopted rules with an independent oracle")
    s.add_argument("archive", nargs="?", default=None,
                   help="consolidation archive directory (default $AGENTDESCENT_HOME/archive)")
    s.add_argument("--domain", required=True,
                   help="module:attribute builder returning the domain dict "
                        "(oracle, tasks_for, run, reward, propose, strategy)")
    s.add_argument("--slot", default="task_sampler",
                   help="the decision slot the rules evolved (default task_sampler)")
    s.add_argument("--min-gain", type=float, default=0.0, dest="min_gain")
    s.add_argument("--max-flags", type=int, default=2, dest="max_flags")
    s.add_argument("--config", default=None,
                   help="JSON config dict used to rank the archive's rules")
    s.set_defaults(fn=cmd_exam)

    s = sub.add_parser("audit",
                       help="trace one archived adoption through its evidence rings")
    s.add_argument("run_id")
    s.add_argument("--archive", default=None,
                   help="consolidation archive directory (default $AGENTDESCENT_HOME/archive)")
    s.add_argument("--meta-audit", default=None, dest="meta_audit",
                   help="a MetaAuditStore path for the meta-gain ring")
    s.add_argument("--inner-audit", default=None, dest="inner_audit",
                   help="an inner AuditStore path for the inner-reward ring")
    s.add_argument("--harness", default=None,
                   help="module:attribute harness_evidence callable for the harness ring")
    s.add_argument("--bias-threshold", type=float, default=0.1, dest="bias_threshold")
    s.set_defaults(fn=cmd_audit)

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

    s = sub.add_parser("safety", help="is this archive safe to keep seeding from?")
    s.add_argument("--archive", help="consolidation archive (default: $AGENTDESCENT_HOME/archive)")
    s.add_argument("--audit", help="MetaAuditStore JSONL to check verifier bias")
    s.add_argument("--bias-threshold", type=float, default=0.1,
                   dest="bias_threshold",
                   help="worst |gain bias| allowed by the audit check")
    s.add_argument("--min-exam", choices=("passed", "flagged", "none"),
                   default="passed", dest="min_exam",
                   help="seeding bar to check (default: passed)")
    s.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                   help="machine-readable output")
    s.set_defaults(fn=cmd_safety)

    s = sub.add_parser("serve", help="serve a read-only run panel on loopback, for a host UI")
    s.add_argument("--host", default=runstore.DEFAULT_HTTP_HOST)
    s.add_argument("--port", type=int, default=runstore.DEFAULT_HTTP_PORT)
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("mcp", help="serve the tools over stdio (needs agentdescent[mcp])")
    s.set_defaults(fn=cmd_mcp)

    sub.metavar = "{" + ",".join(n for n in sub.choices if n not in _INTERNAL_COMMANDS) + "}"
    _accept_json_after_verb(sub)
    return p


def _accept_json_after_verb(sub: "argparse._SubParsersAction") -> None:
    """Let ``--json`` follow the verb as well as precede it (#513).

    The top-level parser owns ``--json``; each leaf subcommand also gets one
    whose default is SUPPRESS, so a flag written after the verb sets the same
    ``json`` attribute and a missing one leaves the top-level value alone.
    Subcommands that already declare ``--json`` keep their own definition.
    """
    for parser in dict.fromkeys(sub.choices.values()):
        nested = [a for a in parser._actions
                  if isinstance(a, argparse._SubParsersAction)]
        if nested:
            for n in nested:
                _accept_json_after_verb(n)
            continue
        if "--json" in parser._option_string_actions:
            continue
        parser.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                            help="machine-readable output")


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
