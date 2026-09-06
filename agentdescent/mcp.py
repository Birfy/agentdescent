"""The MCP server: the CLI's verbs as tools a host agent can call.

``agentdescent mcp`` serves over stdio. DeepSeek Harness, Claude Code, Codex and
the other hosts all speak MCP, so this one server is the whole runtime
integration; the per-host material in ``integrations/`` is only manifests and a
skill file telling the host *when* to call these.

Three things about the shape, each a consequence of how hosts behave:

**Every tool returns quickly.** A run takes minutes to hours and a tool call has
a timeout measured in seconds, so ``start`` launches a detached run
(:mod:`agentdescent.runstore`) and returns a ``run_id``; ``status`` and ``show``
read the run directory. Nothing here blocks on ``evolve()``.

**``plan`` is separate from ``start``.** A skill can *ask* the model to show the
user the spec and the price before running; two tools *make* it. ``plan``
resolves every ref and builds the policy bundle, so a bad spec fails here, not
in round one of a detached process.

**``apply`` is its own tool, and says so.** Writing the evolved artifact over
the user's real directory is the one irreversible step, so it is never a side
effect of anything else and its description tells the calling model to confirm.

The tool bodies are plain functions (:class:`Tools`) so they can be tested and
reused without the SDK; :func:`build_server` wraps them. The SDK is imported
lazily: ``pip install "agentdescent[mcp]"``; the core stays dependency-free.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from . import runstore
from .cli import NESTED_ENV, apply_payload, doctor_report, plan_payload, show_payload, status_payload
from .evolvespec import EvolveSpec, SpecError, compose

__all__ = ["Tools", "TOOL_DESCRIPTIONS", "build_server", "serve"]

#: Written for the calling model: what the tool is for and what to do around it.
TOOL_DESCRIPTIONS: Dict[str, str] = {
    "doctor": (
        "Check what this machine can run: which worker agent CLIs (claude, codex, dsh) "
        "are on PATH, which provider keys are set, whether a container engine and the "
        "git ledger are available. Call this FIRST, before planning a run, and tell the "
        "user what is missing instead of starting a run that fails on round one."),
    "plan": (
        "Validate an EvolveSpec and estimate its cost WITHOUT running anything. Resolves "
        "every agent/scorer/policy reference and loads the data, so a wrong field fails "
        "here with its name. Returns the composed evolve() arguments, the task count, an "
        "upper bound on agent calls per round and in total, and notes. Show the user the "
        "spec and the estimate and get a yes before calling start."),
    "start": (
        "Start an evolution run in the background and return its run_id at once. The run "
        "is a detached process; poll `status` about once per round (not more often) and "
        "read `show` when it is done. Never call start without having shown the user the "
        "plan. A run costs real agent calls: rounds x n_workers x tasks."),
    "status": (
        "Progress of one run (round, best held-out reward, calls, dollars if priced, "
        "state, the last few rounds) or, with no run_id, a list of all runs. Cheap; safe "
        "to poll. Summarise round-to-round deltas for the user rather than pasting JSON."),
    "show": (
        "The evolved artifact when a run is done: for a directory, the list of files, a "
        "unified diff against the original, and the plan `apply` would execute (files "
        "written, extra files, backups); for text, the evolved instruction. Also the "
        "outcomes() histogram that says why proposals were committed or refused. Explain "
        "what changed and why; do not paste the whole tree."),
    "apply": (
        "DESTRUCTIVE: write the evolved artifact back over the target directory (or `to`). "
        "Backs the original up first unless backup=false. Ask the user explicitly before "
        "calling this, after they have seen `show`; use dry_run=true to preview. Tell them "
        "the backup path afterwards."),
    "cancel": (
        "Stop a running evolution and every worker agent it started. The run keeps its "
        "ledger and can be resumed."),
    "resume": (
        "Continue a stopped, failed or cancelled run on its existing ledger (the engine "
        "picks up where it left off). Returns the new status."),
}


class Tools:
    """The tool bodies, SDK-free. One instance per server, bound to a run store."""

    def __init__(self, store: Optional[str] = None) -> None:
        self.store = store
        #: Set by :func:`build_server` on the first tool call, once there is a
        #: live session to borrow. Stays None on a host without sampling, and
        #: `host_model` then fails with a message that says which case it is.
        self.bridge: Any = None
        #: Why there is no bridge, for the `start` payload. Silence here reads
        #: as "the run learned nothing" instead of "it could not ask".
        self.bridge_error: Optional[str] = None
        #: The host's own CLI (`claude_code`, `codex`, `dsh`, `opencode`), from
        #: the name it sent at `initialize`. The fallback route for `host_model`.
        self.host_cli: Optional[str] = None

    def host_model_env(self) -> Dict[str, str]:
        """What a launched run needs to reach the host's model, by either route.

        The bridge when the host can sample; otherwise the name of the host's
        own CLI, which `host_model` runs with the user's real configuration. No
        host measured so far implements sampling, so the second is the one that
        actually carries this feature.
        """
        if self.bridge is not None:
            return self.bridge.env()
        if self.host_cli:
            from .host_sampling import HOST_CLI_ENV

            return {HOST_CLI_ENV: self.host_cli}
        return {}

    # -- read-only -------------------------------------------------------------

    def doctor(self) -> Dict[str, Any]:
        return doctor_report()

    def plan(self, spec: Dict[str, Any], usd_per_call: Optional[float] = None) -> Dict[str, Any]:
        try:
            # Relative paths mean "where this server was launched"; the detached
            # run has a different cwd, so they are resolved here, once.
            return plan_payload(EvolveSpec.from_dict(spec).absolutise(),
                                usd_per_call=usd_per_call)
        except SpecError as e:
            return {"ok": False, "error": str(e)}

    def status(self, run_id: Optional[str] = None) -> Any:
        try:
            return status_payload(run_id, store=self.store)
        except runstore.RunStoreError as e:
            return {"error": str(e)}

    def show(self, run_id: str, diff: bool = True) -> Dict[str, Any]:
        try:
            return show_payload(run_id, store=self.store, diff=diff)
        except runstore.RunStoreError as e:
            return {"error": str(e)}

    # -- side effects -----------------------------------------------------------

    def start(self, spec: Dict[str, Any], budget_usd: Optional[float] = None,
              usd_per_call: Optional[float] = None) -> Dict[str, Any]:
        try:
            es = EvolveSpec.from_dict(spec).absolutise()
            compose(es)                       # fail here, with the field, not in the child
        except SpecError as e:
            return {"ok": False, "error": str(e)}
        if os.environ.get(NESTED_ENV):
            # This server is running inside a worker of another run (the plugin
            # that hosts it is being evolved). The transcript should still show
            # that the host *called* start -- that is what the grader looks for --
            # but nothing may actually run, or every rollout would start a run.
            return {"ok": True, "run_id": "nested-stub", "state": "refused", "nested": True,
                    "note": f"{NESTED_ENV} is set: this session is itself a worker of an "
                            "evolution run, so no nested run was started."}
        rd = runstore.create(es.to_dict(), store=self.store)
        try:
            # The run is a detached process, so a `host_model` reference in the
            # spec resolves there, not here -- it needs the bridge's address in
            # its environment or it has no session to borrow.
            st = runstore.launch(rd, budget_usd=budget_usd, usd_per_call=usd_per_call,
                                 env=self.host_model_env())
        except runstore.RunStoreError as e:
            return {"ok": False, "error": str(e), "run_id": rd.run_id, "dir": rd.path}
        return {"ok": True, "run_id": rd.run_id, "state": st.state, "pid": st.pid,
                "dir": rd.path, "notes": rd.status().notes,
                # Whether a `host_model` reference in this spec can work. A run
                # that silently could not borrow the session's model would
                # otherwise just look like one that proposed nothing.
                "host_model_available": bool(self.bridge or self.host_cli),
                "host_model_route": ("sampling" if self.bridge
                                     else (self.host_cli or None)),
                **({} if (self.bridge or self.host_cli)
                   else {"host_model_unavailable": self.bridge_error})}

    def apply(self, run_id: str, to: Optional[str] = None, dry_run: bool = False,
              backup: bool = True) -> Dict[str, Any]:
        try:
            return apply_payload(run_id, to=to, store=self.store, dry_run=dry_run, backup=backup)
        except (runstore.RunStoreError, OSError, ValueError) as e:
            return {"error": str(e)}

    def cancel(self, run_id: str) -> Dict[str, Any]:
        try:
            return runstore.cancel(run_id, store=self.store).to_dict()
        except runstore.RunStoreError as e:
            return {"error": str(e)}

    def resume(self, run_id: str, budget_usd: Optional[float] = None,
               usd_per_call: Optional[float] = None) -> Dict[str, Any]:
        try:
            # A resumed run gets the *current* bridge: the session that started
            # it is likely gone, and its address with it.
            return runstore.resume(run_id, store=self.store, budget_usd=budget_usd,
                                   usd_per_call=usd_per_call,
                                   env=self.host_model_env()).to_dict()
        except runstore.RunStoreError as e:
            return {"error": str(e)}

    # -- resources --------------------------------------------------------------

    def runs_resource(self) -> str:
        return json.dumps([s.to_dict() for s in runstore.list_runs(store=self.store)],
                          indent=2, default=str)

    def rounds_resource(self, run_id: str) -> str:
        try:
            return json.dumps(runstore.get(run_id, store=self.store).rounds(), indent=2,
                              default=str)
        except runstore.RunStoreError as e:
            return json.dumps({"error": str(e)})


def _context_class():
    """The SDK's ``Context``, whose *annotation* is how a tool asks for a session.

    Injection is by type, not by name: a parameter annotated ``Any`` stays in
    the tool's input schema, and the calling model then sees -- and tries to
    fill -- an argument that is meant to be invisible. Measured: with
    ``ctx: Any`` the `start` schema advertised ``ctx`` alongside ``spec``.
    """
    try:
        from mcp.server.mcpserver import Context  # mcp >= 2
        return Context
    except ImportError:
        from mcp.server.fastmcp import Context  # mcp 1.x
        return Context


def _server_class():
    """mcp 2.x (``MCPServer``) or 1.x (``FastMCP``), whichever is installed."""
    try:
        from mcp.server.mcpserver import MCPServer  # mcp >= 2
        return MCPServer
    except ImportError:
        pass
    try:
        from mcp.server.fastmcp import FastMCP  # mcp 1.x
        return FastMCP
    except ImportError:
        raise ImportError(
            "the MCP server needs the 'mcp' package: pip install \"agentdescent[mcp]\". "
            "The CLI (agentdescent plan / evolve / status ...) works without it.") from None


def build_server(store: Optional[str] = None, *, name: str = "agentdescent"):
    """An MCP server with every tool in :data:`TOOL_DESCRIPTIONS` and two resources."""
    server_cls = _server_class()
    server = server_cls(name, instructions=(
        "AgentDescent evolves skills, agent definitions, prompts, code and host plugins "
        "against examples with a parallel, merge-based optimiser. Workflow: doctor -> "
        "write an EvolveSpec -> plan (show the user) -> start -> status (once a round) -> "
        "show -> ask -> apply."))
    t = Tools(store)

    def attach_bridge(ctx: Any, loop: Any) -> None:
        """Stand up the sampling bridge the first time a launching tool runs.

        Not at construction: there is no session until a client connects, and
        the capability that decides whether a bridge is worth having is one the
        client declares at initialise. A tool call is the first moment both
        exist.

        ``loop`` is passed in rather than looked up, because the lookup has to
        happen where a loop is actually running. A **sync** tool body does not
        qualify -- the SDK runs those on a worker thread, where
        ``asyncio.get_running_loop()`` raises "no running event loop" and the
        bridge silently never attached.
        """
        if (t.bridge is not None or t.host_cli is not None) or ctx is None:
            return
        try:
            import shutil

            from . import agents
            from .host_sampling import bridge_for_session, host_cli_for_client

            t.bridge = bridge_for_session(ctx.session, loop)
            if t.bridge is not None:
                return
            t.bridge_error = "this host did not declare the sampling capability"
            # No sampling anywhere yet, so the CLI route is what makes
            # `host_model` usable at all. Which CLI is decided by the name the
            # client sent at `initialize`, and only accepted if it is on PATH --
            # naming a CLI that is not there would turn every reflection into a
            # FileNotFoundError deep inside a detached run.
            params = getattr(ctx.session, "client_params", None)
            # `client_info` on mcp 2.x, `clientInfo` on 1.x -- the wire name is
            # camelCase and only 2.x renamed the Python attribute. Reading one
            # of them silently returned None, and the fallback never fired.
            info = (getattr(params, "client_info", None)
                    or getattr(params, "clientInfo", None))
            factory = host_cli_for_client(getattr(info, "name", None))
            if factory and shutil.which(getattr(agents, factory)().command[0]):
                t.host_cli = factory
                t.bridge_error += f"; falling back to its CLI ({factory})"
        except Exception as e:  # noqa: BLE001 - sampling is a bonus, never a failure
            # Kept, not discarded: a run whose reflector cannot reach the host
            # proposes nothing and looks like a run that simply learned nothing.
            t.bridge, t.bridge_error = None, f"{type(e).__name__}: {e}"

    @server.tool(description=TOOL_DESCRIPTIONS["doctor"])
    def doctor() -> Dict[str, Any]:
        return t.doctor()

    @server.tool(description=TOOL_DESCRIPTIONS["plan"])
    def plan(spec: Dict[str, Any], usd_per_call: Optional[float] = None) -> Dict[str, Any]:
        return t.plan(spec, usd_per_call)

    # `start` and `resume` are the two tools that launch a process, so they are
    # the two that take a Context. The annotation is set after the definition
    # and the decorator applied by hand, because the SDK is imported lazily and
    # `Context` cannot be a module-level name here.
    context_cls = _context_class()

    async def start(spec: Dict[str, Any], ctx, budget_usd: Optional[float] = None,
                    usd_per_call: Optional[float] = None) -> Dict[str, Any]:
        import asyncio

        import anyio.to_thread

        attach_bridge(ctx, asyncio.get_running_loop())
        # The body forks a process, so it stays off the loop the bridge needs.
        return await anyio.to_thread.run_sync(
            lambda: t.start(spec, budget_usd, usd_per_call))

    start.__annotations__["ctx"] = context_cls
    server.tool(description=TOOL_DESCRIPTIONS["start"])(start)

    @server.tool(description=TOOL_DESCRIPTIONS["status"])
    def status(run_id: Optional[str] = None) -> Any:
        return t.status(run_id)

    @server.tool(description=TOOL_DESCRIPTIONS["show"])
    def show(run_id: str, diff: bool = True) -> Dict[str, Any]:
        return t.show(run_id, diff)

    @server.tool(description=TOOL_DESCRIPTIONS["apply"])
    def apply(run_id: str, to: Optional[str] = None, dry_run: bool = False,
              backup: bool = True) -> Dict[str, Any]:
        return t.apply(run_id, to, dry_run, backup)

    @server.tool(description=TOOL_DESCRIPTIONS["cancel"])
    def cancel(run_id: str) -> Dict[str, Any]:
        return t.cancel(run_id)

    async def resume(run_id: str, ctx, budget_usd: Optional[float] = None,
                     usd_per_call: Optional[float] = None) -> Dict[str, Any]:
        import asyncio

        import anyio.to_thread

        attach_bridge(ctx, asyncio.get_running_loop())
        return await anyio.to_thread.run_sync(
            lambda: t.resume(run_id, budget_usd, usd_per_call))

    resume.__annotations__["ctx"] = context_cls
    server.tool(description=TOOL_DESCRIPTIONS["resume"])(resume)

    @server.resource("agentdescent://runs", mime_type="application/json",
                     description="Every run in the store, newest first, with its status.")
    def runs() -> str:
        return t.runs_resource()

    @server.resource("agentdescent://runs/{run_id}/rounds", mime_type="application/json",
                     description="One run's rounds as they completed: reward, commits, reasons.")
    def rounds(run_id: str) -> str:
        return t.rounds_resource(run_id)

    return server


def serve(store: Optional[str] = None) -> None:
    """Serve over stdio until the host closes the pipe."""
    build_server(store).run("stdio")
