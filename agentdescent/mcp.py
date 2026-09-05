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
from typing import Any, Dict, List, Optional

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

    # -- read-only -------------------------------------------------------------

    def doctor(self) -> Dict[str, Any]:
        return doctor_report()

    def plan(self, spec: Dict[str, Any], usd_per_call: Optional[float] = None) -> Dict[str, Any]:
        try:
            return plan_payload(EvolveSpec.from_dict(spec), usd_per_call=usd_per_call)
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
            es = EvolveSpec.from_dict(spec)
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
            st = runstore.launch(rd, budget_usd=budget_usd, usd_per_call=usd_per_call)
        except runstore.RunStoreError as e:
            return {"ok": False, "error": str(e), "run_id": rd.run_id, "dir": rd.path}
        return {"ok": True, "run_id": rd.run_id, "state": st.state, "pid": st.pid,
                "dir": rd.path, "notes": rd.status().notes}

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
            return runstore.resume(run_id, store=self.store, budget_usd=budget_usd,
                                   usd_per_call=usd_per_call).to_dict()
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

    @server.tool(description=TOOL_DESCRIPTIONS["doctor"])
    def doctor() -> Dict[str, Any]:
        return t.doctor()

    @server.tool(description=TOOL_DESCRIPTIONS["plan"])
    def plan(spec: Dict[str, Any], usd_per_call: Optional[float] = None) -> Dict[str, Any]:
        return t.plan(spec, usd_per_call)

    @server.tool(description=TOOL_DESCRIPTIONS["start"])
    def start(spec: Dict[str, Any], budget_usd: Optional[float] = None,
              usd_per_call: Optional[float] = None) -> Dict[str, Any]:
        return t.start(spec, budget_usd, usd_per_call)

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

    @server.tool(description=TOOL_DESCRIPTIONS["resume"])
    def resume(run_id: str, budget_usd: Optional[float] = None,
               usd_per_call: Optional[float] = None) -> Dict[str, Any]:
        return t.resume(run_id, budget_usd, usd_per_call)

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
