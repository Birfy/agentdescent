"""The MCP server: the CLI's verbs as tools a host agent can call.

``agentdescent mcp`` serves over stdio (or, with ``--transport streamable-http``,
over HTTP for URL-only hosts). DeepSeek Harness, Claude Code, Codex and
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

import inspect
import json
import os
from typing import Annotated, Any, Dict, Optional, Sequence, Tuple

from . import __version__, runstore
from .cli import (NESTED_ENV, apply_from_payload, apply_payload, doctor_report,
                 plan_payload, require_cli_on_path, require_resume_host_on_path,
                 show_payload, status_payload)
from .evolvespec import EvolveSpec, SpecError, compose

__all__ = ["DEFAULT_HTTP_HOST", "DEFAULT_HTTP_PORT", "RESCAN_ALLOW_ENV", "TRANSPORTS",
           "Tools", "TOOL_DESCRIPTIONS", "build_server", "is_loopback_host",
           "rescan_allow_from_env", "serve"]

#: Comma-separated import prefixes `audit_rescan` may resolve beyond this
#: package, read from the operator's environment.
RESCAN_ALLOW_ENV = "AGENTDESCENT_RESCAN_ALLOW"


def rescan_allow_from_env(environ: Optional[Dict[str, str]] = None
                          ) -> Tuple[str, ...]:
    """The operator's rescan allowlist, or an empty one.

    An environment variable rather than a tool parameter, and the difference is
    the entire control. `audit_rescan` resolves a "module:attribute" reference
    by **importing** the module, which runs its top level; the allowlist is what
    stands between that and any importable module on the box. A tool parameter
    is filled in by the model calling the tool, so an allowlist passed that way
    is one the caller can widen for itself -- the boundary would come down to
    the tool description asking it not to. The environment is set by the person
    running the server, before any model is in the room.
    """
    raw = (environ if environ is not None else os.environ).get(RESCAN_ALLOW_ENV, "")
    return tuple(part.strip() for part in raw.split(",") if part.strip())

#: Written for the calling model: what the tool is for and what to do around it.
TOOL_DESCRIPTIONS: Dict[str, str] = {
    "doctor": (
        "Check what this machine can run: which worker agent CLIs (claude, codex, dsh, "
        "opencode) "
        "are on PATH, which provider keys are set, whether a container engine and the "
        "git ledger are available, and which AgentDescent version and Python interpreter "
        "this server runs (compare with the shell's `agentdescent doctor` when results "
        "differ). Call this FIRST, before planning a run, and tell the "
        "user what is missing instead of starting a run that fails on round one."),
    "list_algorithms": (
        "List the named algorithms a spec can ask for (`algorithm` field), with a "
        "one-line summary, the kinds each applies to and its fidelity. Call this when "
        "the user names a method (GEPA, ACE, RRSI, ...) or asks what methods exist; if "
        "they named none, the default recipe for the kind is usually right. Do not "
        "invent a name -- an unknown one fails in plan."),
    "rsi": (
        "The archive's longitudinal self-improvement trend: per decision slot, "
        "the first and best adopted reward, the gain, the adoption that set the "
        "best, and how many adoptions have happened since. Use it when the user "
        "asks whether the system is still getting better or has plateaued; it is "
        "read-only and costs no model calls. `--archive` is the consolidation "
        "archive, not a run store."),
    "describe_algorithm": (
        "One algorithm's card: summary, kinds, fidelity, source, extra cost and every "
        "parameter with its type, default and bounds. Call this before setting "
        "`algorithm: {name, params}` so the values you pass are the ones plan expects. "
        "The same data is in `agentdescent algorithms show <name>`."),
    "plan": (
        "Validate an EvolveSpec and estimate its cost WITHOUT running anything. Resolves "
        "every agent/scorer/policy reference and loads the data, so a wrong field fails "
        "here with its name. Returns the composed evolve() arguments, the task count, an "
        "upper bound on agent calls per round and in total, and notes. When the spec has "
        "an `algorithm`, the resolved name/params are echoed. Show the user the spec and "
        "the estimate and get a yes before calling start."),
    "start": (
        "Start an evolution run in the background and return its run_id at once. The run "
        "is a detached process; poll `status` about once per round (not more often) and "
        "read `show` when it is done. Never call start without having shown the user the "
        "plan. A run costs real agent calls: rounds x n_workers x tasks. When the spec has "
        "an `audit` block, `status` reports `audit_store` once records exist -- pass that "
        "path to audit_status to find out how far the run's scorer is from the truth."),
    "status": (
        "Progress of one run (round, best held-out reward, calls, dollars if priced, "
        "state, the last few rounds, and `audit_store` when the run has one) or, with no "
        "run_id, {store, runs} where `runs` is "
        "every run newest first and is empty when there are none. Cheap; safe to poll. "
        "Summarise round-to-round deltas for the user rather than pasting JSON."),
    "show": (
        "The evolved artifact when a run is done: for a directory, the list of files, a "
        "unified diff against the original, and the plan `apply` would execute (files "
        "written, extra files, backups); for text, the evolved instruction. Also the "
        "outcomes() histogram that says why proposals were committed or refused. Explain "
        "what changed and why; do not paste the whole tree."),
    "diagnose": (
        "Explain why a run ended where it did: solved, partial-improvement, running, "
        "cancelled, run-failed, reflector-error, scoring-mismatch, agent-failures, "
        "gate-rejected, no-improvement or unknown, with bounded evidence and the next "
        "step. Call this after `show` when the run learned little, instead of guessing "
        "from the outcome histogram."),
    "apply": (
        "DESTRUCTIVE: write an evolved artifact back over a target. Pass `run_id` to "
        "install a finished run's result (optional `to` overrides the run's target), OR "
        "pass `from` + `to` to install an on-disk artifact such as a campaign "
        "`final_target` -- same rules as CLI `apply --from` (from requires to; mutually "
        "exclusive with run_id; symlinks refused). Backs the original up first unless "
        "backup=false. Ask the user explicitly before calling this; use dry_run=true to "
        "preview. Returns written / unchanged / backup; `noop: true` means the target "
        "already matched, so nothing was written and no backup exists -- tell the user "
        "nothing changed. Otherwise tell them the backup path afterwards."),
    "cancel": (
        "Stop a running evolution and every worker agent it started. The run keeps its "
        "ledger and can be resumed."),
    "resume": (
        "Continue a stopped, failed or cancelled run on its existing ledger (the engine "
        "picks up where it left off). Returns the new status."),
    "audit_status": (
        "Read a sparse-audit JSONL: how biased the run's cheap verifier is against ground "
        "truth (`delta_hat`), how *scattered* that error is (`resid_sd` -- the number the "
        "acceptance gate's uncertainty is mostly made of), how many labels it rests on, "
        "and how many units are still waiting on an oracle. Call this before quoting any "
        "reward number from a run whose reward was an agent judging an output: it is the "
        "only thing that can tell the user the loop was optimising a proxy. If "
        "`is_stale` is true the correction must not be applied and `stale_reason` says "
        "why."),
    "audit_pending": (
        "List the audited units still waiting on ground truth -- record_id, task, the "
        "output that was produced -- so a person or an experiment system can go and "
        "measure them and hand the results back through audit_resolve. Outputs are "
        "truncated; raise `limit` deliberately, a full queue is large. "
        "order='priority' puts the units the run's own merge path ranked as "
        "riskiest first -- use it when the oracle is a person or an experiment "
        "and only part of the queue will get done."),
    "audit_resolve": (
        "Attach one ground-truth score to one pending audit record. REFUSES to overwrite "
        "a record that already has a result: a second score for the same unit is either "
        "a duplicate submission or a correction, and only the user can say which -- ask "
        "them rather than retrying. Does not recompute the correction; call "
        "audit_recompute when a batch is in."),
    "audit_recompute": (
        "Re-read the audit store and re-estimate the correction. Call after resolving a "
        "batch, or when audit_status says the rectification is stale because the verifier "
        "changed."),
    "audit_scorecard": (
        "Grade a verifier before it replaces another one. Reports `sigma` first (the "
        "spread of the verifier's error) and `blockers`, which is the verdict. Do NOT "
        "recommend a verifier change on `delta_hat` falling: a mean error goes to zero "
        "when errors cancel, and on real data a correct-looking rule cut it 74% while "
        "making the verifier worse. If `ship` is false, relay the blockers verbatim."),
    "audit_rescan": (
        "Re-score the outputs the audit kept with a different verifier, and report how "
        "much of the run's recorded history it would have scored differently -- including "
        "how many artifact pairs reverse order. `verifier` is a 'module:attribute' "
        "reference; resolving it RUNS the module, so only the agentdescent package and "
        "whatever prefixes the operator set in AGENTDESCENT_RESCAN_ALLOW can be named. "
        "That list is not something this tool can widen: if the reference you need is "
        "refused, tell the user which prefix to add and why."),
    "safety": (
        "Read-only: is a consolidation archive (the rules meta runs adopted) safe to "
        "keep seeding new runs from? Returns `verdict` (safe / caution / unsafe), a "
        "one-line `reason`, counts of rules, unexamined, flagged and demoted ones, and "
        "every `check` with its status. Pass `audit` (a MetaAuditStore JSONL) to check "
        "verifier bias -- without it that check is a warning, not a pass. `archive` "
        "defaults to $AGENTDESCENT_HOME/archive; a missing one is an error and is never "
        "created. Same data as `agentdescent safety --json`. Relay the verdict and "
        "every non-ok check before the user seeds from the archive."),
    "portfolio": (
        "Read-only: rank a consolidation archive's layers by return per cost and "
        "propose how to split the next budget window between them (`reallocated`, "
        "with a `reasons` line per layer). Only reports the suggestion; nothing is "
        "written or applied. `archive` defaults to $AGENTDESCENT_HOME/archive; a missing "
        "or empty one is an error. Same data as `agentdescent portfolio --json`."),
    "audit_trace": (
        "Read-only: trace one archived adoption (`run_id`) through its evidence rings "
        "-- archive record, meta-gain audit, inner-reward audit -- and say whether the "
        "chain is `verified`, with each link's status and reason. Pass `meta_audit` / "
        "`inner_audit` store paths for those rings; a ring with no store is reported as "
        "missing evidence, not as passing. Same data as `agentdescent audit <run_id> "
        "--json` (the CLI's --harness ring needs a module reference and is CLI-only)."),
    "audit_drift": (
        "Chart the correction across verifier versions with an EWMA control chart. A "
        "`bias-up` signal means the loop is finding answers the verifier likes and the "
        "truth does not. A `signal-lost` signal means the verifier no longer predicts the "
        "truth -- the fix is a different verifier, not more labels. If `overlapping` is "
        "true the limits do not apply and an alarm is not evidence; say so."),
}


class Tools:
    """The tool bodies, SDK-free. One instance per server, bound to a run store."""

    def __init__(self, store: Optional[str] = None, *,
                 rescan_allow: Optional[Sequence[str]] = None) -> None:
        self.store = store
        #: Import prefixes `audit_rescan` may resolve beyond this package. It is
        #: read from the operator's environment and never from a tool call: see
        #: `audit_rescan` for why that distinction is the whole control.
        self.rescan_allow: Tuple[str, ...] = (
            tuple(rescan_allow) if rescan_allow is not None
            else rescan_allow_from_env())
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

    def list_algorithms(self, kind: Optional[str] = None) -> Dict[str, Any]:
        from . import algorithms
        from .evolvespec import KINDS

        if kind is not None and kind not in KINDS:
            return {"error": f"unknown kind {kind!r}; choose from {list(KINDS)}"}
        payload: Dict[str, Any] = {
            "algorithms": [a.describe() for a in algorithms.listing(kind)]}
        errors = algorithms.external_errors()
        if errors:
            payload["load_errors"] = errors
        if kind == "policy_slot" and not payload["algorithms"]:
            payload["note"] = ("no algorithm applies to kind 'policy_slot': that "
                               "kind evolves a policy slot, and `plan` refuses an "
                               "algorithm for it")
        return payload

    def rsi(self, archive: Optional[str] = None, slot: Optional[str] = None,
            stagnation_window: int = 3) -> Dict[str, Any]:
        from .cli import _archive_root
        from .rsi import rsi_report

        try:
            return rsi_report(_archive_root(archive), slot=slot,
                              stagnation_window=stagnation_window).to_dict()
        except Exception as e:  # noqa: BLE001 - a bad archive is a report, not a crash
            return {"error": str(e)}

    def describe_algorithm(self, name: str) -> Dict[str, Any]:
        from . import algorithms

        try:
            return algorithms.get(name, where="describe_algorithm").describe()
        except algorithms.AlgorithmError as e:
            return {"error": str(e)}

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
            payload = status_payload(run_id, store=self.store)
        except runstore.RunStoreError as e:
            return {"error": str(e)}
        if run_id is not None:
            return payload
        # Wrapped, because a bare list serialises to *zero* content blocks when
        # it is empty: an agent asking "what is running?" on a fresh machine got
        # back nothing at all and could not tell that from a call that failed.
        # The CLI prints "no runs under <dir>"; this is the same sentence in the
        # shape a tool returns, and it names the store either way.
        return {"store": runstore.store_path(self.store), "runs": payload}

    def show(self, run_id: str, diff: bool = True) -> Dict[str, Any]:
        try:
            return show_payload(run_id, store=self.store, diff=diff)
        except runstore.RunStoreError as e:
            return {"error": str(e)}

    def diagnose(self, run_id: str) -> Dict[str, Any]:
        from .diagnose import diagnose_payload

        try:
            return diagnose_payload(run_id, store=self.store).to_dict()
        except runstore.RunStoreError as e:
            return {"error": str(e)}

    # -- side effects -----------------------------------------------------------

    def start(self, spec: Dict[str, Any], budget_usd: Optional[float] = None,
              usd_per_call: Optional[float] = None) -> Dict[str, Any]:
        try:
            es = EvolveSpec.from_dict(spec).absolutise()
            comp = compose(es)                # fail here, with the field, not in the child
            # Before create: same rule as CLI -- budget without price must not
            # return ok/started and then fail only in log.txt (#342).
            runstore.require_budget_pair(budget_usd, usd_per_call)
        except SpecError as e:
            return {"ok": False, "error": str(e)}
        except runstore.RunStoreError as e:
            return {"ok": False, "error": str(e)}
        if os.environ.get(NESTED_ENV):
            # This server is running inside a worker of another run (the plugin
            # that hosts it is being evolved). The transcript should still show
            # that the host *called* start -- that is what the grader looks for --
            # but nothing may actually run, or every rollout would start a run.
            # The PATH CLI gate stays *after* this stub: nested start must never
            # launch, and the grader-visible refused response must not become
            # ok:false just because the proposed CLI is absent here (#351).
            return {"ok": True, "run_id": "nested-stub", "state": "refused", "nested": True,
                    "note": f"{NESTED_ENV} is set: this session is itself a worker of an "
                            "evolution run, so no nested run was started."}
        try:
            # Same gate as plan / CLI evolve: missing CLI must not return ok/running (#346).
            # After the nested stub above so #346 stays fail-closed for real starts.
            require_cli_on_path(es)
        except SpecError as e:
            return {"ok": False, "error": str(e)}
        rd = runstore.create(es.to_dict(), store=self.store)
        # Recorded before the child is up, so the reply below and the first
        # `status` poll carry what `plan` said -- a too-small held-out set among
        # it -- instead of an empty list until the child has composed (#414).
        rd.update_status(notes=comp.notes)
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

    def apply(self, run_id: Optional[str] = None, to: Optional[str] = None,
              dry_run: bool = False, backup: bool = True, *,
              source: Optional[str] = None) -> Dict[str, Any]:
        """Install a run result (`run_id`) or an on-disk artifact (`source` + `to`).

        ``source`` is the MCP/`CLI` ``from`` path (campaign ``final_target``, etc.).
        ``run_id`` and ``source`` are mutually exclusive; ``source`` requires ``to``.
        """
        try:
            if source:
                if run_id:
                    raise runstore.RunStoreError(
                        "pass a run_id or from PATH, not both")
                if not to:
                    raise runstore.RunStoreError(
                        "from needs to PATH (where to install the artifact)")
                return apply_from_payload(source, to=to, dry_run=dry_run, backup=backup)
            if not run_id:
                raise runstore.RunStoreError(
                    "apply needs a run_id, or from PATH and to PATH")
            return apply_payload(run_id, to=to, store=self.store, dry_run=dry_run,
                                backup=backup)
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
            runstore.require_budget_pair(budget_usd, usd_per_call)
            # Same host gate as CLI resume: an absent plugin host must not
            # relaunch and report running (#370).
            require_resume_host_on_path(run_id, store=self.store)
            return runstore.resume(run_id, store=self.store, budget_usd=budget_usd,
                                   usd_per_call=usd_per_call,
                                   env=self.host_model_env()).to_dict()
        except (runstore.RunStoreError, SpecError) as e:
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

    # -- the audit ------------------------------------------------------------
    #
    # These take a JSONL **path**, not a run_id. Nothing writes an audit store
    # into a run directory: `AuditedReward` is built by the caller and points
    # wherever they said, and the resolving process is usually not this one at
    # all -- a wet-lab result comes back next week, to a machine that has only
    # the file. The bodies live in `agentdescent.audit.service` for the same
    # reason the run-store bodies live in `cli`: so they can be driven without
    # an MCP host.

    def audit_status(self, path: str, version: Optional[str] = None) -> Dict[str, Any]:
        from .audit import service

        return service.audit_status(path, version)

    def audit_pending(self, path: str, limit: int = 50,
                      older_than: Optional[float] = None,
                      version: Optional[str] = None,
                      order: str = "dispatched") -> Dict[str, Any]:
        from .audit import service

        return service.audit_pending(path, limit, older_than, version, order)

    def audit_resolve(self, path: str, record_id: str,
                      oracle_score: float) -> Dict[str, Any]:
        from .audit import service

        return service.audit_resolve(path, record_id, oracle_score)

    def audit_recompute(self, path: str, version: Optional[str] = None) -> Dict[str, Any]:
        from .audit import service

        return service.audit_recompute(path, version)

    def audit_scorecard(self, path: str, version: Optional[str] = None,
                        previous: Optional[str] = None,
                        max_false_negative: float = 0.05,
                        verifier_seconds: Optional[float] = None,
                        oracle_seconds: Optional[float] = None) -> Dict[str, Any]:
        from .audit import service

        return service.audit_scorecard(path, version, previous,
                                       max_false_negative, verifier_seconds,
                                       oracle_seconds)

    def audit_rescan(self, path: str, verifier: str,
                     version: Optional[str] = None,
                     pairs: Optional[list] = None) -> Dict[str, Any]:
        """Re-score with another verifier, within the operator's allowlist.

        There is deliberately no ``allow`` parameter. Resolving a reference
        **imports** the module, which runs whatever is at its top level, so the
        allowlist is the boundary between "this server can run the agentdescent
        package" and "this server can run anything importable". A parameter is
        the wrong shape for that: the tool is called by a model, so a widening
        parameter is one the model fills in, and the boundary would be enforced
        by asking it nicely not to. `service.audit_rescan` still takes `allow`
        -- there the caller is the operator, writing a script.
        """
        from .audit import service

        return service.audit_rescan(path, verifier, version,
                                    self.rescan_allow, pairs)

    def audit_drift(self, path: str,
                    versions: Optional[list] = None) -> Dict[str, Any]:
        from .audit import service

        return service.audit_drift(path, versions)

    # -- the meta stack (read-only) ---------------------------------------------
    #
    # The bodies are the CLI's own (`cli.*_payload`), so a tool and the
    # `--json` verb cannot drift apart. None of them writes: a missing archive
    # is an error, never a directory created on the way (#502).

    def safety(self, archive: Optional[str] = None, audit: Optional[str] = None,
               bias_threshold: float = 0.1,
               min_exam: str = "passed") -> Dict[str, Any]:
        from .cli import MetaCommandError, safety_payload

        try:
            return safety_payload(archive, audit=audit,
                                  bias_threshold=bias_threshold, min_exam=min_exam)
        except MetaCommandError as e:
            return {"error": str(e)}

    def portfolio(self, archive: Optional[str] = None, floor: float = 0.05,
                  min_samples: int = 3, grow: float = 0.5, shrink: float = 0.5,
                  cap: Optional[float] = None) -> Dict[str, Any]:
        from .cli import MetaCommandError, portfolio_payload

        try:
            return portfolio_payload(archive, floor=floor, min_samples=min_samples,
                                     grow=grow, shrink=shrink, cap=cap)
        except (MetaCommandError, ValueError) as e:
            return {"error": str(e)}

    def audit_trace(self, run_id: str, archive: Optional[str] = None,
                    meta_audit: Optional[str] = None,
                    inner_audit: Optional[str] = None,
                    bias_threshold: float = 0.1) -> Dict[str, Any]:
        from .cli import MetaCommandError, audit_trace_payload

        try:
            return audit_trace_payload(run_id, archive=archive,
                                       meta_audit=meta_audit,
                                       inner_audit=inner_audit,
                                       bias_threshold=bias_threshold)
        except MetaCommandError as e:
            return {"error": str(e)}


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


def build_server(store: Optional[str] = None, *, name: str = "agentdescent",
                 rescan_allow: Optional[Sequence[str]] = None):
    """An MCP server with every tool in :data:`TOOL_DESCRIPTIONS` and two resources."""
    server_cls = _server_class()
    # `version` only where the constructor takes it: mcp 2.x does, and without it
    # `initialize` answers with an empty string, so every host that shows the
    # server's version shows a blank. mcp 1.x routes unknown keywords into its
    # `Settings` model, where an unexpected one is an error, not a no-op.
    extra: Dict[str, Any] = {}
    if "version" in inspect.signature(server_cls.__init__).parameters:
        extra["version"] = __version__
    server = server_cls(name, **extra, instructions=(
        "AgentDescent evolves skills, agent definitions, prompts, code and host plugins "
        "against examples with a parallel, merge-based optimiser. Workflow: doctor -> "
        "write an EvolveSpec -> plan (show the user) -> start -> status (once a round) -> "
        "show -> ask -> apply. The audit_* tools are a separate axis and take a JSONL "
        "path, not a run_id: when a run's reward was an agent judging an output rather "
        "than a fact about it, the loop was optimising a proxy, and audit_status is the "
        "only thing that can tell the user how far that proxy is from the truth."))
    t = Tools(store, rescan_allow=rescan_allow)

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

    @server.tool(description=TOOL_DESCRIPTIONS["list_algorithms"])
    def list_algorithms(kind: Optional[str] = None) -> Dict[str, Any]:
        return t.list_algorithms(kind)

    @server.tool(description=TOOL_DESCRIPTIONS["describe_algorithm"])
    def describe_algorithm(name: str) -> Dict[str, Any]:
        return t.describe_algorithm(name)

    @server.tool(description=TOOL_DESCRIPTIONS["rsi"])
    def rsi(archive: Optional[str] = None, slot: Optional[str] = None,
            stagnation_window: int = 3) -> Dict[str, Any]:
        return t.rsi(archive, slot, stagnation_window)

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

    @server.tool(description=TOOL_DESCRIPTIONS["diagnose"])
    def diagnose(run_id: str) -> Dict[str, Any]:
        return t.diagnose(run_id)

    # `from` is a Python keyword; expose it as the MCP arg name via pydantic
    # validation_alias so hosts pass `{"from": ..., "to": ...}` like the CLI.
    # Annotation is set after the def (like Context on start/resume) because
    # `from __future__ import annotations` would stringify a local Field import
    # and fail when the SDK evaluates it.
    from pydantic import Field as _Field

    def apply(run_id: Optional[str] = None, to: Optional[str] = None,
              dry_run: bool = False, backup: bool = True,
              from_: Optional[str] = None) -> Dict[str, Any]:
        return t.apply(run_id, to, dry_run, backup, source=from_)

    apply.__annotations__["from_"] = Annotated[
        Optional[str],
        _Field(validation_alias="from", serialization_alias="from")]
    server.tool(description=TOOL_DESCRIPTIONS["apply"])(apply)

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

    @server.tool(description=TOOL_DESCRIPTIONS["audit_status"])
    def audit_status(path: str, version: Optional[str] = None) -> Dict[str, Any]:
        return t.audit_status(path, version)

    @server.tool(description=TOOL_DESCRIPTIONS["audit_pending"])
    def audit_pending(path: str, limit: int = 50, older_than: Optional[float] = None,
                      version: Optional[str] = None,
                      order: str = "dispatched") -> Dict[str, Any]:
        return t.audit_pending(path, limit, older_than, version, order)

    @server.tool(description=TOOL_DESCRIPTIONS["audit_resolve"])
    def audit_resolve(path: str, record_id: str, oracle_score: float) -> Dict[str, Any]:
        return t.audit_resolve(path, record_id, oracle_score)

    @server.tool(description=TOOL_DESCRIPTIONS["audit_recompute"])
    def audit_recompute(path: str, version: Optional[str] = None) -> Dict[str, Any]:
        return t.audit_recompute(path, version)

    @server.tool(description=TOOL_DESCRIPTIONS["audit_scorecard"])
    def audit_scorecard(path: str, version: Optional[str] = None,
                        previous: Optional[str] = None,
                        max_false_negative: float = 0.05,
                        verifier_seconds: Optional[float] = None,
                        oracle_seconds: Optional[float] = None) -> Dict[str, Any]:
        return t.audit_scorecard(path, version, previous, max_false_negative,
                                 verifier_seconds, oracle_seconds)

    @server.tool(description=TOOL_DESCRIPTIONS["audit_rescan"])
    def audit_rescan(path: str, verifier: str, version: Optional[str] = None,
                     pairs: Optional[list] = None) -> Dict[str, Any]:
        return t.audit_rescan(path, verifier, version, pairs)

    @server.tool(description=TOOL_DESCRIPTIONS["audit_drift"])
    def audit_drift(path: str, versions: Optional[list] = None) -> Dict[str, Any]:
        return t.audit_drift(path, versions)

    @server.tool(description=TOOL_DESCRIPTIONS["safety"])
    def safety(archive: Optional[str] = None, audit: Optional[str] = None,
               bias_threshold: float = 0.1, min_exam: str = "passed") -> Dict[str, Any]:
        return t.safety(archive, audit, bias_threshold, min_exam)

    @server.tool(description=TOOL_DESCRIPTIONS["portfolio"])
    def portfolio(archive: Optional[str] = None, floor: float = 0.05,
                  min_samples: int = 3, grow: float = 0.5, shrink: float = 0.5,
                  cap: Optional[float] = None) -> Dict[str, Any]:
        return t.portfolio(archive, floor, min_samples, grow, shrink, cap)

    @server.tool(description=TOOL_DESCRIPTIONS["audit_trace"])
    def audit_trace(run_id: str, archive: Optional[str] = None,
                    meta_audit: Optional[str] = None, inner_audit: Optional[str] = None,
                    bias_threshold: float = 0.1) -> Dict[str, Any]:
        return t.audit_trace(run_id, archive, meta_audit, inner_audit, bias_threshold)

    @server.resource("agentdescent://runs", mime_type="application/json",
                     description="Every run in the store, newest first, with its status.")
    def runs() -> str:
        return t.runs_resource()

    @server.resource("agentdescent://runs/{run_id}/rounds", mime_type="application/json",
                     description="One run's rounds as they completed: reward, commits, reasons.")
    def rounds(run_id: str) -> str:
        return t.rounds_resource(run_id)

    return server


#: Transports ``agentdescent mcp --transport`` accepts. stdio is what hosts that
#: spawn a subprocess speak; streamable-http is for hosts that only take a URL
#: (Grok Bot's custom MCP connectors, other remote hosts) behind a tunnel.
TRANSPORTS = ("stdio", "streamable-http")
DEFAULT_HTTP_HOST = "127.0.0.1"
DEFAULT_HTTP_PORT = 8000


def is_loopback_host(host: str) -> bool:
    """True when ``host`` only accepts connections from this machine."""
    h = (host or "").strip().strip("[]").lower()
    if h == "localhost":
        return True
    try:
        import ipaddress
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def serve(store: Optional[str] = None, *, transport: str = "stdio",
          host: str = DEFAULT_HTTP_HOST, port: int = DEFAULT_HTTP_PORT) -> None:
    """Serve until the host closes the pipe (stdio) or the process is stopped.

    ``streamable-http`` listens on ``host:port`` at ``/mcp``. The host/port go
    where each SDK generation reads them: mcp 2.x takes them as keywords to
    ``run``; mcp 1.x's ``FastMCP`` reads them from ``server.settings``. Callers
    are responsible for the loopback check (the CLI refuses a non-loopback host
    without ``--allow-remote``): ``start`` and ``apply`` write to disk.
    """
    if transport not in TRANSPORTS:
        raise ValueError(f"unknown transport {transport!r}; expected one of {TRANSPORTS}")
    server = build_server(store)
    if transport == "stdio":
        server.run("stdio")
        return
    runner = getattr(server, "run_streamable_http_async", None)
    params = inspect.signature(runner).parameters if runner is not None else {}
    if "host" in params and "port" in params:  # mcp >= 2
        server.run("streamable-http", host=host, port=port)
    else:  # mcp 1.x FastMCP
        server.settings.host = host
        server.settings.port = port
        server.run("streamable-http")
