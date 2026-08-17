"""The sidecar: :func:`~agentdescent.evolution.evolve`, driven by the harness.

The plugin owns the surface -- `/evolve`, the tool, the artifact registrations --
and this owns the optimiser. Between them is :mod:`agentdescent.dsh.bridge`.

**The engine never talks to a model.** Not to reflect, not to judge. Every
completion goes back over the bridge as ``llm.complete``, and every rollout as
``rollout.execute``, so the harness stays the single place that holds
credentials, enforces the token budget, and applies the sandbox and approval
policy. An engine with its own API key would be a second, unpoliced path to the
same provider, and the budget would be advisory the moment it existed.

What the engine keeps is what it is good at: the ledger, staleness, conflict
resolution, fusion, and the Beta-posterior acceptance test.

Run it as ``python -m agentdescent.dsh.daemon`` -- stdin and stdout are the
bridge, so nothing else may write to stdout.
"""

from __future__ import annotations

import hashlib
import re
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from .. import __version__
from ..evolution import Task, evolve
from ..filetree import parse_tree
from ..treestrategy import FileTree, tree_reflector
from .bridge import PROTOCOL_VERSION, Peer

__all__ = ["Engine", "ENGINE_NAME", "serve_stdio"]

ENGINE_NAME = "agentdescent"

#: How many rollout results are remembered so a scorer can see the trajectory
#: behind an output. Bounded because a long run would otherwise hold every
#: rollout it ever ran.
_ROLLOUT_MEMORY = 4096


class RunCancelled(RuntimeError):
    """Raised inside a rollout once the run is cancelled."""


class BudgetExhausted(RuntimeError):
    """Raised inside a rollout once the token budget is spent."""


@dataclass
class RunState:
    """What `run.status` reports, and what the guards read."""

    run_id: str
    artifact: str
    phase: str = "starting"
    round: int = 0
    accepted: int = 0
    rejected: int = 0
    tokens_spent: int = 0
    error: Optional[str] = None
    token_budget: Optional[int] = None
    cancel: threading.Event = field(default_factory=threading.Event)

    def wire(self) -> Dict[str, Any]:
        status: Dict[str, Any] = {
            "runId": self.run_id, "artifact": self.artifact, "phase": self.phase,
            "round": self.round, "accepted": self.accepted, "rejected": self.rejected,
            "tokensSpent": self.tokens_spent,
        }
        if self.error is not None:
            status["error"] = self.error
        return status


class Engine:
    """Serves the harness's half of the protocol.

    One engine, many runs; each run owns a thread and drives ``evolve()``, which
    calls back for every rollout, every reflection and every score.
    """

    def __init__(self, peer: Peer, *, rounds: int = 4, workers: int = 2,
                 task_count: int = 16, rollout_timeout: float = 900.0) -> None:
        self._peer = peer
        self._defaults = {"rounds": rounds, "workers": workers,
                          "taskCount": task_count}
        self._rollout_timeout = rollout_timeout
        self._runs: Dict[str, RunState] = {}
        self._heads: Dict[str, Dict[str, Any]] = {}
        self._history: Dict[str, List[Dict[str, Any]]] = {}
        self._rollouts: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._seq = 0
        for method, handler in (
            ("handshake", self.handshake),
            ("run.start", self.run_start),
            ("run.status", self.run_status),
            ("run.cancel", self.run_cancel),
            ("ledger.head", self.ledger_head),
            ("ledger.history", self.ledger_history),
        ):
            peer.handle(method, handler)

    # -- handlers ----------------------------------------------------------

    def handshake(self, _params: Any = None) -> Dict[str, Any]:
        """Version first. A mismatch refuses rather than negotiating.

        Two versions that half-understand each other produce measurements that
        look fine and are not, which is worse than a process that will not start.
        """
        return {"protocol": PROTOCOL_VERSION, "engine": ENGINE_NAME,
                "version": __version__}

    def run_start(self, params: Mapping[str, Any]) -> str:
        artifact = str(params["artifact"])
        with self._lock:
            self._seq += 1
            run_id = f"run-{self._seq}"
            budget = params.get("tokenBudget")
            state = RunState(run_id, artifact,
                             token_budget=int(budget) if budget else None)
            self._runs[run_id] = state
        thread = threading.Thread(target=self._execute, args=(state, dict(params)),
                                  name=f"evolve-{run_id}", daemon=True)
        thread.start()
        return run_id

    def run_status(self, params: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        with self._lock:
            state = self._runs.get(str(params.get("runId")))
        return state.wire() if state is not None else None

    def run_cancel(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        """Cooperative: the next rollout raises, and the run stops there.

        There is no way to abandon a rollout already in flight without leaving
        the harness holding a live child agent, so the honest contract is "no
        further work", not "stopped now".
        """
        with self._lock:
            state = self._runs.get(str(params.get("runId")))
        if state is None:
            return {"cancelled": False}
        state.cancel.set()
        if state.phase in ("starting", "running"):
            state.phase = "stopping"
        return {"cancelled": True}

    def ledger_head(self, params: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._heads.get(str(params.get("artifact")))

    def ledger_history(self, params: Mapping[str, Any]) -> List[Dict[str, Any]]:
        limit = params.get("limit")
        with self._lock:
            records = list(self._history.get(str(params.get("artifact")), ()))
        return records[-int(limit):] if limit else records

    # -- the run -----------------------------------------------------------

    def _execute(self, state: RunState, params: Mapping[str, Any]) -> None:
        try:
            initial = dict(self._peer.call("artifact.load",
                                           {"artifact": state.artifact}) or {})
            tasks = self._fetch_tasks(state, params)
            strategy = FileTree(initial_files=initial,
                                max_files_per_diff=int(params.get("maxFilesPerDiff", 2)))
            state.phase = "running"
            result = evolve(
                tasks, self._reward_for(state), run=self._run_for(state),
                propose=tree_reflector(self._completion_for(state), strategy=strategy),
                strategy=strategy, initial_state=dict(initial),
                artifact_id=_ledger_id(state.artifact),
                blast_radius=float(params.get("blastRadius", 0.2)),
                rounds=int(params.get("rounds", self._defaults["rounds"])),
                n_workers=int(params.get("workers", self._defaults["workers"])),
                max_concurrency=int(params.get("workers", self._defaults["workers"])),
                # Both defaults are the ones `skilldir` already settled on for an
                # agent-executed rollout: re-running every proposal's trajectory
                # doubles the cost of the run, and ranking every candidate on the
                # whole held-out set is the dominant cost when `eval_fn` is an
                # agent rather than a string comparison.
                self_verify=bool(params.get("selfVerify", False)),
                cheap_eval_tasks=int(params.get("cheapEvalTasks", 4)),
                on_round=self._on_round(state),
            )
        except RunCancelled:
            state.phase = "cancelled"
            return
        except BudgetExhausted:
            state.phase = "budget-exhausted"
            return
        except Exception as exc:                  # noqa: BLE001 - reported, not raised
            state.phase, state.error = "failed", f"{type(exc).__name__}: {exc}"
            return
        self._finish(state, initial, result)

    def _finish(self, state: RunState, initial: Mapping[str, str], result: Any) -> None:
        # `evolve()` does not let a worker failure escape: it ends the run and
        # reports it on `result.error`. So a stop we *asked for* arrives here
        # looking exactly like a crash, and the flags are the only thing that
        # can tell them apart. Checking them first is what keeps a cancelled run
        # from being reported as failed.
        if state.cancel.is_set():
            state.phase = "cancelled"
            return
        if state.token_budget is not None and state.tokens_spent >= state.token_budget:
            state.phase = "budget-exhausted"
            return
        if result.error is not None:
            state.phase, state.error = "failed", str(result.error)
            return
        final = dict(result.state)
        if final == dict(initial):
            # Nothing was accepted. Publishing here would be a no-op, but saying
            # "done" without a commit is the honest report: the gate declined,
            # which on a small held-out set is the expected outcome, not a fault.
            state.phase = "done"
            return
        before = result.history[0].held_out_reward if result.history else 0.0
        record = {
            "commitId": f"{state.run_id}-final", "runId": state.run_id,
            "artifact": state.artifact, "blastRadius": 0.2,
            "version": self._next_version(state.artifact),
            "heldOutBefore": before, "heldOutAfter": result.final_reward,
        }
        self._peer.call("artifact.publish", {
            "artifact": state.artifact, "state": final,
            "version": record["version"], "commit": record,
        })
        with self._lock:
            self._heads[state.artifact] = {
                "artifact": state.artifact, "version": record["version"],
                "heldOut": result.final_reward, "state": final,
            }
            self._history.setdefault(state.artifact, []).append(record)
        state.phase = "done"

    def _next_version(self, artifact: str) -> int:
        with self._lock:
            head = self._heads.get(artifact)
        return int(head["version"]) + 1 if head else 1

    def _fetch_tasks(self, state: RunState, params: Mapping[str, Any]) -> List[Task]:
        count = int(params.get("taskCount", self._defaults["taskCount"]))
        rows = self._peer.call("tasks.fetch", {
            "source": params.get("taskSource"), "count": count,
            "runId": state.run_id}) or []
        tasks = [Task(id=str(row.get("id") or f"t{i}"), prompt=str(row.get("prompt", "")),
                      meta=dict(row.get("meta") or {}))
                 for i, row in enumerate(rows)]
        if not tasks:
            raise RuntimeError(
                f"task source {params.get('taskSource')!r} returned nothing; there "
                f"is no run to make without tasks")
        return tasks

    # -- the callbacks evolve() drives ------------------------------------

    def _guard(self, state: RunState) -> None:
        if state.cancel.is_set():
            raise RunCancelled(state.run_id)
        if state.token_budget is not None and state.tokens_spent >= state.token_budget:
            raise BudgetExhausted(state.run_id)

    def _run_for(self, state: RunState):
        def run(rendered: str, task: Task) -> str:
            self._guard(state)
            reply = self._peer.call("rollout.execute", {
                "runId": state.run_id, "artifact": state.artifact,
                "state": parse_tree(rendered),
                "task": {"id": task.id, "prompt": task.prompt, "meta": task.meta},
            }, timeout=self._rollout_timeout) or {}
            output = str(reply.get("output") or "")
            self._spend(state, reply.get("tokens"))
            self._remember(task.id, output, reply)
            return output
        return run

    def _reward_for(self, state: RunState):
        def reward(task: Task, output: str) -> float:
            # The scorer wants the trajectory, not just the text -- steps, tool
            # errors and how the turn ended are what an efficiency or replay
            # scorer reads. `reward` is only handed (task, output), so the run
            # side stashes its result and this looks it up.
            observed = self._recall(task.id, output) or {"output": output}
            score = self._peer.call("reward.score", {
                "runId": state.run_id, "artifact": state.artifact,
                "task": {"id": task.id, "prompt": task.prompt, "meta": task.meta},
                "run": observed,
            })
            return _clamp(score)
        return reward

    def _completion_for(self, state: RunState):
        def complete(prompt: str) -> str:
            self._guard(state)
            reply = self._peer.call("llm.complete", {
                "runId": state.run_id, "prompt": prompt}, timeout=self._rollout_timeout)
            if isinstance(reply, Mapping):
                self._spend(state, reply.get("tokens"))
                return str(reply.get("text") or "")
            return str(reply or "")
        return complete

    def _on_round(self, state: RunState):
        def on_round(info: Any) -> None:
            state.round = int(getattr(info, "round", state.round))
            state.accepted += int(getattr(info, "committed", 0))
            state.rejected += int(getattr(info, "rejected", 0))
            # A notification, not a call: progress must never block a rollout.
            self._peer.notify("log.progress", {
                "runId": state.run_id, "artifact": state.artifact,
                "round": state.round, "heldOut": getattr(info, "held_out_reward", None),
                "accepted": state.accepted, "rejected": state.rejected,
                "tokensSpent": state.tokens_spent,
                "reasons": dict(getattr(info, "reasons", {}) or {}),
            })
        return on_round

    # -- plumbing ----------------------------------------------------------

    def _spend(self, state: RunState, tokens: Any) -> None:
        try:
            state.tokens_spent += max(0, int(tokens or 0))
        except (TypeError, ValueError):
            pass                                   # a peer that reports junk is not fatal

    @staticmethod
    def _key(task_id: str, output: str) -> str:
        digest = hashlib.sha1(output.encode("utf-8", "replace")).hexdigest()[:16]
        return f"{task_id}:{digest}"

    def _remember(self, task_id: str, output: str, reply: Mapping[str, Any]) -> None:
        with self._lock:
            if len(self._rollouts) >= _ROLLOUT_MEMORY:
                self._rollouts.pop(next(iter(self._rollouts)), None)
            self._rollouts[self._key(task_id, output)] = dict(reply)

    def _recall(self, task_id: str, output: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._rollouts.get(self._key(task_id, output))


#: Characters the ledger accepts in an artifact id, which becomes a filename.
_UNSAFE_ID = re.compile(r"[^A-Za-z0-9_.-]+")


def _ledger_id(artifact: str) -> str:
    """The on-disk name for a wire id.

    The harness addresses artifacts as ``skill:sql`` -- a namespace and a name --
    while the ledger turns its artifact id into a filename, where a colon is not
    allowed (and on Windows is not merely ugly). They are two different names for
    the same thing and conflating them fails at the first commit, so the wire id
    stays exactly as the harness sent it and only the ledger sees this one.
    """
    return _UNSAFE_ID.sub("-", artifact).strip("-") or "artifact"


def _clamp(value: Any) -> float:
    """Rewards outside [0, 1] are a contract violation the engine must not inherit."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number:                           # NaN
        return 0.0
    return 0.0 if number < 0.0 else (1.0 if number > 1.0 else number)


def serve_stdio() -> int:
    """Run the engine on stdin/stdout until the harness goes away."""
    peer = Peer(sys.stdin, sys.stdout)
    Engine(peer)
    peer.start()
    peer.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(serve_stdio())
