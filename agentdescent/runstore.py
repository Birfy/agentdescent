"""Where a run lives when the caller is not waiting for it.

A run started from an agent session outlives the tool call that started it: MCP
tool calls time out in seconds, hosts restart their servers between sessions, and
an evolution takes minutes to hours. So a run is a **directory** plus a
**detached process**, and every question about it is answered from the
directory::

    ~/.agentdescent/runs/<run_id>/
        spec.json        what was asked (an EvolveSpec)
        status.json      state, round, best reward, calls, pid -- rewritten atomically
        rounds.jsonl     one RoundInfo per line, as they complete
        result.json      EvolutionResult.save() when the run ends
        tree/            the evolved directory, materialised (directory kinds)
        ledger/          the git ledger; repo_path= to evolve(), which is what resume is
        log.txt          stdout + stderr of the detached process

Three decisions, and why:

**A process, not a thread.** The MCP server that starts a run is itself a child
of the host and dies with it. A thread would take the run down; a process with
its own session (``start_new_session``) does not, and its process group is also
what :func:`cancel` signals, so the worker agents it spawned die with it.

**Status is written atomically, from the ``on_round`` hook.** A reader can arrive
at any moment, so the file is replaced (``os.replace``), never appended in
place. The hook is the engine's own progress seam; nothing here polls the engine.

**Resume is the ledger.** ``evolve(repo_path=<run>/ledger)`` resumes a ledger
that exists, so re-launching the same spec on the same directory continues the
run. Nothing is serialised beyond what the engine already persists.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterator, List, Optional

from .evolution import EvolutionResult, RoundInfo

__all__ = [
    "STATES",
    "RunDir",
    "RunStatus",
    "RunStoreError",
    "cancel",
    "create",
    "execute",
    "get",
    "launch",
    "list_runs",
    "resume",
    "root",
]

STATES = ("created", "running", "done", "failed", "cancelled")

_ENV_ROOT = "AGENTDESCENT_HOME"


class RunStoreError(RuntimeError):
    """A run directory that is missing, malformed, or in the wrong state."""


def root() -> str:
    """``$AGENTDESCENT_HOME/runs`` or ``~/.agentdescent/runs``.

    Outside the repository on purpose: an installed package may be read-only, and
    a run store inside a checkout would be committed by accident."""
    home = os.environ.get(_ENV_ROOT) or os.path.expanduser("~/.agentdescent")
    return os.path.join(home, "runs")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


@dataclass
class RunStatus:
    """The one file a host polls. Small on purpose: it is read once a round."""

    run_id: str
    state: str = "created"
    round: int = 0
    rounds: Optional[int] = None
    best_reward: Optional[float] = None
    last_reward: Optional[float] = None
    committed: int = 0
    calls: int = 0
    rollouts: int = 0
    usd: Optional[float] = None
    pid: Optional[int] = None
    started: Optional[float] = None
    updated: Optional[float] = None
    finished: Optional[float] = None
    stop_reason: Optional[str] = None
    error: Optional[str] = None
    kind: Optional[str] = None
    target: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunStatus":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})

    def alive(self) -> bool:
        """Whether the recorded pid is still a live process (POSIX).

        A zombie is not alive: ``kill(pid, 0)`` succeeds on one, so on Linux the
        state letter in ``/proc/<pid>/stat`` is checked as well."""
        if self.pid is None:
            return False
        try:
            os.kill(self.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        try:
            with open(f"/proc/{self.pid}/stat") as fh:
                return fh.read().rsplit(")", 1)[1].split()[0] not in ("Z", "X")
        except OSError:
            return True


def _write_json(path: str, payload: Any) -> None:
    tmp = f"{path}.{uuid.uuid4().hex}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def _read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# the run directory
# ---------------------------------------------------------------------------


@dataclass
class RunDir:
    """One run's directory and the files in it."""

    path: str

    @property
    def run_id(self) -> str:
        return os.path.basename(self.path.rstrip(os.sep))

    def _file(self, name: str) -> str:
        return os.path.join(self.path, name)

    # -- files ---------------------------------------------------------------

    @property
    def spec_path(self) -> str:
        return self._file("spec.json")

    @property
    def status_path(self) -> str:
        return self._file("status.json")

    @property
    def rounds_path(self) -> str:
        return self._file("rounds.jsonl")

    @property
    def result_path(self) -> str:
        return self._file("result.json")

    @property
    def tree_path(self) -> str:
        return self._file("tree")

    @property
    def ledger_path(self) -> str:
        return self._file("ledger")

    @property
    def log_path(self) -> str:
        return self._file("log.txt")

    # -- reads ---------------------------------------------------------------

    def spec_dict(self) -> Dict[str, Any]:
        return _read_json(self.spec_path)

    def status(self) -> RunStatus:
        try:
            st = RunStatus.from_dict(_read_json(self.status_path))
        except FileNotFoundError:
            raise RunStoreError(f"{self.run_id}: no status.json; not a run directory") from None
        # A process that died without writing a terminal state -- killed, OOM,
        # machine rebooted -- must not read as "running" forever.
        if st.state == "running" and not st.alive():
            st.state = "failed"
            st.error = st.error or "process ended without reporting (killed, or the machine went away)"
            st.finished = st.finished or st.updated
            _write_json(self.status_path, st.to_dict())
        return st

    def rounds(self) -> List[Dict[str, Any]]:
        try:
            with open(self.rounds_path, encoding="utf-8") as fh:
                return [json.loads(line) for line in fh if line.strip()]
        except FileNotFoundError:
            return []

    def result(self) -> Optional[EvolutionResult]:
        if not os.path.exists(self.result_path):
            return None
        return EvolutionResult.load(self.result_path)

    def log_tail(self, lines: int = 40) -> str:
        try:
            with open(self.log_path, encoding="utf-8", errors="replace") as fh:
                return "".join(fh.readlines()[-lines:])
        except FileNotFoundError:
            return ""

    # -- writes --------------------------------------------------------------

    def update_status(self, **changes: Any) -> RunStatus:
        try:
            st = RunStatus.from_dict(_read_json(self.status_path))
        except FileNotFoundError:
            st = RunStatus(run_id=self.run_id)
        for k, v in changes.items():
            setattr(st, k, v)
        st.updated = time.time()
        _write_json(self.status_path, st.to_dict())
        return st

    def append_round(self, info: RoundInfo) -> None:
        with open(self.rounds_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(info), default=str) + "\n")


# ---------------------------------------------------------------------------
# the store
# ---------------------------------------------------------------------------


def _new_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def create(spec_dict: Dict[str, Any], *, run_id: Optional[str] = None,
           store: Optional[str] = None) -> RunDir:
    """Make a run directory and write the spec and an initial status. Runs nothing."""
    base = store or root()
    os.makedirs(base, exist_ok=True)
    rid = run_id or _new_id()
    rd = RunDir(os.path.join(base, rid))
    if os.path.exists(rd.path):
        raise RunStoreError(f"run {rid!r} already exists")
    os.makedirs(rd.path)
    _write_json(rd.spec_path, spec_dict)
    rd.update_status(state="created", kind=spec_dict.get("kind"),
                     target=spec_dict.get("target"),
                     rounds=(spec_dict.get("evolve") or {}).get("rounds"))
    return rd


def get(run_id: str, *, store: Optional[str] = None) -> RunDir:
    rd = RunDir(os.path.join(store or root(), run_id))
    if not os.path.isdir(rd.path):
        raise RunStoreError(f"no run {run_id!r} under {store or root()}")
    return rd


def list_runs(*, store: Optional[str] = None) -> List[RunStatus]:
    """Every run's status, newest first."""
    base = store or root()
    if not os.path.isdir(base):
        return []
    out: List[RunStatus] = []
    for name in sorted(os.listdir(base), reverse=True):
        rd = RunDir(os.path.join(base, name))
        if os.path.exists(rd.status_path):
            try:
                out.append(rd.status())
            except (RunStoreError, ValueError):
                continue
    return out


def _python() -> str:
    return sys.executable or "python"


def _child_pythonpath() -> str:
    """Make sure the child can import *this* copy of the package.

    Installed, this is a no-op; from a checkout (`pytest` puts the repo root on
    `sys.path`, `pip` did not) the detached child has neither, and would die on
    its first import with nothing in the store to say why."""
    import agentdescent

    here = os.path.dirname(os.path.dirname(os.path.abspath(agentdescent.__file__)))
    existing = os.environ.get("PYTHONPATH", "")
    return here if not existing else f"{here}{os.pathsep}{existing}"


#: The intermediate that makes the run a grandchild. The parent waits for this
#: one-liner and never for the run, so the run is re-parented to init and reaped
#: there: a child the parent never `wait()`s for is a zombie, and a zombie still
#: answers `kill(pid, 0)`, which read as "running" forever.
_BOOTSTRAP = (
    "import subprocess,sys\n"
    "log=open(sys.argv[1],'a')\n"
    "p=subprocess.Popen(sys.argv[2:],stdin=subprocess.DEVNULL,stdout=log,"
    "stderr=subprocess.STDOUT,start_new_session=True)\n"
    "print(p.pid)\n"
)


def launch(rd: RunDir, *, budget_usd: Optional[float] = None,
           usd_per_call: Optional[float] = None,
           env: Optional[Dict[str, str]] = None) -> RunStatus:
    """Start the run as a **detached** process and return at once.

    The run is ``python -m agentdescent.cli run --run-dir <path>``, started by a
    short-lived intermediate so it is a grandchild: in its own session (so it
    survives the parent -- an MCP server the host restarts -- and so
    :func:`cancel` can signal the whole group) and re-parented to init (so it is
    reaped when it ends instead of lingering as a zombie that still looks alive).
    Its stdout and stderr go to ``log.txt``.
    """
    st = rd.status()
    if st.state == "running" and st.alive():
        raise RunStoreError(f"{rd.run_id} is already running (pid {st.pid})")
    argv = [_python(), "-m", "agentdescent.cli", "run", "--run-dir", rd.path]
    if budget_usd is not None:
        argv += ["--budget", str(budget_usd)]
    if usd_per_call is not None:
        argv += ["--usd-per-call", str(usd_per_call)]
    child_env = {**os.environ, "PYTHONPATH": _child_pythonpath(), **(env or {})}
    boot = subprocess.run([_python(), "-c", _BOOTSTRAP, rd.log_path, *argv],
                          capture_output=True, text=True, cwd=rd.path, env=child_env)
    if boot.returncode != 0 or not boot.stdout.strip().isdigit():
        raise RunStoreError(f"could not start the run: {boot.stderr.strip()[:500]}")
    pid = int(boot.stdout.strip())
    return rd.update_status(state="running", pid=pid, started=time.time(),
                            finished=None, error=None, stop_reason=None)


def resume(run_id: str, *, store: Optional[str] = None, **launch_kwargs: Any) -> RunStatus:
    """Re-launch a run on its existing ledger. The engine picks up where it stopped."""
    rd = get(run_id, store=store)
    st = rd.status()
    if st.state == "running" and st.alive():
        raise RunStoreError(f"{run_id} is still running (pid {st.pid})")
    return launch(rd, **launch_kwargs)


def cancel(run_id: str, *, store: Optional[str] = None, grace: float = 5.0) -> RunStatus:
    """SIGTERM the run's process group, then SIGKILL what is left after ``grace``.

    The group, not the pid: ``runners._sh`` gives every worker its own session so
    a timeout can kill *its* children, but the run process itself is the parent of
    the agent CLIs it spawns directly, and a cancel that left eight ``claude -p``
    processes running would not be a cancel.
    """
    rd = get(run_id, store=store)
    st = rd.status()
    if st.pid is None or not st.alive():
        if st.state == "running":
            st = rd.update_status(state="cancelled", finished=time.time())
        return st
    pgid = None
    try:
        pgid = os.getpgid(st.pid)
    except (ProcessLookupError, PermissionError, AttributeError):
        pass
    _signal(st.pid, pgid, signal.SIGTERM)
    deadline = time.time() + grace
    while time.time() < deadline and st.alive():
        time.sleep(0.1)
    if st.alive():
        _signal(st.pid, pgid, getattr(signal, "SIGKILL", signal.SIGTERM))
    return rd.update_status(state="cancelled", finished=time.time(),
                            stop_reason="cancelled")


def _signal(pid: int, pgid: Optional[int], sig: int) -> None:
    try:
        if pgid is not None and hasattr(os, "killpg"):
            os.killpg(pgid, sig)
        else:
            os.kill(pid, sig)
    except ProcessLookupError:
        pass


# ---------------------------------------------------------------------------
# the body of the detached process
# ---------------------------------------------------------------------------


def execute(rd: RunDir, *, budget_usd: Optional[float] = None,
            usd_per_call: Optional[float] = None) -> EvolutionResult:
    """Run the spec in ``rd`` **in this process**, keeping the directory current.

    This is what the detached child calls; it is also usable directly for a
    foreground run (``agentdescent evolve`` without ``--detach``). The engine's
    ``on_round`` hook writes ``status.json`` and ``rounds.jsonl``; the ledger
    lives under the run so a later :func:`resume` continues it; the result and
    the materialised tree are written at the end whatever the outcome.
    """
    from .agents import Usage
    from .evolvespec import EvolveSpec, compose, estimate
    from .filetree import materialize

    spec = EvolveSpec.from_dict(rd.spec_dict())
    usage = Usage()
    rd.update_status(state="running", pid=os.getpid(), started=time.time(),
                     kind=spec.kind, target=spec.target)
    best: List[float] = []

    def on_round(info: RoundInfo) -> None:
        rd.append_round(info)
        best.append(info.held_out_reward)
        changes: Dict[str, Any] = dict(
            round=info.round + 1, last_reward=info.held_out_reward,
            best_reward=max(best), calls=usage.calls)
        st = rd.status()
        changes["committed"] = st.committed + info.committed
        changes["rollouts"] = st.rollouts + info.rollouts
        if usd_per_call is not None:
            changes["usd"] = round(usage.calls * usd_per_call, 4)
        rd.update_status(**changes)

    overrides: Dict[str, Any] = {}
    if budget_usd is not None:
        if not usd_per_call:
            raise RunStoreError("budget_usd needs usd_per_call: the engine counts calls, "
                                "and a tool-using agent's price per call is not known "
                                "in advance")

        def stop_when(info: RoundInfo) -> bool:
            # Asked between rounds, where max_calls is; the run overshoots by at
            # most one round and reports the spend it actually incurred.
            return usage.calls * usd_per_call >= budget_usd

        overrides["stop_when"] = stop_when
    try:
        comp = compose(spec, usage=usage, on_round=on_round, repo_path=rd.ledger_path,
                       **overrides)
        est = estimate(comp, usd_per_call=usd_per_call)
        rd.update_status(rounds=comp.kwargs.get("rounds"), notes=comp.notes,
                         usd=None if usd_per_call is None else 0.0)
        result = comp.run()
    except BaseException as e:  # noqa: BLE001 - the child must report before dying
        rd.update_status(state="failed", finished=time.time(),
                         error=f"{type(e).__name__}: {e}"[:2000])
        raise
    result.save(rd.result_path)
    if comp.tree is not None:
        shutil.rmtree(rd.tree_path, ignore_errors=True)
        materialize(result.state, rd.tree_path)
    rd.update_status(
        state="failed" if result.error else "done", finished=time.time(),
        stop_reason=result.stop_reason, error=result.error,
        best_reward=max(best) if best else result.final_reward,
        last_reward=result.final_reward, calls=usage.calls,
        usd=None if usd_per_call is None else round(usage.calls * usd_per_call, 4))
    return result
