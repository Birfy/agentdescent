"""Meta-campaign: continuous meta-learning as a bounded, multi-session loop.

:func:`~agentdescent.meta.continuous_meta_run` makes one session durable -- it
evolves a decision slot, validates it on unseen problems, and archives the
adopted rule. :class:`~agentdescent.rsi_control.RSIController` owns the bounded
loop around a sequence of sessions: target, cost, patience, and session count.
:class:`~agentdescent.consolidation.ConsolidationStore` is the memory that makes
one session seed the next.

This module composes those three pieces without reimplementing any of them::

    report = meta_campaign_run(
        store, "task_sampler", spec,
        config=MetaCampaignConfig(max_sessions=4, target=0.9, patience=2),
        tasks_for=tasks_for, run=run, reward=reward,
        inner_propose=propose, inner_strategy=SingleSlot("cap:0"),
        problems={"a": problem_a, "b": problem_b},
        val_problems={"z": problem_z})

Each session is a fresh ``continuous_meta_run`` against the same archive and a
fresh ledger repo (``<repo_dir>/session-<index>``), because an engine resuming a
prior ledger would ignore the archive's seed and turn the campaign into a
replay. A session that raises is reported as a zero-reward error and stops the
campaign: a failed session is never promoted as if it had measured something.
A rejected rule is an ordinary measured session and the loop continues until the
controller's own stop condition fires.
"""

from __future__ import annotations

import inspect
import math
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .consolidation import ConsolidationStore
from .meta import ContinuousMetaReport, SlotSpec, continuous_meta_run
from .rsi_control import ControlConfig, ControlReport, RSIController, SessionResult

__all__ = ["MetaCampaignConfig", "MetaCampaignReport", "meta_campaign_run"]

_REAL_CONTINUOUS_META_RUN = continuous_meta_run


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, not bool")
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def _integer(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an int, not bool")
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}")
    return value


@dataclass(frozen=True)
class MetaCampaignConfig:
    """Bounds for one :func:`meta_campaign_run`.

    ``target``, ``max_cost``, ``min_gain`` and ``patience`` are
    :class:`~agentdescent.rsi_control.ControlConfig`'s stop conditions.
    ``max_sessions`` caps the campaign even when no other stop fires, and
    ``base_seed`` is the controller's seed for session ``index``
    (``base_seed + index``).
    """

    max_sessions: int = 3
    target: Optional[float] = None
    max_cost: Optional[float] = None
    min_gain: float = 0.0
    patience: int = 1
    base_seed: int = 0

    def __post_init__(self) -> None:
        _integer(self.max_sessions, "max_sessions", 1)
        if self.target is not None:
            if _finite(self.target, "target") < 0:
                raise ValueError("target must be >= 0 or None")
        if self.max_cost is not None:
            if _finite(self.max_cost, "max_cost") <= 0:
                raise ValueError("max_cost must be > 0 or None")
        if _finite(self.min_gain, "min_gain") < 0:
            raise ValueError("min_gain must be >= 0")
        _integer(self.patience, "patience", 1)
        _integer(self.base_seed, "base_seed", 0)

    def control_config(self) -> ControlConfig:
        """The :class:`ControlConfig` these bounds compile to."""
        return ControlConfig(
            target=self.target, max_sessions=self.max_sessions,
            max_cost=self.max_cost, min_gain=self.min_gain,
            patience=self.patience)


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_value(item) for item in value]
    return str(value)


def _control_dict(control: ControlReport) -> Dict[str, Any]:
    return {
        "decisions": [{
            "index": decision.index,
            "action": decision.action,
            "reason": decision.reason,
            "reward": _json_value(decision.reward),
            "gain": _json_value(decision.gain),
            "cost": _json_value(decision.cost),
            "budget_scale": _json_value(decision.budget_scale),
            "stop": bool(decision.stop),
            "evidence": _json_value(dict(decision.evidence)),
        } for decision in control.decisions],
        "best_reward": _json_value(control.best_reward),
        "total_cost": _json_value(control.total_cost),
        "accepted": int(control.accepted),
        "stopped_reason": control.stopped_reason,
        "reason": control.reason,
    }


@dataclass(frozen=True)
class MetaCampaignReport:
    """What one :func:`meta_campaign_run` did, and why.

    ``sessions`` holds the successful :class:`ContinuousMetaReport` rows in the
    order they ran. ``best_session`` is the index of the highest ``mean_gain``
    (the first session on a tie). ``control`` is the controller's decision log
    and stop reason, and ``archive_count`` is the archive size after the last
    session. ``complete`` is true only when a session ran, the controller did not
    stop on an error, and no session raised.
    """

    sessions: Tuple[ContinuousMetaReport, ...]
    control: ControlReport
    best_session: Optional[int]
    archive_count: int
    complete: bool
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-safe view of the campaign and its per-session decisions."""
        return {
            "sessions": [{
                "index": index,
                "verdict": report.verdict,
                "mean_gain": _json_value(report.mean_gain),
                "fresh_gain": _json_value(report.fresh_gain),
                "seeded_from": report.seeded_from,
                "archived_run_id": report.archived_run_id,
                "archived": bool(report.archived),
            } for index, report in enumerate(self.sessions)],
            "control": _control_dict(self.control),
            "best_session": self.best_session,
            "archive_count": self.archive_count,
            "complete": self.complete,
            "detail": self.detail,
        }


def _repo_keyword(session_dir: str) -> Dict[str, str]:
    """The ledger keyword ``continuous_meta_run`` understands here.

    The real function forwards this to ``evolve`` as ``repo_path``. A test
    double injected for a campaign receives the campaign's own ``repo_dir``
    spelling, as the composition contract names it.
    """
    function = continuous_meta_run
    if function is _REAL_CONTINUOUS_META_RUN:
        return {"repo_path": session_dir}
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return {"repo_dir": session_dir}
    if "repo_dir" in parameters:
        return {"repo_dir": session_dir}
    if "repo_path" in parameters:
        return {"repo_path": session_dir}
    return {"repo_dir": session_dir}


def _detail(sessions: List[ContinuousMetaReport], control: ControlReport,
            errors: List[Tuple[int, str]]) -> str:
    parts = [f"{len(sessions)} session(s) recorded",
             f"stopped: {control.stopped_reason or 'unknown'}"]
    if control.reason:
        parts.append(control.reason)
    for index, message in errors:
        parts.append(f"session {index} error: {message}")
    return "; ".join(parts)


def meta_campaign_run(
    store: ConsolidationStore,
    slot: str,
    spec: SlotSpec,
    *,
    config: Optional[MetaCampaignConfig] = None,
    run_id: Optional[str] = None,
    repo_dir: Optional[str] = None,
    **kwargs: Any,
) -> MetaCampaignReport:
    """Run continuous meta-learning as a bounded multi-session campaign.

    Each session calls :func:`~agentdescent.meta.continuous_meta_run` with its
    own ``run_id`` and its own fresh ledger directory
    (``<repo_dir>/session-<index>``), against the same
    :class:`~agentdescent.consolidation.ConsolidationStore`. The archive is not
    reimplemented: a session's adopted rule is what the next session seeds from,
    exactly as it is outside this loop.

    Parameters
    ----------
    store:
        The durable archive shared by every session.
    slot, spec:
        The decision slot being evolved and how its value is represented.
    config:
        The campaign's bounds. ``None`` uses :class:`MetaCampaignConfig`'s
        defaults.
    run_id:
        The base for each session's archived ``run_id``; the session id is
        ``<base>-<index>-<uuid4 hex>``. When omitted the base is
        ``meta-campaign-<slot>-<unix time>``.
    repo_dir:
        Base directory for the per-session ledger repos. It is created when
        given; when omitted a temporary campaign directory is used.
    **kwargs:
        The domain functions and settings ``continuous_meta_run`` needs --
        ``tasks_for``, ``run``, ``reward``, ``inner_propose``,
        ``inner_strategy``, ``propose`` / ``model``, ``problems``,
        ``val_problems``, and so on. ``run_id``, ``repo_dir`` and ``repo_path``
        are set by the campaign, never by the caller.

    Returns
    -------
    MetaCampaignReport
        The successful sessions in order, the controller's report, the index of
        the best session, the archive size, and whether the campaign completed
        without an error.
    """
    if config is None:
        config = MetaCampaignConfig()
    elif not isinstance(config, MetaCampaignConfig):
        raise TypeError("config must be a MetaCampaignConfig or None")
    for taken in ("run_id", "repo_dir", "repo_path"):
        if taken in kwargs:
            raise TypeError(
                f"meta_campaign_run() sets {taken}= itself -- pass the campaign "
                "run_id= or repo_dir= instead")
    if run_id is not None:
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-empty string or None")
        base = run_id
    else:
        base = f"meta-campaign-{slot}-{int(time.time())}"
    if repo_dir is not None:
        try:
            base_repo = os.fspath(repo_dir)
        except TypeError:
            raise TypeError("repo_dir must be a path-like string or None")
        if isinstance(base_repo, bytes):
            base_repo = os.fsdecode(base_repo)
        os.makedirs(base_repo, exist_ok=True)
    else:
        base_repo = tempfile.mkdtemp(prefix="agentdescent-meta-campaign-")

    sessions: List[ContinuousMetaReport] = []
    errors: List[Tuple[int, str]] = []

    def run_session(index: int, seed: int) -> SessionResult:
        rid = f"{base}-{index}-{uuid.uuid4().hex}"
        session_dir = os.path.join(base_repo, f"session-{index}")
        try:
            os.makedirs(session_dir, exist_ok=True)
            outcome = continuous_meta_run(
                store, slot, spec, run_id=rid,
                **_repo_keyword(session_dir), **kwargs)
            report = outcome[0] if isinstance(outcome, tuple) else outcome
            if report is None:
                raise TypeError("continuous_meta_run returned no report")
            result = SessionResult(
                reward=float(report.mean_gain), cost=0.0, error=None)
        except Exception as error:  # noqa: BLE001 - one session must not pass silently
            message = f"{type(error).__name__}: {error}"
            errors.append((index, message))
            return SessionResult(0.0, error=message)
        sessions.append(report)
        return result

    control = RSIController(config.control_config()).run(
        run_session, base_seed=config.base_seed)

    best_session: Optional[int] = None
    best_gain = -math.inf
    for index, report in enumerate(sessions):
        gain = float(report.mean_gain)
        if gain > best_gain:
            best_gain = gain
            best_session = index

    return MetaCampaignReport(
        sessions=tuple(sessions), control=control, best_session=best_session,
        archive_count=store.count(),
        complete=bool(sessions) and control.stopped_reason != "error"
        and not errors,
        detail=_detail(sessions, control, errors))
