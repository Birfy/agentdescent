"""Human approval for high-risk self-modifications.

Automatic gates are good at local edits and bad at judging a change that rewrites
the harness, the verifier or a large part of the artifact. This module turns that
risk into a queue: a high-risk candidate is *not* committed, a request is written
to a JSONL store, and a human approves or rejects it out of band. The next merge
of the same diff reads the decision.

    Policies(acceptance=ApprovalGate(store, risk_threshold=0.5))

``risk_threshold`` compares the artifact's ``blast_radius`` (L1/L0 governance);
``ops_threshold`` is a second, size-based trigger. Low-risk candidates delegate
straight to the inner acceptance policy. A pending request rejects the merge with
``category="needs-approval"``; an explicit rejection is ``approval-rejected``.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .advantage import _ForwardsInstall
from .policies import AcceptDecision, MergeContext

__all__ = ["ApprovalGate", "ApprovalRequest", "ApprovalStore"]


@dataclass(frozen=True)
class ApprovalRequest:
    """One high-risk candidate's approval request and its latest decision."""

    key: str
    artifact_id: str
    reason: str
    status: str          # "pending" / "approved" / "rejected"
    created_at: float
    decided_at: Optional[float] = None
    note: str = ""
    summary: str = ""


_LOCKS: Dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(path: str) -> threading.Lock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(path, threading.Lock())


class ApprovalStore:
    """Append-only JSONL decisions for high-risk candidates.

    Records are immutable events; ``status`` is the latest event for a key.
    A pending request is written once; re-offering the same content does not
    grow the file. The store is intentionally file-based: a human or another
    process can inspect and decide without a server. Within one process writers
    on the same path share a lock; separate processes rely on single-line
    ``O_APPEND`` writes and should not interleave a hand edit with the store.
    """

    def __init__(self, path: str) -> None:
        self.path = os.path.abspath(os.path.expanduser(path))
        self._lock = _lock_for(self.path)

    def _events(self) -> List[Dict[str, Any]]:
        try:
            with open(self.path, encoding="utf-8") as fh:
                events = []
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except (ValueError, TypeError):
                        continue
                    if isinstance(event, dict):
                        events.append(event)
                return events
        except FileNotFoundError:
            return []

    def _append(self, payload: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def status(self, key: str) -> Optional[str]:
        request = self._requests().get(key)
        return request.status if request is not None else None

    def request(self, key: str, artifact_id: str, reason: str,
                summary: str = "") -> ApprovalRequest:
        with self._lock:
            existing = self._requests().get(key)
            if existing is not None:
                return existing
            created = time.time()
            self._append({"event": "request", "key": key,
                          "artifact_id": artifact_id, "reason": reason,
                          "summary": summary, "created_at": created})
            return ApprovalRequest(key, artifact_id, reason, "pending",
                                   created, summary=summary)

    def decide(self, key: str, *, approve: bool, note: str = "") -> None:
        with self._lock:
            if key not in self._requests():
                raise KeyError(f"no approval request for key {key!r}")
            self._append({"event": "approve" if approve else "reject",
                          "key": key, "note": note, "decided_at": time.time()})

    def _requests(self) -> Dict[str, ApprovalRequest]:
        requests: Dict[str, Dict[str, Any]] = {}
        decisions: Dict[str, Dict[str, Any]] = {}
        for event in self._events():
            key = event.get("key")
            if not isinstance(key, str) or not key:
                continue
            if event.get("event") == "request":
                requests[key] = event
            elif event.get("event") in ("approve", "reject"):
                decisions[key] = event
        out = {}
        for key, request in requests.items():
            decision = decisions.get(key)
            out[key] = ApprovalRequest(
                key=key, artifact_id=str(request.get("artifact_id", "")),
                reason=str(request.get("reason", "")),
                status=("approved" if decision and decision["event"] == "approve"
                        else "rejected" if decision else "pending"),
                created_at=float(request.get("created_at", 0.0) or 0.0),
                decided_at=(float(decision.get("decided_at", 0.0) or 0.0)
                            if decision else None),
                note=str(decision.get("note", "") if decision else ""),
                summary=str(request.get("summary", "")))
        return out

    def pending(self) -> List[ApprovalRequest]:
        return sorted((request for request in self._requests().values()
                       if request.status == "pending"),
                      key=lambda request: request.created_at)

    def all(self) -> List[ApprovalRequest]:
        return sorted(self._requests().values(),
                      key=lambda request: request.created_at)


class ApprovalGate(_ForwardsInstall):
    """Require a human decision for changes above a risk/size threshold."""

    def __init__(self, store: ApprovalStore, *, inner: Optional[Any] = None,
                 risk_threshold: float = 0.5, ops_threshold: int = 6) -> None:
        if (isinstance(risk_threshold, bool)
                or not isinstance(risk_threshold, (int, float))
                or not 0.0 <= float(risk_threshold) <= 1.0):
            raise ValueError("risk_threshold must be in [0, 1]")
        if isinstance(ops_threshold, bool) or not isinstance(ops_threshold, int) \
                or ops_threshold < 1:
            raise ValueError("ops_threshold must be an int >= 1")
        from .defaults import DefaultAcceptance

        self.store = store
        self.inner = inner if inner is not None else DefaultAcceptance()
        self.risk_threshold = float(risk_threshold)
        self.ops_threshold = int(ops_threshold)

    def _key(self, ctx: MergeContext) -> str:
        """Fingerprint what the human is reviewing, not the producer's id.

        ``diff_id`` is often ``author:key:base_version`` and identical text can
        carry different ops. Binding the key to target + ops + base version
        means an approval cannot be replayed for a different change.
        """
        diff = ctx.diff
        if diff is None:
            payload = {"artifact": getattr(ctx.candidate, "id", "unknown"),
                       "base_version": getattr(ctx.artifact, "version", None)}
        else:
            payload = {"target": getattr(diff, "target", ""),
                       "ops": getattr(diff, "ops", {}) or {},
                       "base_version": getattr(ctx.artifact, "version", None)}
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()
        return "diff:" + digest[:32]

    def _summary(self, ctx: MergeContext) -> str:
        diff = ctx.diff
        ops = getattr(diff, "ops", {}) or {} if diff is not None else {}
        return json.dumps(ops, sort_keys=True, default=str)[:600]

    def _risky(self, ctx: MergeContext) -> bool:
        diff = ctx.diff
        ops = len(getattr(diff, "ops", {}) or {}) if diff is not None else 0
        blast = float(getattr(ctx, "trust_radius", 0.0) or 0.0)
        return blast >= self.risk_threshold or ops >= self.ops_threshold

    def accept(self, ctx: MergeContext) -> AcceptDecision:
        if not self._risky(ctx):
            return (self.inner.accept(ctx) if hasattr(self.inner, "accept")
                    else self.inner(ctx))
        key = self._key(ctx)
        status = self.store.status(key)
        if status == "approved":
            return (self.inner.accept(ctx) if hasattr(self.inner, "accept")
                    else self.inner(ctx))
        if status == "rejected":
            return AcceptDecision(False, "approval-rejected",
                                  f"human rejected {key}")
        self.store.request(key, str(getattr(ctx.diff, "target", "")),
                           "high-risk self-modification",
                           summary=self._summary(ctx))
        return AcceptDecision(False, "needs-approval",
                              f"human approval required for {key}")
