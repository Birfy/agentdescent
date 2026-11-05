"""Layered trajectory evidence: turn raw agent traces into a sourced digest.

A run's trajectory is the richest evidence an evolution loop has, and the least
usable: a coding-agent run easily produces millions of tokens across prompts,
tool calls, errors and scores. A reflector handed that stream either cannot fit
it or silently drops the one event that explains the failure. This module is the
small, deterministic bridge between the two -- the same job AHE's Agent Debugger
does for a harness: **summarise by layer, keep a source link for every claim**.

    events = load_trace_events("runs/<run_id>/trajectories.jsonl")
    digest = digest_traces(events, config=TraceDigestConfig(max_digest_chars=4000))
    print(digest.text)
    for signature in digest.signatures:
        print(signature.id, signature.count, signature.message)
    raw = drilldown(events, signature=digest.signatures[0].id)

The layers are deliberately shallow and inspectable:

* **run** -- how many events, which tasks, whether anything failed;
* **task** -- outcome, reward and the failure signatures it hit;
* **signature** -- a normalised failure mode, its count and the event ids that
  produced it (the drill-down source);
* **text** -- a token-bounded prompt block a reflector can consume, with the
  signature ids it can ask to expand.

No model is called and no network is used. Input order is the only ordering
signal; ``event_id`` is content-addressed when absent, so the same JSONL always
produces the same digest.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .data_flywheel import redact

__all__ = [
    "TRACE_DIGEST_SCHEMA_VERSION",
    "TraceDigestError",
    "TraceEvent",
    "FailureSignature",
    "TraceDigestConfig",
    "TraceDigest",
    "load_trace_events",
    "digest_traces",
    "drilldown",
]

TRACE_DIGEST_SCHEMA_VERSION = 1

_FAILURE_KINDS = frozenset({"error", "exception", "failure", "tool_error"})
_TOOL_KINDS = frozenset({"tool", "tool_call", "tool_result", "call"})
_SUCCESS_REWARD = 0.999
_SECRET_VALUE = re.compile(
    r"\b(sk-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{16,}|"
    r"github_pat_[A-Za-z0-9_]{16,}|AKIA[0-9A-Z]{16}|"
    r"Bearer\s+[A-Za-z0-9._-]{16,})\b")
_NUMBER = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?")
_HEX = re.compile(r"\b(?:0x)?[0-9a-f]{8,}\b", re.IGNORECASE)
_PATH = re.compile(r"(?:[A-Za-z]:)?[\\/](?:[^\s\\/]+[\\/])+[^\s\\/]+")
_URL = re.compile(r"https?://[^\s]+")
_WS = re.compile(r"\s+")


class TraceDigestError(ValueError):
    """A trace input the digest cannot use; the message is safe to print."""


@dataclass(frozen=True)
class TraceEvent:
    """One normalised trajectory event."""

    event_id: str
    run_id: str
    task_id: str
    step: int
    kind: str
    name: str
    text: str
    at: float = 0.0
    reward: Optional[float] = None
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_id or not self.run_id:
            raise TraceDigestError("event_id and run_id are required")
        if self.step < 0:
            raise TraceDigestError("step must be >= 0")
        if not math.isfinite(float(self.at)):
            raise TraceDigestError("at must be finite")
        if self.reward is not None:
            value = float(self.reward)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise TraceDigestError("reward must be in [0, 1] or None")
            object.__setattr__(self, "reward", value)
        object.__setattr__(self, "payload", dict(self.payload))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "step": self.step,
            "kind": self.kind,
            "name": self.name,
            "text": self.text,
            "at": self.at,
            "reward": self.reward,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class FailureSignature:
    """One normalised failure mode and the evidence behind it."""

    id: str
    kind: str
    name: str
    message: str
    count: int
    run_ids: Tuple[str, ...]
    task_ids: Tuple[str, ...]
    event_ids: Tuple[str, ...]
    sample: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "name": self.name,
            "message": self.message,
            "count": self.count,
            "run_ids": list(self.run_ids),
            "task_ids": list(self.task_ids),
            "event_ids": list(self.event_ids),
            "sample": self.sample,
        }


@dataclass(frozen=True)
class TraceDigestConfig:
    """Bounds for one digest. All limits are upper bounds, never targets."""

    max_signatures: int = 20
    max_sources_per_signature: int = 20
    max_tasks_per_run: int = 50
    max_event_chars: int = 600
    max_sample_chars: int = 240
    max_digest_chars: int = 4000
    failure_reward: float = 0.5
    redact_secrets: bool = True

    def __post_init__(self) -> None:
        for name in ("max_signatures", "max_sources_per_signature",
                     "max_tasks_per_run", "max_event_chars",
                     "max_sample_chars", "max_digest_chars"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise TraceDigestError(f"{name} must be an int >= 1")
        value = float(self.failure_reward)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise TraceDigestError("failure_reward must be in [0, 1]")
        object.__setattr__(self, "failure_reward", value)


@dataclass(frozen=True)
class TraceDigest:
    """The layered evidence corpus for one trace."""

    schema_version: int
    runs: Tuple[Dict[str, Any], ...]
    tasks: Tuple[Dict[str, Any], ...]
    signatures: Tuple[FailureSignature, ...]
    sources: Mapping[str, Tuple[str, ...]]
    summary: str
    text: str
    events_total: int
    events_used: int
    truncated: bool
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "runs": [dict(row) for row in self.runs],
            "tasks": [dict(row) for row in self.tasks],
            "signatures": [row.to_dict() for row in self.signatures],
            "sources": {key: list(value) for key, value in self.sources.items()},
            "summary": self.summary,
            "text": self.text,
            "events_total": self.events_total,
            "events_used": self.events_used,
            "truncated": self.truncated,
            "detail": self.detail,
        }


def _stable_id(prefix: str, *parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return prefix + hashlib.sha1(payload.encode()).hexdigest()[:12]


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _clean_text(value: Any, limit: int, *, redact_secrets: bool) -> str:
    if isinstance(value, (Mapping, list, tuple)):
        text = json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True)
    else:
        text = "" if value is None else str(value)
    if redact_secrets:
        text = _SECRET_VALUE.sub("[REDACTED]", text)
    text = _WS.sub(" ", text).strip()
    if len(text) > limit:
        text = text[: max(0, limit - 1)].rstrip() + "…"
    return text


def _text_field(row: Mapping[str, Any]) -> Any:
    for key in ("text", "output", "message", "content", "error",
                "prompt", "completion", "result", "detail"):
        if key in row and row[key] is not None:
            return row[key]
    return row.get("payload")


def _kind_field(row: Mapping[str, Any]) -> str:
    raw = row.get("kind") or row.get("type") or row.get("event") or "note"
    kind = str(raw).strip().lower().replace("-", "_")
    if kind in ("tool_call", "tool_result", "tool_use"):
        return "tool"
    if kind in ("llm", "model", "generation", "agent"):
        return "completion"
    if kind in ("failed", "tool_error", "tool_failure"):
        return "error"
    return kind or "note"


def _name_field(row: Mapping[str, Any]) -> str:
    for key in ("name", "tool", "tool_name", "action", "category", "command"):
        value = row.get(key)
        if value:
            return str(value).strip()
    payload = row.get("payload")
    if isinstance(payload, Mapping):
        for key in ("tool", "name", "tool_name", "command", "action"):
            value = payload.get(key)
            if value:
                return str(value).strip()
    return ""


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    number = float(value)
    return number if math.isfinite(number) else default


def _reward(row: Mapping[str, Any]) -> Optional[float]:
    for key in ("reward", "score"):
        if key in row and row[key] is not None:
            if isinstance(row[key], bool) or not isinstance(row[key], (int, float)):
                continue
            number = float(row[key])
            if math.isfinite(number) and 0.0 <= number <= 1.0:
                return number
    payload = row.get("payload")
    if isinstance(payload, Mapping):
        for key in ("reward", "score"):
            if key in payload and payload[key] is not None:
                if isinstance(payload[key], bool) or not isinstance(payload[key], (int, float)):
                    continue
                number = float(payload[key])
                if math.isfinite(number) and 0.0 <= number <= 1.0:
                    return number
    return None


def _is_failure(row: Mapping[str, Any], config: TraceDigestConfig) -> bool:
    kind = _kind_field(row)
    if kind in _FAILURE_KINDS:
        return True
    payload = row.get("payload")
    if isinstance(payload, Mapping):
        status = str(payload.get("status") or payload.get("outcome") or "").lower()
        if status in ("error", "failed", "failure", "timeout"):
            return True
        if payload.get("error") or payload.get("ok") is False:
            return True
    reward = _reward(row)
    return reward is not None and reward < config.failure_reward


def _normalise_message(row: Mapping[str, Any], config: TraceDigestConfig) -> str:
    text = _clean_text(_text_field(row), config.max_event_chars,
                       redact_secrets=config.redact_secrets)
    text = _URL.sub("<url>", text)
    text = _PATH.sub("<path>", text)
    text = _HEX.sub("<hex>", text)
    text = _NUMBER.sub("#", text)
    text = text.lower().strip(" .:;,-")
    return text or "(no message)"


def _event_id(row: Mapping[str, Any], index: int) -> str:
    raw = row.get("event_id") or row.get("id")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return _stable_id("evt_", index, row.get("run_id", ""), row.get("task_id", ""),
                      row.get("step"), _kind_field(row), _name_field(row),
                      _text_field(row))


def _event(row: Mapping[str, Any], index: int, config: TraceDigestConfig) -> TraceEvent:
    run_id = row.get("run_id", row.get("run", "run"))
    task_id = row.get("task_id", row.get("task", ""))
    step = row.get("step", row.get("index", 0))
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        step = 0
    payload = row.get("payload")
    if not isinstance(payload, Mapping):
        payload = {}
    clean_payload = redact(dict(payload)) if config.redact_secrets else dict(payload)
    text_row = dict(row)
    text_row["payload"] = clean_payload
    return TraceEvent(
        event_id=_event_id(row, index),
        run_id=str(run_id) if run_id is not None else "run",
        task_id="" if task_id is None else str(task_id),
        step=step,
        kind=_kind_field(row),
        name=_name_field(row),
        text=_clean_text(_text_field(text_row), config.max_event_chars,
                         redact_secrets=config.redact_secrets),
        at=_number(row.get("at", row.get("timestamp", 0.0))),
        reward=_reward(row),
        payload=clean_payload,
    )


def _rows(value: Any) -> List[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if "events" in value:
            value = value["events"]
        elif any(key in value for key in ("run_id", "kind", "text", "payload")):
            value = [value]
        else:
            raise TraceDigestError("a trace object needs an 'events' list")
    if isinstance(value, (str, bytes, os.PathLike)):
        path = os.path.abspath(os.path.expanduser(os.fspath(value)))
        try:
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
        except OSError as error:
            raise TraceDigestError(f"trace {value!r} cannot be read: {error}") from None
        stripped = text.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            try:
                value = json.loads(stripped)
            except ValueError as error:
                raise TraceDigestError(
                    f"trace {value!r} is not valid JSON: {error}") from None
        else:
            rows = []
            for line_no, line in enumerate(text.splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError as error:
                    raise TraceDigestError(
                        f"trace {value!r} line {line_no} is not valid JSON: "
                        f"{error}") from None
                if not isinstance(row, Mapping):
                    raise TraceDigestError(
                        f"trace {value!r} line {line_no} is not an object")
                rows.append(row)
            value = rows
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TraceDigestError("trace events must be a list, an object or a path")
    rows = []
    for index, row in enumerate(value):
        if not isinstance(row, Mapping):
            raise TraceDigestError(f"trace event {index} is not an object")
        rows.append(row)
    return rows


def load_trace_events(value: Any, *,
                      config: Optional[TraceDigestConfig] = None
                      ) -> Tuple[TraceEvent, ...]:
    """Load JSONL/JSON/list trace rows into normalised :class:`TraceEvent`."""
    config = config or TraceDigestConfig()
    return tuple(_event(row, index, config)
                 for index, row in enumerate(_rows(value)))


def _signature(event: TraceEvent, config: TraceDigestConfig) -> FailureSignature:
    message = _normalise_message(event.to_dict(), config)
    name = event.name
    sid = _stable_id("sig_", event.kind, name, message)
    return FailureSignature(
        id=sid, kind=event.kind, name=name, message=message, count=1,
        run_ids=(event.run_id,), task_ids=(event.task_id,),
        event_ids=(event.event_id,), sample=event.text[:config.max_sample_chars])


def _merge_signatures(signatures: Dict[str, Dict[str, Any]],
                      event: TraceEvent, config: TraceDigestConfig) -> None:
    row = _signature(event, config)
    entry = signatures.setdefault(row.id, {
        "id": row.id, "kind": row.kind, "name": row.name, "message": row.message,
        "count": 0, "run_ids": set(), "task_ids": set(), "event_ids": [],
        "sample": row.sample,
    })
    entry["count"] += 1
    if event.run_id:
        entry["run_ids"].add(event.run_id)
    if event.task_id:
        entry["task_ids"].add(event.task_id)
    if len(entry["event_ids"]) < config.max_sources_per_signature:
        entry["event_ids"].append(event.event_id)


def _task_outcome(events: Sequence[TraceEvent], config: TraceDigestConfig) -> str:
    if any(_is_failure(e.to_dict(), config) for e in events):
        return "failed"
    rewards = [r for e in events for r in [_reward(e.to_dict())] if r is not None]
    if rewards:
        return "succeeded" if max(rewards) >= _SUCCESS_REWARD else "partial"
    return "unknown"


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def digest_traces(events: Iterable[TraceEvent], *,
                  config: Optional[TraceDigestConfig] = None) -> TraceDigest:
    """Summarise traces by run, task and normalised failure signature."""
    config = config or TraceDigestConfig()
    if not isinstance(config, TraceDigestConfig):
        raise TraceDigestError("config must be a TraceDigestConfig")
    rows = tuple(events)
    for event in rows:
        if not isinstance(event, TraceEvent):
            raise TraceDigestError("digest_traces expects TraceEvent objects")

    runs: Dict[str, Dict[str, Any]] = {}
    tasks: Dict[Tuple[str, str], List[TraceEvent]] = {}
    signatures: Dict[str, Dict[str, Any]] = {}
    for event in rows:
        run = runs.setdefault(event.run_id, {
            "run_id": event.run_id, "events": 0, "tasks": set(),
            "failures": 0, "rewards": [], "start": event.at, "end": event.at,
        })
        run["events"] += 1
        if event.task_id:
            run["tasks"].add(event.task_id)
        run["start"] = min(run["start"], event.at)
        run["end"] = max(run["end"], event.at)
        reward = _reward(event.to_dict())
        if reward is not None:
            run["rewards"].append(reward)
        if _is_failure(event.to_dict(), config):
            run["failures"] += 1
            _merge_signatures(signatures, event, config)
        tasks.setdefault((event.run_id, event.task_id), []).append(event)

    task_rows = []
    per_run: Dict[str, int] = {}
    tasks_truncated = False
    for (run_id, task_id), group in sorted(tasks.items()):
        count = per_run.get(run_id, 0)
        if count >= config.max_tasks_per_run:
            tasks_truncated = True
            continue
        per_run[run_id] = count + 1
        rewards = [r for e in group for r in [_reward(e.to_dict())] if r is not None]
        task_rows.append({
            "run_id": run_id,
            "task_id": task_id,
            "events": len(group),
            "outcome": _task_outcome(group, config),
            "reward": max(rewards) if rewards else None,
            "failures": sum(1 for e in group
                            if _is_failure(e.to_dict(), config)),
            "event_ids": [e.event_id for e in group[:config.max_sources_per_signature]],
        })

    ordered = sorted(
        signatures.values(),
        key=lambda row: (-row["count"], row["kind"], row["name"], row["id"]))
    signature_truncated = len(ordered) > config.max_signatures
    signature_rows = tuple(
        FailureSignature(
            id=row["id"], kind=row["kind"], name=row["name"],
            message=row["message"], count=row["count"],
            run_ids=tuple(sorted(row["run_ids"])),
            task_ids=tuple(sorted(row["task_ids"])),
            event_ids=tuple(row["event_ids"]), sample=row["sample"])
        for row in ordered[:config.max_signatures])

    run_rows = tuple({
        "run_id": run_id,
        "events": row["events"],
        "tasks": len(row["tasks"]),
        "failures": row["failures"],
        "reward": max(row["rewards"]) if row["rewards"] else None,
        "start": row["start"],
        "end": row["end"],
    } for run_id, row in sorted(runs.items()))

    sources = {row.id: row.event_ids for row in signature_rows}
    summary = (
        f"{len(run_rows)} run(s), {len(task_rows)} task(s), "
        f"{len(rows)} event(s); {len(signature_rows)} failure signature(s)")
    if signature_rows:
        top = signature_rows[0]
        summary += f"; top: {top.kind}" + (
            f"/{top.name}" if top.name else "") + f" x{top.count}"
    lines = [f"trace digest: {summary}"]
    for row in run_rows:
        lines.append(
            f"run {row['run_id']}: {row['events']} events, {row['tasks']} tasks, "
            f"{row['failures']} failures, reward={row['reward']}")
    if signature_rows:
        lines.append("failure signatures:")
        for row in signature_rows:
            name = f"{row.kind}/{row.name}" if row.name else row.kind
            lines.append(
                f"- {row.id} {name} x{row.count}: {row.message}"
                + (f" | sample: {row.sample}" if row.sample else ""))
    text = _truncate("\n".join(lines), config.max_digest_chars)
    return TraceDigest(
        schema_version=TRACE_DIGEST_SCHEMA_VERSION,
        runs=run_rows, tasks=tuple(task_rows), signatures=signature_rows,
        sources=sources, summary=summary, text=text,
        events_total=len(rows), events_used=len(rows),
        truncated=(len(text) < len("\n".join(lines))
                   or signature_truncated or tasks_truncated),
        detail="event text is truncated per max_event_chars; signatures keep "
               "source event ids for drill-down")


def drilldown(events: Iterable[TraceEvent], *, run_id: Optional[str] = None,
              task_id: Optional[str] = None, signature: Optional[str] = None,
              limit: int = 50,
              config: Optional[TraceDigestConfig] = None) -> Dict[str, Any]:
    """Return the raw events behind a run, a task or one failure signature."""
    config = config or TraceDigestConfig()
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise TraceDigestError("limit must be an int >= 1")
    rows = tuple(events)
    if signature is not None:
        selected = [
            event for event in rows
            if _is_failure(event.to_dict(), config)
            and _signature(event, config).id == signature]
        if not selected:
            raise TraceDigestError(f"unknown signature {signature!r}")
    else:
        selected = [
            event for event in rows
            if (run_id is None or event.run_id == run_id)
            and (task_id is None or event.task_id == task_id)]
    selected = selected[:limit]
    return {
        "schema_version": TRACE_DIGEST_SCHEMA_VERSION,
        "run_id": run_id,
        "task_id": task_id,
        "signature": signature,
        "count": len(selected),
        "events": [event.to_dict() for event in selected],
    }
