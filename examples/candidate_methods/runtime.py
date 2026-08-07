"""Thread-safe model measurement helpers for the candidate-method ports.

Scheduling deliberately does not live here. Serial, synchronous-parallel, and
barrier-free asynchronous execution are provided by AgentDescent's ``evolve``
and ``async_evolve`` runtimes in :mod:`examples.candidate_methods.framework`.
"""

from __future__ import annotations

import json
import statistics
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, TypeVar

from agentdescent.agents import Completion, Usage


MODES = ("serial", "sync_parallel", "async_pipeline")
T = TypeVar("T")


@dataclass(frozen=True)
class CallEvent:
    phase: str
    unit: str
    started_s: float
    ended_s: float
    success: bool

    @property
    def duration_s(self) -> float:
        return self.ended_s - self.started_s


class Recorder:
    """Record live provider calls without retaining prompts or responses."""

    def __init__(self, completion: Completion, usage: Usage) -> None:
        self.completion = completion
        self.usage = usage
        self.started = time.monotonic()
        self.events: List[CallEvent] = []
        self._lock = threading.Lock()

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def call(self, prompt: str, *, phase: str, unit: str) -> str:
        started = self.elapsed()
        success = False
        try:
            response = self.completion(prompt).strip()
            success = True
            return response
        finally:
            ended = self.elapsed()
            with self._lock:
                self.events.append(CallEvent(phase, unit, started, ended, success))

    def phase_summary(self) -> Dict[str, Dict[str, float]]:
        grouped: Dict[str, List[CallEvent]] = {}
        for event in self.events:
            grouped.setdefault(event.phase, []).append(event)
        return {
            phase: {
                "calls": len(rows),
                "wall_span_s": max(row.ended_s for row in rows)
                - min(row.started_s for row in rows),
                "sum_call_s": sum(row.duration_s for row in rows),
            }
            for phase, rows in sorted(grouped.items())
        }


def parse_json_object(text: str) -> Dict[str, Any]:
    """Decode the first JSON object, tolerating prose and Markdown fences."""
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("model response did not contain a JSON object")


def extract_python(text: str) -> str:
    """Extract one fenced Python block, or return an unfenced response."""
    marker = "```"
    if marker not in text:
        return text.strip()
    chunks = text.split(marker)
    for index in range(1, len(chunks), 2):
        block = chunks[index].strip()
        if block.lower().startswith("python"):
            return block[6:].lstrip("\r\n ")
    return chunks[1].strip()


def interval(values: Iterable[float]) -> List[float]:
    rows = list(values)
    if not rows:
        return []
    return [min(rows), statistics.median(rows), max(rows)]


def usage_dict(usage: Usage) -> Dict[str, float]:
    return {
        "calls": usage.calls,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "model_seconds": usage.seconds,
        "failures": usage.failures,
    }


def compact_events(events: Sequence[CallEvent]) -> List[Dict[str, Any]]:
    return [
        {
            "phase": event.phase,
            "unit": event.unit,
            "started_s": event.started_s,
            "ended_s": event.ended_s,
            "duration_s": event.duration_s,
            "success": event.success,
        }
        for event in sorted(events, key=lambda row: row.started_s)
    ]


def rotate(items: Sequence[T], amount: int) -> List[T]:
    if not items:
        return []
    shift = amount % len(items)
    return list(items[shift:]) + list(items[:shift])
