"""Synthesize, validate and select tools instead of hand-writing every skill.

Open-ended agents eventually need a tool that does not exist in the catalog.
This module is the product seam:

* :class:`ToolSpec` describes a candidate tool (source, capabilities, schemas);
* a caller-supplied generator proposes candidates and receives the candidates
  (including rejection reasons) produced earlier in the same synthesis;
* a caller-supplied validator (sandbox, tests, capability gate, static checks)
  scores them;
* :class:`ToolSynthesizer` keeps the best valid candidate and records every
  rejection so the next generation can learn from it.

It does not execute arbitrary code by itself. `validator` is where the sandbox,
`capability_guard`/`safe_selfmod` and property tests live; this module owns the
generate/validate/select loop and the audit trail.

A generator returning ``None`` means "no attempt for this generation"; only
real specs (and generator/validator errors) enter the candidate audit trail.

    synth = ToolSynthesizer(generate, validate)
    report = synth.synthesize(ToolSynthesisRequest(goal="parse a CSV"))
    report.selected.spec.source
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "ToolSpec",
    "ToolValidation",
    "ToolCandidate",
    "ToolSynthesisRequest",
    "ToolSynthesisReport",
    "ToolSynthesizer",
    "EnvironmentSpec",
]

Generator = Callable[["ToolSynthesisRequest", int, Tuple["ToolCandidate", ...]],
                     Optional["ToolSpec"]]
Validator = Callable[["ToolSpec"], Any]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    source: str
    language: str = "python"
    capabilities: Tuple[str, ...] = ()
    input_schema: Mapping[str, Any] = field(default_factory=dict)
    output_schema: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("name", "description", "source", "language"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"ToolSpec.{name} must be a non-empty string")
        if isinstance(self.capabilities, str):
            raise ValueError("capabilities must be a sequence, not a bare string")
        object.__setattr__(self, "capabilities",
                           tuple(str(cap) for cap in self.capabilities))


@dataclass(frozen=True)
class ToolValidation:
    valid: bool
    score: float = 0.0
    reason: str = ""
    capabilities: Tuple[str, ...] = ()
    tests: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        score = float(self.score)
        if not math.isfinite(score):
            raise ValueError("ToolValidation.score must be finite")
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "capabilities",
                           tuple(str(cap) for cap in self.capabilities))


@dataclass(frozen=True)
class ToolCandidate:
    spec: ToolSpec
    validation: ToolValidation
    generation: int


@dataclass(frozen=True)
class ToolSynthesisRequest:
    goal: str
    examples: Tuple[Any, ...] = ()
    constraints: Mapping[str, Any] = field(default_factory=dict)
    max_candidates: int = 4

    def __post_init__(self) -> None:
        if not self.goal.strip():
            raise ValueError("ToolSynthesisRequest.goal is required")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be >= 1")
        object.__setattr__(self, "examples", tuple(self.examples))


@dataclass(frozen=True)
class ToolSynthesisReport:
    request: ToolSynthesisRequest
    candidates: Tuple[ToolCandidate, ...]
    selected: Optional[ToolCandidate]
    detail: str = ""

    @property
    def accepted(self) -> Tuple[ToolCandidate, ...]:
        return tuple(candidate for candidate in self.candidates
                     if candidate.validation.valid)

    @property
    def rejected(self) -> Tuple[ToolCandidate, ...]:
        return tuple(candidate for candidate in self.candidates
                     if not candidate.validation.valid)


@dataclass(frozen=True)
class EnvironmentSpec:
    name: str
    tools: Tuple[ToolSpec, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("EnvironmentSpec.name is required")
        if not self.tools:
            raise ValueError("EnvironmentSpec needs at least one tool")
        for tool in self.tools:
            if not isinstance(tool, ToolSpec):
                raise TypeError("EnvironmentSpec.tools must contain ToolSpec values")
        object.__setattr__(self, "tools", tuple(self.tools))


def _validation(value: Any) -> ToolValidation:
    if isinstance(value, ToolValidation):
        return value
    if isinstance(value, bool):
        return ToolValidation(value, 1.0 if value else 0.0)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        score = float(value)
        if not math.isfinite(score):
            raise ValueError("validator returned a non-finite score")
        return ToolValidation(score >= 0.5, score)
    if isinstance(value, Mapping):
        return ToolValidation(
            bool(value.get("valid", value.get("passed", False))),
            float(value.get("score", 0.0)),
            str(value.get("reason", "")),
            tuple(value.get("capabilities", ())),
            dict(value.get("tests", {})))
    raise TypeError(f"validator returned unsupported {type(value).__name__}")


class ToolSynthesizer:
    """Generate, validate and select the best tool for a request."""

    def __init__(self, generator: Generator, validator: Validator, *,
                 selector: Optional[Callable[[Sequence[ToolCandidate]],
                                            Optional[ToolCandidate]]] = None) -> None:
        if not callable(generator) or not callable(validator):
            raise TypeError("generator and validator must be callable")
        if selector is not None and not callable(selector):
            raise TypeError("selector must be callable or None")
        self.generator = generator
        self.validator = validator
        self.selector = selector

    def synthesize(self, request: ToolSynthesisRequest) -> ToolSynthesisReport:
        candidates: List[ToolCandidate] = []
        for generation in range(request.max_candidates):
            try:
                spec = self.generator(request, generation, tuple(candidates))
            except Exception as exc:  # noqa: BLE001 - recorded, not fatal
                candidates.append(ToolCandidate(
                    ToolSpec(f"error-{generation}", f"generator error: {exc}",
                             "raise RuntimeError('generator failed')"),
                    ToolValidation(False, 0.0, f"{type(exc).__name__}: {exc}"),
                    generation))
                continue
            if spec is None:
                continue
            if not isinstance(spec, ToolSpec):
                raise TypeError(
                    f"generator returned {type(spec).__name__}, expected ToolSpec")
            try:
                validation = _validation(self.validator(spec))
            except Exception as exc:  # noqa: BLE001 - recorded, not fatal
                validation = ToolValidation(False, 0.0,
                                            f"{type(exc).__name__}: {exc}")
            candidates.append(ToolCandidate(spec, validation, generation))
        accepted = [candidate for candidate in candidates
                    if candidate.validation.valid]
        selected = None
        if self.selector is not None:
            chosen = self.selector(tuple(candidates))
            if chosen is None:
                selected = None
            elif (isinstance(chosen, ToolCandidate) and chosen.validation.valid
                  and chosen in candidates):
                selected = chosen
            else:
                selected = _best(accepted)
        else:
            selected = _best(accepted)
        detail = (f"{len(candidates)} candidate(s), {len(accepted)} valid, "
                  f"selected={selected.spec.name if selected else 'none'}")
        return ToolSynthesisReport(request, tuple(candidates), selected, detail)


def _best(accepted: Sequence[ToolCandidate]) -> Optional[ToolCandidate]:
    if not accepted:
        return None
    return max(accepted, key=lambda candidate: (
        candidate.validation.score, -candidate.generation))
