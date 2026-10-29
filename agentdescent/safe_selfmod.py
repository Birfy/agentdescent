"""A capability gate and canary/rollback path for self-modifying code.

`self_modify` can evaluate and apply a source patch, but a self-improving loop
needs more than a diff: it needs to know **what the patch is allowed to do**,
whether it needs a canary, and when to roll it back. This module is that policy
seam. It is **defense-in-depth**, not a complete sandbox: execution must still go
through `sandbox` / `sandbox_container` or an equivalent OS boundary.

A modification declares the capabilities it wants (`files:write`,
`network:outbound`, ...). The gate grants a fixed set, rejects forbidden
imports/calls/dunder access, and routes by risk:

* ``allow`` -- low risk, no canary;
* ``canary`` -- above the canary threshold but below the hard ceiling;
* ``deny`` -- forbidden syntax, missing capability, or risk above the ceiling.

`promote()` then combines the review with the measured before/after (and, for a
canary, the canary metric): a regression rolls back, an unmeasured canary is
held, otherwise it promotes.

    gate = SafeSelfModifier({"files:write"}, max_risk=0.4, canary_risk=0.1)
    decision = gate.review(spec)
    final = gate.promote(spec, before=0.7, after=0.75, canary=0.72)
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Mapping, Optional, Set, Tuple

__all__ = [
    "ModificationSpec",
    "SafetyDecision",
    "SafeSelfModifier",
]

_DEFAULT_ALLOWED_IMPORTS = frozenset({
    "base64", "collections", "dataclasses", "datetime", "functools",
    "hashlib", "http", "io", "itertools", "json", "math", "pathlib", "random",
    "re", "statistics", "tempfile", "textwrap", "typing", "urllib",
})
_DEFAULT_FORBIDDEN_IMPORTS = frozenset({
    "builtins", "code", "ctypes", "importlib", "marshal", "multiprocessing",
    "nt", "os", "pickle", "posix", "pty", "runpy", "shutil", "signal",
    "socket", "subprocess", "sys", "threading", "_posixsubprocess",
})
_DEFAULT_FORBIDDEN_CALLS = frozenset({
    "__import__", "breakpoint", "compile", "delattr", "eval", "exec",
    "format", "format_map", "get_field", "getattr", "globals", "input",
    "locals", "open", "setattr", "vars", "vformat", "Formatter",
})


@dataclass(frozen=True)
class ModificationSpec:
    """One proposed self-modification and the authority it asks for."""

    modification_id: str
    source: str
    target: str
    requested_capabilities: Tuple[str, ...] = ()
    risk: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("modification_id", "source", "target"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"ModificationSpec.{name} must be a non-empty string")
        if isinstance(self.risk, bool):
            raise ValueError("ModificationSpec.risk must be a number, not bool")
        risk = float(self.risk)
        if not math.isfinite(risk) or not 0.0 <= risk <= 1.0:
            raise ValueError("ModificationSpec.risk must be in [0, 1]")
        object.__setattr__(self, "risk", risk)
        if isinstance(self.requested_capabilities, str):
            raise ValueError("requested_capabilities must be a sequence, not a bare string")
        object.__setattr__(self, "requested_capabilities",
                           tuple(str(cap) for cap in self.requested_capabilities))


@dataclass(frozen=True)
class SafetyDecision:
    """The gate's verdict and the reason a caller can audit."""

    modification_id: str
    action: str
    allowed: bool
    reason: str
    risk: float
    violations: Tuple[str, ...] = ()
    capabilities: Tuple[str, ...] = ()

    @property
    def needs_canary(self) -> bool:
        return self.action == "canary"


class SafeSelfModifier:
    """A capability gate plus a canary/rollback promotion rule."""

    def __init__(self, granted_capabilities: Iterable[str], *,
                 max_risk: float = 0.4, canary_risk: float = 0.1,
                 tolerance: float = 0.0,
                 allowed_imports: Optional[Iterable[str]] = None,
                 forbidden_imports: Optional[Iterable[str]] = None,
                 forbidden_calls: Optional[Iterable[str]] = None) -> None:
        self.granted = frozenset(str(cap) for cap in granted_capabilities)
        self.max_risk = self._risk(max_risk, "max_risk")
        self.canary_risk = self._risk(canary_risk, "canary_risk")
        self.tolerance = float(tolerance)
        if not math.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("tolerance must be finite and >= 0")
        if self.canary_risk > self.max_risk:
            raise ValueError("canary_risk must be <= max_risk")
        if isinstance(allowed_imports, str) or isinstance(forbidden_imports, str)                 or isinstance(forbidden_calls, str):
            raise ValueError("import/call policies must be sequences, not bare strings")
        self.allowed_imports = frozenset(
            _DEFAULT_ALLOWED_IMPORTS if allowed_imports is None else allowed_imports)
        self.forbidden_imports = frozenset(
            _DEFAULT_FORBIDDEN_IMPORTS if forbidden_imports is None else forbidden_imports)
        self.forbidden_calls = frozenset(
            _DEFAULT_FORBIDDEN_CALLS if forbidden_calls is None else forbidden_calls)

    @staticmethod
    def _risk(value: float, name: str) -> float:
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
        return number

    def _scan(self, source: str) -> Tuple[List[str], Set[str]]:
        try:
            tree = ast.parse(source)
        except SyntaxError as error:
            return [f"syntax: {error.msg}"], set()
        violations: List[str] = []
        imports: Set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    imports.add(root)
                    if root in self.forbidden_imports or root not in self.allowed_imports:
                        violations.append(f"import:{alias.name}")
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                if root:
                    imports.add(root)
                if root in self.forbidden_imports or (root and root not in self.allowed_imports):
                    violations.append(f"import:{node.module}")
            elif isinstance(node, ast.Name):
                if node.id.startswith("__"):
                    violations.append(f"dunder:{node.id}")
                if node.id in self.forbidden_calls:
                    violations.append(f"ref:{node.id}")
            elif isinstance(node, ast.Attribute):
                if node.attr.startswith("_"):
                    violations.append(f"private-attr:{node.attr}")
                if node.attr in self.forbidden_calls:
                    violations.append(f"ref:{node.attr}")
                if node.attr in ("sys", "os", "modules", "builtins"):
                    violations.append(f"module-attr:{node.attr}")
            elif isinstance(node, ast.Call):
                name = ""
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    name = node.func.attr
                if name in self.forbidden_calls:
                    violations.append(f"call:{name}")
        return sorted(set(violations)), imports

    def _inferred_capabilities(self, imports: Iterable[str]) -> Set[str]:
        required: Set[str] = set()
        for root in imports:
            if root in {"pathlib", "io", "tempfile", "shutil"}:
                required.add("files:write")
            if root in {"urllib", "http", "socket", "requests", "aiohttp"}:
                required.add("network:outbound")
            if root in {"os", "posix", "nt", "subprocess"}:
                required.update({"files:write", "process:spawn"})
        return required

    def review(self, spec: ModificationSpec) -> SafetyDecision:
        """Check syntax, capability grant and risk; choose allow/canary/deny."""
        if not isinstance(spec, ModificationSpec):
            raise TypeError(f"review() expects ModificationSpec, got {type(spec).__name__}")
        requested = tuple(sorted(set(str(cap) for cap in spec.requested_capabilities)))
        missing = tuple(cap for cap in requested if cap not in self.granted)
        violations, imports = self._scan(spec.source)
        inferred = self._inferred_capabilities(imports)
        undeclared = tuple(sorted(inferred - set(requested)))
        if missing:
            violations.append("capability:" + ",".join(missing))
        if undeclared:
            violations.append("capability-required:" + ",".join(undeclared))
        if violations:
            return SafetyDecision(spec.modification_id, "deny", False,
                                  "policy violation", spec.risk,
                                  tuple(sorted(violations)), requested)
        if spec.risk > self.max_risk:
            return SafetyDecision(spec.modification_id, "deny", False,
                                  f"risk {spec.risk:.3f} > max_risk {self.max_risk:.3f}",
                                  spec.risk, (), requested)
        if spec.risk >= self.canary_risk:
            return SafetyDecision(spec.modification_id, "canary", True,
                                  "risk requires a canary", spec.risk, (), requested)
        return SafetyDecision(spec.modification_id, "allow", True,
                              "within policy", spec.risk, (), requested)

    def promote(self, spec: ModificationSpec, *, before: float, after: float,
                canary: Optional[float] = None) -> SafetyDecision:
        """Promote, hold, or roll back based on measured outcomes.

        A canary decision needs a canary measurement; without one it is held.
        Either the full metric or the canary metric falling below ``before -
        tolerance`` rolls back.
        """
        decision = self.review(spec)
        if not decision.allowed:
            return SafetyDecision(spec.modification_id, "deny", False,
                                  decision.reason, spec.risk,
                                  decision.violations, decision.capabilities)
        before = float(before)
        after = float(after)
        if not math.isfinite(before) or not math.isfinite(after):
            raise ValueError("before/after must be finite")
        if decision.needs_canary and canary is None:
            return SafetyDecision(spec.modification_id, "hold", False,
                                  "canary measurement required", spec.risk,
                                  (), decision.capabilities)
        if after < before - self.tolerance:
            return SafetyDecision(spec.modification_id, "rollback", False,
                                  f"after {after:.6g} < before {before:.6g}",
                                  spec.risk, (), decision.capabilities)
        if canary is not None:
            canary = float(canary)
            if not math.isfinite(canary):
                raise ValueError("canary must be finite")
            if canary < before - self.tolerance:
                return SafetyDecision(spec.modification_id, "rollback", False,
                                      f"canary {canary:.6g} < before {before:.6g}",
                                      spec.risk, (), decision.capabilities)
        return SafetyDecision(spec.modification_id, "promote", True,
                              "canary and full metric passed", spec.risk,
                              (), decision.capabilities)

    @staticmethod
    def rollback_plan(decision: SafetyDecision) -> Tuple[str, ...]:
        """A deterministic checklist a caller can hand to an operator."""
        if decision.action == "deny":
            return (
                f"do not apply {decision.modification_id}",
                "record the denial and its violations in the decision log",
            )
        if decision.action == "rollback":
            return (
                f"restore the last-known-good artifact for {decision.modification_id}",
                "verify the restored artifact against the protected suite",
                "record the rollback in the decision log and manifest",
            )
        return ()
