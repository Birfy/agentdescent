"""Metamorphic testing for artifacts whose correct output is not known.

A held-out task set gives one labelled example per behaviour. It cannot cover
the transformations a real deployment will see: casing changes, reordered
options, paraphrased instructions, a different file name, a longer context.
Metamorphic testing checks a **relation between outputs** instead of an absolute
label: if the task is transformed in a way that should not change the answer,
the answer must stay equivalent; if the transform should change it in a known
way, the output must change in that way.

That makes it a cheap, model-agnostic acceptance probe for self-evolution. A
candidate that scores higher on the target task but breaks an invariance is not
an improvement -- it learned the surface form of the training set.

    suite = MetamorphicSuite([
        invariance("case", lambda task: task.lower()),
        relation("duplicate", lambda task: task + "\n" + task,
                 lambda original, transformed, task: transformed.count("x") == 2 * original.count("x")),
    ])
    report = suite.evaluate(run, tasks)
    assert report.ok          # report.passed is the count; report.ok is the gate

`run(task)` is any callable returning a string-like output; `transform(task)` is
the relation's input transform; `check` receives the original and transformed
outputs plus the original task. The suite never calls a model on its own.

For a non-deterministic model, exact ``==`` invariants are flaky. Pass a
normaliser to ``invariance(equal=...)`` (for example compare ``str.strip()``
or a parsed structure), not only the transform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "MetamorphicRelation",
    "MetamorphicSuite",
    "MetamorphicReport",
    "invariance",
    "relation",
]

Transform = Callable[[Any], Any]
Check = Callable[[Any, Any, Any], bool]


@dataclass(frozen=True)
class MetamorphicRelation:
    """One input transform and the output relation it must preserve.

    ``check(original_output, transformed_output, original_task)`` returns
    truthy when the relation holds. ``weight`` only affects the aggregate
    weighted pass rate; violations are always reported individually.
    """

    name: str
    transform: Transform
    check: Check
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("MetamorphicRelation.name must be a non-empty string")
        if not callable(self.transform) or not callable(self.check):
            raise TypeError("transform and check must be callable")
        if self.weight <= 0:
            raise ValueError("MetamorphicRelation.weight must be > 0")


def invariance(name: str, transform: Transform, *,
               equal: Optional[Callable[[Any, Any], bool]] = None,
               weight: float = 1.0) -> MetamorphicRelation:
    """A relation whose transform must leave the output unchanged.

    ``equal`` defaults to ``==``. Use a normaliser (``str.strip``) when the
    output is allowed to differ only in whitespace.
    """
    compare = equal or (lambda left, right: left == right)

    def check(original: Any, transformed: Any, _task: Any) -> bool:
        return bool(compare(original, transformed))

    return MetamorphicRelation(name=name, transform=transform, check=check,
                               weight=weight)


def relation(name: str, transform: Transform,
             check: Check, *, weight: float = 1.0) -> MetamorphicRelation:
    """A relation with a caller-supplied check (a convenience constructor)."""
    return MetamorphicRelation(name=name, transform=transform, check=check,
                               weight=weight)


@dataclass(frozen=True)
class MetamorphicReport:
    """Check counts and the concrete violations, per relation."""

    checks: int
    passed: int
    weighted_rate: float
    per_relation: Dict[str, Dict[str, float]] = field(default_factory=dict)
    violations: Tuple[Dict[str, Any], ...] = ()
    verdict: str = "pass"
    detail: str = ""

    @property
    def failed(self) -> int:
        return self.checks - self.passed

    @property
    def ok(self) -> bool:
        return self.verdict == "pass"


class MetamorphicSuite:
    """Run a set of relations over tasks and score the candidate."""

    def __init__(self, relations: Sequence[MetamorphicRelation]) -> None:
        if not relations:
            raise ValueError("MetamorphicSuite needs at least one relation")
        names = [item.name for item in relations]
        if len(set(names)) != len(names):
            raise ValueError(f"relation names must be unique: {names}")
        self.relations = tuple(relations)

    def evaluate(self, run: Callable[[Any], Any], tasks: Iterable[Any], *,
                 max_violations: int = 20) -> MetamorphicReport:
        """Check every relation on every task.

        Exceptions from ``run`` or ``transform`` are recorded as violations and
        do not abort the suite; the report's detail names the first one. A
        relation is never allowed to silently disappear because a transform
        failed on one row.
        """
        if max_violations < 0:
            raise ValueError("max_violations must be >= 0")
        rows = list(tasks)
        counts = {item.name: {"passed": 0, "total": 0, "weight": float(item.weight)}
                  for item in self.relations}
        violations: List[Dict[str, Any]] = []
        checks = passed = 0
        weighted_total = weighted_passed = 0.0
        for task in rows:
            try:
                original = run(task)
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                for item in self.relations:
                    counts[item.name]["total"] += 1
                    checks += 1
                    weighted_total += item.weight
                    if len(violations) < max_violations:
                        violations.append({"relation": item.name, "task": task,
                                           "error": f"{type(exc).__name__}: {exc}"})
                continue
            for item in self.relations:
                checks += 1
                counts[item.name]["total"] += 1
                weighted_total += item.weight
                try:
                    transformed_task = item.transform(task)
                    transformed = run(transformed_task)
                    ok = bool(item.check(original, transformed, task))
                except Exception as exc:  # noqa: BLE001
                    ok = False
                    if len(violations) < max_violations:
                        violations.append({"relation": item.name, "task": task,
                                           "error": f"{type(exc).__name__}: {exc}"})
                else:
                    if not ok and len(violations) < max_violations:
                        violations.append({"relation": item.name, "task": task,
                                           "original": original,
                                           "transformed": transformed})
                if ok:
                    counts[item.name]["passed"] += 1
                    passed += 1
                    weighted_passed += item.weight
        weighted_rate = weighted_passed / weighted_total if weighted_total else 0.0
        if checks == 0:
            verdict = "insufficient"
        elif passed == checks:
            verdict = "pass"
        else:
            verdict = "fail"
        if checks == 0:
            detail = "no tasks"
        elif passed == checks:
            detail = f"{checks}/{checks} checks passed"
        else:
            names = sorted({row["relation"] for row in violations})
            detail = f"{checks - passed} failed; relations={','.join(names)}"
        per_relation = {
            name: {
                "passed": data["passed"],
                "total": data["total"],
                "rate": (data["passed"] / data["total"] if data["total"] else 0.0),
            }
            for name, data in counts.items()
        }
        return MetamorphicReport(checks=checks, passed=passed,
                                 weighted_rate=weighted_rate,
                                 per_relation=per_relation,
                                 violations=tuple(violations),
                                 verdict=verdict, detail=detail)

    def accept(self, run: Callable[[Any], Any], tasks: Iterable[Any], *,
               min_rate: float = 1.0, max_violations: int = 20) -> bool:
        """A boolean gate: pass when the weighted rate clears ``min_rate``.

        Unlike ``report.verdict == "pass"`` (which requires *every* check), this
        lets a caller accept a bounded failure rate, for example ``min_rate=0.8``
        when one relation is known to be noisy. An empty task set still fails.
        """
        if not 0.0 <= min_rate <= 1.0:
            raise ValueError("min_rate must be in [0, 1]")
        report = self.evaluate(run, tasks, max_violations=max_violations)
        return report.checks > 0 and report.weighted_rate >= min_rate
