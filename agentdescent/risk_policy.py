"""Declarative risk policy: turn a change's context into a safety action.

An acceptance gate answers "is this better?", not "what should we do about the
risk?". A product needs a policy layer that maps a change's risk context -- tier,
blast radius, diff size, capability domain, tenant -- to one action:
``allow``, ``approve``, ``canary``, ``rollback`` or ``deny``. Rules are ordered;
the first match wins; no match uses the policy's ``default`` (``deny``). Every
unknown/invalid action is rejected at construction, so a typo cannot silently
allow a change.

    policy = RiskPolicy([
        RiskRule("never-touch-verifier", "deny", domain="verifier"),
        RiskRule("l1-approval", "approve", tiers=(1,), min_blast_radius=0.5),
        RiskRule("low-risk-canary", "canary", min_ops=6),
    ])
    policy.decide(RiskContext(tier=2, blast_radius=0.2, ops=8)).action
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

__all__ = ["RiskContext", "RiskDecision", "RiskPolicy", "RiskRule",
           "RISK_ACTIONS"]

#: The actions a policy may return; they are alternatives, not a total order.
RISK_ACTIONS = ("allow", "approve", "canary", "rollback", "deny")


@dataclass(frozen=True)
class RiskContext:
    """The facts a policy may match on.

    A missing fact is ``None``, and an unknown value **matches** any rule that
    tests it -- fail-closed. Do not substitute ``0`` for "unknown": zero is the
    most permissive context and would silently allow a change whose risk was
    never measured.
    """

    tier: Optional[int] = None
    artifact_id: str = ""
    blast_radius: Optional[float] = None
    ops: Optional[int] = None
    domain: Optional[str] = None
    tenant: Optional[str] = None

    def __post_init__(self) -> None:
        if self.tier is not None and (isinstance(self.tier, bool)
                                      or not isinstance(self.tier, int)
                                      or self.tier < 0):
            raise ValueError("tier must be a non-negative int or None")
        if self.blast_radius is not None:
            value = float(self.blast_radius)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("blast_radius must be finite and in [0, 1]")
        if self.ops is not None and (isinstance(self.ops, bool)
                                     or not isinstance(self.ops, int)
                                     or self.ops < 0):
            raise ValueError("ops must be a non-negative int or None")


@dataclass(frozen=True)
class RiskRule:
    """One ordered rule; every supplied condition must match."""

    name: str
    action: str
    tiers: Tuple[int, ...] = ()
    min_blast_radius: Optional[float] = None
    min_ops: Optional[int] = None
    domain: Optional[str] = None
    tenant: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("risk rule name must not be empty")
        if self.action not in RISK_ACTIONS:
            raise ValueError(
                f"risk rule {self.name!r}: action must be one of "
                f"{list(RISK_ACTIONS)}, got {self.action!r}")
        object.__setattr__(self, "tiers", tuple(self.tiers or ()))
        if any(isinstance(t, bool) or not isinstance(t, int) or t < 0
               for t in self.tiers):
            raise ValueError(f"risk rule {self.name!r}: tiers must be non-negative ints")
        if self.min_blast_radius is not None:
            value = float(self.min_blast_radius)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"risk rule {self.name!r}: min_blast_radius must be in [0, 1]")
        if self.min_ops is not None and (isinstance(self.min_ops, bool)
                                         or not isinstance(self.min_ops, int)
                                         or self.min_ops < 0):
            raise ValueError(
                f"risk rule {self.name!r}: min_ops must be a non-negative int")

    def matches(self, ctx: RiskContext) -> bool:
        # A missing fact matches a rule that tests it: unknown is worst-case.
        if self.tiers and ctx.tier is not None and ctx.tier not in self.tiers:
            return False
        if (self.min_blast_radius is not None and ctx.blast_radius is not None
                and ctx.blast_radius < self.min_blast_radius):
            return False
        if (self.min_ops is not None and ctx.ops is not None
                and ctx.ops < self.min_ops):
            return False
        if self.domain is not None and ctx.domain != self.domain:
            return False
        if self.tenant is not None and ctx.tenant != self.tenant:
            return False
        return True


@dataclass(frozen=True)
class RiskDecision:
    """The action a policy chose and the rule that produced it."""

    action: str
    rule: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {"action": self.action, "rule": self.rule,
                "reason": self.reason}


class RiskPolicy:
    """First-match risk rules with a fail-closed default."""

    def __init__(self, rules: Sequence[RiskRule], *,
                 default: str = "deny") -> None:
        if default not in RISK_ACTIONS:
            raise ValueError(
                f"default must be one of {list(RISK_ACTIONS)}, got {default!r}")
        names = [rule.name for rule in rules]
        if len(set(names)) != len(names):
            raise ValueError(f"risk rule names must be unique: {names}")
        self.rules = tuple(rules)
        self.default = default

    def decide(self, ctx: RiskContext) -> RiskDecision:
        for rule in self.rules:
            if rule.matches(ctx):
                return RiskDecision(
                    rule.action, rule.name,
                    f"matched rule {rule.name!r} (tier={ctx.tier}, "
                    f"blast={ctx.blast_radius}, ops={ctx.ops}, "
                    f"domain={ctx.domain}, tenant={ctx.tenant})")
        return RiskDecision(self.default, "<default>",
                            f"no rule matched; default is {self.default!r}")

    def describe(self) -> Tuple[Dict[str, Any], ...]:
        return tuple({"name": rule.name, "action": rule.action,
                      "tiers": list(rule.tiers),
                      "min_blast_radius": rule.min_blast_radius,
                      "min_ops": rule.min_ops, "domain": rule.domain,
                      "tenant": rule.tenant} for rule in self.rules)
