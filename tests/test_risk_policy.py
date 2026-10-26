"""Risk actions must be ordered, explainable and fail-closed."""

from __future__ import annotations

import pytest

from agentdescent import RiskContext, RiskPolicy, RiskRule


def test_first_matching_rule_wins():
    policy = RiskPolicy([
        RiskRule("verifier", "deny", domain="verifier"),
        RiskRule("l1", "approve", tiers=(1,), min_blast_radius=0.5),
        RiskRule("large", "canary", min_ops=6),
    ])
    decision = policy.decide(RiskContext(tier=1, blast_radius=0.7,
                                         ops=8, domain="verifier"))
    assert decision.action == "deny" and decision.rule == "verifier"
    assert policy.decide(RiskContext(tier=1, blast_radius=0.7)).action == "approve"
    assert policy.decide(RiskContext(tier=2, ops=8)).action == "canary"


def test_no_match_uses_the_fail_closed_default():
    policy = RiskPolicy([RiskRule("only-l1", "allow", tiers=(1,))])
    decision = policy.decide(RiskContext(tier=3))
    assert decision.action == "deny" and decision.rule == "<default>"


def test_the_default_can_be_set_but_must_be_valid():
    assert RiskPolicy([], default="allow").decide(RiskContext()).action == "allow"
    with pytest.raises(ValueError, match="default"):
        RiskPolicy([], default="maybe")


def test_an_invalid_rule_action_is_refused():
    with pytest.raises(ValueError, match="action"):
        RiskRule("bad", "maybe")


def test_duplicate_rule_names_and_describe():
    with pytest.raises(ValueError, match="unique"):
        RiskPolicy([RiskRule("a", "allow"), RiskRule("a", "deny")])
    policy = RiskPolicy([RiskRule("a", "approve", tenant="t1")])
    assert policy.describe()[0]["tenant"] == "t1"


def test_tenant_and_domain_are_conjunctive():
    rule = RiskRule("r", "deny", domain="verifier", tenant="acme")
    policy = RiskPolicy([rule], default="allow")
    assert policy.decide(RiskContext(domain="verifier",
                                     tenant="acme")).rule == "r"
    assert policy.decide(RiskContext(domain="verifier",
                                     tenant="other")).rule == "<default>"
    assert policy.decide(RiskContext(domain="verifier")).rule == "<default>"


def test_unknown_risk_facts_match_threshold_rules():
    policy = RiskPolicy([
        RiskRule("big", "approve", min_blast_radius=0.5),
        RiskRule("rest", "allow")], default="deny")
    # missing blast radius is not "zero" -- it matches the risk rule
    assert policy.decide(RiskContext()).action == "approve"


def test_invalid_context_and_rule_values_are_refused():
    with pytest.raises(ValueError, match="blast_radius"):
        RiskContext(blast_radius=float("nan"))
    with pytest.raises(ValueError, match="ops"):
        RiskContext(ops=-1)
    with pytest.raises(ValueError, match="tier"):
        RiskContext(tier=True)
    with pytest.raises(ValueError, match="name"):
        RiskRule("", "deny")
    with pytest.raises(ValueError, match="tiers"):
        RiskRule("t", "deny", tiers=(1, "x"))
    with pytest.raises(ValueError, match="min_ops"):
        RiskRule("t", "deny", min_ops=-1)


def test_tiers_are_normalized_to_a_tuple():
    rule = RiskRule("t", "deny", tiers=[1, 2])
    assert rule.tiers == (1, 2)
