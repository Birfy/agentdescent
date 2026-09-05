"""A wrapper around a default rule must be installable through ``Policies`` alone.

Two of the four merge-side defaults need something only the engine has -- the
verifier, for the rules that rank -- and the other two read thresholds off the
aggregator's config. Before this the aggregator handed the verifier to *fusion*
and to nothing else, so ``AdvantageConflict(DefaultConflict(verifier))`` could
only be written from inside an ``aggregator_factory`` (which is what
``tests/test_advantage.py`` did), and the acceptance wrapper's inner gate had to
copy three numbers out of ``AggregatorConfig`` by hand -- silently diverging the
moment ``agg_config=`` changed one of them.

Now every installed policy is offered ``bind(verifier)`` and
``configure(config)``, wrappers forward both, and a default used *without*
being installed says so instead of dying on a ``NoneType``.
"""

import pytest

from agentdescent import Policies, SingleSlot, Task, evolve
from agentdescent.advantage import (
    AdvantageAcceptance, AdvantageConflict, StableDistanceAcceptance,
)
from agentdescent.aggregator import Aggregator, AggregatorConfig, install_policy
from agentdescent.defaults import (
    DefaultAcceptance, DefaultConflict, DefaultFusion, DefaultPromotion,
    PolicyUnboundError,
)
from agentdescent.evolvable import Diff
from agentdescent.ledger import Ledger
from agentdescent.scheduler import AuditScheduler
from agentdescent.verifier import ThreeLayerVerifier, VerifierBudget


def _verifier():
    return ThreeLayerVerifier(eval_fn=lambda a, ts: 1.0, held_out=[1, 2, 3],
                              budget=VerifierBudget(oracle_calls_remaining=5))


def _card(value, advantage=None):
    c = type("C", (), {})()
    c.diff = Diff(diff_id=f"d{value}", target="a", ops={"k": value})
    c.advantage = advantage
    return c


# -- a contradiction reaching the inner rule, through Policies alone ----------


def _contradicting_run(**kw):
    """Three workers rewrite one slot with three different values every round,
    so every merge carries contradictions and none of the cards has an
    advantage (the group is too small) -- the exact path that used to die."""
    tasks = [Task(id=str(i), prompt="q", meta={"gold": f"g{i % 3}"})
             for i in range(12)]
    return evolve(tasks, lambda t, o: 1.0 if o == t.meta["gold"] else 0.0,
                  run=lambda rendered, t: rendered.strip().split()[-1],
                  propose=lambda rendered, t, o, s: t.meta["gold"],
                  strategy=SingleSlot(initial_value="g0"), rounds=3, n_workers=3,
                  max_concurrency=1, held_out_frac=0.5, seed=0, **kw)


def test_a_wrapped_default_conflict_rule_installs_through_policies():
    seen = []

    class Watching(AdvantageConflict):
        def resolve(self, artifact, cards):
            seen.append(len(cards))
            return super().resolve(artifact, cards)

    policy = Watching()
    r = _contradicting_run(policies=Policies(conflict=policy))
    assert r.error is None
    assert max(seen) >= 2, "the run never produced a contradiction to resolve"
    assert policy.inner.verifier is not None, "the inner rule was never bound"


def test_the_bare_default_conflict_rule_installs_through_policies():
    policy = DefaultConflict()
    r = _contradicting_run(policies=Policies(conflict=policy))
    assert r.error is None and policy.verifier is not None


# -- thresholds come from the run's config, not from a hand copy --------------


def test_a_wrapped_default_gate_takes_the_runs_thresholds():
    policy = AdvantageAcceptance()
    _contradicting_run(policies=Policies(acceptance=policy),
                       agg_config=AggregatorConfig(base_delta=0.9, anneal_half_life=7,
                                                   accept_samples=123))
    inner = policy.inner
    assert (inner.base_delta, inner.anneal_half_life, inner.accept_samples) == (0.9, 7, 123)


def test_a_pinned_threshold_is_not_overwritten_by_configure():
    gate = DefaultAcceptance(base_delta=0.25)
    gate.configure(AggregatorConfig(base_delta=0.9, anneal_half_life=7, accept_samples=123))
    assert (gate.base_delta, gate.anneal_half_life, gate.accept_samples) == (0.25, 7, 123)


def test_from_config_is_the_rule_the_aggregator_builds():
    cfg = AggregatorConfig(base_delta=0.4, anneal_half_life=9, accept_samples=50,
                           promote_after_k=5)
    gate = DefaultAcceptance.from_config(cfg)
    assert (gate.base_delta, gate.anneal_half_life, gate.accept_samples) == (0.4, 9, 50)
    assert DefaultPromotion.from_config(cfg).promote_after_k == 5


def test_stable_distance_wrapper_forwards_configure():
    policy = StableDistanceAcceptance()
    install_policy(policy, _verifier(), AggregatorConfig(base_delta=0.33))
    assert policy.inner.base_delta == 0.33


# -- an uninstalled default names what it is missing --------------------------


def test_an_unbound_conflict_rule_says_so_on_the_first_contradiction():
    with pytest.raises(PolicyUnboundError, match="DefaultConflict.*bind"):
        DefaultConflict().resolve(None, [_card("x"), _card("y")])


def test_an_unbound_conflict_rule_is_fine_without_contradictions():
    kept, dropped = DefaultConflict().resolve(None, [_card("x")])
    assert len(kept) == 1 and dropped == 0


def test_a_wrapper_around_an_unbound_rule_reports_the_inner_rule():
    with pytest.raises(PolicyUnboundError, match="DefaultConflict"):
        AdvantageConflict().resolve(None, [_card("x"), _card("y")])


def test_an_unconfigured_gate_says_so():
    from agentdescent.policies import MergeContext
    ctx = MergeContext(artifact=None, candidate=None, cards=(),
                       base_counts=(1.0, 1.0), cand_counts=(2.0, 0.0))
    with pytest.raises(PolicyUnboundError, match="DefaultAcceptance.*configure"):
        DefaultAcceptance().accept(ctx)


def test_an_unconfigured_promotion_rule_says_so():
    with pytest.raises(PolicyUnboundError, match="DefaultPromotion.*configure"):
        DefaultPromotion().observe([])


def test_an_unbound_tournament_says_so():
    art = type("A", (), {"apply": lambda self, d: self})()
    with pytest.raises(PolicyUnboundError, match="DefaultFusion.*bind"):
        DefaultFusion(tournament=True).select(art, [_card("x").diff, _card("y").diff])


# -- the aggregator installs all four, and a bound value stays bound ----------


def test_the_aggregator_installs_every_merge_side_policy(tmp_path):
    verifier = _verifier()
    conflict, fusion = AdvantageConflict(), DefaultFusion()
    acceptance, promotion = StableDistanceAcceptance(), DefaultPromotion()
    cfg = AggregatorConfig(base_delta=0.6, promote_after_k=4)
    ledger = Ledger(str(tmp_path / "repo"), serialize=lambda a: {"state": {}},
                    deserialize=lambda aid, v, d: None)
    Aggregator(ledger, verifier, AuditScheduler(), cfg,
               conflict=conflict, fusion=fusion, acceptance=acceptance,
               promotion=promotion)
    assert conflict.inner.verifier is verifier
    assert fusion.verifier is verifier
    assert acceptance.inner.base_delta == 0.6
    assert promotion.promote_after_k == 4


def test_a_caller_supplied_verifier_is_kept_over_the_engines():
    mine, engines = _verifier(), _verifier()
    policy = DefaultConflict(mine)
    install_policy(policy, engines, AggregatorConfig())
    assert policy.verifier is mine


def test_a_policy_without_hooks_is_left_alone():
    class Bare:
        def resolve(self, artifact, cards):
            return list(cards), 0

    install_policy(Bare(), _verifier(), AggregatorConfig())     # no error, no effect
