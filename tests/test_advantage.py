"""Three RL decision rules, and the guards that keep them honest.

None of these tests claim the mechanisms *work*. That is what an A/B on a real
dataset is for, and until one has run, each of these is an option and not a
mechanism -- which is why the tests below also pin that installing none of them
changes nothing.

What is checked here is the part an A/B cannot check: that `None` never
masquerades as a measurement, that the adaptive region moves for the right reason
and stops at its bounds, and that the stable-distance penalty refuses a commit
rather than quietly editing the score it was judged on.
"""

import pytest

from agentdescent.advantage import (
    AdaptiveTrustRegion, AdvantageAcceptance, AdvantageConflict, GroupAdvantage,
    StableDistanceAcceptance, TrustRegion, state_distance,
)
from agentdescent.aggregator import AggregatorConfig
from agentdescent.evolution import AppendRules, Task, evolve
from agentdescent.policies import AcceptDecision, MergeContext, Policies


# -- 1. group-relative advantage --------------------------------------------


def test_a_group_too_small_to_have_a_spread_reports_nothing():
    """`None`, not `0.0`. Zero is a real advantage -- 'did exactly as well as its
    peers' -- so returning it for 'we do not know yet' feeds noise to the gate
    dressed as a measurement."""
    ga = GroupAdvantage(min_group=4)
    assert [ga.observe("g", r) for r in (1.0, 0.0, 1.0)] == [None, None, None]
    assert ga.observe("g", 0.0) is not None


def test_a_group_with_no_variance_reports_nothing():
    """The zero-advantage case `DifficultyWeighted` already filters on. Dividing
    by that variance is how a rounding error becomes a large advantage."""
    ga = GroupAdvantage(min_group=2)
    assert ga.observe("g", 1.0) is None
    assert ga.observe("g", 1.0) is None
    assert ga.observe("g", 1.0) is None


def test_the_advantage_is_signed_around_the_group_mean():
    ga = GroupAdvantage(min_group=2)
    ga.observe("g", 0.0)
    ga.observe("g", 0.0)
    ga.observe("g", 0.0)
    high = ga.observe("g", 1.0)
    assert high is not None and high > 0
    low = ga.observe("g", 0.0)
    assert low is not None and low < 0


def test_groups_are_independent():
    """Same base version, different task cluster, is a different group -- that is
    the whole point: an easy group and a hard one must not share a mean."""
    ga = GroupAdvantage(min_group=2)
    for _ in range(3):
        ga.observe(ga.key(1, "easy"), 1.0)
        ga.observe(ga.key(1, "hard"), 0.0)
    assert ga.group_size(ga.key(1, "easy")) == 3
    assert ga.group_size(ga.key(1, "hard")) == 3


def test_a_group_of_one_is_refused_at_construction():
    with pytest.raises(ValueError):
        GroupAdvantage(min_group=1)


def test_the_group_table_is_bounded():
    """A key contains the base version, which moves on every commit -- so the
    number of groups grows with the length of the run and nothing removes one.
    Evicting the oldest is safe: a superseded base version gets no more rollouts.
    """
    ga = GroupAdvantage(min_group=2, max_groups=8)
    for version in range(50):
        ga.observe(ga.key(version), 1.0)
    assert len(ga._n) == 8
    assert ga.group_size(ga.key(49)) == 1, "the newest group must survive"
    assert ga.group_size(ga.key(0)) == 0, "the oldest must have been evicted"


# -- the signal reaches the card --------------------------------------------


def _tasks(n=24, clusters=1):
    return [Task(id=str(i), prompt="q",
                 meta={"h": f"H{i % 6}", "cluster": f"c{i % clusters}"})
            for i in range(n)]


def _evolve(clusters=1, **kw):
    kw.setdefault("n_workers", 4)
    # Seeded with two of the six rules, so a round's rollouts do not all score
    # the same. Without that the group has zero variance -- which is the GRPO
    # zero-advantage case and correctly reports nothing, but then the test would
    # be measuring that filter rather than the signal.
    kw.setdefault("initial_state", {"r0": "H0", "r1": "H1"})
    return evolve(_tasks(clusters=clusters), lambda t, o: 1.0 if o == "good" else 0.0,
                  run=lambda rendered, t: "good" if t.meta["h"] in rendered else "bad",
                  propose=lambda rendered, t, o, s: t.meta["h"],
                  strategy=AppendRules(), rounds=8,
                  max_concurrency=1, held_out_frac=0.4, seed=2, **kw)


def test_recording_the_signal_changes_no_default_run():
    """The whole module is off by default, and this is what that means."""
    a, b = _evolve(), _evolve()
    assert a.rendered == b.rendered and a.final_reward == b.final_reward


def _advantages_seen(**kw):
    """Every card's advantage, as the aggregator receives it."""
    seen = []

    class Watching:
        def resolve(self, artifact, cards):
            seen.extend(getattr(c, "advantage", "missing") for c in cards)
            from agentdescent.defaults import DefaultConflict
            return DefaultConflict(self.verifier).resolve(artifact, cards)

    from agentdescent.aggregator import Aggregator

    def factory(ledger, verifier, audit, config, policy):
        agg = Aggregator(ledger, verifier, audit, config, staleness_policy=policy)
        watcher = Watching()
        watcher.verifier = verifier
        agg.conflict_policy = watcher
        return agg

    _evolve(aggregator_factory=factory, **kw)
    return seen


def test_the_engine_puts_an_advantage_on_the_cards_it_ingests():
    seen = _advantages_seen()
    assert seen, "the aggregator saw no cards at all"
    assert "missing" not in seen, "EvidenceCard lost its advantage field"
    assert any(v is not None for v in seen), \
        "no group ever grew large enough to report an advantage"


def test_the_group_is_at_most_one_round_of_workers_per_cluster():
    """Measured, and it bounds what this signal can ever be worth.

    A group is one base version and one cluster, and the base version moves on
    every commit -- so the group is at most the workers of a single round, split
    across however many clusters those workers landed in. With four workers and
    four clusters the largest possible group is one, and `min_group=4` therefore
    records nothing at all.

    Not a bug to tune away: the group definition is the one that makes the
    comparison meaningful, and this is what it costs. It says where the signal
    can exist -- wide rounds, few clusters, or a head that has stopped moving --
    and any A/B on `AdvantageAcceptance` has to be run somewhere it does.
    """
    wide = _advantages_seen(clusters=1)
    split = _advantages_seen(clusters=4)
    assert any(v is not None for v in wide)
    assert all(v is None for v in split), \
        "four clusters over four workers cannot fill a group of four"


# -- the consumers ----------------------------------------------------------


class _Recording:
    """An acceptance policy that only reports what it was handed."""

    def __init__(self):
        self.ctx = None

    def accept(self, ctx):
        self.ctx = ctx
        return AcceptDecision(accept=True, observed_delta=0.0)


class _Card:
    def __init__(self, advantage):
        self.advantage = advantage


def _ctx(cards=(), **kw):
    base = dict(artifact=None, candidate=None, cards=cards,
                base_counts=(1.0, 1.0), cand_counts=(2.0, 1.0))
    base.update(kw)
    return MergeContext(**base)


def test_advantage_acceptance_shifts_the_prior_it_passes_on():
    inner = _Recording()
    AdvantageAcceptance(inner, strength=2.0).accept(_ctx([_Card(1.5)]))
    assert inner.ctx.prior is not None
    assert inner.ctx.prior.successes == pytest.approx(3.0)
    assert inner.ctx.prior.failures == pytest.approx(0.0)


def test_a_negative_advantage_shifts_the_other_way():
    inner = _Recording()
    AdvantageAcceptance(inner, strength=2.0).accept(_ctx([_Card(-1.5)]))
    assert inner.ctx.prior.failures == pytest.approx(3.0)


def test_cards_without_an_advantage_are_skipped_not_counted_as_zero():
    inner = _Recording()
    sentinel = object()
    AdvantageAcceptance(inner).accept(_ctx([_Card(None)], prior=sentinel))
    assert inner.ctx.prior is sentinel, \
        "with nothing to say it must hand the context through untouched"


def test_advantage_acceptance_keeps_the_guards_it_wraps():
    """Wrapping, not replacing: the Beta test and the regression guard have a
    recorded history of bugs behind them, and a replacement drops all of it."""
    calls = []

    class Guarding:
        def accept(self, ctx):
            calls.append(ctx)
            return AcceptDecision(accept=False, category="below-threshold")

    decision = AdvantageAcceptance(Guarding()).accept(_ctx([_Card(9.0)]))
    assert calls, "the wrapped policy must still decide"
    assert decision.accept is False, "a huge advantage must not override a refusal"


def test_advantage_strength_must_not_be_negative():
    with pytest.raises(ValueError):
        AdvantageAcceptance(_Recording(), strength=-1.0)


# -- 2. the adaptive trust region -------------------------------------------


def test_it_does_not_move_before_it_has_a_full_window():
    region = AdaptiveTrustRegion(window=10)
    start = region.current
    for _ in range(9):
        assert region.observe("committed") == start


def test_a_clean_window_widens_it():
    region = AdaptiveTrustRegion(window=4, widen=2.0)
    for _ in range(4):
        region.observe("committed")
    assert region.current.ops == 12
    assert len(region.history) == 2


def test_a_reversal_tightens_it_even_among_successes():
    """Reversals are the expensive direction, so one outweighs a window of
    accepted merges."""
    region = AdaptiveTrustRegion(window=4, tighten=0.5)
    for outcome in ("committed", "committed", "committed", "oracle-rejected"):
        region.observe(outcome)
    assert region.current.ops == 3


def test_a_gate_that_is_refusing_does_not_move_the_region_either_way():
    """The merges are being refused on *quality*. Widening buys bigger proposals
    for a gate already saying no; tightening blames the region for something it
    did not do."""
    region = AdaptiveTrustRegion(window=4)
    start = region.current
    for _ in range(4):
        region.observe("below-threshold")
    assert region.current == start


def test_a_window_of_oversized_refusals_widens_the_region():
    """``oversized`` means the trust region itself rejected the diffs. Leaving
    the cap unchanged would keep refusing the same sizes; widen instead. This
    is the opposite of a gate refusal (``below-threshold``), which must not
    move the region either way."""
    region = AdaptiveTrustRegion(window=4, widen=2.0,
                                 initial=TrustRegion(ops=4, chars=1000))
    for _ in range(4):
        region.observe("oversized")
    assert region.current.ops == 8
    assert region.current.chars == 2000
    assert len(region.history) == 2


def test_a_minority_of_oversized_refusals_does_not_widen_by_itself():
    """Same threshold as accept_rate_to_widen: a single oversized among gate
    refusals is not enough evidence that the region is the limiter."""
    region = AdaptiveTrustRegion(window=4, widen=2.0,
                                 initial=TrustRegion(ops=4, chars=1000),
                                 accept_rate_to_widen=0.5)
    for outcome in ("below-threshold", "below-threshold",
                    "below-threshold", "oversized"):
        region.observe(outcome)
    assert region.current.ops == 4
    assert region.current.chars == 1000


def test_it_stops_at_its_bounds():
    """Unbounded growth ends with a trust region that constrains nothing, which
    is the same as not having one."""
    region = AdaptiveTrustRegion(window=2, widen=10.0,
                                 maximum=TrustRegion(8, 1_000))
    for _ in range(20):
        region.observe("committed")
    assert region.current.ops == 8 and region.current.chars == 1_000


def test_the_aggregator_honours_an_adaptive_region_rather_than_the_constants():
    """A region that moved while the size check read `config` would be recorded
    as adapting and change nothing."""
    from agentdescent.aggregator import Aggregator

    policy = AdaptiveTrustRegion(initial=TrustRegion(ops=1, chars=10))
    config = AggregatorConfig(trust_region_ops=99, trust_region_chars=10 ** 6,
                              trust_region_policy=policy)
    region = Aggregator._trust_region(type("A", (), {"config": config})())
    assert (region.ops, region.chars) == (1, 10)


def test_a_run_with_an_adaptive_region_still_finishes():
    result = _evolve(agg_config=AggregatorConfig(
        trust_region_policy=AdaptiveTrustRegion(window=2)))
    assert result.error is None


def test_no_policy_leaves_the_constants_in_charge():
    from agentdescent.aggregator import Aggregator

    config = AggregatorConfig(trust_region_ops=7, trust_region_chars=11)
    region = Aggregator._trust_region(type("A", (), {"config": config})())
    assert (region.ops, region.chars) == (7, 11)


# -- 3. distance from stable ------------------------------------------------


def test_distance_counts_a_missing_key_as_a_difference():
    """A deletion moves the artifact as much as an edit does."""
    assert state_distance({"a": "1"}, {}) == 1.0
    assert state_distance({"a": "1", "b": "2"}, {"a": "1"}) == 0.5
    assert state_distance({}, {}) == 0.0
    assert state_distance({"a": "1"}, {"a": "1"}) == 0.0


def test_the_penalty_refuses_rather_than_editing_the_score():
    """Scores here feed a Beta posterior. Quietly lowering one corrupts every
    number downstream that reads it."""
    class Accepting:
        def accept(self, ctx):
            return AcceptDecision(accept=True, observed_delta=0.01,
                                  detail="P(delta>0)=0.99")

    decision = StableDistanceAcceptance(Accepting(), strength=0.1).accept(
        _ctx(stable_distance=0.5))
    assert decision.accept is False
    assert decision.category == "below-threshold"
    assert "stable-distance penalty" in decision.detail
    assert decision.observed_delta == 0.01, "the measurement itself is untouched"


def test_a_large_enough_improvement_still_lands():
    class Accepting:
        def accept(self, ctx):
            return AcceptDecision(accept=True, observed_delta=0.5)

    assert StableDistanceAcceptance(Accepting(), strength=0.1).accept(
        _ctx(stable_distance=0.5)).accept is True


def test_no_stable_branch_means_no_penalty():
    """A branch that does not exist is not a reference to be far from."""
    class Accepting:
        def accept(self, ctx):
            return AcceptDecision(accept=True, observed_delta=0.0)

    assert StableDistanceAcceptance(Accepting()).accept(_ctx()).accept is True


def test_a_refusal_is_never_turned_into_an_acceptance():
    class Refusing:
        def accept(self, ctx):
            return AcceptDecision(accept=False, category="below-threshold")

    assert StableDistanceAcceptance(Refusing()).accept(
        _ctx(stable_distance=0.0)).accept is False


def test_the_aggregator_measures_the_distance_it_hands_to_the_policy():
    seen = []

    class Watching:
        def accept(self, ctx):
            seen.append(ctx.stable_distance)
            from agentdescent.defaults import DefaultAcceptance
            return DefaultAcceptance(0.5, 64, 4000).accept(ctx)

    from agentdescent.aggregator import Aggregator

    def factory(ledger, verifier, audit, config, policy):
        agg = Aggregator(ledger, verifier, audit, config, staleness_policy=policy)
        agg.acceptance_policy = Watching()
        return agg

    _evolve(aggregator_factory=factory)
    assert seen, "no candidate reached the acceptance gate"
    assert all(0.0 <= d <= 1.0 for d in seen)


# -- the conflict tie-break --------------------------------------------------


def test_advantage_conflict_defers_when_it_has_nothing_to_add():
    """It only fires where it knows something the default does not; otherwise the
    wrapped rule decides, unchanged."""
    calls = []

    class Inner:
        def resolve(self, artifact, cards):
            calls.append(cards)
            return list(cards)[:1], 1

    from agentdescent.evolvable import Diff

    def card(advantage, value):
        c = type("C", (), {})()
        c.diff = Diff(diff_id=f"d{value}", target="a", ops={"k": value})
        c.advantage = advantage
        return c

    policy = AdvantageConflict(Inner(), margin=0.5)
    kept, dropped = policy.resolve(None, [card(None, "x"), card(None, "y")])
    assert calls, "with no advantage on either side the inner rule must decide"
    assert dropped == 1 and len(kept) == 1


def test_advantage_conflict_keeps_the_higher_advantage_side():
    class Inner:
        def resolve(self, artifact, cards):
            raise AssertionError("should not be consulted for a clear margin")

    from agentdescent.evolvable import Diff

    def card(advantage, value):
        c = type("C", (), {})()
        c.diff = Diff(diff_id=f"d{value}", target="a", ops={"k": value})
        c.advantage = advantage
        return c

    kept, dropped = AdvantageConflict(Inner(), margin=0.5).resolve(
        None, [card(0.1, "weak"), card(2.0, "strong")])
    assert dropped == 1
    assert [c.diff.ops["k"] for c in kept] == ["strong"]


def test_an_inner_rule_that_keeps_both_does_not_spin():
    """A wrapped rule is user code. One that returns a contradicting pair intact
    would otherwise be asked the same question forever."""
    from agentdescent.evolvable import Diff

    class KeepsBoth:
        def resolve(self, artifact, cards):
            return list(cards), 0

    def card(value):
        c = type("C", (), {})()
        c.diff = Diff(diff_id=f"d{value}", target="a", ops={"k": value})
        c.advantage = None
        return c

    kept, dropped = AdvantageConflict(KeepsBoth()).resolve(
        None, [card("x"), card("y")])
    assert dropped == 0 and len(kept) == 2


# -- 2b. every move is a move, and a reversal never loosens (#282) ----------


@pytest.mark.parametrize("floor_ops", [1, 2])
def test_a_tightened_op_cap_can_widen_again(floor_ops):
    """``round(2 * 1.25) == 2`` and ``round(1 * 1.25) == 1``: with the default
    widen factor the op cap used to be stuck at its floor after two reversals,
    while ``chars`` kept growing and ``history`` kept saying it moved."""
    region = AdaptiveTrustRegion(window=1,
                                 initial=TrustRegion(ops=floor_ops, chars=8_000))
    region.observe("committed")
    assert region.current.ops == floor_ops + 1
    for _ in range(10):
        region.observe("committed")
    assert region.current.ops > floor_ops + 5


def test_two_reversals_then_clean_windows_recover_the_default_cap():
    region = AdaptiveTrustRegion(window=1)
    region.observe("oracle-rejected")
    region.observe("rolled-back")
    assert region.current.ops == 2
    for _ in range(4):
        region.observe("committed")
    assert [r.ops for r in region.history] == [6, 3, 2, 3, 4, 5, 6]


def test_tightening_moves_by_at_least_one_unit():
    region = AdaptiveTrustRegion(window=1, tighten=0.9,
                                 initial=TrustRegion(ops=3, chars=20_000))
    region.observe("oracle-rejected")
    # round(3 * 0.9) == 3 would have been "tightened" to the same cap.
    assert region.current == TrustRegion(ops=2, chars=18_000)


def test_a_reversal_never_loosens_a_cap_below_the_minimum():
    """``initial`` below ``minimum`` (as the #224/#226 regressions build it):
    clamping a tighten to ``minimum`` used to *quadruple* the char cap on a
    rollback."""
    region = AdaptiveTrustRegion(window=1,
                                 initial=TrustRegion(ops=3, chars=500))
    region.observe("oracle-rejected")
    assert region.current.chars == 500
    assert region.current.ops == 2
    region.observe("rolled-back")
    assert region.current.chars == 500 and region.current.ops == 1


def test_a_widen_still_respects_the_maximum():
    region = AdaptiveTrustRegion(window=1, widen=1.01,
                                 maximum=TrustRegion(7, 32_001))
    region.observe("committed")
    assert region.current == TrustRegion(7, 32_001)
    region.observe("committed")
    assert region.current == TrustRegion(7, 32_001)


@pytest.mark.parametrize("kwargs", [
    {"window": 0}, {"window": -1}, {"window": 2.5}, {"window": True},
    {"widen": 0.5}, {"widen": float("inf")}, {"widen": float("nan")},
    {"tighten": 0.0}, {"tighten": 1.5}, {"tighten": float("nan")},
    {"accept_rate_to_widen": -0.1}, {"accept_rate_to_widen": 1.1},
    {"accept_rate_to_widen": float("nan")},
    {"minimum": TrustRegion(0, 2_000)},
    {"maximum": TrustRegion(64, 0)},
    {"minimum": TrustRegion(1.5, 2_000)},
])
def test_invalid_knobs_are_refused_at_construction(kwargs):
    with pytest.raises(ValueError):
        AdaptiveTrustRegion(**kwargs)


def test_neutral_factors_are_allowed_and_hold_the_region():
    region = AdaptiveTrustRegion(window=1, widen=1.0, tighten=1.0)
    start = region.current
    region.observe("committed")
    region.observe("oracle-rejected")
    assert region.current == start and region.history == [start]


def test_a_maximum_below_the_default_minimum_still_caps_a_tighten():
    """``maximum`` wins where the bounds disagree, as before #282."""
    region = AdaptiveTrustRegion(window=1, maximum=TrustRegion(8, 1_000))
    region.observe("oracle-rejected")
    assert region.current == TrustRegion(3, 1_000)
