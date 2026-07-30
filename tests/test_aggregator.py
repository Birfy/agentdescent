from agentdescent.aggregator import (
    Aggregator,
    AggregatorConfig,
    diffs_contradict,
    fuse_diffs,
)
from agentdescent.domains.router import (
    RouterSkill,
    Task,
    deserialize_router,
    router_eval,
    serialize_router,
)
from agentdescent.evolvable import Diff, EvidenceCard
from agentdescent.ledger import Ledger
from agentdescent.scheduler import AuditScheduler
from agentdescent.verifier import ThreeLayerVerifier, VerifierBudget


def _card(target, ops, base_v, tasks, delta, author="w"):
    diff = Diff(diff_id=f"{author}:{'-'.join(ops)}", target=target, ops=ops, author=author)
    return EvidenceCard(
        diff=diff,
        base_version={target: base_v},
        touched=[target],
        before_after_delta=delta,
        trajectory_refs=tasks,
    )


def _build(tmp_path, held_out):
    led = Ledger(str(tmp_path / "repo"), serialize_router, deserialize_router)
    led.register(RouterSkill("s", table={}))
    verifier = ThreeLayerVerifier(router_eval, held_out, seed=0,
                                  budget=VerifierBudget(oracle_calls_remaining=100))
    agg = Aggregator(led, verifier, AuditScheduler(),
                     AggregatorConfig(batch_trigger=2, base_delta=0.4))
    return led, agg


def test_fuse_is_union_of_complementary_ops():
    a = Diff("a", "s", {"kw00": "L1"})
    b = Diff("b", "s", {"kw01": "L2"})
    fused = fuse_diffs([a, b])
    assert fused.ops == {"kw00": "L1", "kw01": "L2"}
    assert not diffs_contradict(a, b)


def test_contradiction_detected():
    a = Diff("a", "s", {"kw00": "L1"})
    b = Diff("b", "s", {"kw00": "L2"})
    assert diffs_contradict(a, b)


def test_complementary_diffs_get_fused_and_committed(tmp_path):
    held = [Task(f"t-kw00-{i}", "acidbase", "kw00") for i in range(6)] + \
           [Task(f"t-kw01-{i}", "kinetics", "kw01") for i in range(6)]
    led, agg = _build(tmp_path, held)
    tasks0 = [Task("t-kw00-0", "acidbase", "kw00")]
    tasks1 = [Task("t-kw01-0", "kinetics", "kw01")]
    agg.ingest(_card("s", {"kw00": "acidbase"}, 1, tasks0, 1.0, "w0"))
    agg.ingest(_card("s", {"kw01": "kinetics"}, 1, tasks1, 1.0, "w1"))
    reports = agg.step()
    assert len(reports) == 1
    rep = reports[0]
    assert rep.committed_version is not None
    committed = led.snapshot(Ledger.DEV).get("s")
    # the merged skill carries BOTH fixes -- the whole point of merge-over-fork.
    assert committed.table == {"kw00": "acidbase", "kw01": "kinetics"}


def test_contradictory_diff_is_dropped(tmp_path):
    held = [Task(f"t-kw00-{i}", "acidbase", "kw00") for i in range(8)]
    led, agg = _build(tmp_path, held)
    tasks = [Task("t-kw00-0", "acidbase", "kw00")]
    # correct proposal + wrong proposal for the same keyword.
    agg.ingest(_card("s", {"kw00": "acidbase"}, 1, tasks, 1.0, "good"))
    agg.ingest(_card("s", {"kw00": "thermo"}, 1, tasks, -1.0, "bad"))
    reports = agg.step()
    assert reports[0].conflicts_dropped >= 1
    committed = led.snapshot(Ledger.DEV).get("s")
    # the correct label wins.
    assert committed.table.get("kw00") == "acidbase"


def test_conflict_resolution_leaves_no_contradictions():
    """A card that displaces one survivor may contradict another.

    Resolution stopped at the first conflict, so a replacement was never
    re-checked against the rest of `kept`. Mutually contradicting cards then
    survived together, which made the tournament's "no contradictions" guard
    false and silently skipped building the fused candidate -- losing the
    model-soup benefit the aggregator is built around.
    """
    from agentdescent.aggregator import Aggregator, diffs_contradict
    from agentdescent.evolvable import Diff, EvidenceCard
    from agentdescent.evolution import EvolvingArtifact, KeyedRules

    art = EvolvingArtifact("a", {}, 1, 0.2, None, KeyedRules(categories=["k1", "k2"]))

    def card(did, ops):
        return EvidenceCard(diff=Diff(diff_id=did, target="a", ops=ops, author=did),
                            base_version={"a": 1}, touched=["a"],
                            before_after_delta=0.1, trajectory_refs=[])

    class _Stub:                       # 'z' wins, so C displaces A
        def cheap_eval(self, artifact):
            return 0.9 if artifact.state.get("k1") == "z" else 0.1

    agg = Aggregator.__new__(Aggregator)
    agg.verifier = _Stub()

    # A and B are complementary (different keys); C contradicts BOTH.
    cards = [card("A", {"k1": "x"}), card("B", {"k2": "y"}),
             card("C", {"k1": "z", "k2": "w"})]
    kept, dropped = Aggregator._resolve_conflicts(agg, art, cards)

    residual = [(a.diff.diff_id, b.diff.diff_id)
                for i, a in enumerate(kept) for b in kept[i + 1:]
                if diffs_contradict(a.diff, b.diff)]
    assert not residual, f"contradicting cards survived together: {residual}"
    assert dropped == 2


def test_conflict_resolution_keeps_complementary_cards():
    """Cards touching different keys must both survive -- that is what fusion needs."""
    from agentdescent.aggregator import Aggregator
    from agentdescent.evolvable import Diff, EvidenceCard
    from agentdescent.evolution import EvolvingArtifact, KeyedRules

    art = EvolvingArtifact("a", {}, 1, 0.2, None, KeyedRules(categories=["k1", "k2"]))

    def card(did, ops):
        return EvidenceCard(diff=Diff(diff_id=did, target="a", ops=ops, author=did),
                            base_version={"a": 1}, touched=["a"],
                            before_after_delta=0.1, trajectory_refs=[])

    class _Stub:
        def cheap_eval(self, artifact):
            return 0.5

    agg = Aggregator.__new__(Aggregator)
    agg.verifier = _Stub()
    kept, dropped = Aggregator._resolve_conflicts(
        agg, art, [card("A", {"k1": "x"}), card("B", {"k2": "y"})])
    assert len(kept) == 2 and dropped == 0


def test_oversized_diffs_are_counted_and_settled(tmp_path):
    """Trust-region rejects used to vanish from both the report and the pool.

    They were filtered out before `considered` was computed and never settled, so
    a diff dropped for being too large left no trace anywhere.
    """
    from agentdescent.aggregator import Aggregator, AggregatorConfig
    from agentdescent.evolvable import Diff, EvidenceCard
    from agentdescent.evolution import AppendRules, EvolvingArtifact
    from agentdescent.ledger import Ledger
    from agentdescent.scheduler import AuditScheduler
    from agentdescent.verifier import ThreeLayerVerifier, VerifierBudget

    lg = Ledger(str(tmp_path), lambda a: {"state": dict(a.state)},
                lambda aid, v, s: EvolvingArtifact(aid, s.get("state", {}), v,
                                                   0.2, None, AppendRules()))
    lg.register(EvolvingArtifact("a", {}, 1, 0.2, None, AppendRules()))
    verifier = ThreeLayerVerifier(eval_fn=lambda art, tasks: 0.5, held_out=[1, 2, 3],
                                  budget=VerifierBudget())
    agg = Aggregator(lg, verifier, AuditScheduler(),
                     AggregatorConfig(batch_trigger=1, trust_region_ops=2))

    # one diff inside the trust region, one far outside it
    small = Diff(diff_id="small", target="a", ops={"k1": "v"}, author="w0")
    huge = Diff(diff_id="huge", target="a",
                ops={f"k{i}": "v" for i in range(10)}, author="w1")
    for d in (small, huge):
        agg.ingest(EvidenceCard(diff=d, base_version={"a": 1}, touched=["a"],
                                before_after_delta=0.1, trajectory_refs=[]))

    reports = agg.step()
    assert reports, "expected a merge report"
    assert reports[0].considered == 2, (
        f"the oversized diff should still be counted, got {reports[0].considered}")
    assert any(c.diff.diff_id == "huge" for c in agg.buffer.settled), \
        "the oversized diff's evidence should be settled back into the pool"
