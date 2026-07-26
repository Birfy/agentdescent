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
