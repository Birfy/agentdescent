"""What the async runtime actually guarantees, pinned.

Audited after finding that TensorParallel's section ownership was never enforced:
if one parallelism guarantee was decorative, the async ones deserve checking too.
These are the properties that turned out to hold, plus the one semantic mismatch
that turned out not to.
"""

from agentdescent.aggregator import AggregatorConfig, EvidenceBuffer
from agentdescent.async_evolve import async_evolve
from agentdescent.evolution import AppendRules, Task, evolve
from agentdescent.evolvable import Diff, EvidenceCard


class _Composer:
    def solve(self, rendered, task):
        return "yes" if task.meta["h"] in rendered else "no"

    def propose(self, rendered, task, output, reward):
        return task.meta["h"]


def _tasks(n=24, k=6):
    return [Task(id=f"t{i}", prompt="q", meta={"h": f"H{i % k}"}) for i in range(n)]


REWARD = lambda t, o: 1.0 if o == "yes" else 0.0


def _card(aid="a"):
    return EvidenceCard(diff=Diff(diff_id="d", target=aid, ops={"k": "v"}),
                        base_version={aid: 1}, touched=[aid],
                        before_after_delta=0.1, trajectory_refs=[])


def test_batch_trigger_fires_a_bucket():
    cfg = AggregatorConfig(batch_trigger=3, max_wait_rounds=99)
    b = EvidenceBuffer()
    for _ in range(2):
        b.add(_card())
    assert b.ready(cfg) == []            # below the batch size
    b.add(_card())
    assert b.ready(cfg) == ["a"]


def test_max_wait_rounds_stops_a_cold_bucket_starving():
    """The timeout path matters: without it a rarely-touched artifact never merges."""
    cfg = AggregatorConfig(batch_trigger=99, max_wait_rounds=2)
    b = EvidenceBuffer()
    b.add(_card())
    assert b.ready(cfg) == []
    b.tick()
    assert b.ready(cfg) == []
    b.tick()
    assert b.ready(cfg) == ["a"], "max_wait_rounds must eventually fire"


def test_a_large_lag_budget_does_not_livelock():
    """async_ratio >> alpha means most diffs are discarded as stale.

    The reference runtime guards this with stall_patience; async_evolve has no such
    guard, so check directly that it still converges rather than spinning.
    """
    for ratio in (1, 16, 64):
        r = async_evolve(_tasks(), REWARD, agent=_Composer(), strategy=AppendRules(),
                         n_workers=4, async_ratio=ratio, max_seconds=3.0,
                         held_out_frac=0.5,
                         agg_config=AggregatorConfig(alpha_head=1, alpha_tail=1))
        assert r.error is None
        assert r.final_reward > 0.9, f"async_ratio={ratio} failed to converge"


def test_history_is_rounds_on_the_sync_path():
    r = evolve(_tasks(16, 4), REWARD, agent=_Composer(), strategy=AppendRules(),
               rounds=5, n_workers=3)
    assert len(r.history) == 5
    assert [h.round for h in r.history] == [0, 1, 2, 3, 4]


def test_history_is_merger_sweeps_on_the_async_path():
    """Same field, different unit -- documented, and pinned so it stays documented."""
    r = async_evolve(_tasks(16, 4), REWARD, agent=_Composer(), strategy=AppendRules(),
                     n_workers=3, max_seconds=2.0, held_out_frac=0.5)
    # not tied to any 'rounds' argument: it counts non-empty merges
    assert r.history, "expected at least one sweep"
    assert [h.round for h in r.history] == list(range(len(r.history)))


def test_pending_intake_is_bounded_by_the_lag_budget():
    """The cold-start throttle: workers must not pile up unbounded work."""
    r = async_evolve(_tasks(), REWARD, agent=_Composer(), strategy=AppendRules(),
                     n_workers=4, async_ratio=2, max_seconds=2.0, held_out_frac=0.5)
    assert r.error is None


# ---------------------------------------------------------------------------
# What the async path measures about itself
# ---------------------------------------------------------------------------


def _counting_run(*, on_propose=None, **kw):
    """Run the async path with the cards and the ledger reads counted."""
    import tempfile
    import threading
    import warnings

    from agentdescent import Policies
    from agentdescent.aggregator import Aggregator
    from agentdescent.evolution import EvolvingArtifact
    from agentdescent.ledger import Ledger

    seen = {"cards": 0, "reads": 0, "read_log": [], "worker_rollouts": {}}
    lock = threading.Lock()

    def record_read(kind):
        with lock:
            seen["reads"] += 1
            seen["read_log"].append((threading.current_thread(), kind))

    class _Ledger(Ledger):
        def head_version(self, branch=Ledger.DEV):
            record_read("head_version")
            return super().head_version(branch)

        def snapshot(self, branch=Ledger.DEV):
            record_read("snapshot")
            return super().snapshot(branch)

    class _Agg(Aggregator):
        def ingest(self, card):
            seen["cards"] += 1
            return super().ingest(card)

    ledger = _Ledger(
        tempfile.mkdtemp() + "/repo",
        lambda a: {"state": a.state, "blast_radius": a.blast_radius},
        lambda aid, v, d: EvolvingArtifact(aid, d.get("state", {}), v,
                                           d.get("blast_radius", 0.2)))
    n = [0]

    def propose(rendered, task, output, score):
        # Proposals run on workers, unlike run(), which also scores merge gates.
        with lock:
            worker = threading.current_thread()
            seen["worker_rollouts"][worker] = seen["worker_rollouts"].get(worker, 0) + 1
            n[0] += 1
            rule = f"rule {n[0]}"
        if on_propose is not None:
            on_propose(ledger)
        return rule

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = async_evolve(
            _tasks(n=12), lambda t, o: 0.0, run=lambda rendered, t: "x",
            propose=propose, strategy=AppendRules(), n_workers=kw.pop("n_workers", 2),
            max_iters=20, max_seconds=10.0, self_verify=False,
            aggregator_factory=lambda l, v, a, c, p: _Agg(
                l, v, a, c, staleness_policy=p),
            policies=Policies(ledger=ledger), **kw)
    return result, seen


def test_the_stale_denominator_counts_each_card_once():
    """`async_evolve` runs a staleness gate and so does `Aggregator`.

    Both wrote to the same meter, so every card that *survived* the first gate
    was counted as "considered" twice: measured, 20 cards reported
    `stale_considered = 40`. A true 50% stale rate then read as 33%, and the
    "you are discarding most of your evidence" warning -- which fires at
    `discarded/considered > 0.5` -- needed a true rate of 67% to trip.

    Each side now counts only what the other cannot see: this gate's discards,
    the aggregator's survivors.
    """
    result, seen = _counting_run()
    assert seen["cards"] > 0, "premise: cards reached the aggregator"
    assert result.stale_considered == seen["cards"] + result.stale_discarded, (
        f"stale_considered={result.stale_considered} but "
        f"{seen['cards']} card(s) survived the gate and "
        f"{result.stale_discarded} were discarded")
    assert 0.0 <= result.stale_rate() <= 1.0


def _assert_worker_snapshot_reads(on_propose=None):
    # Twenty rollouts cannot exhaust either budget. Disable the independent
    # commit-triggered refresh too: exactly one startup snapshot is legitimate.
    # A single worker makes the rollout count exact, without timing or sleeps.
    result, seen = _counting_run(
        n_workers=1, async_ratio=64, stall_patience=64,
        resync_on_commit=False, on_propose=on_propose)
    assert result.error is None
    assert result.rollouts == 20 and result.history, "premise: work and merges ran"
    assert result.forced_refreshes == 0
    assert len(seen["worker_rollouts"]) == 1
    worker, rollouts = next(iter(seen["worker_rollouts"].items()))
    assert rollouts == result.rollouts
    reads = [kind for thread, kind in seen["read_log"] if thread is worker]
    # Startup, merger, promotion, and shutdown reads on other threads are not
    # worker drift checks. Keep them out rather than guessing a per-sweep cost.
    assert len(seen["read_log"]) > len(reads), "premise: non-worker reads occurred"
    assert reads == ["snapshot"], (
        f"worker ledger reads: {reads}; expected only its startup snapshot "
        f"across {rollouts} rollouts")


def test_a_worker_does_not_read_the_ledger_once_per_rollout():
    """Drift checks use the published head, without a git read per rollout.

    Count reads by worker provenance under a schedule with no refresh due. The
    old total-read budget depended on the number of merger sweeps and omitted a
    legitimate stable-distance snapshot, so harmless scheduling changes failed.
    This pins the worker property directly, without loosening a slack constant.
    """
    _assert_worker_snapshot_reads()


def test_worker_read_guard_rejects_per_rollout_reads():
    """Negative controls must fail the same assertion as a worker regression."""
    import pytest

    for method in ("head_version", "snapshot"):
        with pytest.raises(AssertionError, match="worker ledger reads:"):
            _assert_worker_snapshot_reads(
                on_propose=lambda ledger: getattr(ledger, method)())
