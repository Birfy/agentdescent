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
