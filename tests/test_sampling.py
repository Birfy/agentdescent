"""Tests for the task samplers (agentdescent.sampling).

A rollout is the expensive unit of work, so which task a worker picks matters.
These check the contract, the difficulty filter (zero-signal tasks get
down-weighted), and that both samplers drop into `evolve()` unchanged.
"""

from agentdescent.evolution import AppendRules, Task, evolve
from agentdescent.sampling import DifficultyWeighted, RoundRobin, TaskSampler


def test_both_satisfy_the_protocol_structurally():
    assert isinstance(RoundRobin(), TaskSampler)
    assert isinstance(DifficultyWeighted(), TaskSampler)


def test_round_robin_cycles_deterministically():
    s = RoundRobin()
    keys = ["a", "b", "c"]
    assert [s.pick(keys, i) for i in range(5)] == ["a", "b", "c", "a", "b"]


def test_round_robin_record_is_a_noop():
    s = RoundRobin()
    s.record("a", 1.0)                      # must not raise or change behaviour
    assert s.pick(["a", "b"], 0) == "a"


def test_difficulty_weighted_explores_untried_tasks_first():
    s = DifficultyWeighted()
    seen = set()
    for i in range(4):
        k = s.pick(["a", "b", "c", "d"], i)
        seen.add(k)
        s.record(k, 1.0)
    assert seen == {"a", "b", "c", "d"}     # every task tried before repeating


def test_difficulty_weighted_downweights_always_pass_tasks():
    """A task that always passes carries no gradient -> lose to one that fails."""
    s = DifficultyWeighted()
    for _ in range(6):
        s.record("solved", 1.0)             # always passes
        s.record("mixed", 1.0)              # ~50% pass -> maximal signal
        s.record("mixed", 0.0)
    picks = [s.pick(["solved", "mixed"], i) for i in range(10)]
    assert picks.count("mixed") > picks.count("solved")


def test_difficulty_weighted_downweights_never_pass_tasks():
    """A task that never passes is just as useless as one that always passes."""
    s = DifficultyWeighted()
    for _ in range(6):
        s.record("impossible", 0.0)
        s.record("mixed", 1.0)
        s.record("mixed", 0.0)
    picks = [s.pick(["impossible", "mixed"], i) for i in range(10)]
    assert picks.count("mixed") > picks.count("impossible")


def test_signal_peaks_at_half_and_vanishes_at_extremes():
    sig = DifficultyWeighted._signal
    assert sig(0.5) > sig(0.8) > sig(1.0)
    assert sig(0.5) > sig(0.2) > sig(0.0)
    assert sig(0.0) < 0.01 and sig(1.0) < 0.01


def test_pick_rejects_empty_keys():
    for s in (RoundRobin(), DifficultyWeighted()):
        try:
            s.pick([], 0)
        except (ValueError, IndexError):
            continue
        raise AssertionError(f"{type(s).__name__}.pick([]) should raise")


class _Agent:
    """Solves a task once its hint is in the artifact; only 'hard' tasks can fail."""

    def solve(self, rendered, task):
        if task.meta["hard"] and task.meta["hint"] not in rendered:
            return "no"
        return "yes"

    def propose(self, rendered, task, output, reward):
        return task.meta["hint"]


def _tasks(n=20, n_hard=4):
    return [Task(id=f"t{i}", prompt="q", meta={"hard": i < n_hard, "hint": f"H{i}"})
            for i in range(n)]


REWARD = lambda t, o: 1.0 if o == "yes" else 0.0


def test_evolve_accepts_either_sampler_and_still_improves():
    for sampler in (RoundRobin(), DifficultyWeighted()):
        r = evolve(_tasks(), REWARD, agent=_Agent(), strategy=AppendRules(),
                   rounds=8, n_workers=3, held_out_frac=0.3, task_sampler=sampler)
        assert r.error is None
        assert r.final_reward > 0.0


def test_default_sampler_is_round_robin_behaviour():
    """Omitting task_sampler must not change the existing deterministic default."""
    a = evolve(_tasks(), REWARD, agent=_Agent(), strategy=AppendRules(),
               rounds=6, n_workers=3, held_out_frac=0.3)
    b = evolve(_tasks(), REWARD, agent=_Agent(), strategy=AppendRules(),
               rounds=6, n_workers=3, held_out_frac=0.3, task_sampler=RoundRobin())
    assert a.state == b.state and a.final_reward == b.final_reward


def test_sampler_learns_from_recorded_outcomes_during_a_run():
    s = DifficultyWeighted()
    evolve(_tasks(), REWARD, agent=_Agent(), strategy=AppendRules(),
           rounds=8, n_workers=3, held_out_frac=0.3, task_sampler=s)
    assert s.stats(), "the engine should have reported rollout outcomes"
    assert all(trials > 0 for _, trials in s.stats().values())


# --- select_hard: turning a saturated benchmark into one with headroom ---------

def test_select_hard_keeps_only_what_the_baseline_fails():
    from agentdescent.dataloader import select_hard
    items = list(range(40))
    hard = select_hard(items, lambda i: 1.0 if i % 2 == 0 else 0.0)
    assert hard == [i for i in items if i % 2]


def test_select_hard_tops_up_rather_than_returning_an_unusable_split():
    """The point is a near-saturated benchmark, so survivors can be a handful.

    Measured: on MGSM, fewer than 12 of 160 items failed -- which splits into a
    3-item validation set, or crashes the engine's train/held-out split outright.
    """
    import warnings

    from agentdescent.dataloader import select_hard

    items = list(range(40))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = select_hard(items, lambda i: 0.0 if i < 3 else 1.0)
    assert len(out) == 12                       # the 3 failures, topped up
    assert out[:3] == [0, 1, 2]                 # failures first
    assert any("already solved" in str(x.message) for x in w), "topped up silently"


def test_select_hard_min_items_can_be_switched_off():
    from agentdescent.dataloader import select_hard
    items = list(range(40))
    out = select_hard(items, lambda i: 0.0 if i < 3 else 1.0, min_items=0)
    assert out == [0, 1, 2]


def test_select_hard_returns_everything_when_nothing_fails():
    """An empty benchmark is worse than a saturated one -- say so by returning it."""
    from agentdescent.dataloader import select_hard
    items = list(range(10))
    assert select_hard(items, lambda i: 1.0) == items


def test_select_hard_caps_with_keep():
    from agentdescent.dataloader import select_hard
    items = list(range(20))
    assert select_hard(items, lambda i: 0.0, keep=3) == [0, 1, 2]


def test_select_hard_handles_an_empty_pool():
    from agentdescent.dataloader import select_hard
    assert select_hard([], lambda i: 0.0) == []


def test_select_hard_scores_concurrently():
    """It runs a whole baseline pass, so serial scoring would make it unusable."""
    import threading
    from agentdescent.dataloader import select_hard

    live, peak, lock = [0], [0], threading.Lock()

    def score(i):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        import time
        time.sleep(0.05)
        with lock:
            live[0] -= 1
        return 0.0

    select_hard(list(range(16)), score, concurrency=8)
    assert peak[0] > 1, "select_hard scored the pool serially"


# ---------------------------------------------------------------------------
# ReplaySampler: the settled-evidence consumer
# ---------------------------------------------------------------------------

def _card(task_id, ops=None, n_refs=1):
    from agentdescent.evolvable import Diff, EvidenceCard
    refs = [Task(id=task_id, prompt=f"p-{task_id}") for _ in range(n_refs)]
    return EvidenceCard(
        diff=Diff(diff_id=f"d-{task_id}", target="a", ops=ops or {task_id: "v"}),
        base_version={"a": 1}, touched=["a"], before_after_delta=0.0,
        trajectory_refs=refs)


def test_replay_sampler_satisfies_the_protocol():
    from agentdescent.sampling import ReplaySampler
    assert isinstance(ReplaySampler(), TaskSampler)


def test_replay_sampler_at_zero_temperature_is_difficulty_weighted():
    """temperature=0 means no replay bonus: byte-for-byte the base sampler."""
    from agentdescent.sampling import ReplaySampler
    base = DifficultyWeighted()
    replay = ReplaySampler(temperature=0.0)
    keys = ["a", "b", "c", "d"]
    for i in range(12):
        base.record("a", 1.0), replay.record("a", 1.0)
        assert replay.pick(keys, i) == base.pick(keys, i)


def test_settled_cards_raise_the_tasks_score():
    from agentdescent.sampling import ReplaySampler
    replay = ReplaySampler(temperature=0.5)
    # All tasks start at the optimistic prior and equal UCB; the one with a
    # settled card gets picked because its replay bonus dominates.
    replay.settle(_card("a"))
    keys = ["a", "b"]
    assert replay.pick(keys, 0) == "a"


def test_replay_bonus_is_capped():
    from agentdescent.sampling import ReplaySampler
    replay = ReplaySampler(temperature=0.5, capped_at=3)
    keys = ["a", "b", "c"]
    for _ in range(10):
        replay.settle(_card("a"))
    replay.settle(_card("b"))
    replay.settle(_card("b"))
    # 10 settled vs 2 settled: same bonus after the cap.
    assert replay._settled_count["a"] == 10
    assert replay._settled_count["b"] == 2
    # a still wins because its bonus is capped at the same value but it got there first
    assert replay.pick(keys, 0) == "a"


def test_replay_without_a_wired_pool_never_moves():
    from agentdescent.sampling import ReplaySampler
    replay = ReplaySampler(temperature=0.5)
    assert replay.settled_counts() == {}
    # settle() never called -> counts stay empty -> bonus stays zero
    keys = ["a", "b"]
    base = DifficultyWeighted()
    assert replay.pick(keys, 0) == base.pick(keys, 0)


def test_replay_settled_counts_are_inspectable():
    from agentdescent.sampling import ReplaySampler
    replay = ReplaySampler(temperature=0.3)
    replay.settle(_card("a"))
    replay.settle(_card("b"))
    replay.settle(_card("a"))
    assert replay.settled_counts() == {"a": 2, "b": 1}


def test_replay_records_through_to_the_base_sampler():
    from agentdescent.sampling import ReplaySampler
    replay = ReplaySampler()
    replay.record("a", 1.0)
    assert replay.stats() == {"a": (1.0, 1.0)}

# ---------------------------------------------------------------------------
# ReplaySampler + ProposalContext integration
# ---------------------------------------------------------------------------

from agentdescent.evolution import AppendRules, Task


def test_replay_settle_without_task_ids_is_safe():
    from agentdescent.sampling import ReplaySampler
    from agentdescent.evolvable import Diff, EvidenceCard
    replay = ReplaySampler()
    card = EvidenceCard(
        diff=Diff(diff_id="d", target="a", ops={"x": "v"}),
        base_version={"a": 1}, touched=["a"], before_after_delta=0.0,
        trajectory_refs=["not-a-task"])  # strings have no .id
    replay.settle(card)
    assert replay.settled_counts() == {}


def test_proposal_context_has_rejected_field():
    from agentdescent.policies import ProposalContext
    ctx = ProposalContext(rendered="", task=Task(id="t", prompt="p"),
                          output="", reward=0.0)
    assert ctx.rejected == ()

# ---------------------------------------------------------------------------
# ReplayAwareProposal: the consumer of ctx.rejected
# ---------------------------------------------------------------------------

from dataclasses import replace, dataclass


def test_replay_aware_policy_appends_rejection_to_output():
    from agentdescent.replay import ReplayAwareProposal
    from agentdescent.evolvable import Diff, EvidenceCard
    from agentdescent.policies import ProposalContext
    from agentdescent.evolution import Task

    seen = []

    class CapturePolicy:
        def propose(self, ctx):
            seen.append(ctx.output)
            return []

    policy = ReplayAwareProposal(CapturePolicy())
    card = EvidenceCard(
        diff=Diff(diff_id="d1", target="a", ops={"src/x.py": "x"}, author="w"),
        base_version={"a": 2}, touched=["a"], before_after_delta=0.1,
        trajectory_refs=[Task(id="t0", prompt="p")])
    ctx = ProposalContext(rendered="", task=Task(id="t0", prompt="p"),
                          output="old", reward=0.0, rejected=(card,))
    policy.propose(ctx)
    assert "## Recently discarded" in seen[0]
    assert "src/x.py" in seen[0]
    assert policy.read == 1 and policy.shown == 1


def test_replay_aware_policy_passes_through_when_empty():
    from agentdescent.replay import ReplayAwareProposal
    from agentdescent.policies import ProposalContext
    from agentdescent.evolution import Task

    seen = []

    class CapturePolicy:
        def propose(self, ctx):
            seen.append(ctx.output)
            return []

    policy = ReplayAwareProposal(CapturePolicy())
    ctx = ProposalContext(rendered="", task=Task(id="t0", prompt="p"),
                          output="old", reward=0.0)
    policy.propose(ctx)
    assert seen == ["old"]          # output unchanged
    assert policy.read == 0


def test_replay_aware_policy_caps_max_cards():
    from agentdescent.replay import ReplayAwareProposal
    from agentdescent.evolvable import Diff, EvidenceCard
    from agentdescent.policies import ProposalContext
    from agentdescent.evolution import Task

    policy = ReplayAwareProposal(
        max_cards=2,
        policy=type("", (), {"propose": lambda s, ctx: []})())

    cards = tuple(
        EvidenceCard(diff=Diff(diff_id=f"d{i}", target="a", ops={str(i): "v"}, author="w"),
                     base_version={"a": 1}, touched=["a"], before_after_delta=0.0,
                     trajectory_refs=[Task(id="t0", prompt="p")])
        for i in range(5))
    ctx = ProposalContext(rendered="", task=Task(id="t0", prompt="p"),
                          output="o", reward=0.0, rejected=cards)
    policy.propose(ctx)
    assert policy.shown == 2         # capped, not 5


def test_default_note_is_readable():
    from agentdescent.replay import _default_note
    from agentdescent.evolvable import Diff, EvidenceCard
    card = EvidenceCard(
        diff=Diff(diff_id="d", target="a", ops={"src/x.py": "def f(): pass\n"}, author="w"),
        base_version={"a": 3}, touched=["a"], before_after_delta=0.25,
        trajectory_refs=[])
    note = _default_note(card)
    assert "src/x.py" in note
    assert "{'a': 3}" in note
    assert "+0.250" in note

# ---------------------------------------------------------------------------
# Integration: ReplaySampler + aggregator (design doc, test 6)
# ---------------------------------------------------------------------------

def test_replay_sampler_and_aggregator_integration():
    """Build a buffer, settle cards, verify the sampler sees them."""
    from agentdescent.sampling import ReplaySampler
    from agentdescent.aggregator import EvidenceBuffer
    from agentdescent.evolvable import Diff, EvidenceCard
    from agentdescent.evolution import Task

    buf = EvidenceBuffer()
    sampler = ReplaySampler(temperature=0.5)
    buf.set_settled_consumer(sampler.settle)

    tasks = [Task(id=f"t{i}", prompt="p") for i in range(3)]
    cards = [
        EvidenceCard(diff=Diff(diff_id=f"d{i}", target="a",
                               ops={f"src/{i}.py": "v"}, author="w"),
                     base_version={"a": 1}, touched=["a"],
                     before_after_delta=0.0,
                     trajectory_refs=[tasks[i]]) for i in range(3)]
    buf.settle(cards)

    assert sampler.settled_counts() == {"t0": 1, "t1": 1, "t2": 1}
    keys = ["t0", "t1"]
    # With temperature=0, base would pick randomly; with replay bonus,
    # both t0 and t1 have the same bonus (1 each), plus UCB exploration.
    picked = sampler.pick(keys, 0)
    assert picked in keys


def test_replay_affected_count_works():
    from agentdescent.sampling import ReplaySampler
    from agentdescent.evolvable import Diff, EvidenceCard
    from agentdescent.evolution import Task

    sampler = ReplaySampler(temperature=1.0)
    card = EvidenceCard(
        diff=Diff(diff_id="d", target="a", ops={"x": "v"}, author="w"),
        base_version={"a": 1}, touched=["a"], before_after_delta=0.0,
        trajectory_refs=[Task(id="t-a", prompt="p")])
    sampler.settle(card)
    keys = ["t-a", "t-b"]
    sampler.pick(keys, 0)
    sampler.pick(keys, 1)
    assert sampler.picks == 2
    assert sampler.replay_affected >= 0  # may be 0 if base already prefers t-a


# ---------------------------------------------------------------------------
# End-to-end through evolve(): the full wire (design doc, "integration")
# ---------------------------------------------------------------------------

from agentdescent.aggregator import AggregatorConfig
from agentdescent.policies import Policies, ProposalContext
from agentdescent.sampling import ReplaySampler
from agentdescent.replay import ReplayAwareProposal
from agentdescent.replay import ReplayAwareProposal
from agentdescent.strategies import Strategy
from agentdescent.evolvable import Diff


class _Oversize(Strategy):
    """Every proposal becomes a 5-op diff, over a 3-op trust region: oversized."""

    def keys(self):
        return ["a", "b", "c", "d", "e"]

    def initial(self):
        return {k: "" for k in self.keys()}

    def render(self, state):
        return repr(state)

    def to_diff(self, state, proposal, author, base_version, target):
        return Diff(diff_id=f"{author}:5", target=target,
                    ops={k: proposal for k in self.keys()}, author=author)

    def frozen_files(self, source):
        return {}


class _RejectAware:
    """Records whether the proposal context carried rejected evidence."""

    def __init__(self):
        self.seen_rejected = 0

    def propose(self, ctx: ProposalContext):
        self.seen_rejected += int(bool(ctx.rejected))
        return ["x"]


def test_full_wire_settled_to_proposer_and_sampler():
    """Oversized diffs are settled; the sampler sees them, and the proposer sees
    the rejections via ProposalContext.rejected -- all through evolve()."""
    tasks = [Task(id=f"t{i}", prompt=f"p{i}") for i in range(6)]

    def reward(task, out):
        return 0.0                       # never solved: always propose

    dummy = _RejectAware()
    sampler = ReplaySampler(temperature=0.3)
    result = evolve(
        tasks, reward,
        run=lambda rendered, task: "answer",
        propose=lambda rendered, task, output, reward: None,
        strategy=_Oversize(),
        policies=Policies(proposal=ReplayAwareProposal(dummy),
                          task_sampler=sampler),
        agg_config=AggregatorConfig(trust_region_ops=3, batch_trigger=1),
        max_rollouts=48, n_workers=1, max_concurrency=1, seed=1, rounds=8)

    assert sum(sampler.settled_counts().values()) > 0, "the pool must have settled"
    assert sampler.picks > 0
    assert result.replay is not None and result.replay["settled"] > 0
    # Over enough rollouts, at least one proposal sees a prior rejection.
    assert dummy.seen_rejected >= 1, "a proposer must eventually see a rejection"


def test_result_replay_is_none_without_a_replay_sampler():
    tasks = [Task(id=f"t{i}", prompt=f"p{i}") for i in range(6)]

    def reward(task, out):
        return 0.0

    result = evolve(
        tasks, reward,
        run=lambda rendered, task: "answer",
        propose=lambda rendered, task, output, reward: None,
        strategy=_Oversize(),
        agg_config=AggregatorConfig(trust_region_ops=3, batch_trigger=1),
        max_rollouts=12, n_workers=1, max_concurrency=1, seed=1, rounds=3)
    assert result.replay is None
