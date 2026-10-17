"""The persistent-island population engine has to be right in the places a
parallel population search is dangerous.

The point of an island is that its lineage *survives* across steps -- that is
what lets a tree-shaped policy (CladeThompson, FlatPuct) score a clade instead
of a single node. So the fixtures pin: the default (``island_count=0``) is the
old per-step behaviour with no forks left behind, persistent islands are
created and checkpointed, and a resumed run keeps its islands and lineages.
"""

import pytest

from agentdescent import Task, evolve
from agentdescent.population import PopulationAggregator, population_factory
from agentdescent.selection import CladeThompson, SingleHead


def _cap(rendered):
    import re
    m = re.search(r"cap:(\d+)", rendered)
    return int(m.group(1)) if m else 0


def _run(rendered, task):
    return str(task.meta["gold"]) if task.meta.get("a", 0) <= _cap(rendered) else "?"


def _reward(task, output):
    return 1.0 if str(output).strip() == str(task.meta["gold"]).strip() else 0.0


def _propose(rendered, task, output, reward):
    if reward >= 0.999:
        return None
    if task.meta.get("a", 0) > _cap(rendered):
        return f"cap:{_cap(rendered) + 1}"
    return None


def _tasks(prefix="s", n=8):
    order = [1, 2, 3, 4, 8, 5, 6, 7][:n]
    return [Task(id=f"{prefix}{i}", prompt=f"item {i}",
                 meta={"gold": str(2 * a), "a": a}) for i, a in enumerate(order)]


def _run_evolve(*, islands, repo, rounds=10, n_workers=3):
    from agentdescent import Policies
    from agentdescent.evolution import SingleSlot
    return evolve(
        _tasks(), _reward, run=_run, propose=_propose,
        strategy=SingleSlot(initial_value="cap:0"),
        policies=Policies(selection=CladeThompson()),
        rounds=rounds, n_workers=n_workers,
        max_concurrency=n_workers, repo_path=repo,
        population_islands=islands, target_reward=1.0)


def _live_heads(repo):
    from agentdescent.ledger import Ledger
    ledger = Ledger(repo, serialize=lambda a: {"state": a.state},
                    deserialize=lambda aid, v, s: None)
    return ledger.live_heads()


# -- the default is the old behaviour --------------------------------------------


def test_zero_islands_keeps_the_ephemeral_behaviour(tmp_path):
    repo = str(tmp_path / "repo")
    result = _run_evolve(islands=0, repo=repo, rounds=10, n_workers=2)
    assert result.outcomes().get("committed", 0) >= 1
    # forks were discarded at finalize: the ledger ends with dev only
    assert _live_heads(repo) == ["dev"]


def test_zero_islands_works_with_the_default_single_head(tmp_path):
    repo = str(tmp_path / "repo")
    result = evolve(_tasks(), _reward, run=_run, propose=_propose,
                    strategy=_slots(), rounds=10, n_workers=2, repo_path=repo,
                    population_islands=0, target_reward=1.0)
    assert result.outcomes().get("committed", 0) >= 1


# -- persistent islands ----------------------------------------------------------


def test_islands_run_end_to_end_and_are_reaped_at_finalize(tmp_path):
    repo = str(tmp_path / "repo")
    result = _run_evolve(islands=2, repo=repo, rounds=8, n_workers=3)
    assert result.outcomes().get("committed", 0) >= 1
    assert _live_heads(repo) == ["dev"]          # islands reaped cleanly

# -- the aggregator in isolation ------------------------------------------------


def _make_aggregator(tmp_path, island_count, selection=None):
    from agentdescent.aggregator import AggregatorConfig
    from agentdescent.evolution import EvolvingArtifact, SingleSlot, _Runtime
    from agentdescent.evalcache import MemoryCache
    from agentdescent.ledger import Ledger
    from agentdescent.policies import Policies
    from agentdescent.scheduler import AuditScheduler
    from agentdescent.selection import CladeThompson
    from agentdescent.verifier import ThreeLayerVerifier

    repo = str(tmp_path / "repo")
    runtime = _Runtime(run=_run, reward=_reward, cache=MemoryCache())
    strategy = SingleSlot(initial_value="cap:0")

    def serialize(a):
        return {"state": a.state}

    def deserialize(aid, version, state):
        return EvolvingArtifact(aid, state.get("state", {}), version,
                                runtime=runtime, strategy=strategy)

    ledger = Ledger(repo, serialize, deserialize)
    art = EvolvingArtifact("artifact", strategy.initial(), runtime=runtime,
                           strategy=strategy)
    ledger.register(art)
    verifier = ThreeLayerVerifier(
        eval_fn=lambda a, ts: a.score(ts),
        held_out=[_tasks()[5]], rule_subset=1)
    selection = selection or CladeThompson(seed=0)
    agg = PopulationAggregator(
        ledger, verifier, AuditScheduler(), AggregatorConfig(batch_trigger=1,
                                                             max_wait_rounds=1),
        selection=selection, artifact_id="artifact", island_count=island_count)
    return agg, ledger, runtime, strategy


def _seed_archive(agg, runtime, strategy):
    # pre-populate the archive so the selection has a population to choose
    # from (the direct-drive test has no workers committing new candidates)
    from agentdescent.evolution import EvolvingArtifact
    for i, cap in enumerate([5, 6, 7]):
        art = EvolvingArtifact("artifact", {"value": f"cap:{cap}"},
                               runtime=runtime, strategy=strategy)
        agg._admit(art, version=i + 1)


def test_islands_are_created_and_persist_across_steps(tmp_path):
    agg, ledger, runtime, strategy = _make_aggregator(tmp_path, island_count=2)
    _seed_archive(agg, runtime, strategy)
    for _ in range(3):
        agg.step(n_workers=3)
    assert len(agg._forks) == 2
    assert agg._forks == ["head/0", "head/1"]
    # the forks persist across steps (not recreated each step)
    prev = list(agg._forks)
    agg.step(n_workers=3)
    assert agg._forks == prev
    # every island has a lineage tip recorded
    assert "dev" in agg._branch_parent
    assert all(f in agg._branch_parent for f in agg._forks)
    # a worker mutation on an island becomes a child of that island's tip
    from agentdescent.evolvable import Diff
    snap = ledger.snapshot("head/0")
    head = snap.get("artifact")
    assert head is not None
    mutated = head.apply(Diff(diff_id="w", target="artifact",
                              ops={"value": "cap:8"}))
    ledger.commit(mutated, {"artifact": snap.version.get("artifact", 0)},
                  branch="head/0")
    n_before = len(agg._archive)
    tip = agg._branch_parent["head/0"]
    agg.step(n_workers=3)
    assert len(agg._archive) == n_before + 1
    # the new entry is a child of head/0's recorded tip *at admit time*
    child = max(agg._archive, key=lambda e: e["version"])
    assert child["parent"] == tip


def test_checkpoint_serializes_islands_and_lineage(tmp_path):
    agg, ledger, runtime, strategy = _make_aggregator(tmp_path, island_count=2)
    _seed_archive(agg, runtime, strategy)
    for _ in range(3):
        agg.step(n_workers=3)
    payload = agg.checkpoint()
    assert payload["forks"] == ["head/0", "head/1"]
    assert payload["island_count"] == 2
    assert "dev" in payload["branch_parent"]
    assert isinstance(payload["archive"], list) and len(payload["archive"]) >= 2


def test_restore_keeps_islands(tmp_path):
    agg, ledger, runtime, strategy = _make_aggregator(tmp_path, island_count=2)
    _seed_archive(agg, runtime, strategy)
    for _ in range(3):
        agg.step(n_workers=3)
    payload = agg.checkpoint()
    fresh, ledger2, _, _ = _make_aggregator(tmp_path, island_count=2)
    fresh.restore(payload)
    assert fresh._forks == ["head/0", "head/1"]
    assert fresh._branch_parent == agg._branch_parent
    assert len(fresh._archive) == len(agg._archive)




def _slots():
    from agentdescent.evolution import SingleSlot
    return SingleSlot(initial_value="cap:0")


def test_bad_island_count_is_refused():
    with pytest.raises(ValueError):
        population_factory(SingleHead(), "artifact", island_count=-1)


def test_island_count_requires_a_selection_policy():
    # population_islands without a declared selection policy: the engine has no
    # population layer, so the knob is silently inert -- which the
    # "honour or refuse" rule forbids. A declared policy is the gate.
    from agentdescent.evolution import SingleSlot
    with pytest.raises(ValueError):
        evolve(_tasks(), _reward, run=_run, propose=_propose,
               strategy=SingleSlot(initial_value="cap:0"),
               rounds=2, n_workers=2, population_islands=2)
