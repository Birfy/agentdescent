"""Cross-run consolidation has to be right in the places continuing self-
improvement is dangerous.

The archive stores what a run produced; a later run may adopt it -- and the
one failure mode that matters is seeding a run with a skill learned on a
*different* problem, which helps nothing and hurts the cold start. Every
fixture pins one of the guards: the archive is idempotent and small (links,
not copies), retrieval stays inside the same configuration and strategy,
and the mini-eval gate refuses a regression instead of installing it.
"""

import json

import pytest

from agentdescent import (
    AppendRules,
    ConsolidationStore,
    SingleSlot,
    Task,
    config_fingerprint,
    seed_from_archive,
    evolve,
)


# -- a deterministic capability-capped domain -----------------------------------

CAP_OPEN = "cap:{0}"


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
    a = task.meta.get("a", 0)
    if a > _cap(rendered):
        return CAP_OPEN.format(a)
    return None


def _tasks(prefix="s", n=8):
    # order puts a=8 in TRAIN (the last 3 are held-out 5,6,7), so the cap-based
    # skill can learn the full range and the held-out actually passes
    order = [1, 2, 3, 4, 8, 5, 6, 7][:n]
    return [Task(id=f"{prefix}{i}", prompt=f"item {i}",
                 meta={"gold": str(2 * a), "a": a}) for i, a in enumerate(order)]


DOMAIN_CONFIG = {"dataset": "cap-arithmetic", "scorer": "exact"}


def _run_evolve(tasks, *, repo, rounds=6, seed=0):
    return evolve(tasks, _reward, run=_run, propose=_propose,
                  strategy=SingleSlot(initial_value="cap:0"),
                  rounds=rounds, n_workers=2, repo_path=repo, seed=seed,
                  target_reward=1.0)


# -- the archive -----------------------------------------------------------------


def test_archive_round_trips(tmp_path):
    store = ConsolidationStore(str(tmp_path))
    result = _run_evolve(_tasks(), repo=str(tmp_path / "repo"))
    store.archive(result, run_id="run-a", strategy="SingleSlot",
                  config=DOMAIN_CONFIG)
    rec = store.find("run-a")
    assert rec is not None
    assert rec.strategy == "SingleSlot"
    assert rec.config_hash == config_fingerprint(DOMAIN_CONFIG)
    assert rec.final_reward == result.final_reward
    assert rec.state == result.state
    assert list(store.iter_archive()) == [rec]
    assert store.count() == 1


def test_archive_is_idempotent_by_run_id(tmp_path):
    store = ConsolidationStore(str(tmp_path))
    a = _run_evolve(_tasks(), repo=str(tmp_path / "r1"))
    store.archive(a, run_id="run-x", config=DOMAIN_CONFIG)
    b = _run_evolve(_tasks(prefix="q"), repo=str(tmp_path / "r2"))
    store.archive(b, run_id="run-x", config=DOMAIN_CONFIG)
    assert store.count() == 1
    assert store.find("run-x").state == b.state   # last write wins, atomically


def test_non_finite_reward_is_refused(tmp_path):
    from agentdescent.evolution import EvolutionResult
    store = ConsolidationStore(str(tmp_path))
    bogus = EvolutionResult(state={}, rendered="", final_reward=float("nan"),
                            history=[], ledger_log=[])
    with pytest.raises(ValueError):
        store.archive(bogus, run_id="bad", config=DOMAIN_CONFIG)


def test_config_fingerprint():
    a = config_fingerprint({"dataset": "x"})
    b = config_fingerprint({"dataset": "x"})
    c = config_fingerprint({"dataset": "y"})
    assert a == b and a != c
    assert config_fingerprint(None) == "no-config"


# -- seeding and the gate ---------------------------------------------------------


def test_a_same_config_seed_is_accepted_and_helps_cold_start(tmp_path):
    store = ConsolidationStore(str(tmp_path))
    run1 = _run_evolve(_tasks(), repo=str(tmp_path / "r1"), rounds=6)
    assert run1.final_reward > 0.0
    store.archive(run1, run_id="run-1", strategy="SingleSlot",
                  config=DOMAIN_CONFIG)
    # run 2: new tasks, same domain/config
    tasks2 = _tasks(prefix="t")
    state, report = seed_from_archive(
        store, tasks2, _run, _reward, strategy=SingleSlot(),
        config=DOMAIN_CONFIG, sample=6, seed=1)
    assert report.accepted is True
    assert report.matched_run == "run-1"
    assert state is not None
    # the seeded run reaches quality earlier than the empty one
    seeded = evolve(tasks2, _reward, run=_run, propose=_propose,
                    strategy=SingleSlot(initial_value="cap:0"),
                    initial_state=state, rounds=4, n_workers=2,
                    repo_path=str(tmp_path / "s"), seed=1, target_reward=1.0)
    empty = evolve(tasks2, _reward, run=_run, propose=_propose,
                   strategy=SingleSlot(initial_value="cap:0"), rounds=4,
                   n_workers=2, repo_path=str(tmp_path / "e"), seed=1,
                   target_reward=1.0)
    # the seeded run reaches the bar earlier than the empty one
    st = seeded.time_to_quality(1.0)
    et = empty.time_to_quality(1.0)
    assert st is not None and et is not None
    assert st <= et
    assert seeded.final_reward >= empty.final_reward


def test_a_cross_domain_seed_is_refused(tmp_path):
    # an archive entry from a different dataset regresses the new run's sample
    store = ConsolidationStore(str(tmp_path))
    # archive a skill that answers "42" to everything (harmful on this domain)
    result = evolve(_tasks(), lambda t, o: 1.0 if o == "42" else 0.0,
                    run=lambda r, t: "42",
                    propose=lambda *a: None,
                    strategy=SingleSlot(initial_value="answer 42"), rounds=1,
                    n_workers=1, repo_path=str(tmp_path / "r1"))
    store.archive(result, run_id="other-domain", strategy="SingleSlot",
                  config={"dataset": "forty-two"})
    # The new domain rewards a skill that speaks its language: the run answers
    # correctly only when the skill text carries "cap:" (the empty seed does);
    # a cross-domain skill ("answer 42") does not, so it fails where the empty
    # seed passes -- a genuine regression the gate refuses.
    new_run = lambda rendered, task: task.meta["gold"] if "cap:" in rendered else "wrong"
    state, report = seed_from_archive(
        store, _tasks(prefix="z"), new_run, _reward,
        strategy=SingleSlot(initial_value="cap:0"),
        config=DOMAIN_CONFIG, sample=6, seed=2)
    assert state is None
    assert report.accepted is False
    assert report.seeded_score < report.empty_score
    assert "rejected" in report.reason


def test_no_archive_means_no_seed(tmp_path):
    store = ConsolidationStore(str(tmp_path))
    state, report = seed_from_archive(
        store, _tasks(), _run, _reward, strategy=SingleSlot(),
        config=DOMAIN_CONFIG, sample=4, seed=0)
    assert state is None
    assert report.matched_run is None
    assert report.accepted is False


def test_retrieval_stays_inside_the_same_strategy(tmp_path):
    # an AppendRules archive is not adoptable through SingleSlot
    store = ConsolidationStore(str(tmp_path))
    result = _run_evolve(_tasks(), repo=str(tmp_path / "r1"), rounds=2)
    store.archive(result, run_id="rules-run", strategy="AppendRules",
                  config=DOMAIN_CONFIG)
    state, report = seed_from_archive(
        store, _tasks(), _run, _reward, strategy=SingleSlot(),
        config=DOMAIN_CONFIG, sample=4, seed=0)
    assert state is None
    assert report.matched_run is None
    assert "no archived run" in report.reason
