"""Continuous meta-learning has to be right where a two-level loop touches
durable state.

meta_run() evolves a decision slot and decides; continuous_meta_run() makes
that decision durable -- the archive holds the adopted rules and their problem
descriptors, and the next session is *seeded* by the best one. The fixtures
pin the surfaces that make this honest: a rule record carries a problem
descriptor and can be rebuilt into a runnable Problem; an adoption lands in
the archive and the next session starts from it (and does not regress); train
and validation problems stay disjoint; an empty archive is an ordinary
meta_run; and the report carries both the gain over the prior and the gain
over the fresh seed -- the cross-session number.
"""

import re

import pytest

from agentdescent import (
    ConsolidationStore,
    ContinuousMetaReport,
    EvolutionResult,
    MetaOutcome,
    MetaRunReport,
    ParamSlot,
    SingleSlot,
    Task,
    archive_rule,
    config_fingerprint,
    continuous_meta_run,
    evolve_problem,
    problem_descriptor,
    problem_from_archive,
)
from agentdescent.sampling import DifficultyWeighted


# -- a deterministic capability domain (the inner problem family) ---------------


def _cap(rendered):
    m = re.search(r"cap:(\d+)", rendered)
    return int(m.group(1)) if m else 0


def _c_of(rendered):
    m = re.search(r"^c: ([0-9.]+)", rendered, re.M)
    return float(m.group(1)) if m else 0.2


def _run(rendered, task):
    return str(task.meta["gold"]) if task.meta.get("a", 0) <= _cap(rendered) else "?"


def _reward(task, output):
    return 1.0 if str(output).strip() == str(task.meta["gold"]).strip() else 0.0


def _propose(rendered, task, output, reward):
    if reward >= 0.999:
        return "cap:0"
    a = task.meta.get("a", 0)
    if a > _cap(rendered):
        return f"cap:{a}"
    return None


def _tasks(prefix="s", n=8):
    order = [1, 2, 3, 4, 8, 5, 6, 7][:n]
    return [Task(id=f"{prefix}{i}", prompt=f"item {i}",
                 meta={"gold": str(2 * a), "a": a}) for i, a in enumerate(order)]


def _inner_problem(tasks, *, rounds=5):
    return evolve_problem(
        tasks, _reward, slot="task_sampler", run=_run, propose=_propose,
        strategy=SingleSlot(initial_value="cap:0"),
        rounds=rounds, n_workers=1, target_reward=1.0)


def _sampler_slot():
    # the evolved slot: the UCB exploration constant of DifficultyWeighted
    return ParamSlot(
        factory=lambda c: DifficultyWeighted(c=c),
        params={"c": 0.2}, bounds={"c": (0.05, 1.0)})


def _deterministic_propose(rendered, task, output, reward):
    # an offline hill-climb: nudge up on a low meta-reward, keep otherwise
    c = _c_of(rendered)
    if reward < 0.7:
        return f"c: {min(1.0, c + 0.1):.2f}"
    return f"c: {c:.2f}"


CONFIG = {"dataset": "cap-arithmetic", "scorer": "exact"}

INNER = {"rounds": 5, "n_workers": 1, "target_reward": 1.0}


# -- synthetic problems, for verdicts that must not depend on the sampler ------


def _helping(value, seed):
    # the evolved value (c > 0.2) beats the seed (c == 0.2) on this problem
    if getattr(value, "c", 0.2) > 0.2:
        return MetaOutcome(curve=[0.5, 1.0], final=1.0)
    return MetaOutcome(curve=[0.2, 0.3], final=0.3)


def _helping_val(value, seed):
    # a distinct object with the same behaviour as _helping: the isolation
    # guard compares problems by identity, so a validation problem that *is*
    # a training problem (same object) is refused even under an anonymous
    # sequence API.
    return _helping(value, seed)


def _train_stepped(value, seed):
    # only a value that reached c >= 0.4 scores well, so the offline proposer
    # is forced to keep climbing from a lower prior
    if getattr(value, "c", 0.2) >= 0.4:
        return MetaOutcome(curve=[0.5, 1.0], final=1.0)
    return MetaOutcome(curve=[0.2, 0.3], final=0.3)


def _val_stepped(value, seed):
    c = getattr(value, "c", 0.2)
    if c >= 0.4:
        return MetaOutcome(curve=[0.5, 1.0], final=1.0)
    if c >= 0.3:
        return MetaOutcome(curve=[0.3, 0.6], final=0.6)
    return MetaOutcome(curve=[0.1, 0.2], final=0.2)


def _session_common(tmp_path, *, repo="repo", tasks_for=None):
    calls = []

    def spy(rec):
        calls.append(rec.run_id)
        return _tasks()

    return {
        "store": ConsolidationStore(str(tmp_path / "archive")),
        "slot": "task_sampler", "spec": _sampler_slot(),
        "propose": _deterministic_propose,
        "tasks_for": tasks_for or spy,
        "run": _run, "reward": _reward, "inner_propose": _propose,
        "inner_strategy": SingleSlot(initial_value="cap:0"),
        "config": CONFIG, "inner": dict(INNER),
        "seeds": (0, 1, 2, 3), "val_seeds": (10, 11),
        "rounds": 1, "n_workers": 1, "repo_path": str(tmp_path / repo),
    }, calls


# -- the problem descriptor -----------------------------------------------------


def test_problem_descriptor_shape():
    desc = problem_descriptor("task_sampler", rounds=5, n_workers=1)
    assert desc == {"slot": "task_sampler",
                    "inner": {"rounds": 5, "n_workers": 1}}
    assert set(desc) == {"slot", "inner"}


def test_problem_from_archive_rebuilds_a_runnable_problem(tmp_path):
    store = ConsolidationStore(str(tmp_path))
    store.archive(
        EvolutionResult(state={"c": "0.3"},
                        rendered="# Policy parameters\nc: 0.3",
                        final_reward=0.5, history=[], ledger_log=[]),
        run_id="run-a", strategy="ParamSlot", config=CONFIG,
        problem=problem_descriptor("task_sampler", **INNER),
        artifact_id="meta-rule")
    problem = problem_from_archive(
        store, "run-a", tasks_for=lambda rec: _tasks(), spec=_sampler_slot(),
        run=_run, reward=_reward, propose=_propose,
        strategy=SingleSlot(initial_value="cap:0"))
    # a rebuilt problem runs to a MetaOutcome under a compiled value
    outcome = problem(_sampler_slot().compile("# Policy parameters\nc: 0.3"),
                      seed=0)
    assert isinstance(outcome, MetaOutcome)
    assert isinstance(outcome.curve, list) and len(outcome.curve) > 0
    assert outcome.final >= 0.0


def test_problem_from_archive_refuses_a_foreign_spec(tmp_path):
    store = ConsolidationStore(str(tmp_path))
    store.archive(
        EvolutionResult(state={"value": "cap:4"}, rendered="cap:4",
                        final_reward=0.6, history=[], ledger_log=[]),
        run_id="rules", strategy="AppendRules", config=CONFIG,
        problem=problem_descriptor("task_sampler"), artifact_id="meta-rule")
    with pytest.raises(ValueError):
        problem_from_archive(
            store, "rules", tasks_for=lambda rec: _tasks(), spec=_sampler_slot(),
            run=_run, reward=_reward, propose=_propose,
            strategy=SingleSlot(initial_value="cap:0"))


def test_problem_from_archive_unknown_run(tmp_path):
    store = ConsolidationStore(str(tmp_path))
    with pytest.raises(KeyError):
        problem_from_archive(
            store, "nope", tasks_for=lambda rec: _tasks(), spec=_sampler_slot(),
            run=_run, reward=_reward, propose=_propose,
            strategy=SingleSlot(initial_value="cap:0"))


# -- archiving a rule -----------------------------------------------------------


def test_archive_rule_round_trips_the_adopted_value(tmp_path):
    store = ConsolidationStore(str(tmp_path))
    result = EvolutionResult(state={"c": "0.3"},
                             rendered="# Policy parameters\nc: 0.3",
                             final_reward=0.8, history=[], ledger_log=[])
    report = MetaRunReport(slot="task_sampler", seed_value="", value=result.rendered,
                           validation={}, mean_gain=0.0, transfer=None,
                           verdict="adopt", reason="")
    rec = archive_rule(
        store, report, result, run_id="run-1", slot="task_sampler",
        spec=_sampler_slot(), config=CONFIG,
        problem=problem_descriptor("task_sampler", **INNER))
    assert rec.problem == {"slot": "task_sampler", "inner": dict(INNER)}
    assert rec.strategy == "ParamSlot"            # the spec that owns the state
    assert rec.state == {"c": "0.3"}              # the seedable evolved value
    assert rec.rendered == result.rendered
    assert rec.config_hash == config_fingerprint(CONFIG)
    # a later session's retrieval sees the record with its descriptor
    assert store.find("run-1").problem["slot"] == "task_sampler"


# -- the session loop -----------------------------------------------------------


def test_empty_archive_is_an_ordinary_meta_run(tmp_path):
    common, _ = _session_common(tmp_path)
    report, result = continuous_meta_run(
        **common, problems={"ta": _helping, "tb": _helping},
        val_problems={"va": _helping_val})
    assert isinstance(report, ContinuousMetaReport)
    assert report.verdict in ("adopt", "reject")
    assert report.seeded_from is None            # nothing archived yet
    assert report.seed_value == report.fresh_seed_value
    assert report.fresh_gain == pytest.approx(report.mean_gain)   # same baseline
    assert report.archived == (report.verdict == "adopt")
    assert report.archive_count == (1 if report.archived else 0)
    # the report rides the run result like meta_run's does
    assert result.meta["verdict"] == report.verdict
    assert result.meta["seeded_from"] is None


def test_an_adopted_rule_archives_and_the_next_session_starts_from_it(tmp_path):
    common, _ = _session_common(tmp_path, repo="r1")
    s1, _ = continuous_meta_run(
        **common, run_id="run-1",
        problems={"ta": _helping, "tb": _helping}, val_problems={"va": _helping_val})
    assert s1.verdict == "adopt"
    assert s1.archived and s1.archived_run_id == "run-1"
    prior_c = _c_of(s1.value)
    assert prior_c > 0.2                         # the seed value was improved

    # session 2, same store: seeded by run-1's rule, train derived from its
    # problem descriptor, and the monotone nudge-up proposer never regresses
    common2, calls = _session_common(tmp_path, repo="r2")
    s2, _ = continuous_meta_run(
        **common2, train_runs=1,
        problems={"ta": _train_stepped, "tb": _train_stepped},
        val_problems={"va": _val_stepped})
    assert s2.seeded_from == "run-1"
    assert _c_of(s2.seed_value) == pytest.approx(prior_c)
    assert _c_of(s2.value) >= prior_c
    # the train problem was rebuilt from the archive (the spy was asked for
    # run-1's tasks), not the caller's fallback alone
    assert "run-1" in calls


def test_an_improved_rule_is_archived_and_visible_next_round(tmp_path):
    # session 1: empty archive -> the caller's problems stand in; adopt archives
    # c:0.3. session 2 asks for more train problems than the archive holds, so
    # its train falls back to the stepped closure, climbs to c:0.4, adopts, and
    # archives a strictly better rule than the one it started from.
    common, _ = _session_common(tmp_path, repo="a")
    s1, _ = continuous_meta_run(
        **common, run_id="run-1",
        problems={"ta": _helping, "tb": _helping}, val_problems={"va": _helping_val})
    assert s1.verdict == "adopt"
    assert s1.archived_run_id == "run-1"
    assert s1.archive_count == 1

    common2, _ = _session_common(tmp_path, repo="b")
    s2, _ = continuous_meta_run(
        **common2, run_id="run-2", train_runs=2, val_runs=1,
        problems={"ta": _train_stepped, "tb": _train_stepped},
        val_problems={"va": _val_stepped})
    # the cross-session numbers are the honest ones
    assert _c_of(s2.seed_value) == pytest.approx(_c_of(s1.value))  # from run-1
    assert _c_of(s2.value) == pytest.approx(0.4)                   # climbed one rung
    assert s2.fresh_gain > s2.mean_gain        # vs the fresh seed, a bigger win
    assert s2.verdict == "adopt"
    assert s2.archived and s2.archived_run_id == "run-2"
    assert s2.archive_count == 2
    # the archive now holds a strictly better rule than it started with
    assert _c_of(store_rule(common2["store"], "run-2")) \
        > _c_of(store_rule(common2["store"], "run-1"))


def store_rule(store, run_id):
    rec = store.find(run_id)
    return rec.rendered if rec else ""


def test_train_and_validation_problems_stay_disjoint(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    store.archive(
        EvolutionResult(state={"c": "0.3"},
                        rendered="# Policy parameters\nc: 0.3",
                        final_reward=0.8, history=[], ledger_log=[]),
        run_id="run-1", strategy="ParamSlot", config=CONFIG,
        problem=problem_descriptor("task_sampler", **INNER),
        artifact_id="meta-rule")
    common, calls = _session_common(tmp_path)
    report, _ = continuous_meta_run(
        store=store, slot=common["slot"], spec=common["spec"],
        propose=common["propose"], tasks_for=common["tasks_for"],
        run=common["run"], reward=common["reward"],
        inner_propose=common["inner_propose"],
        inner_strategy=common["inner_strategy"], config=common["config"],
        train_runs=1, val_runs=1, inner=common["inner"],
        seeds=common["seeds"], val_seeds=common["val_seeds"],
        rounds=1, n_workers=1, repo_path=str(tmp_path / "repo"),
        problems=None, val_problems={"va": _helping, "vb": _helping})
    # train came from the archive ("run-1" asked for via tasks_for); the
    # validation problems are the caller's, disjoint from it by construction
    assert "run-1" in calls
    assert set(report.validation) == {"va", "vb"}
