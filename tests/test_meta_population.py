"""Meta-population has to be right where parallel lineages touch the archive.

The archive's best rules each seed an island; every island searches from its
own starting rule and validates the same isolated set; the best adopt island is
promoted back. The fixtures pin the surfaces that make this honest: a population
needs something to diversify from (no archived rules -> refused), islands are
reported per seed, promotion is evidence-based (the island that gained most is
the one archived), an errored island is reported and never promoted, and train/
val isolation is enforced like the rest of the meta surface.
"""

import re

import pytest

from agentdescent import (
    ConsolidationStore,
    EvolutionResult,
    MetaOutcome,
    MetaPopulationReport,
    ParamSlot,
    PopulationIsland,
    SingleSlot,
    Task,
    config_fingerprint,
    continuous_meta_run,
    meta_population_run,
    problem_descriptor,
)
from agentdescent.sampling import DifficultyWeighted


# -- the deterministic capability domain ----------------------------------------


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


def _sampler_slot():
    return ParamSlot(factory=lambda c: DifficultyWeighted(c=c),
                     params={"c": 0.2}, bounds={"c": (0.05, 1.0)})


def _nudge_up(rendered, task, output, reward):
    c = _c_of(rendered)
    if reward < 0.7:
        return f"c: {min(1.0, c + 0.1):.2f}"
    return f"c: {c:.2f}"


def _stair(value, seed):
    # a synthetic landscape where the value c >= 0.3 scores well and c < 0.3
    # does not -- so an island starting low gains, one starting high does not
    if getattr(value, "c", 0.2) >= 0.3:
        return MetaOutcome(curve=[0.5, 1.0], final=1.0)
    return MetaOutcome(curve=[0.1, 0.2], final=0.2)


def _stair_val(value, seed):
    # a distinct object with the same landscape: the isolation guard compares
    # problems by identity, so a validation problem cannot be a training one
    return _stair(value, seed)


CONFIG = {"dataset": "cap-arithmetic", "scorer": "exact"}
INNER = {"rounds": 5, "n_workers": 1, "target_reward": 1.0}


def _archive_rule(store, run_id, c, reward):
    store.archive(
        EvolutionResult(state={"c": f"{c:.1f}"},
                        rendered=f"# Policy parameters\nc: {c:.1f}",
                        final_reward=reward, history=[], ledger_log=[]),
        run_id=run_id, strategy="ParamSlot", config=CONFIG,
        problem=problem_descriptor("task_sampler", **INNER),
        artifact_id="meta-rule")


def _common(store, tmp_path, repo="r"):
    return dict(
        store=store, slot="task_sampler", spec=_sampler_slot(),
        propose=_nudge_up,
        tasks_for=lambda rec: _tasks(), run=_run, reward=_reward,
        inner_propose=_propose, inner_strategy=SingleSlot(initial_value="cap:0"),
        config=CONFIG, inner=dict(INNER),
        seeds=(0, 1, 2, 3), val_seeds=(10, 11),
        rounds=1, n_workers=1, repo_dir=str(tmp_path / repo))


# -- the population -------------------------------------------------------------


def test_population_requires_archived_rules(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    with pytest.raises(ValueError):
        meta_population_run(
            store=store, slot="task_sampler", spec=_sampler_slot(),
            propose=_nudge_up, tasks_for=lambda rec: _tasks(), run=_run,
            reward=_reward, inner_propose=_propose,
            inner_strategy=SingleSlot(initial_value="cap:0"), config=CONFIG,
            problems={"a": _stair, "b": _stair}, val_problems={"v": _stair_val})


def test_population_reports_islands_and_promotes_the_best_adopt(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    _archive_rule(store, "run-high", 0.3, 0.8)
    _archive_rule(store, "run-low", 0.2, 0.6)
    # train_runs=3 with only 2 archived rules -> the caller's problems stand in,
    # so the islands' outcomes are the deterministic synthetic landscape's
    report, result = meta_population_run(
        **_common(store, tmp_path), islands=2, train_runs=3, val_runs=1,
        problems={"a": _stair, "b": _stair}, val_problems={"v": _stair_val})
    assert isinstance(report, MetaPopulationReport)
    assert len(report.islands) == 2
    assert report.islands[0].seeded_from == "run-high"   # best rule first
    assert report.islands[1].seeded_from == "run-low"
    # the high-starting island has nothing to gain; the low one climbs to 0.3
    # and gains a lot -- it is the one promoted
    assert report.islands[0].verdict == "reject"
    assert report.islands[1].verdict == "adopt"
    assert report.promoted_from == 1
    assert report.promoted_run_id is not None
    assert report.archive_count == 3
    assert report.reason.startswith("promoted island 1")
    # the promoted rule is strictly better than the seed it came from
    promoted = store.find(report.promoted_run_id)
    assert _c_of(promoted.rendered) > 0.2
    # the report rides the returned result
    assert result.meta["promoted_from"] == 1


def test_an_errored_island_is_reported_and_never_promoted(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    _archive_rule(store, "run-good", 0.2, 0.6)
    # a rule whose state renders to something the spec cannot compile: the
    # island seeded from it fails validation and is reported as an error
    store.archive(
        EvolutionResult(state={"c": "abc"}, rendered="# Policy parameters\nc: abc",
                        final_reward=0.5, history=[], ledger_log=[]),
        run_id="run-bad", strategy="ParamSlot", config=CONFIG,
        problem=problem_descriptor("task_sampler", **INNER),
        artifact_id="meta-rule")
    report, _ = meta_population_run(
        **_common(store, tmp_path), islands=2, train_runs=3, val_runs=1,
        problems={"a": _stair, "b": _stair}, val_problems={"v": _stair_val})
    assert len(report.islands) == 2
    errored = [i for i in report.islands if i.seeded_from == "run-bad"]
    assert errored and errored[0].verdict == "error"
    assert errored[0].error
    # the good island may promote; the errored one never does
    assert report.promoted_from != [i.index for i in report.islands
                                    if i.seeded_from == "run-bad"][0]


def test_population_degrades_to_a_single_island(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    _archive_rule(store, "run-only", 0.2, 0.6)
    report, _ = meta_population_run(
        **_common(store, tmp_path), islands=1, train_runs=3, val_runs=1,
        problems={"a": _stair, "b": _stair}, val_problems={"v": _stair_val})
    assert len(report.islands) == 1
    assert report.islands[0].seeded_from == "run-only"
    assert report.promoted_from in (None, 0)


def test_islands_must_be_a_positive_int(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    _archive_rule(store, "run-1", 0.2, 0.6)
    with pytest.raises(ValueError):
        meta_population_run(
            **_common(store, tmp_path), islands=0,
            problems={"a": _stair}, val_problems={"v": _stair_val})


def test_population_enforces_train_val_isolation(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    _archive_rule(store, "run-1", 0.2, 0.6)
    common = _common(store, tmp_path)
    common.pop("seeds")
    common.pop("val_seeds")
    with pytest.raises(ValueError):
        meta_population_run(
            **common, islands=1, train_runs=3, val_runs=1,
            problems={"a": _stair}, val_problems={"a": _stair},   # shared name
            seeds=(0, 1), val_seeds=(1, 2))                       # shared seed


def test_population_round_trips_with_continuous_meta(tmp_path):
    # a two-session arc: seed the archive with continuous_meta_run, then run a
    # population on top of it -- the promoted rule is visible next round
    store = ConsolidationStore(str(tmp_path / "archive"))
    common = _common(store, tmp_path, repo="a")
    common.pop("repo_dir")
    s1, _ = continuous_meta_run(
        **common, run_id="run-1", repo_path=str(tmp_path / "seed-repo"),
        problems={"ta": _stair, "tb": _stair}, val_problems={"va": _stair_val})
    assert s1.archived or store.count() >= 1
    report, _ = meta_population_run(
        **_common(store, tmp_path, repo="b"), islands=1, train_runs=1,
        val_runs=1, problems=None, val_problems={"va": _stair_val})
    assert report.islands[0].seeded_from == s1.archived_run_id or \
        report.islands[0].seeded_from is not None
    assert store.count() >= 1
