"""Async meta-population has to be right where islands run concurrently.

The async variant of `meta_population_run` is the same call with a thread pool:
islands start from distinct archived rules, train and validate on the shared
isolated sets, each in its own fresh repo, and the archive is only read while
they run (it is thread-safe) and written once at promotion. The fixtures pin
the surfaces that make this honest: the decision and report are identical to
the sequential call's on the same archive, `workers` bounds concurrency,
validation is a real concurrency test (islands share the archive), an errored
island is never promoted, and an empty archive is refused.
"""

import pytest

from agentdescent import (
    ConsolidationStore,
    EvolutionResult,
    MetaOutcome,
    ParamSlot,
    SingleSlot,
    Task,
    async_meta_population_run,
    meta_population_run,
    problem_descriptor,
)
from agentdescent.sampling import DifficultyWeighted


def _cap(rendered):
    import re
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


def _nudge_up():
    import re
    def propose(rendered, task, output, reward):
        c = _c_of(rendered)
        if reward < 0.7:
            return f"c: {min(1.0, c + 0.1):.2f}"
        return f"c: {c:.2f}"
    return propose


def _stair(value, seed):
    if getattr(value, "c", 0.2) >= 0.3:
        return MetaOutcome(curve=[0.5, 1.0], final=1.0)
    return MetaOutcome(curve=[0.1, 0.2], final=0.2)


def _stair_val(value, seed):
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


def _call(store, tmp_path, *, repo, **kw):
    base = dict(store=store, slot="task_sampler", spec=_sampler_slot(),
                propose=_nudge_up(), tasks_for=lambda rec: _tasks(), run=_run,
                reward=_reward, inner_propose=_propose,
                inner_strategy=SingleSlot(initial_value="cap:0"), config=CONFIG,
                inner=dict(INNER), seeds=(0, 1, 2, 3), val_seeds=(10, 11),
                rounds=1, n_workers=1, repo_dir=str(tmp_path / repo))
    base.update(kw)
    return base


def test_async_reports_islands_and_promotes_like_sequential(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    _archive_rule(store, "run-high", 0.3, 0.8)
    _archive_rule(store, "run-low", 0.2, 0.6)
    common = dict(islands=2, train_runs=3, val_runs=1,
                  problems={"a": _stair, "b": _stair},
                  val_problems={"v": _stair_val})
    rep_seq, _ = meta_population_run(**_call(store, tmp_path, repo="s", **common))
    store2 = ConsolidationStore(str(tmp_path / "archive"))
    rep_async, _ = async_meta_population_run(
        **_call(store2, tmp_path, repo="a", **common))
    # the decision and the per-island verdicts are identical
    assert rep_async.promoted_from == rep_seq.promoted_from
    assert [i.verdict for i in rep_async.islands] == \
        [i.verdict for i in rep_seq.islands]
    assert rep_async.islands[0].seeded_from == "run-high"
    assert rep_async.archive_count == rep_seq.archive_count


def test_workers_bounds_concurrency(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    _archive_rule(store, "run-1", 0.2, 0.6)
    _archive_rule(store, "run-2", 0.4, 0.9)
    report, _ = async_meta_population_run(
        **_call(store, tmp_path, repo="r"), islands=2, workers=1,
        train_runs=3, val_runs=1, problems={"a": _stair, "b": _stair},
        val_problems={"v": _stair_val})
    assert len(report.islands) == 2
    assert report.promoted_from in (None, 0, 1)


def test_an_errored_island_is_never_promoted(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    _archive_rule(store, "run-good", 0.2, 0.6)
    store.archive(
        EvolutionResult(state={"c": "abc"}, rendered="# Policy parameters\nc: abc",
                        final_reward=0.5, history=[], ledger_log=[]),
        run_id="run-bad", strategy="ParamSlot", config=CONFIG,
        problem=problem_descriptor("task_sampler", **INNER),
        artifact_id="meta-rule")
    report, _ = async_meta_population_run(
        **_call(store, tmp_path, repo="r"), islands=2, train_runs=3, val_runs=1,
        problems={"a": _stair, "b": _stair}, val_problems={"v": _stair_val})
    errored = [i for i in report.islands if i.seeded_from == "run-bad"]
    assert errored and errored[0].verdict == "error"
    assert report.promoted_from != [i.index for i in report.islands
                                    if i.seeded_from == "run-bad"][0]


def test_empty_archive_is_refused(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    with pytest.raises(ValueError):
        async_meta_population_run(
            **_call(store, tmp_path, repo="r"), islands=2,
            problems={"a": _stair}, val_problems={"v": _stair_val})


def test_workers_must_be_positive(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    _archive_rule(store, "run-1", 0.2, 0.6)
    with pytest.raises(ValueError):
        async_meta_population_run(
            **_call(store, tmp_path, repo="r"), islands=1, workers=0,
            problems={"a": _stair}, val_problems={"v": _stair_val})
