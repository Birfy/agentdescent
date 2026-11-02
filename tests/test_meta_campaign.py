"""A meta-campaign is the bounded loop around continuous meta-learning.

The fixture set pins the composition rather than re-testing meta evolution:
each session gets a unique run id and its own fresh ledger repo, the archive is
the only memory across sessions, and the controller's stop conditions bound the
loop. A failed session is reported and stops the campaign; a rejected rule is a
measured session and is not an error. The end-to-end case runs the real
``continuous_meta_run`` offline so the second session is seeded by the first and
the archive actually grows.
"""

import json
import os
import re
from types import SimpleNamespace

import pytest

from agentdescent import (
    ConsolidationStore,
    EvolutionResult,
    MetaOutcome,
    ParamSlot,
    SingleSlot,
    Task,
    evolve_problem,
)
from agentdescent.meta_campaign import (
    MetaCampaignConfig,
    MetaCampaignReport,
    meta_campaign_run,
)
from agentdescent.sampling import DifficultyWeighted


def _fake_run(monkeypatch, gains, *, error_at=None):
    calls = []

    def fake(store, slot, spec, *, run_id=None, repo_dir=None, **kwargs):
        index = len(calls)
        calls.append({"run_id": run_id, "repo_dir": repo_dir, "kwargs": kwargs})
        if error_at is not None and index == error_at:
            raise RuntimeError("boom")
        gain = gains[min(index, len(gains) - 1)]
        return SimpleNamespace(
            slot=slot, verdict="adopt", mean_gain=gain,
            fresh_gain=gain + 0.1,
            seeded_from=None if index == 0 else f"prior-{index - 1}",
            archived_run_id=f"arch-{index}", archive_count=index + 1,
            archived=True)

    monkeypatch.setattr(
        "agentdescent.meta_campaign.continuous_meta_run", fake)
    return calls


def test_control_config_round_trips_the_campaign_bounds():
    config = MetaCampaignConfig(
        max_sessions=4, target=0.8, max_cost=2.0, min_gain=0.1,
        patience=2, base_seed=3)
    control = config.control_config()
    assert control.target == 0.8
    assert control.max_sessions == 4
    assert control.max_cost == 2.0
    assert control.min_gain == 0.1
    assert control.patience == 2


def test_target_stops_after_n_sessions_and_tracks_the_best(tmp_path, monkeypatch):
    store = ConsolidationStore(str(tmp_path / "archive"))
    store.archive(
        EvolutionResult(state={"x": "1"}, rendered="x: 1", final_reward=0.5,
                        history=[], ledger_log=[]),
        run_id="pre-existing")
    calls = _fake_run(monkeypatch, [0.3, 0.9, 0.5])
    report = meta_campaign_run(
        store, "task_sampler", object(),
        config=MetaCampaignConfig(
            max_sessions=5, target=0.9, patience=5, base_seed=7),
        run_id="camp", repo_dir=str(tmp_path / "repos"),
        tasks_for=lambda rec: [])

    assert isinstance(report, MetaCampaignReport)
    assert len(report.sessions) == 2
    assert report.best_session == 1
    assert report.archive_count == 1
    assert report.complete is True
    assert report.control.stopped_reason == "target"
    assert [entry["run_id"].split("-")[1] for entry in calls] == ["0", "1"]
    assert len({entry["run_id"] for entry in calls}) == 2
    assert len({entry["repo_dir"] for entry in calls}) == 2
    assert os.path.basename(calls[0]["repo_dir"]) == "session-0"
    assert os.path.basename(calls[1]["repo_dir"]) == "session-1"
    assert os.path.isdir(calls[0]["repo_dir"])
    assert os.path.isdir(calls[1]["repo_dir"])


def test_patience_stops_a_flat_campaign(tmp_path, monkeypatch):
    _fake_run(monkeypatch, [0.5, 0.5, 0.5, 0.5])
    report = meta_campaign_run(
        ConsolidationStore(str(tmp_path / "archive")), "task_sampler", object(),
        config=MetaCampaignConfig(max_sessions=10, patience=2),
        tasks_for=lambda rec: [])

    assert len(report.sessions) == 3
    assert report.control.stopped_reason == "patience"
    assert report.complete is True


def test_an_exception_stops_the_campaign_and_is_visible(tmp_path, monkeypatch):
    _fake_run(monkeypatch, [0.4, 0.7], error_at=1)
    report = meta_campaign_run(
        ConsolidationStore(str(tmp_path / "archive")), "task_sampler", object(),
        config=MetaCampaignConfig(max_sessions=4, patience=4),
        tasks_for=lambda rec: [])

    assert len(report.sessions) == 1
    assert report.best_session == 0
    assert report.complete is False
    assert report.control.stopped_reason == "error"
    assert "boom" in report.detail
    assert "boom" in report.control.reason
    payload = report.to_dict()
    assert any("boom" in row["reason"]
               for row in payload["control"]["decisions"])


def test_a_reject_is_a_valid_measured_session(tmp_path, monkeypatch):
    calls = []

    def fake(store, slot, spec, *, run_id=None, repo_dir=None, **kwargs):
        calls.append({"run_id": run_id, "repo_dir": repo_dir, "kwargs": kwargs})
        return SimpleNamespace(
            slot=slot, verdict="reject", mean_gain=0.6, fresh_gain=0.6,
            seeded_from=None, archived_run_id=None, archive_count=0,
            archived=False)

    monkeypatch.setattr(
        "agentdescent.meta_campaign.continuous_meta_run", fake)
    report = meta_campaign_run(
        ConsolidationStore(str(tmp_path / "archive")), "task_sampler", object(),
        config=MetaCampaignConfig(max_sessions=2, patience=5),
        tasks_for=lambda rec: [])

    assert len(report.sessions) == 2
    assert report.control.stopped_reason == "max_sessions"
    assert report.complete is True
    assert report.sessions[1].verdict == "reject"


@pytest.mark.parametrize("kwargs", [
    {"max_sessions": 0},
    {"max_sessions": True},
    {"max_sessions": 1.5},
    {"max_sessions": "3"},
    {"target": float("nan")},
    {"target": float("inf")},
    {"target": -0.1},
    {"target": "0.5"},
    {"max_cost": 0},
    {"max_cost": -1},
    {"max_cost": float("nan")},
    {"max_cost": "1"},
    {"min_gain": -0.1},
    {"min_gain": float("nan")},
    {"min_gain": True},
    {"patience": 0},
    {"patience": True},
    {"patience": 1.5},
    {"base_seed": -1},
    {"base_seed": True},
    {"base_seed": 1.5},
])
def test_config_rejects_invalid_values(kwargs):
    with pytest.raises((TypeError, ValueError)):
        MetaCampaignConfig(**kwargs)


def test_bad_config_type_fails_closed(tmp_path):
    with pytest.raises(TypeError):
        meta_campaign_run(
            ConsolidationStore(str(tmp_path / "archive")), "task_sampler",
            object(), config={}, tasks_for=lambda rec: [])


def test_to_dict_is_json_safe_and_carries_the_session_facts(tmp_path, monkeypatch):
    _fake_run(monkeypatch, [0.2, 0.6])
    report = meta_campaign_run(
        ConsolidationStore(str(tmp_path / "archive")), "task_sampler", object(),
        config=MetaCampaignConfig(max_sessions=2, target=0.6),
        tasks_for=lambda rec: [])

    payload = report.to_dict()
    text = json.dumps(payload)
    assert isinstance(text, str)
    assert payload["best_session"] == 1
    assert payload["complete"] is True
    assert len(payload["sessions"]) == 2
    for row in payload["sessions"]:
        for key in ("verdict", "mean_gain", "fresh_gain", "seeded_from",
                    "archived_run_id"):
            assert key in row
    assert payload["sessions"][1]["seeded_from"] == "prior-0"
    assert payload["sessions"][0]["archived_run_id"] == "arch-0"
    assert payload["control"]["stopped_reason"] == "target"


def _cap(rendered):
    match = re.search(r"cap:(\d+)", rendered)
    return int(match.group(1)) if match else 0


def _run(rendered, task):
    return (str(task.meta["gold"])
            if task.meta.get("a", 0) <= _cap(rendered) else "?")


def _reward(task, output):
    return 1.0 if str(output).strip() == str(task.meta["gold"]).strip() else 0.0


def _inner_propose(rendered, task, output, reward):
    if reward >= 0.999:
        return "cap:0"
    a = task.meta.get("a", 0)
    if a > _cap(rendered):
        return f"cap:{a}"
    return None


def _tasks(prefix="s", n=8):
    order = [1, 2, 3, 4, 8, 5, 6, 7][:n]
    return [Task(id=f"{prefix}{i}", prompt=f"item {i}",
                 meta={"gold": str(2 * a), "a": a})
            for i, a in enumerate(order)]


def _sampler_slot():
    return ParamSlot(
        factory=lambda c: DifficultyWeighted(c=c),
        params={"c": 0.2}, bounds={"c": (0.05, 1.0)})


def _climb_propose(rendered, task, output, reward):
    match = re.search(r"^c: ([0-9.]+)", rendered, re.M)
    c = float(match.group(1)) if match else 0.2
    return f"c: {min(0.9, c + 0.1):.2f}"


def _climbing(value, seed):
    c = getattr(value, "c", 0.2)
    reward = min(1.0, 0.3 + 0.5 * c)
    return MetaOutcome(curve=[reward, min(1.0, reward + 0.05)], final=reward)


def _climbing_val(value, seed):
    return _climbing(value, seed)


def test_real_campaign_runs_two_seeded_sessions_and_grows_the_archive(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    report = meta_campaign_run(
        store, "task_sampler", _sampler_slot(),
        config=MetaCampaignConfig(max_sessions=2, patience=5),
        run_id="real", repo_dir=str(tmp_path / "repos"),
        tasks_for=lambda rec: _tasks(), propose=_climb_propose,
        run=_run, reward=_reward, inner_propose=_inner_propose,
        inner_strategy=SingleSlot(initial_value="cap:0"),
        problems={"ta": _climbing, "tb": _climbing, "tc": _climbing},
        val_problems={"va": _climbing_val},
        inner={"rounds": 5, "n_workers": 1, "target_reward": 1.0},
        seeds=(0, 1, 2, 3), val_seeds=(10, 11), rounds=1, n_workers=1)

    assert report.complete is True
    assert len(report.sessions) == 2
    assert store.count() == 2
    assert report.archive_count == 2
    first, second = report.sessions
    assert first.verdict == "adopt"
    assert second.verdict == "adopt"
    assert second.seeded_from == first.archived_run_id
    assert first.archive_count == 1
    assert second.archive_count == 2
    assert report.control.stopped_reason == "max_sessions"
    assert os.path.isdir(str(tmp_path / "repos" / "session-0"))
    assert os.path.isdir(str(tmp_path / "repos" / "session-1"))
