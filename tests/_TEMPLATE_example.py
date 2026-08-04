"""Test template for ``examples/_TEMPLATE.py``.

Copy to ``tests/test_<algorithm>_example.py`` and change the import. Keep the
tests offline: patch dataset rows and completions instead of using real services.
"""

from examples import _TEMPLATE as port
from agentdescent.evolution import Task


def test_strategy_turns_one_proposal_into_one_diff():
    diff = port.STRATEGY.to_diff({}, "one general rule", "test", 0, "artifact")
    assert diff is not None
    assert len(diff.ops) == 1


def test_reward_matches_the_benchmark_contract():
    task = Task(id="1", prompt="question", meta={"gold": "answer"})
    assert port.REWARD(task, "answer") == 1.0
    assert port.REWARD(task, "wrong") == 0.0


def test_loader_shapes_cached_rows(monkeypatch):
    monkeypatch.setattr(port, "DATASET_NAME", "test/dataset")
    monkeypatch.setattr(
        port, "hf_rows",
        lambda *_args, **_kwargs: [{"prompt": "q", "gold": "a"}],
    )
    tasks = port.load_tasks(1)
    assert [(task.prompt, task.meta["gold"]) for task in tasks] == [("q", "a")]


def test_dry_run_stops_before_loading_data(monkeypatch, capsys):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("dry-run must be offline")

    monkeypatch.setattr(port, "load_tasks", forbidden)
    monkeypatch.setattr(port, "claude", forbidden)
    monkeypatch.setattr(port, "openai_compatible", forbidden)
    port.main(["--dry-run"])
    assert "dry-run" in capsys.readouterr().out.lower()
