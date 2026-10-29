"""The console read model must paginate and route approvals correctly."""

import pytest

from agentdescent.ops_console import (
    ApprovalItem,
    ConsoleSnapshot,
    CostSnapshot,
    LineageSnapshot,
    OperationsConsole,
    QDSnapshot,
    RunSummary,
)


def _console():
    return OperationsConsole(clock=lambda: 100.0)


def _run(i, status="running", reward=0.1):
    return RunSummary(f"r{i}", "t", "w", status, reward=reward,
                      cost=1.0, updated_at=float(i))


def test_upsert_list_filter_and_pagination():
    console = _console()
    for i in range(5):
        console.upsert_run(_run(i, "running" if i % 2 == 0 else "completed"))
    console.upsert_run(RunSummary("other", "t2", "w2", "running", reward=0.9,
                                  updated_at=10.0))
    page = console.list_runs(limit=2)
    assert [run.run_id for run in page.items] == ["other", "r4"]
    assert page.next_cursor == "r4"
    next_page = console.list_runs(limit=10, cursor="r4")
    assert [run.run_id for run in next_page.items] == ["r3", "r2", "r1", "r0"]
    assert [run.run_id for run in console.list_runs(status="completed").items] == [
        "r3", "r1"]
    assert [run.run_id for run in console.list_runs(workspace="w2").items] == ["other"]
    with pytest.raises(KeyError):
        console.list_runs(cursor="missing")
    with pytest.raises(ValueError):
        console.list_runs(status="magic")
    with pytest.raises(ValueError):
        console.list_runs(limit=0)


def test_approval_queue_is_risk_sorted_and_decisions_are_terminal():
    console = _console()
    console.add_approval(ApprovalItem("a", "art-a", risk=0.2, requested_at=1))
    console.add_approval(ApprovalItem("b", "art-b", risk=0.9, requested_at=1))
    console.add_approval(ApprovalItem("c", "art-c", risk=0.5, status="approved"))
    assert [item.approval_id for item in console.pending_approvals()] == ["b", "a"]
    decided = console.decide_approval("b", "denied", actor="alice")
    assert decided.status == "denied" and decided.decided_by == "alice"
    with pytest.raises(ValueError):
        console.decide_approval("b", "approved", actor="alice")
    with pytest.raises(ValueError):
        console.decide_approval("a", "maybe", actor="alice")
    with pytest.raises(KeyError):
        console.decide_approval("missing", "approved", actor="alice")


def test_snapshot_joins_runs_approvals_qd_lineage_and_cost():
    console = _console()
    console.upsert_run(_run(1, reward=0.4))
    console.upsert_run(_run(2, reward=0.8))
    console.add_approval(ApprovalItem("a", "art", risk=0.7))
    console.set_qd(QDSnapshot(coverage=0.5, qd_score=1.2, filled=4, best=0.8,
                              cells=("0,1",)))
    console.set_lineage(LineageSnapshot(nodes=3, edges=2, roots=("r",)))
    console.set_cost(CostSnapshot(tokens=100, cost=1.5,
                                  by_model={"m": 100}, by_project={"p": 1.5}))
    snapshot = console.snapshot()
    assert snapshot.runs == 2
    assert snapshot.by_status == {"running": 2}
    assert snapshot.pending_approvals == 1
    assert snapshot.top_runs[0].run_id == "r2"
    data = console.to_dict()
    assert data["qd"]["coverage"] == 0.5
    assert data["lineage"]["nodes"] == 3
    assert data["cost"]["tokens"] == 100
    assert data["generated_at"] == 100.0


def test_invalid_records_and_cursors_are_refused():
    with pytest.raises(ValueError):
        RunSummary("", "t", "w", "running")
    with pytest.raises(ValueError):
        RunSummary("r", "t", "w", "magic")
    with pytest.raises(ValueError):
        RunSummary("r", "t", "w", "running", risk=1.5)
    with pytest.raises(ValueError):
        ApprovalItem("", "art")
    with pytest.raises(ValueError):
        ApprovalItem("a", "art", status="maybe")
    with pytest.raises(ValueError):
        QDSnapshot(coverage=1.5, qd_score=0, filled=0)
    with pytest.raises(ValueError):
        LineageSnapshot(nodes=-1, edges=0)
    with pytest.raises(ValueError):
        CostSnapshot(tokens=float("nan"))
    console = _console()
    with pytest.raises(KeyError):
        console.get_run("missing")
    console.add_approval(ApprovalItem("a", "art"))
    with pytest.raises(ValueError):
        console.add_approval(ApprovalItem("a", "art"))


def test_cursor_outside_the_filtered_set_is_a_clear_error():
    console = _console()
    console.upsert_run(_run(1, "running"))
    console.upsert_run(_run(2, "completed"))
    with pytest.raises(ValueError):
        console.list_runs(status="running", cursor="r2")
    with pytest.raises(KeyError):
        console.list_runs(cursor="missing")


def test_cost_and_approval_validation():
    with pytest.raises(ValueError):
        CostSnapshot(tokens=-1)
    with pytest.raises(ValueError):
        CostSnapshot(cost=-1)
    console = _console()
    console.add_approval(ApprovalItem("a", "art"))
    with pytest.raises(ValueError):
        console.decide_approval("a", "approved", actor="")
