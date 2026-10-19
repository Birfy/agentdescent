"""Harness composition has to be right where evidence meets routing.

A harness is a named ``(tasks, repo) -> result`` callable; the archive is the
leaderboard (one record per (config, harness) with the measured reward); and the
selector routes a domain to the harness the evidence says wins there, falling
back to round-robin rather than guessing. The fixtures pin the surfaces that
make this honest: harness names are validated, outcomes land in the archive as
ordinary ConsolidationRecords, the leaderboard respects config fingerprints,
the selector is evidence-based with a clear round-robin fallback, and
compose_run records what it runs so the next call routes on the updated
evidence.
"""

import pytest

from agentdescent import (
    ConsolidationStore,
    EvolutionResult,
    HarnessReport,
    SingleSlot,
    Task,
    compose_run,
    config_fingerprint,
    evolve,
    harness,
    harness_leaderboard,
    harness_score,
    record_harness,
    select_harness,
)


# -- a deterministic capability-capped domain -----------------------------------


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
        return "cap:0"
    a = task.meta.get("a", 0)
    if a > _cap(rendered):
        return f"cap:{a}"
    return None


def _tasks(prefix="s", n=8):
    order = [1, 2, 3, 4, 8, 5, 6, 7][:n]
    return [Task(id=f"{prefix}{i}", prompt=f"item {i}",
                 meta={"gold": str(2 * a), "a": a}) for i, a in enumerate(order)]


def _make_harness(name, rounds):
    def run(tasks, repo):
        return evolve(list(tasks), _reward, run=_run, propose=_propose,
                      strategy=SingleSlot(initial_value="cap:0"),
                      rounds=rounds, n_workers=2, repo_path=repo,
                      target_reward=1.0)
    return harness(name, run)


QUICK = _make_harness("quick", rounds=4)
DEEP = _make_harness("deep", rounds=8)
CONFIG_A = {"dataset": "cap-a", "scorer": "exact"}
CONFIG_B = {"dataset": "cap-b", "scorer": "exact"}


# -- the surface ----------------------------------------------------------------


def test_harness_names_are_validated():
    with pytest.raises(ValueError):
        harness("bad name", lambda tasks, repo: None)
    with pytest.raises(ValueError):
        harness("", lambda tasks, repo: None)


def test_compose_run_all_records_every_harness(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    report, winner = compose_run(
        store, _tasks(), [QUICK, DEEP], CONFIG_A, mode="all",
        repo_dir=str(tmp_path / "repos"))
    assert isinstance(report, HarnessReport)
    assert len(report.runs) == 2
    assert {r.harness for r in report.runs} == {"quick", "deep"}
    assert report.winner in ("quick", "deep")
    assert report.reason == "running all harnesses (leaderboard training)"
    assert winner is not None
    # both outcomes are ordinary archive records under the harness artifact ids
    board = harness_leaderboard(store, CONFIG_A)
    assert set(board) == {"quick", "deep"}
    assert all(v["rewards"] for v in board.values())
    # the winner's score matches the report
    assert report.runs[report.runs.index(next(
        r for r in report.runs if r.harness == report.winner))].reward == \
        pytest.approx(winner.final_reward)


def test_select_harness_is_evidence_based_with_round_robin_fallback(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    # no evidence yet -> round-robin (first), never a guess
    chosen, reason = select_harness(store, [QUICK, DEEP], CONFIG_A)
    assert chosen.name == "quick"
    assert "round-robin" in reason
    chosen2, reason2 = select_harness(store, [QUICK, DEEP], CONFIG_A, default="deep")
    assert chosen2.name == "deep"
    assert "caller default" in reason2

    compose_run(store, _tasks(), [QUICK, DEEP], CONFIG_A, mode="all",
                repo_dir=str(tmp_path / "repos"))
    # now evidence decides, and the reason says so
    chosen3, reason3 = select_harness(store, [QUICK, DEEP], CONFIG_A)
    assert "evidence-based" in reason3
    assert chosen3.name == report_winner_of(store, CONFIG_A)


def report_winner_of(store, config):
    return max(("quick", "deep"), key=lambda n: harness_score(store, n, config) or -1)


def test_leaderboard_respects_config_fingerprints(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    compose_run(store, _tasks(), [QUICK, DEEP], CONFIG_A, mode="all",
                repo_dir=str(tmp_path / "repos"))
    # no runs on CONFIG_B -> its leaderboard is empty and scores are None
    assert harness_leaderboard(store, CONFIG_B) == {}
    assert harness_score(store, "quick", CONFIG_B) is None
    # ...and the evidence on A does not leak into a B selection
    chosen, _ = select_harness(store, [QUICK, DEEP], CONFIG_B)
    assert chosen.name == "quick"          # round-robin, not A's winner


def test_compose_run_best_runs_only_the_selected_harness(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    compose_run(store, _tasks(), [QUICK, DEEP], CONFIG_A, mode="all",
                repo_dir=str(tmp_path / "repos"))
    before = store.count()
    chosen, _ = select_harness(store, [QUICK, DEEP], CONFIG_A)
    report, _ = compose_run(store, _tasks(), [QUICK, DEEP], CONFIG_A,
                            mode="best", repo_dir=str(tmp_path / "repos"))
    assert len(report.runs) == 1
    assert report.runs[0].harness == chosen.name
    assert store.count() == before + 1      # one new record for the rerun
    assert "evidence-based" in report.reason


def test_record_harness_is_idempotent_by_run_id(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    result = EvolutionResult(state={}, rendered="quick", final_reward=0.8,
                             history=[], ledger_log=[])
    rid = record_harness(store, harness_name="quick", config=CONFIG_A, result=result,
                         run_id="r1")
    assert rid == "r1"
    record_harness(store, harness_name="quick", config=CONFIG_A, result=result,
                   run_id="r1")
    assert store.count() == 1                # last write wins, atomically
