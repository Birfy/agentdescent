"""Offline tests for examples/metasearch: the search rule as the artifact.

No model, no network, no sandbox. The tree is the real `EraTree`, the engine is
the real `evolve()`, and the proposer is a script.
"""

import json

import pytest

from agentdescent.selection import FlatPuct

from examples.era import era_empirical_software as era
from examples.era._era_support import Program
from examples.metasearch import evolve_search_policy as port
from examples.metasearch._landscape import SOURCE, TARGET, search
from examples.metasearch._policy_source import (SEED_SOURCE, EvolvedSelection,
                                                SearchPolicySlot, compile_priority)


# -- the gate -----------------------------------------------------------------


def test_the_seed_rule_compiles_and_is_finite_at_the_root():
    priority = compile_priority(SEED_SOURCE)
    assert priority(0.5, 0, 0, 1.0, 0, 1) == 0.5


@pytest.mark.parametrize("source, reason", [
    ("import os\ndef priority(rank, visits, total, prior, depth, n_nodes):\n    return 1.0",
     "module level"),
    ("def priority(rank, visits, total, prior, depth, n_nodes):\n    return rank / visits",
     "ZeroDivisionError"),
    ("def priority(rank, visits, total, prior, depth, n_nodes):\n    return open('x')",
     "forbidden call"),
    ("def priority(rank, visits, total, prior, depth, n_nodes):\n"
     "    for i in range(3): pass\n    return 1.0", "forbidden syntax"),
    ("def priority(rank, visits, total, prior, depth):\n    return 1.0", "exactly"),
    ("def priority(rank, visits, total, prior, depth, n_nodes):\n    return 1e308 * 10.0",
     "not a finite number"),
    ("def priority(rank, visits, total, prior, depth, n_nodes):\n"
     "    return rank.__class__", "only math"),
    ("def priority(rank, visits, total, prior, depth, n_nodes):\n"
     "    return __builtins__", "dunder"),
])
def test_the_gate_refuses_what_a_scoring_rule_must_not_do(source, reason):
    with pytest.raises(ValueError, match=reason):
        compile_priority(source)


def test_the_slot_validates_at_to_diff_and_counts_the_rejection():
    slot = SearchPolicySlot()
    assert slot.to_diff(slot.initial(), "```python\nimport os\n```", "w", 0, "a") is None
    assert slot.invalid_proposals == 1
    diff = slot.to_diff(
        slot.initial(),
        "```python\ndef priority(rank, visits, total, prior, depth, n_nodes):\n"
        "    return rank * 2\n```", "w", 0, "a")
    assert diff is not None and diff.ops["value"].startswith("def priority")


# -- the wrapper is upstream's rule when given upstream's source ---------------


def _drive(tree):
    """Expand a deterministic, non-monotone landscape through `tree`."""
    trace = []
    while True:
        selection = tree.select_parent()
        if selection is None:
            break
        iteration, parent = selection
        score = float((iteration * 7) % 11) - (2.0 if iteration % 4 == 0 else 0.0)
        tree.add_node(Program(f"n{iteration}", iteration, parent.program.program_id,
                              f"v{iteration}", "", {"rmse": None}, True),
                      score, parent.index)
        trace.append((parent.index, score))
    return trace, [node.num_visits for node in tree.nodes]


def test_the_seed_rule_expands_the_same_nodes_as_flat_puct():
    root = Program("root", 0, None, "v0", "", {"rmse": None}, True)
    ours = era.EraTree(candidate_limit=16, policy=EvolvedSelection(SEED_SOURCE))
    theirs = era.EraTree(c_puct=1.0, candidate_limit=16)
    ours.seed(root, 0.0)
    theirs.seed(root, 0.0)
    assert _drive(ours) == _drive(theirs)
    assert ours.summary()["selection"] == "PrioritySelection"
    assert theirs.summary()["selection"] == "FlatPuct"


def test_a_different_rule_changes_what_is_expanded():
    root = Program("root", 0, None, "v0", "", {"rmse": None}, True)
    greedy = era.EraTree(candidate_limit=16, policy=EvolvedSelection(
        "def priority(rank, visits, total, prior, depth, n_nodes):\n    return rank\n"))
    default = era.EraTree(candidate_limit=16)
    greedy.seed(root, 0.0)
    default.seed(root, 0.0)
    assert _drive(greedy) != _drive(default)


def test_run_agentdescent_era_accepts_a_selection_policy():
    import inspect

    assert "selection" in inspect.signature(era.run_agentdescent_era).parameters


# -- the landscape ------------------------------------------------------------


def test_the_landscape_is_deterministic_and_the_target_is_harder():
    a = search(None, SOURCE, 3, 16)
    b = search(None, SOURCE, 3, 16)
    assert a.curve == b.curve and a.expanded_depths == b.expanded_depths
    assert len(a.curve) == 16 and a.nodes == 17
    assert all(0.0 <= v <= 1.0 for v in a.curve)
    src = sum(search(None, SOURCE, s, 16).auc for s in range(12)) / 12
    tgt = sum(search(None, TARGET, s, 16).auc for s in range(12)) / 12
    assert tgt < src


def test_the_seed_rule_matches_flat_puct_on_the_landscape():
    for family in (SOURCE, TARGET):
        assert [search(FlatPuct(1.0), family, s, 12).curve for s in range(5)] == \
               [search(EvolvedSelection(), family, s, 12).curve for s in range(5)]


# -- the outer loop, end to end ------------------------------------------------


PROPOSAL = """```python
def priority(rank, visits, total, prior, depth, n_nodes):
    # exploit harder
    return rank * rank + 0.5 * (1.0 / n_nodes) * math.sqrt(total) / (1 + visits)
```"""


def test_the_outer_loop_evolves_the_rule_and_reports_both_families():
    calls = {"n": 0}

    def scripted(prompt):
        calls["n"] += 1
        assert "def priority(rank, visits, total, prior, depth, n_nodes)" in prompt
        assert '"curve"' in prompt          # the trace reaches the reflector
        return PROPOSAL

    result = port.run_outer(scripted, rounds=2, workers=2, tasks=10, seed=0,
                            inner_budget=12, mode="sync", max_seconds=120,
                            async_ratio=1)
    assert result.error is None
    assert calls["n"] >= 1
    compile_priority(result.rendered)
    assert result.final_reward >= 0.0
    report = port.validate(SEED_SOURCE, result.rendered, seeds=range(500, 508), budget=12)
    assert set(report) == {"source", "target"}
    for row in report.values():
        assert row["n"] == 8 and 0.0 <= row["seed_rule"] <= 1.0
    text = port.format_report(report)
    assert "transfer ratio" in text


def test_reward_reads_the_trace_and_era_auc_reads_a_history():
    trace = search(None, SOURCE, 1, 8)
    outcome = port.landscape_problem(SOURCE, 8)(None, 1)
    assert outcome.curve == trace.curve
    assert port.reward(port.build_tasks(SOURCE, 1)[0], outcome.to_json()) == pytest.approx(trace.auc)
    assert port.reward(port.build_tasks(SOURCE, 1)[0], "not json") == 0.0

    class Row:
        def __init__(self, r):
            self.held_out_reward = r

    assert port.era_auc([Row(0.2), Row(0.1), Row(0.5)]) == pytest.approx((0.2 + 0.2 + 0.5) / 3)
    assert port.era_auc([]) == 0.0


def test_dry_run_touches_nothing(capsys):
    assert port.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "[dry-run]" in out and "blast_radius=0.6" in out
    assert port.main(["--dry-run", "--serial"]) == 0
    assert "mode=serial" in capsys.readouterr().out
