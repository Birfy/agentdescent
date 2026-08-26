"""Offline contract tests for the ERA AgentDescent port.

The load-bearing one is `test_serial_tree_reproduces_upstream_futs`: it drives
this port's tree and a line-by-line transcription of upstream's `futs.search`
with the same mock generator and executor, and asserts they expand the same
node in the same order. Without it, "the selection rule is untouched" is a claim
resting on someone reading two files side by side.
"""

from __future__ import annotations

import json
import math
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from agentdescent.selection import Candidate, FlatPuct, SelectionContext
from examples.era import _era_support as support
from examples.era import era_empirical_software as port


# --------------------------------------------------------------------------
# A transcription of google-research/era @ b836730 implementation/futs.py
# --------------------------------------------------------------------------


class _UpstreamNode:
    def __init__(self, index, parent_index, program, score):
        self.index = index
        self.parent_index = parent_index
        self.program = program
        self.score = score
        self.num_visits = 0
        self.rank_score = 0.5
        self.puct = 0.5


def _upstream_rank_scores(nodes):
    if len(nodes) == 1:
        nodes[0].rank_score = 0.5
        return
    for rank, node in enumerate(sorted(nodes, key=lambda n: n.score)):
        node.rank_score = rank / (len(nodes) - 1)


def _upstream_pucts(nodes, c_puct):
    prior = 1 / len(nodes)
    total = sum(n.num_visits for n in nodes)
    for node in nodes:
        node.puct = node.rank_score + c_puct * prior * math.sqrt(total) / (
            1 + node.num_visits)


def _upstream_backpropagate(nodes, node):
    node.num_visits += 1
    if node.parent_index is not None:
        _upstream_backpropagate(nodes, nodes[node.parent_index])


def _upstream_search(initial_program, initial_score, generate, execute,
                     num_iterations, c_puct=1.0):
    """`futs.search`, returning the expansion trace as well as the winner."""
    nodes = [_UpstreamNode(0, None, initial_program, initial_score)]
    trace = []
    for _ in range(num_iterations):
        _upstream_rank_scores(nodes)
        _upstream_pucts(nodes, c_puct)
        best = max(nodes, key=lambda n: n.puct)
        program = generate(best.program, best.score)
        score = execute(program)
        node = _UpstreamNode(len(nodes), best.index, program, score)
        nodes.append(node)
        _upstream_backpropagate(nodes, node)
        trace.append((best.index, program, score))
    winner = max(nodes, key=lambda n: n.score)
    return trace, winner.program, winner.score, [n.num_visits for n in nodes]


# --------------------------------------------------------------------------
# The selection rule
# --------------------------------------------------------------------------


def _rows(spec):
    return tuple(
        Candidate(artifact_id="era", version=i, score=score, selected=visits,
                  parent=parent)
        for i, (score, visits, parent) in enumerate(spec)
    )


def test_rank_scores_match_the_upstream_unit_test():
    # futs_test.py::test_compute_rank_scores
    ranks = FlatPuct._rank_scores(_rows([(1.0, 0, None), (3.0, 0, 0), (2.0, 0, 0)]))
    assert ranks == [0.0, 1.0, 0.5]


def test_a_lone_node_ranks_one_half():
    assert FlatPuct._rank_scores(_rows([(1.0, 0, None)])) == [0.5]


def test_puct_matches_the_upstream_unit_test():
    # futs_test.py::test_compute_pucts -- same nodes, same expected values.
    rows = _rows([(1.0, 1, None), (3.0, 4, 0), (2.0, 0, 0)])
    ranks = [0.0, 1.0, 0.5]
    total, prior = 5, 1 / 3
    expected = [
        ranks[i] + prior * math.sqrt(total) / (1 + visits)
        for i, visits in enumerate((1, 4, 0))
    ]
    assert FlatPuct._rank_scores(rows) == ranks
    policy = FlatPuct(1.0)
    # The policy picks the largest PUCT; assert the ordering it implies.
    assert expected[2] == pytest.approx(0.5 + prior * math.sqrt(5))
    assert policy.select(SelectionContext(head=rows[0], candidates=rows), 1)[0].version == (
        max(range(3), key=lambda i: expected[i]))


def test_a_second_pick_reserves_a_visit_up_the_parent_chain():
    """Upstream is serial, so `n > 1` is this port's generalisation.

    The reservation is what stops N workers from all being sent to the same
    node on evidence that has not arrived yet. At `c_puct=10` the exploration
    term is large enough that one reserved visit moves the winner, so the
    second pick differs -- and it differs for the documented reason, which the
    hand-applied comparison below is what actually pins.
    """
    rows = _rows([(1.0, 1, None), (3.0, 1, 0), (2.0, 1, 0)])
    policy = FlatPuct(10.0)
    ctx = SelectionContext(head=rows[0], candidates=rows)
    picked = [candidate.version for candidate in policy.select(ctx, 2)]

    first = policy.select(ctx, 1)[0].version
    # `first` and every ancestor of it take the visit upstream would have
    # backpropagated once that expansion finished.
    bumped = list(rows)
    cursor: Optional[int] = first
    while cursor is not None:
        row = bumped[cursor]
        bumped[cursor] = Candidate(artifact_id=row.artifact_id, version=row.version,
                                   score=row.score, selected=row.selected + 1,
                                   parent=row.parent)
        cursor = bumped[cursor].parent
    second = policy.select(
        SelectionContext(head=bumped[0], candidates=tuple(bumped)), 1)[0].version

    assert picked == [first, second]
    assert picked[0] != picked[1]


def test_serial_tree_reproduces_upstream_futs():
    """The port's tree and upstream's loop expand the same node every time."""

    def generate(program, _score):
        return f"v{int(program[1:]) + 1}"

    def execute(program):
        # Non-monotone on purpose: a version number alone would make every new
        # node the best one and hide any selection difference.
        return float((int(program[1:]) * 7) % 11)

    trace, best_program, best_score, visits = _upstream_search(
        "v0", 0.0, generate, execute, num_iterations=12)

    tree = port.EraTree(c_puct=1.0, candidate_limit=12)
    tree.seed(support.Program("root", 0, None, "v0", "", {"rmse": None}, True), 0.0)
    ours: List[Tuple[int, str, float]] = []
    while True:
        selection = tree.select_parent()
        if selection is None:
            break
        _, parent, _ = selection
        program = generate(parent.program.code, parent.score)
        score = execute(program)
        tree.add_node(
            support.Program(program, 0, parent.program.program_id, program, "",
                            {"rmse": None}, True),
            score,
            parent.index,
        )
        ours.append((parent.index, program, score))

    assert ours == trace
    assert tree.best().program.code == best_program
    assert tree.best().score == best_score
    assert [node.num_visits for node in tree.nodes] == visits


# --------------------------------------------------------------------------
# The gate and the parser
# --------------------------------------------------------------------------


def test_gate_accepts_the_upstream_baseline_and_rejects_unsafe_code():
    assert support.validate_source(support.INITIAL_PROGRAM)[0]
    assert not support.validate_source(
        "import subprocess\ndef train_and_predict(a, b): return []")[0]
    assert not support.validate_source(
        "def train_and_predict(a, b):\n    return open('/etc/passwd').read()\n")[0]
    assert not support.validate_source(
        "import pandas as pd\ndef helper(a, b): return []")[0]
    assert not support.validate_source(
        "import xgboost\ndef train_and_predict(a, b): return []")[0]


def test_extract_program_prefers_the_longest_fenced_block():
    reply = (
        "Here is the idea:\n```python\nprint('short')\n```\n"
        "and the solution:\n```python\n\"\"\"Gradient boosting on ratios.\"\"\"\n"
        "import pandas as pd\n\n\ndef train_and_predict(a, b):\n    return []\n```\n"
    )
    code, summary = support.extract_program(reply)
    assert "train_and_predict" in code and "short" not in code
    assert summary == "Gradient boosting on ratios."


def test_extract_program_falls_back_to_the_whole_reply():
    code, summary = support.extract_program("def train_and_predict(a, b):\n    return []\n")
    assert code.startswith("def train_and_predict")
    assert summary == ""


def test_reward_preserves_the_upstream_score_ordering():
    better = {"rmse": 0.51}
    worse = {"rmse": 0.73}
    assert support.framework_score(better) > support.framework_score(worse)
    assert -better["rmse"] > -worse["rmse"]
    assert support.framework_score({"rmse": None}) == 0.0


def test_rmse_matches_the_textbook_definition():
    assert support.rmse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0
    assert support.rmse([0.0, 0.0], [1.0, 1.0]) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# The engine plug-ins
# --------------------------------------------------------------------------


def test_strategy_round_trips_a_proposal_into_a_diff():
    strategy = port.EraStrategy()
    state = strategy.initial()
    assert "train_and_predict" in strategy.render(state)
    proposal = json.dumps({
        "code": "def train_and_predict(a, b):\n    return []\n",
        "change_summary": "ridge",
        "iteration": 3,
        "parent_index": 2,
        "parent_id": "abc",
    })
    diff = strategy.to_diff(state, proposal, "worker-1", 4, port.ARTIFACT_ID)
    assert diff is not None
    assert diff.ops["parent_index"] == "2"
    assert diff.ops["iteration"] == "3"
    assert strategy.to_diff(state, "not json", "w", 0, port.ARTIFACT_ID) is None


    # ------------------------------------------------------------------
    # The damaged-reply guard
    # ------------------------------------------------------------------


DAMAGED_REPLIES = {
    "spliced identifier": "```python\ndef train_and_predict(a, b):\n"
                          "    x = c_orig,0$ zG$C$F1_orig\n    return x\n```",
    "spliced number": "```python\ndef train_and_predict(a, b):\n"
                      "    return val9.3192\n```",
    "no code at all": "I would recommend a gradient boosting model here.",
    "empty": "   ",
}

#: Programs that are wrong, slow or fatal -- and that the guard must let past,
#: because turning them into nodes scoring -inf is the algorithm working.
INTACT_BUT_BAD = {
    "never terminates": "```python\ndef train_and_predict(a, b):\n"
                        "    while True:\n        pass\n```",
    "raises at runtime": "```python\ndef train_and_predict(a, b):\n"
                         "    return 1.0 / 0.0\n```",
    "wrong answer": "```python\ndef train_and_predict(a, b):\n    return []\n```",
    "banned import": "```python\nimport os\ndef train_and_predict(a, b):\n"
                     "    return []\n```",
}


@pytest.mark.parametrize("name", sorted(DAMAGED_REPLIES))
def test_a_reply_the_channel_damaged_is_recognised(name):
    intact, reason = support.reply_is_intact(DAMAGED_REPLIES[name])
    assert not intact and reason


@pytest.mark.parametrize("name", sorted(INTACT_BUT_BAD))
def test_a_bad_program_is_not_treated_as_damage(name):
    """The distinction the guard exists to keep.

    Redrawing a program that merely fails would be the port quietly retrying
    its own failures, which is a different algorithm: upstream appends the
    failed node and this port pins that behaviour elsewhere in this file.
    """
    assert support.reply_is_intact(INTACT_BUT_BAD[name])[0]


def test_the_guard_redraws_damage_counts_it_and_then_gives_up():
    replies = [DAMAGED_REPLIES["spliced number"],
               DAMAGED_REPLIES["spliced identifier"],
               INTACT_BUT_BAD["raises at runtime"]]
    drawn = []

    def channel(prompt):
        drawn.append(prompt)
        return replies[len(drawn) - 1]

    counter = {}
    complete = support.with_intact_replies(channel, attempts=4, counter=counter)
    assert complete("p") == replies[2]
    assert len(drawn) == 3
    assert counter == {"drawn": 3, "damaged": 2}

    # A channel that is always damaged must terminate, hand back the last
    # reply, and say so -- the run then records those expansions as the failed
    # nodes they are rather than hanging.
    calls = {"n": 0}

    def always_damaged(prompt):
        calls["n"] += 1
        return DAMAGED_REPLIES["spliced number"]

    with pytest.warns(RuntimeWarning, match="damaged"):
        out = support.with_intact_replies(always_damaged, attempts=3)("p")
    assert calls["n"] == 3 and out == DAMAGED_REPLIES["spliced number"]


def test_the_guard_is_off_when_one_attempt_is_allowed():
    calls = {"n": 0}

    def channel(prompt):
        calls["n"] += 1
        return DAMAGED_REPLIES["spliced number"]

    support.with_intact_replies(channel, attempts=1)("p")
    assert calls["n"] == 1


def test_an_empty_reply_still_becomes_a_node():
    """Upstream appends a node for every expansion, including a failed one.

    `generate_fn` returning an empty program means `execute_fn` scores it
    `-inf` and `futs.search` appends it anyway. Dropping it would shrink the
    rank denominator and raise the `1/N` prior on every later iteration.
    """
    strategy = port.EraStrategy()
    diff = strategy.to_diff(
        strategy.initial(),
        json.dumps({"code": "", "iteration": 1, "parent_index": 0}),
        "w", 0, port.ARTIFACT_ID)
    assert diff is not None and diff.ops["code"] == ""
    # And the gate is what makes it score -inf rather than reach the ledger.
    assert not support.validate_source("")[0]
    assert support._zero_metrics("empty source")["score"] == -math.inf


def test_reward_program_reads_the_runner_payload():
    good = json.dumps({"valid": True, "metrics": {"rmse": 0.5}, "error": ""})
    bad = json.dumps({"valid": False, "metrics": {"rmse": None}, "error": "boom"})
    assert port.reward_program(port.Task(id="t", prompt=""), good) == pytest.approx(1 / 1.5)
    assert port.reward_program(port.Task(id="t", prompt=""), bad) == 0.0
    assert port.reward_program(port.Task(id="t", prompt=""), "garbage") == 0.0


def test_tasks_are_one_per_scoring_shard():
    tasks = port.build_tasks(5)
    assert [task.meta["shard"] for task in tasks] == [0, 1, 2, 3, 4]


def test_dry_run_touches_no_network_data_or_sandbox(capsys):
    assert port.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "[dry-run]" in out
    assert "FUTS" in out


def test_serial_is_visible_in_the_printed_plan(capsys):
    assert port.main(["--dry-run", "--serial"]) == 0
    assert "mode=serial" in capsys.readouterr().out


def test_reflective_merge_is_refused_rather_than_ignored():
    with pytest.raises(SystemExit) as excinfo:
        port.main(["--dry-run", "--reflective-merge"])
    assert "delete tree nodes" in str(excinfo.value)


def test_budget_rollouts_sets_the_expansion_budget():
    args = port.build_parser().parse_args(["--budget-rollouts", "24"])
    assert args.budget_rollouts == 24


# --------------------------------------------------------------------------
# The dataset, on a fixture rather than the network
# --------------------------------------------------------------------------


_FIXTURE_HEADER = "id,MedInc,HouseAge,MedHouseVal"


def _fixture_csv(rows: int = 200) -> str:
    lines = [_FIXTURE_HEADER]
    for index in range(rows):
        lines.append(f"{index},{index * 0.1:.3f},{index % 40}.0,{index * 0.01:.3f}")
    return "\n".join(lines) + "\n"


def test_splits_reproduce_upstreams_eighty_twenty_head_tail(monkeypatch, tmp_path):
    monkeypatch.setattr(support, "fetch_text", lambda *a, **k: _fixture_csv(1000))
    monkeypatch.setattr(support, "cache_path",
                        lambda subdir, name: str(tmp_path / subdir / name))
    splits = support.prepare_splits(shards=4, test_shards=2, url="fixture://s3e1")
    assert splits.train_rows == 800                    # upstream: int(0.8 * len)
    assert len(splits.shard_paths) == 6                # 4 scoring + 2 test
    assert sum(len(truth) for truth in splits.shard_truth) == 198  # 6 x (200 // 6)
    # The target is dropped from what the candidate reads, exactly as upstream's
    # `test_features = test_df.drop('MedHouseVal', axis=1)` does.
    header = splits.shard_paths[0].read_text(encoding="utf-8").splitlines()[0]
    assert support.TARGET_COLUMN not in header
    assert support.TARGET_COLUMN in splits.train_path.read_text(encoding="utf-8").splitlines()[0]
    # The truth values are the column that was dropped: the first held-out row
    # is fixture row 800, whose target is 800 * 0.01.
    assert splits.shard_truth[0][0] == pytest.approx(8.00)


def test_a_shard_too_small_to_score_is_refused(monkeypatch, tmp_path):
    monkeypatch.setattr(support, "fetch_text", lambda *a, **k: _fixture_csv(200))
    monkeypatch.setattr(support, "cache_path",
                        lambda subdir, name: str(tmp_path / subdir / name))
    with pytest.raises(ValueError, match="too few for a stable RMSE"):
        support.prepare_splits(shards=8, test_shards=4, url="fixture://s3e1")


# --------------------------------------------------------------------------
# The sandbox, checked against the kernel rather than by reading the profile
# --------------------------------------------------------------------------


@pytest.mark.skipif(support.sandbox_backend() is None,
                    reason="no Bubblewrap or Seatbelt on this host")
def test_the_sandbox_blocks_the_writes_and_network_it_claims_to_block(tmp_path):
    """Run a probe under the real profile and assert the kernel agreed.

    The ERA gate allows scikit-learn, so it cannot be the boundary -- which
    makes this the test that decides whether the port is safe to run at all.

    Two probes for "outside", because the two backends refuse differently and
    one of them used to make this test fail on an ordinary Linux host:

    * ``outside`` sits beside the scratch directory, which pytest puts under
      ``/tmp``. The Bubblewrap profile mounts a **fresh tmpfs over `/tmp`**, so
      the write there succeeds *inside the sandbox* and reaches nothing: the
      guarantee is that it never lands on the host, which is what
      ``outside.exists()`` checks from here. Asserting the write itself was
      refused would be asserting an implementation detail that Seatbelt happens
      to have and Bubblewrap deliberately does not.
    * ``readonly`` sits under ``/usr``, which no profile makes writable on
      either platform. That one has to be refused outright.
    """
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    outside = tmp_path / "outside.txt"
    readonly = f"/usr/lib/era-sandbox-probe-{os.getpid()}"
    probe = scratch / "probe.py"
    probe.write_text(
        "import json, socket\n"
        "result = {}\n"
        "try:\n"
        f"    open({str(outside)!r}, 'w').write('x')\n"
        "    result['outside_write'] = 'allowed'\n"
        "except Exception as exc:\n"
        "    result['outside_write'] = type(exc).__name__\n"
        "try:\n"
        f"    open({readonly!r}, 'w').write('x')\n"
        "    result['readonly_write'] = 'allowed'\n"
        "except Exception as exc:\n"
        "    result['readonly_write'] = type(exc).__name__\n"
        "try:\n"
        f"    open({str(scratch / 'inside.txt')!r}, 'w').write('x')\n"
        "    result['inside_write'] = 'allowed'\n"
        "except Exception as exc:\n"
        "    result['inside_write'] = type(exc).__name__\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 53), timeout=3)\n"
        "    result['network'] = 'allowed'\n"
        "except Exception as exc:\n"
        "    result['network'] = type(exc).__name__\n"
        "print(json.dumps(result))\n",
        encoding="utf-8",
    )
    command, env = support.sandbox_command(
        probe, train=probe, test=probe, rows=0, timeout=30.0, nproc_limit=64)
    # Swap the runner out for the probe: the isolation under test is the
    # wrapper, not what it happens to execute.
    command = [part for part in command if part != str(support.RUNNER)]
    completed = subprocess.run(command, capture_output=True, text=True,
                               timeout=90, env=env, cwd=str(scratch))
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["readonly_write"] != "allowed"
    assert payload["inside_write"] == "allowed"
    assert payload["network"] != "allowed"
    assert not outside.exists()
    assert not os.path.exists(readonly)


# ---------------------------------------------------------------------------
# Recall: the snapshot the tree hands a mutation prompt
# ---------------------------------------------------------------------------


def _seeded_tree(scores, *, valid=None):
    """A tree holding one node per score, so recall has something to rank."""
    tree = port.EraTree(c_puct=1.0, metric_key="rmse")
    tree.seed(support.Program("root", 0, None, "root", "", {"rmse": None}, True),
              scores[0])
    for index, score in enumerate(scores[1:], start=1):
        ok = True if valid is None else valid[index]
        tree.add_node(
            support.Program(f"p{index}", 0, "root", f"p{index}", "",
                            {"rmse": None, "n": index}, ok),
            score, 0)
    return tree


def test_recall_is_off_by_default_which_is_upstreams_single_parent_view():
    """`futs.search` shows a mutation one parent and one score. So does this."""
    tree = _seeded_tree([0.0, 1.0, 2.0])
    iteration, parent, recalled = tree.select_parent()
    assert isinstance(iteration, int) and parent is not None
    assert recalled == ()


def test_recall_returns_the_best_nodes_first():
    tree = _seeded_tree([0.0, 2.0, 1.0, 3.0])
    _, _, recalled = tree.select_parent(2)
    assert [score for _, score in recalled] == [3.0, 2.0]
    assert [metrics["n"] for metrics, _ in recalled] == [3, 1]


def test_recall_leaves_out_a_node_that_never_ran():
    """A refused or crashed program has no structure worth telling the model about."""
    tree = _seeded_tree([0.0, 1.0, 2.0], valid=[True, False, True])
    _, _, recalled = tree.select_parent(8)
    assert [score for _, score in recalled] == [2.0, 0.0]


def test_recall_leaves_out_a_node_scored_minus_infinity():
    tree = port.EraTree(c_puct=1.0, metric_key="rmse")
    tree.seed(support.Program("root", 0, None, "root", "", {"rmse": None}, True), 0.5)
    tree.add_node(support.Program("bad", 0, "root", "bad", "", {"rmse": None}, True),
                  float("-inf"), 0)
    _, _, recalled = tree.select_parent(8)
    assert [score for _, score in recalled] == [0.5]


def test_the_snapshot_comes_from_the_same_view_of_the_tree_as_the_parent():
    """One lock, so a prompt can never call a node "already tried" too early.

    The snapshot is built while the parent is chosen, which is why it is on
    `select_parent` rather than a second reader a caller has to remember to
    take at the right moment.
    """
    tree = _seeded_tree([0.0, 1.0])
    _, _, before = tree.select_parent(8)
    tree.add_node(support.Program("late", 0, "root", "late", "", {"rmse": None}, True),
                  9.0, 0)
    _, _, after = tree.select_parent(8)
    assert len(before) == 2 and len(after) == 3
    assert max(score for _, score in after) == 9.0


def test_propose_hands_the_snapshot_to_the_domains_prompt():
    """The wiring: what the tree collected has to reach the prompt that uses it."""
    tree = _seeded_tree([0.0, 1.0, 2.0])
    seen = {}

    def prompt_for(program, recalled=()):
        seen["recalled"] = recalled
        return "write me a program"

    reply = "```python\ndef discover(x, y, spec):\n    return '1.0'\n```"
    propose = port.make_propose(tree, lambda prompt: reply,
                                prompt_for=prompt_for, recall=2)
    propose("", port.Task(id="t0", prompt="p"), "", 0.0)
    assert [score for _, score in seen["recalled"]] == [2.0, 1.0]
