"""Offline tests for the GEPA (Reflective Prompt Evolution) example.

Focus on the faithful Algorithm-2 Pareto selection, plus prompt representation,
answer normalisation, and scoring -- no network or LLM calls.
"""

import random

from concordia.evolution import Task, evolve
from examples.gepa_prompt_evolution import (
    InstructionSlot,
    _extract_answer,
    build_tasks,
    estimate_calls,
    make_reward,
    normalize_answer,
    pareto_aggregator_factory,
    pareto_frontier,
    pareto_select,
)


def test_pareto_dominated_candidates_are_pruned():
    # cand2 wins both instances -> strictly dominates cand0 and cand1.
    scores = [[1, 0], [0, 1], [1, 1]]
    kept, freq = pareto_frontier(scores)
    assert kept == {2}
    assert freq == {2: 2}


def test_pareto_keeps_complementary_specialists():
    # two specialists, neither dominated -> both stay on the frontier.
    scores = [[1, 0], [0, 1]]
    kept, freq = pareto_frontier(scores)
    assert kept == {0, 1}
    assert freq == {0: 1, 1: 1}


def test_pareto_select_is_frequency_weighted():
    # cand0 wins 3 instances, cand1 wins 1 -> ~75% / 25% sampling.
    scores = [[1, 1, 1, 0], [0, 0, 0, 1]]
    rng = random.Random(0)
    picks = [pareto_select(scores, rng) for _ in range(4000)]
    frac0 = picks.count(0) / len(picks)
    assert 0.70 < frac0 < 0.80


def test_pareto_select_degenerate_falls_back_to_best_average():
    scores = [[0, 0], [0, 0]]  # nobody strictly wins -> best average (tie -> index 0)
    assert pareto_select(scores, random.Random(0)) in (0, 1)


def test_instruction_slot_replaces_and_dedupes():
    s = InstructionSlot()
    state = s.initial()
    assert s.to_diff(state, state["instruction"], "w", 1, "a") is None  # unchanged
    d = s.to_diff(state, "A sharper instruction.", "w", 1, "a")
    assert d is not None and d.ops["instruction"] == "A sharper instruction."


def test_answer_extraction_and_normalisation():
    assert _extract_answer("reasoning...\nAnswer: Ed Wood") == "Ed Wood"
    assert normalize_answer("The Ed Wood.") == "ed wood"
    assert normalize_answer("Yes!") == "yes"


def test_reward_is_exact_match():
    reward = make_reward()
    task = Task(id="t", prompt="q", meta={"target": "Ed Wood"})
    assert reward(task, "blah\nAnswer: ed wood") == 1.0
    assert reward(task, "Answer: Edward Wood") == 0.0


def test_build_tasks_shapes_context():
    rows = [{
        "question": f"Q{i}?", "answer": f"A{i}",
        "context": {"title": ["T1", "T2"],
                    "sentences": [["s1. ", "s2. "], ["s3. "]]},
    } for i in range(10)]
    tasks = build_tasks(rows, limit=5, seed=1)
    assert len(tasks) == 5
    assert all(t.prompt.startswith("Context:") and "Question:" in t.prompt for t in tasks)
    # deterministic given the seed
    assert [t.id for t in tasks] == [t.id for t in build_tasks(rows, limit=5, seed=1)]


def test_pareto_optimizer_integrates_with_evolve():
    """The custom aggregator swaps evolve()'s greedy head for Pareto selection
    and composes complementary improvements (illumination)."""
    tasks = [Task(id=f"t{i}", prompt=f"q{i}",
                  meta={"target": "yes", "hint": f"H{i % 3}"}) for i in range(20)]

    class StubAgent:
        def solve(self, rendered, task):
            return "yes" if task.meta["hint"] in rendered else "no"

        def propose(self, rendered, task, output, reward):
            return (rendered + " " + task.meta["hint"]).strip()

    reward = lambda t, o: 1.0 if o.strip().lower() == "yes" else 0.0
    factory = pareto_aggregator_factory(artifact_id="gepa_prompt", seed=1)
    evolve(tasks, reward, agent=StubAgent(), strategy=InstructionSlot(),
           initial_state={"instruction": "start"}, artifact_id="gepa_prompt",
           rounds=8, n_workers=3, held_out_frac=0.5, aggregator_factory=factory)
    agg = factory.holder["agg"]
    seed_avg = sum(agg.scores[0]) / len(agg.scores[0])
    assert agg.best_avg > seed_avg          # illumination improved held-out
    assert agg.best_avg == 1.0              # composed H0+H1+H2


def test_estimate_calls_scales():
    assert estimate_calls(10, 3, 12) > estimate_calls(5, 3, 12)
    assert estimate_calls(10, 4, 12) > estimate_calls(10, 2, 12)
