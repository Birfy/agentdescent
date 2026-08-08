"""Offline tests for the GEPA (Reflective Prompt Evolution) example.

Focus on the faithful Algorithm-2 Pareto selection, plus prompt representation,
answer normalisation, and scoring -- no network or LLM calls.
"""

import random

from agentdescent.evolution import Task, evolve
from examples.gepa.gepa_prompt_evolution import (
    InstructionSlot,
    ParetoAggregator,
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


def test_admitting_a_round_of_candidates_scores_their_rows_together():
    """The port's largest cost was serial in two dimensions, not one.

    `_score_row` swept D_pareto one instance at a time, and `step()` admitted one
    card at a time -- so a round from eight workers queued eight full sweeps back
    to back. On a 20-instance D_pareto at ~34s a call that is the whole run.

    Both are flattened into one pool now, which must not change the result: the
    pool contents, their order, and every row have to match one-at-a-time
    admission exactly. Dedup stays sequential so *which* candidates enter, and in
    what order, never depends on thread timing.
    """
    import threading
    import time

    class _Verifier:
        def __init__(self):
            self.held_out = list(range(6))
            self.live = self.peak = 0
            self.lock = threading.Lock()

        def eval_fn(self, artifact, tasks):
            with self.lock:
                self.live += 1
                self.peak = max(self.peak, self.live)
            # A real eval is a model call; without some duration the threads
            # never overlap and `peak` cannot observe concurrency that is there.
            time.sleep(0.01)
            try:
                return float((hash((artifact.render(), tasks[0])) % 2))
            finally:
                with self.lock:
                    self.live -= 1

    class _Artifact:
        def __init__(self, key):
            self.key = key
            self.state = {"instruction": key}

        def render(self):
            return self.key

    def _blank(concurrency):
        agg = ParetoAggregator.__new__(ParetoAggregator)
        agg.verifier = _Verifier()
        agg.eval_concurrency = concurrency
        agg.states, agg.scores, agg._seen = [], [], set()
        return agg

    candidates = [_Artifact(f"c{i}") for i in range(5)]

    one_at_a_time = _blank(1)
    for candidate in candidates:
        one_at_a_time._admit(candidate)

    batched = _blank(8)
    assert batched._admit_many(candidates) == len(candidates)

    assert batched.scores == one_at_a_time.scores, "batching changed a score row"
    assert ([s["instruction"] for s in batched.states]
            == [s["instruction"] for s in one_at_a_time.states]), \
        "pareto_frontier reads states/scores as parallel lists; order must hold"
    assert batched.verifier.peak > 1, "the batch did not actually run concurrently"
    assert batched.verifier.peak <= 8, \
        "nesting a pool per row inside a pool per candidate would put " \
        "concurrency^2 requests on the endpoint"


def test_duplicate_candidates_are_dropped_before_they_are_scored():
    """Dedup is what keeps the pool from paying twice for the same instruction."""
    class _V:
        held_out = [0, 1]
        calls = 0

        def eval_fn(self, artifact, tasks):
            type(self).calls += 1
            return 1.0

    class _A:
        def __init__(self, key):
            self.key, self.state = key, {"instruction": key}

        def render(self):
            return self.key

    agg = ParetoAggregator.__new__(ParetoAggregator)
    agg.verifier, agg.eval_concurrency = _V(), 4
    agg.states, agg.scores, agg._seen = [], [], set()

    assert agg._admit_many([_A("x"), _A("x"), _A("y")]) == 2
    assert [s["instruction"] for s in agg.states] == ["x", "y"]
    assert _V.calls == 4, "a duplicate must not be scored"
