"""PromptBreeder's Algorithm 1, and the three ways this port used to miss it.

Each test here exists because the port ran, produced a number, and was wrong in
a way no assertion could see: an archive that only grows still returns a best
unit, a tournament ranked on held-out still picks a winner, and three operators
in rotation still mutate. None of those raise.
"""
from __future__ import annotations

import json
import threading

import pytest

from examples._method_policy import PopulationContext
from examples.promptbreeder import promptbreeder_genetic_prompts as pb
from examples.promptbreeder._promptbreeder_operators import (
    OPERATORS, OperatorSampler, build_prompt)
from examples.promptbreeder._promptbreeder_population import (
    PopulationView, PromptBreederPopulation)


# -- the operators -----------------------------------------------------------


def test_all_nine_operators_are_present_and_none_is_a_duplicate():
    assert len(OPERATORS) == 9 == len(set(OPERATORS))


def test_sampling_is_uniform_rather_than_a_rotation():
    """The port cycled three operators round-robin. Rotation guarantees each
    operator's share and sampling does not, so with rotation a run's operator mix
    was an assumption; the paper samples uniformly per replication event."""
    sampler = OperatorSampler(seed=0)
    drawn = [sampler.draw(lambda name: f"prompt for {name}")[0] for _ in range(200)]

    assert set(drawn) == set(OPERATORS), "some operator was never sampled"
    counts = [drawn.count(name) for name in OPERATORS]
    assert max(counts) - min(counts) > 0, "a perfectly even split is a rotation"
    # A rotation of period 9 repeats exactly; sampling does not.
    assert drawn[0:9] != drawn[9:18]


def test_the_sampler_is_reproducible_from_its_seed():
    a = OperatorSampler(seed=7)
    b = OperatorSampler(seed=7)
    pull = lambda s: [s.draw(lambda n: n)[0] for _ in range(30)]
    assert pull(a) == pull(b)


def test_concurrent_draws_do_not_corrupt_the_histogram():
    """`random.Random` is not thread-safe and the workers draw concurrently."""
    sampler = OperatorSampler(seed=1)
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        for _ in range(50):
            sampler.draw(lambda name: f"prompt for {name}")

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(sampler.counts.values()) == 8 * 50


def _genome():
    return {"task_prompt": "answer it", "mutation_prompt": "improve it"}


def _task():
    from agentdescent.evolution import Task
    return Task(id="train:0", prompt="A pen costs 3 dollars. Two pens cost?",
                meta={"answer": "6", "split": "train"})


@pytest.mark.parametrize("operator", OPERATORS)
def test_every_operator_either_builds_a_prompt_or_says_why_not(operator):
    """An operator whose input does not exist returns None so the sampler draws
    again. Degrading it to a first-order mutation instead would make the operator
    histogram a fiction -- the run would report nine operators and have run one.
    """
    view = PopulationView()
    view.publish(
        [{"state": {"task_prompt": f"prompt {i}"}, "score": i / 4.0}
         for i in range(4)],
        [{"task_prompt": "elite a"}, {"task_prompt": "elite b"}])
    sampler = OperatorSampler(seed=0)
    prompt = build_prompt(operator, _genome(), _task(), "6", 1.0, view,
                          sampler.rng())
    assert prompt, f"{operator} could not build a prompt from a full population"
    # The gold answer appears only where the paper's operator is *defined* over
    # execution feedback, and only for a train item -- see
    # `test_proposals_only_ever_see_the_train_split` for why that is safe.
    if "'6'" in prompt or "It wanted: '6'" in prompt:
        assert operator in ("first_order", "lamarckian"), (
            f"{operator} shows the item's answer without consuming feedback")


@pytest.mark.parametrize("operator,missing", [
    ("eda", "an empty population"),
    ("eda_rank_index", "an empty population"),
    ("lineage", "no elite history"),
    ("context_shuffle", "no crossover partner"),
])
def test_population_operators_decline_when_the_population_is_empty(operator, missing):
    sampler = OperatorSampler(seed=0)
    assert build_prompt(operator, _genome(), _task(), "x", 0.0,
                        PopulationView(), sampler.rng()) is None, missing


def test_lamarckian_needs_a_correct_working_not_just_any_output():
    """Its input is a working-out the grader *accepted*; reverse-engineering an
    instruction from a wrong answer teaches the wrong instruction."""
    view, sampler = PopulationView(), OperatorSampler(seed=0)
    assert build_prompt("lamarckian", _genome(), _task(), "6", 0.0, view,
                        sampler.rng()) is None
    assert build_prompt("lamarckian", _genome(), _task(), "", 1.0, view,
                        sampler.rng()) is None
    assert build_prompt("lamarckian", _genome(), _task(), "6", 1.0, view,
                        sampler.rng())


# -- Algorithm 1's population ------------------------------------------------


class _Artifact:
    def __init__(self, state):
        self.state = dict(state)

    def render(self):
        return json.dumps(self.state, sort_keys=True)


def _population(fitness, size=4, seed_units=(), batch=4):
    view = PopulationView()
    pop = PromptBreederPopulation.__new__(PromptBreederPopulation)
    # The aggregator's own __init__ needs a ledger, verifier and audit trail;
    # the population half under test needs none of them, so it is initialised
    # directly rather than behind a stack of fakes that would themselves need
    # asserting.
    pop._archive, pop._seen = [], set()
    pop._archive_lock = threading.Lock()
    pop._fitness, pop._view, pop._size = fitness, view, size
    pop._batch = batch
    import random as _random
    pop._rng = _random.Random(0)
    pop._pending_seed_units = list(seed_units)
    pop._loser, pop._elites, pop._best = None, [], -1.0
    return pop, view


def test_the_population_is_fixed_size_and_the_loser_is_the_slot_reused():
    """The port's archive only grew, so no unit ever died and the selection
    pressure Algorithm 1's replacement step creates was simply absent."""
    scores = {"a": 0.9, "b": 0.1, "c": 0.5, "d": 0.4, "e": 0.8, "f": 0.2}
    pop, _ = _population(lambda s, b: scores[s["task_prompt"]], size=4)

    for name in "abcd":
        pop._admit(_Artifact({"task_prompt": name}), 1)
    assert len(pop._archive) == 4

    winner = pop._replication_event()
    assert winner is not None
    loser = pop._loser
    assert loser is not None, "the tournament did not name a slot to reuse"
    doomed = pop._archive[loser]["state"]["task_prompt"]

    pop._admit(_Artifact({"task_prompt": "e"}), 2)
    living = [u["state"]["task_prompt"] for u in pop._archive]
    assert len(pop._archive) == 4, "the population grew past N"
    assert doomed not in living, "the loser survived its own replacement"
    assert "e" in living


def test_the_tournament_scores_on_training_data_not_the_held_out_gate():
    """Ranking a population on the acceptance gate's own signal makes the
    tournament a second gate rather than a fitness measure."""
    seen = []
    pop, _ = _population(
        lambda s, b: (seen.append((s["task_prompt"], b)), 0.5)[1], size=4)
    for name in "abcd":
        pop._admit(_Artifact({"task_prompt": name}), 1)
    seen.clear()
    pop._replication_event()
    assert len(seen) == 2, "both sampled units must be re-scored, not cached"
    assert {b for _, b in seen} == {4}, (
        "the tournament asked for the whole training split, not a batch -- the "
        "paper's fitness is a sample, and scoring every item is both a departure "
        "and the dominant cost of a run")
    # `fitness` is the runner's train-split evaluator; the held-out verifier is
    # never consulted here, and `_population` gives this object no verifier at
    # all -- if the shared archive's `_admit` were still in play this would raise.
    assert not hasattr(pop, "verifier")


def test_the_winner_is_the_higher_scoring_of_the_two_sampled():
    scores = {"lo": 0.1, "hi": 0.9}
    pop, _ = _population(lambda s, b: scores[s["task_prompt"]], size=2)
    pop._admit(_Artifact({"task_prompt": "lo"}), 1)
    pop._admit(_Artifact({"task_prompt": "hi"}), 1)
    winner = pop._replication_event()
    assert winner["state"]["task_prompt"] == "hi"
    assert pop._archive[pop._loser]["state"]["task_prompt"] == "lo"


def test_a_population_of_one_cannot_hold_a_tournament():
    pop, _ = _population(lambda s, b: 1.0, size=4)
    pop._admit(_Artifact({"task_prompt": "only"}), 1)
    assert pop._replication_event() is None


def test_the_view_the_operators_read_tracks_the_population():
    scores = {"a": 0.2, "b": 0.9, "c": 0.5}
    pop, view = _population(lambda s, b: scores[s["task_prompt"]], size=4)
    for name in "abc":
        pop._admit(_Artifact({"task_prompt": name}), 1)
    assert view.task_prompts() and set(view.task_prompts()) == {"a", "b", "c"}
    assert view.task_prompts(ranked=True) == ["a", "c", "b"], "not ascending by fitness"
    assert view.elite_history()[-1] == "b", "the best unit is not the last elite"


# -- initialisation ----------------------------------------------------------


def test_seeding_produces_a_population_rather_than_one_instruction():
    """A run that starts from a single instruction has nothing to sample for its
    first tournaments, and three of the nine operators have no input until enough
    children commit -- that is a different algorithm for the first stretch of the
    budget, not a slow start."""
    calls = []

    def llm(prompt, **kw):
        calls.append(kw.get("unit"))
        return json.dumps({"task_prompt": f"instruction {len(calls)}"})

    units = pb.seed_population(llm, size=6, seed=0)
    assert len(units) == 5, "N-1 generated units, the seed itself being the Nth"
    assert len({u["task_prompt"] for u in units}) == 5, "the units are identical"
    assert all(u["mutation_prompt"] for u in units), "no mutation-prompt drawn"
    assert set(calls) == {"seed-population"}


def test_seeding_survives_a_reply_it_cannot_parse():
    units = pb.seed_population(lambda prompt, **kw: "not json at all", size=6,
                               seed=0)
    assert units == []


def test_seed_calls_are_not_billed_as_proposals():
    """`observed_proposal_calls <= expected` is what makes the matrix rows
    comparable. Seeding is initialisation, and recording it under `proposal:`
    would both break that check and hide setup cost inside the search's cost."""
    import inspect

    from examples import _method_runner

    source = inspect.getsource(_method_runner.run_port)
    assert 'init_llm = PhasedLLM' in inspect.getsource(_method_runner)
    assert "llm=lambda prompt, **kw: init_llm(prompt, **kw)" in source, (
        "the population hook is being handed the proposal-phase LLM")


def test_proposals_only_ever_see_the_train_split():
    """Two operators put the grader's answer in the prompt, which is the domain's
    whole teaching signal -- and is only safe because the item is always a
    training item.

    The engine splits `tasks` positionally: `cut = round(len(tasks) * (1 -
    held_out_frac))`. The runner passes `engine_tasks = train + held_out` with
    `shuffle=False` and `held_out_frac = len(held_out)/len(engine_tasks)`, so the
    cut lands exactly on the boundary between the two lists. Shuffle them, or
    size the splits unevenly, and held-out answers start reaching proposal
    prompts -- silently, and the gate would still report a rising score.
    """
    policy = pb.build(0)
    tasks = policy.engine_tasks
    cut = max(1, round(len(tasks) * (1 - policy.held_out_frac)))
    assert [t.id for t in tasks[:cut]] == [t.id for t in policy.train_tasks]
    assert [t.id for t in tasks[cut:]] == [t.id for t in policy.held_out_tasks]
    assert all(t.meta["split"] == "train" for t in tasks[:cut])


def test_the_population_hook_is_what_the_port_declares():
    policy = pb.build(0)
    assert policy.population is not None, "the port fell back to the shared archive"
    ctx = PopulationContext(
        selection=policy.engine.selection, artifact_id="candidate-promptbreeder",
        conflict=None, fusion=None, acceptance=None,
        llm=lambda prompt, **kw: json.dumps({"task_prompt": "x"}),
        fitness=lambda state, batch: 0.5, train_size=16, seed=0)
    factory = policy.population(ctx)
    assert callable(factory)


def test_hypermutation_is_two_steps_and_ends_at_the_task_prompt():
    """The paper's hypermutation writes a mutation-prompt and then **applies it
    to the task-prompt**. Stopping after step one returns a candidate whose
    task-prompt is unchanged, which cannot move the score by construction -- one
    seed spent 29 of its 80 draws on exactly that and finished at 0.000.
    """
    from examples.promptbreeder._promptbreeder_operators import TWO_STEP

    assert set(TWO_STEP) == {"zero_order_hyper", "first_order_hyper"}

    policy = pb.build(0)
    calls = []

    def llm(prompt, **kw):
        calls.append((kw.get("unit", ""), prompt))
        if 'with a "mutation_prompt" string' in prompt:
            return json.dumps({"mutation_prompt": "sharpen the output convention"})
        return json.dumps({"task_prompt": "state the amount in integer cents"})

    rendered = policy.strategy.render(policy.strategy.initial())
    # Draw until a two-step operator comes up; the sampler is uniform, so this
    # terminates quickly and does not assume an order.
    for _ in range(200):
        before = len(calls)
        payload = policy.propose(llm, rendered, _task(), "6", 1.0)
        drawn = [u for u, _ in calls[before:]]
        if any(u.endswith(":apply") for u in drawn):
            assert len(drawn) == 2, "the second step did not run"
            assert json.loads(payload)["task_prompt"] == (
                "state the amount in integer cents"), (
                "the two-step operator returned only its new mutation-prompt, so "
                "the task-prompt would be unchanged")
            assert json.loads(payload)["mutation_prompt"] == (
                "sharpen the output convention")
            break
    else:
        raise AssertionError("no two-step operator was drawn in 200 attempts")


def test_the_declared_proposal_budget_covers_the_second_step():
    """`observed_proposal_calls <= expected` is what makes these rows comparable,
    and hypermutation legitimately costs two calls."""
    assert pb.build(0).proposal_calls_per_candidate == 2
