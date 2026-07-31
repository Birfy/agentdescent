"""The one-call path: dataset in, evolved skill out.

It must stay a *thin* wrapper -- same engine, same result object, every default
overridable -- or it becomes a second system with its own semantics.
"""
import pytest

from agentdescent import evolve_skill
from agentdescent.agents import echo
from agentdescent.evolution import EvolutionResult, SingleSlot, Task

ROWS = [{"q": f"value {i}", "a": str(i * 100)} for i in range(1, 13)]


def _model(prompt):
    """Answers correctly only once the skill mentions cents."""
    n = int(prompt.rsplit(" ", 1)[-1])
    return str(n * 100) if "cents" in prompt else str(n)


def _evolve(**kw):
    kw.setdefault("propose", lambda r, t, o, s: "answer in cents")
    kw.setdefault("score", "last_number")
    kw.setdefault("instruction", "answer plainly")
    return evolve_skill(ROWS, model=echo(_model), prompt="q", gold="a", **kw)


def test_a_dataset_of_dicts_is_enough():
    res = _evolve()
    assert isinstance(res, EvolutionResult)
    assert res.final_reward == 1.0
    assert "cents" in res.rendered


def test_it_returns_the_same_result_object_as_evolve():
    """Not a different system -- outcomes(), save(), error all still work."""
    res = _evolve()
    assert res.outcomes()
    assert res.error is None
    assert res.history


def test_ready_made_tasks_are_accepted():
    tasks = [Task(id=str(i), prompt=f"value {i}", meta={"gold": str(i * 100)})
             for i in range(1, 13)]
    res = evolve_skill(tasks, model=echo(_model), score="last_number",
                       instruction="answer plainly",
                       propose=lambda r, t, o, s: "answer in cents")
    assert res.final_reward == 1.0


def test_a_callable_scorer_is_accepted():
    res = _evolve(score=lambda task, out: 1.0 if "cents" in out else 0.0)
    assert isinstance(res, EvolutionResult)


def test_every_default_is_overridable():
    res = _evolve(rounds=2, n_workers=1, max_concurrency=1, held_out_frac=0.5)
    assert len(res.history) <= 2


def test_the_caller_can_replace_the_actors():
    """run=/propose=/strategy= must win over the ones it builds."""
    seen = []
    res = _evolve(run=lambda rendered, task: seen.append(rendered) or "0",
                  strategy=SingleSlot(initial_value="mine"))
    assert seen and seen[0] == "mine"
    assert isinstance(res, EvolutionResult)


def test_the_async_path_works_and_is_not_handed_a_rejected_knob():
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = _evolve(asynchronous=True, max_seconds=5.0)
    assert isinstance(res, EvolutionResult)
    assert not [x for x in w if "ignores max_concurrency" in str(x.message)]


def test_template_decides_where_the_skill_lands():
    """The skill can go after the question, not only before it."""
    captured = []

    def spy(prompt):
        captured.append(prompt)
        return "0"

    evolve_skill(ROWS, model=echo(spy), prompt="q", gold="a", score="last_number",
                 instruction="BE BRIEF", rounds=1,
                 template="Question: {prompt}\n\nGuidance: {skill}",
                 propose=lambda r, t, o, s: None)
    assert captured and captured[0].startswith("Question: ")
    assert captured[0].rstrip().endswith("BE BRIEF")


def test_a_template_missing_its_placeholders_is_rejected():
    with pytest.raises(ValueError, match=r"\{skill\}"):
        _evolve(template="no placeholders here")


def test_unknown_scorer_names_the_alternatives():
    with pytest.raises(ValueError, match="last_number"):
        _evolve(score="fuzzy-match")


def test_empty_data_is_rejected_clearly():
    with pytest.raises(ValueError, match="no data"):
        evolve_skill([], model=echo(_model))


# --- the scorers, and the trap that made a working model look broken -----------

def test_last_number_reads_the_gold_the_same_way_as_the_output():
    """A dataset's answer column is often the whole worked solution.

    GSM8K's ends "#### 72". Parsing that as a bare number fails, and it fails
    *silently*: every item scores 0 and it reads as a hopeless model rather than a
    scorer mismatch. Measured on real GSM8K, this was the difference between a
    reported 0/7 and the true 7/7.
    """
    from agentdescent.rewards import last_number
    gold = "Natalia sold 48+24 = <<48+24=72>>72 clips altogether.\n#### 72"
    task = Task(id="0", prompt="q", meta={"gold": gold})
    assert last_number()(task, "The answer is 72.") == 1.0
    assert last_number()(task, "The answer is 70.") == 0.0


def test_a_gold_with_no_number_is_a_loud_error_not_a_zero():
    """Scoring every item 0 is indistinguishable from a model that cannot answer."""
    from agentdescent.rewards import last_number
    with pytest.raises(ValueError, match="contains no number"):
        last_number()(Task(id="7", prompt="q", meta={"gold": "Paris"}), "Paris")


def test_exact_match_ignores_trailing_punctuation():
    """'Henry J. Kaiser.' vs 'Henry J. Kaiser' is not a reasoning failure."""
    from agentdescent.rewards import exact_match
    task = Task(id="0", prompt="q", meta={"gold": "Henry J. Kaiser"})
    assert exact_match()(task, "Henry J. Kaiser.") == 1.0


def test_a_scorer_without_gold_names_the_fix():
    from agentdescent.rewards import exact_match
    with pytest.raises(KeyError, match="gold_key"):
        exact_match()(Task(id="3", prompt="q"), "anything")
