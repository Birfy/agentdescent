"""The dataset path: rows in, evolved instruction out -- through ``evolve()``.

There used to be a one-call wrapper for this (`evolve_skill`). It was removed so
the package has one entry point; what it assembled is now three public building
blocks -- `tasks_from` for the rows, `scorer` for the reward, `SingleSlot` +
`reflector` for the artifact and its reflection -- and this file is the proof
that they add up to the same run.
"""
import pytest

from agentdescent import SingleSlot, evolve, reflector, scorer, tasks_from
from agentdescent.agents import echo
from agentdescent.evolution import EvolutionResult, Task

ROWS = [{"q": f"value {i}", "a": str(i * 100)} for i in range(1, 13)]


def _model(prompt):
    """Answers correctly only once the skill mentions cents."""
    n = int(prompt.rsplit(" ", 1)[-1])
    return str(n * 100) if "cents" in prompt else str(n)


def _evolve(tasks=None, *, model=None, template="{skill}\n\n{prompt}", **kw):
    """The dataset path, spelled out: what the removed wrapper used to build."""
    model = model or echo(_model)
    tasks = tasks if tasks is not None else tasks_from(ROWS, prompt="q", gold="a")

    def run(rendered, task):
        return model(template.format(skill=rendered, prompt=task.prompt))

    kw.setdefault("run", run)
    kw.setdefault("propose", lambda r, t, o, s: "answer in cents")
    kw.setdefault("strategy", SingleSlot(initial_value="answer plainly"))
    for key, value in (("held_out_frac", 0.3), ("rounds", 8), ("n_workers", 4),
                       ("max_concurrency", 4), ("patience", 3), ("target_reward", 0.98)):
        kw.setdefault(key, value)
    return evolve(tasks, scorer(kw.pop("score", "last_number")), **kw)


def test_a_dataset_of_dicts_is_enough():
    res = _evolve()
    assert isinstance(res, EvolutionResult)
    assert res.final_reward == 1.0
    assert "cents" in res.rendered
    assert res.outcomes() and res.error is None and res.history


def test_ready_made_tasks_are_accepted():
    tasks = [Task(id=str(i), prompt=f"value {i}", meta={"gold": str(i * 100)})
             for i in range(1, 13)]
    assert _evolve(tasks).final_reward == 1.0


def test_a_callable_scorer_is_accepted():
    res = _evolve(score=lambda task, out: 1.0 if "cents" in out else 0.0)
    assert isinstance(res, EvolutionResult)


def test_the_shipped_reflector_is_the_default_proposer():
    """`reflector(model)` is what the wrapper installed; it must still produce a
    proposal the strategy accepts."""
    res = _evolve(propose=reflector(echo(lambda p: "answer in cents")))
    assert "cents" in res.rendered


def test_the_async_path_takes_the_same_pieces():
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = _evolve(asynchronous=True, max_seconds=5.0, max_concurrency=1)
    assert isinstance(res, EvolutionResult)
    assert not [x for x in w if "ignores max_concurrency" in str(x.message)]


def test_template_decides_where_the_skill_lands():
    """The skill can go after the question, not only before it."""
    captured = []

    def spy(prompt):
        captured.append(prompt)
        return "0"

    _evolve(model=echo(spy), rounds=1, strategy=SingleSlot(initial_value="BE BRIEF"),
            template="Question: {prompt}\n\nGuidance: {skill}",
            propose=lambda r, t, o, s: None)
    assert captured and captured[0].startswith("Question: ")
    assert captured[0].rstrip().endswith("BE BRIEF")


def test_unknown_scorer_names_the_alternatives():
    with pytest.raises(ValueError, match="last_number"):
        scorer("fuzzy-match")


def test_the_named_scorers_are_the_reward_module():
    from agentdescent import SCORERS
    from agentdescent.rewards import contains, exact_match, last_number, numeric_close
    assert SCORERS == {"last_number": last_number, "exact": exact_match,
                       "contains": contains, "numeric_close": numeric_close}
    assert callable(scorer("exact"))
    own = lambda t, o: 1.0                                        # noqa: E731
    assert scorer(own) is own


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
