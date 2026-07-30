"""The headline promise: bring any agent, evolve it in parallel or async.

The framework's own `LLMAgent` bundles solving and proposing, which only fits when
it also drives the rollout. The usual case is the reverse -- you already have an
agent and want it improved -- so check that path stays a few lines.
"""

from agentdescent.agents import echo
from agentdescent.async_evolve import async_evolve
from agentdescent.evolution import SingleSlot, Task, evolve, reflector


def my_agent(system_prompt: str, question: str) -> str:
    """Somebody's existing agent -- arbitrary signature, not a Completion."""
    return "PARIS" if "uppercase" in system_prompt.lower() else "Paris"


TASKS = [Task(id=str(i), prompt="capital of France?") for i in range(10)]
REWARD = lambda t, o: 1.0 if o == "PARIS" else 0.0
RUN = lambda rendered, task: my_agent(rendered, task.prompt)
REFLECT = lambda: reflector(echo(lambda p: "Answer in uppercase."))


def test_bring_your_own_agent_synchronously():
    res = evolve(TASKS, REWARD, run=RUN, propose=REFLECT(),
                 strategy=SingleSlot(initial_value="Answer concisely."),
                 rounds=4, n_workers=3, max_concurrency=3)
    assert res.final_reward == 1.0
    assert "uppercase" in res.rendered.lower()
    assert res.error is None


def test_the_same_agent_evolves_asynchronously():
    """Same four arguments, barrier-free -- no restructuring required."""
    res = async_evolve(TASKS, REWARD, run=RUN, propose=REFLECT(),
                       strategy=SingleSlot(initial_value="Answer concisely."),
                       n_workers=3, max_seconds=4.0, held_out_frac=0.4)
    assert res.final_reward == 1.0
    assert res.error is None


def test_evolve_switches_to_async_with_one_flag():
    res = evolve(TASKS, REWARD, run=RUN, propose=REFLECT(),
                 strategy=SingleSlot(), rounds=4, n_workers=3,
                 asynchronous=True, async_ratio=3, max_seconds=4.0)
    assert res.final_reward == 1.0


def test_reflector_is_independent_of_the_agent_being_evolved():
    """A cheap model may reflect on an expensive agent."""
    cheap = reflector(echo(lambda p: "Answer in uppercase."))
    res = evolve(TASKS, REWARD, run=RUN, propose=cheap, strategy=SingleSlot(),
                 rounds=4)
    assert res.final_reward == 1.0


def test_single_slot_is_exported_for_the_common_case():
    import agentdescent

    assert hasattr(agentdescent, "SingleSlot")
    assert hasattr(agentdescent, "reflector")
