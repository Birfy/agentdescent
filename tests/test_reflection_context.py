"""What the reflector is allowed to see.

Without ``task.meta`` a reflector is told only *that* it was wrong, never what
right looks like -- so any output convention it cannot guess is unlearnable.
Meta is where every shipped port already puts the expected answer, so it is
shown by default; these tests pin the bounds on that.
"""
import pytest

from agentdescent.agents import echo
from agentdescent.evolution import LLMAgent, Task, reflector


def _capture(**kw):
    """Return (agent, seen) where seen['p'] is the last prompt built."""
    seen = {}
    def complete(prompt):
        seen["p"] = prompt
        return "a rule"
    return LLMAgent(echo(complete), **kw), seen


def test_meta_reaches_the_reflection_prompt():
    agent, seen = _capture()
    agent.propose("art", Task(id="t", prompt="q", meta={"gold": "2800"}), "out", 0.0)
    assert "2800" in seen["p"]


def test_meta_can_be_withheld():
    agent, seen = _capture(show_meta=False)
    agent.propose("art", Task(id="t", prompt="q", meta={"secret": "2800"}), "out", 0.0)
    assert "2800" not in seen["p"]
    assert "metadata" not in seen["p"]


def test_no_meta_leaves_no_empty_section():
    """A task without meta must not grow a dangling 'expected:' header."""
    agent, seen = _capture()
    agent.propose("art", Task(id="t", prompt="q"), "out", 0.0)
    assert "expected" not in seen["p"].lower()


def test_meta_is_truncated():
    """Meta can hold a whole document; it must not blow up the prompt."""
    agent, seen = _capture(meta_chars=100)
    agent.propose("art", Task(id="t", prompt="q", meta={"doc": "x" * 50_000}), "out", 0.0)
    assert len(seen["p"]) < 2_000
    assert "..." in seen["p"]


def test_prompt_asks_for_a_general_rule():
    """Showing the answer invites 'the answer is 2800'; the template forbids it."""
    agent, seen = _capture()
    agent.propose("art", Task(id="t", prompt="q", meta={"gold": "2800"}), "out", 0.0)
    assert "do NOT mention" in seen["p"]


def test_reflector_helper_plumbs_the_flag():
    seen = {}
    fn = reflector(echo(lambda p: (seen.__setitem__("p", p), "r")[1]), show_meta=False)
    fn("art", Task(id="t", prompt="q", meta={"gold": "2800"}), "out", 0.0)
    assert "2800" not in seen["p"]


def test_custom_template_without_the_field_still_works():
    """Templates predate the field -- an old one must not raise KeyError."""
    agent, seen = _capture(propose_template="fix {artifact} for {prompt}")
    got = agent.propose("art", Task(id="t", prompt="q", meta={"gold": "x"}), "out", 0.0)
    assert got == "a rule"
    assert seen["p"] == "fix art for q"


def test_meta_makes_an_unguessable_convention_learnable():
    """End-to-end: a convention that is *not* inferable from the score alone.

    The agent must answer in cents. Nothing in the prompt or the score says so --
    only ``meta['gold']`` does. A reflector that cannot see meta cannot fix it, so
    this run is the plumbing test the string assertions cannot be.
    """
    from agentdescent.evolution import SingleSlot, evolve

    tasks = [Task(id=str(i), prompt=f"{i} dollars in cents?", meta={"gold": str(i * 100)})
             for i in range(1, 9)]

    def run(rendered, task):
        n = int(task.prompt.split()[0])
        return str(n * 100) if "cents" in rendered else str(n)

    def reward(task, output):
        return 1.0 if output == task.meta["gold"] else 0.0

    def propose(rendered, task, output, reward_value):
        # Reads only what the reflector is given -- no closure over the answer.
        return "answer in cents" if task.meta.get("gold") != output else None

    res = evolve(tasks, reward, run=run, propose=propose,
                 strategy=SingleSlot(initial_value="answer plainly"),
                 rounds=4, n_workers=2, held_out_frac=0.5)
    assert res.final_reward == 1.0, res.rendered
