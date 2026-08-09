"""Voyager against `MineDojo/Voyager@55e45a88`.

The library is the mechanism, and the port described it backwards.
"""
from __future__ import annotations

from examples._measure import canonical_json
from examples.voyager import voyager_skill_library as vy


def test_a_second_skill_for_a_goal_replaces_the_first():
    """`add_new_skill` prints "Skill {name} already exists. Rewriting!", deletes
    the vector-store entry and reassigns `self.skills[program_name]`. The older
    code is dumped to disk as `{name}V2.js` and **retrieval never reads it** --
    `retrieve_skills` returns `self.skills[...]["code"]`. Versioned on disk and
    overwritten in memory are different libraries, and only the second one is
    the algorithm.
    """
    policy = vy.build(0)
    state = policy.strategy.initial()

    first = canonical_json({"steps": ["collect:water", "serve:drink"]})
    diff = policy.strategy.to_diff(state, f"skill generic: {first}", "w", 1,
                                   "candidate-voyager")
    state = {**state, **diff.ops}

    second = canonical_json({"steps": ["sanitize:vessel", "collect:water",
                                       "serve:drink"]})
    diff = policy.strategy.to_diff(state, f"skill generic: {second}", "w", 2,
                                   "candidate-voyager")
    assert diff is not None
    state = {**state, **diff.ops}

    assert state["skill generic"] == second
    assert first not in state.values(), "the replaced skill is still retrievable"


def test_a_skill_for_a_different_goal_does_not_disturb_another():
    """Different `program_name`s coexist; only a repeat of the same name
    rewrites."""
    policy = vy.build(0)
    state = policy.strategy.initial()
    for key in ("skill mint", "skill berry"):
        value = canonical_json({"steps": [f"collect:{key.split()[1]}"]})
        diff = policy.strategy.to_diff(state, f"{key}: {value}", "w", 1,
                                       "candidate-voyager")
        state = {**state, **diff.ops}
    assert "skill mint" in state and "skill berry" in state
    assert state["skill generic"], "the seed entry was displaced"


def test_the_environment_reports_the_first_unmet_step_and_no_gold_trace():
    """Upstream's repair prompt sees the critique and the events, never the
    program that would have worked. A message containing the required sequence
    turns repair into transcription."""
    ok, message = vy.simulate(["collect:water"], "mint")
    assert not ok
    required = vy._required("mint")
    for step in required:
        assert step not in message, f"the feedback hands over {step!r}"
    assert "progress" in message and "primitives" in message


def test_a_complete_sequence_reaches_the_goal():
    ok, message = vy.simulate(vy._required("mint"), "mint")
    assert ok and "goal reached" in message


def test_extra_actions_do_not_break_a_correct_ordering():
    """`simulate` advances a cursor, so a skill that does more than the minimum
    still succeeds -- the world judges the goal, not the transcript."""
    noisy = ["heat:water"] + vy._required("mint") + ["serve:drink"]
    ok, _ = vy.simulate(noisy, "mint")
    assert ok


def test_out_of_order_actions_fail():
    reversed_steps = list(reversed(vy._required("mint")))
    ok, _ = vy.simulate(reversed_steps, "mint")
    assert not ok


def test_the_critic_is_the_engines_self_verify_rollout():
    """`check_task_success` judges from the environment rather than the agent's
    claim, and a skill is added only under `info["success"]`."""
    assert vy.build(0).self_verify is True


def test_the_overwrite_semantics_are_declared_rather_than_called_add_only():
    notes = " ".join(vy.build(0).notes).lower()
    assert "not an accumulation" in notes or "latest accepted skill" in notes
    assert "add_new_skill" in notes


# -- the world has to be learnable from what it says --------------------------


def test_the_required_argument_syntax_is_not_a_guessing_game():
    """`combine:water+mint` was matched by string equality, and that `X+Y` syntax
    appears nowhere the agent can see: not in `PRIMITIVES`, not in the seed
    skill, not in any feedback message -- which names only the failing verb.

    One step of six was therefore a lottery, and the reward is all-or-nothing.
    All three seeds scored 0.000 with `accepted=0/80` and **no invalid
    proposals**: the port was proposing well-formed skills against a target it
    could not reach by reasoning.
    """
    required = vy._required("mint")
    assert "water+mint" in required[4], "the fixture changed; update this test"

    seed = vy.build(0).strategy.initial()["skill generic"]
    assert "+" not in seed
    assert "+" not in vy.PRIMITIVES
    _, message = vy.simulate(required[:4], "mint")
    assert "+" not in message, "the feedback now hands over the syntax"

    # Content, not spelling: each of these performs the step the world wants.
    for phrasing in ("combine:water+mint", "combine:mint+water",
                     "combine:water and mint", "combine:water_with_mint"):
        ok, _ = vy.simulate(required[:4] + [phrasing, "serve:drink"], "mint")
        assert ok, f"{phrasing!r} does the required thing and was refused"


def test_arguments_the_world_never_names_are_matched_on_the_verb_alone():
    """`sanitize:vessel` was the second lottery, and the harder one: "vessel"
    appears nowhere the agent can see, so *every* sanitize it wrote was refused.
    Measured -- the solver discovered all three missing steps and still scored
    zero, writing `sanitize:water` and `sanitize:berry`.

    A step whose argument the world does not name is matched on the verb; a step
    whose argument follows from the goal still has to contain it.
    """
    required = vy._required("mint")
    assert required[0] == "sanitize:vessel" and required[5] == "serve:drink"
    for word in ("vessel", "drink"):
        _, message = vy.simulate([], "mint")
        assert word not in message
    assert word not in vy.PRIMITIVES

    for sanitize in ("sanitize:vessel", "sanitize:water", "sanitize:mint",
                     "sanitize:everything"):
        ok, _ = vy.simulate([sanitize] + required[1:], "mint")
        assert ok, f"{sanitize!r} sanitizes something and was refused"
    for serve in ("serve:drink", "serve:tea", "serve:cup"):
        ok, _ = vy.simulate(required[:5] + [serve], "mint")
        assert ok, f"{serve!r} serves something and was refused"


def test_content_matching_does_not_make_the_world_permissive():
    """The sequence is still the skill. Loosening the *spelling* of an argument
    must not loosen what has to happen or in what order."""
    required = vy._required("mint")
    # Missing the ingredient entirely is still wrong.
    ok, _ = vy.simulate(required[:4] + ["combine:water", "serve:drink"], "mint")
    assert not ok
    # Wrong ingredient is still wrong.
    ok, _ = vy.simulate(required[:4] + ["combine:water+berry", "serve:drink"], "mint")
    assert not ok
    # Order still matters.
    assert not vy.simulate(list(reversed(required)), "mint")[0]
    # And the seed skill, which skips three steps, still fails.
    assert not vy.simulate(["collect:water", "collect:mint", "serve:drink"], "mint")[0]


def test_every_required_step_is_reachable_from_the_verbs_the_world_names():
    """Each feedback message names the verb the world is waiting for, so the
    sequence is discoverable one step at a time. This asserts the chain has no
    rung that cannot be climbed."""
    for ingredient in ("mint", "cocoa", "lychee"):
        required = vy._required(ingredient)
        for i in range(len(required)):
            _, message = vy.simulate(required[:i], ingredient)
            verb = required[i].split(":", 1)[0]
            assert f"'{verb}'" in message, (
                f"{ingredient} step {i}: the world does not say it wants {verb}")


def test_the_feedback_distinguishes_a_missing_step_from_a_late_one():
    """"A 'collect' step never happened", said to an agent that wrote two of
    them, is not a terse error -- it is a wrong one, and the agent does the only
    thing it says, which is add a third.

    Measured: the solver had discovered all six actions and was writing
    `[collect, collect, sanitize, heat, combine, serve]` against a world that
    wants sanitize first. The world consumed the sanitize, looked for a collect
    among the actions *after* it, found none, and reported the collect missing.
    Three seeds scored 0.000 with `accepted=0/80` and no invalid proposals.
    """
    late = ["collect:water", "collect:mint", "sanitize:vessel", "heat:water",
            "combine:water+mint", "serve:drink"]
    ok, message = vy.simulate(late, "mint")
    assert not ok
    assert "out of order" in message
    assert "never happened" not in message

    absent = ["collect:water", "collect:mint", "heat:water",
              "combine:water+mint", "serve:drink"]
    ok, message = vy.simulate(absent, "mint")
    assert not ok
    assert "never happened" in message and "out of order" not in message


def test_the_ordering_message_still_hands_over_nothing():
    """It says a step is late. It does not say which step, where it belongs, or
    what comes after -- the rule the port claims is that repair sees the failure
    and never the required program."""
    late = ["collect:water", "collect:mint", "sanitize:vessel", "heat:water",
            "combine:water+mint", "serve:drink"]
    _, message = vy.simulate(late, "mint")
    # `PRIMITIVES` lists every verb by design, so the check is on the *steps*:
    # no `verb:argument` pair, and no hint of which one is in the wrong place.
    body = message.split("Available primitives:")[0]
    for step in vy._required("mint"):
        assert step not in message
    assert "sanitize" not in body, "the message names the step to move"
