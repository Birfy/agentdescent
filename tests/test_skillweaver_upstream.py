"""SkillWeaver against `OSU-NLP-Group/SkillWeaver@f2a63d65`."""
from __future__ import annotations

from examples._measure import canonical_json
from examples.skillweaver import skillweaver_web_apis as sw


def _task(index: int = 0):
    return sw._tasks()[index]


def test_the_site_reports_the_first_unmet_call_and_no_required_trace():
    """Honing sees execution feedback. A message containing the required calls
    turns repair into transcription."""
    task = _task()
    ok, message = sw.simulate(["open:/settings/profile"], task)
    assert not ok
    for step in sw._required(task):
        assert step not in message, f"the feedback hands over {step!r}"
    assert "progress" in message and "Available calls" in message


def test_the_hydration_step_is_discoverable_only_from_feedback():
    """The seed API omits `wait:hydration-complete`, so the port has to learn it
    from the site -- which is the mechanism under test, not a quirk."""
    seed = sw.build(0).strategy.initial()["api generic"]
    assert "hydration" not in seed
    ok, message = sw.simulate(
        ["open:/settings/profile", "fill:timezone=utc", "click:save"], _task())
    assert not ok
    assert "hydrates" in message


def test_a_complete_sequence_persists():
    task = _task()
    ok, message = sw.simulate(sw._required(task), task)
    assert ok and "saved-toast" in message


def test_a_page_key_holds_the_latest_api_rather_than_an_accumulation():
    """Upstream's knowledge base carries one current definition per function
    name and bumps `metadata["global_version"]`; it does not keep both."""
    policy = sw.build(0)
    state = policy.strategy.initial()
    first = canonical_json({"calls": ["open:{page}", "click:save"]})
    second = canonical_json({"calls": ["open:{page}", "wait:hydration-complete",
                                       "click:save"]})
    for value in (first, second):
        diff = policy.strategy.to_diff(state, f"api generic: {value}", "w", 1,
                                       "candidate-skillweaver")
        state = {**state, **diff.ops}
    assert state["api generic"] == second
    assert first not in state.values()


def test_the_reward_is_the_environment_not_a_model_judge():
    """`check_success_simple` asks a separate LM to judge a trajectory and a
    screenshot. This port's reward is the site itself, so it is a *cleaner*
    signal than the paper's rather than merely a cheaper one -- and that is a
    departure to name, not to fold into "a deterministic service replaces
    WebArena"."""
    policy = sw.build(0)
    task = _task()
    good = canonical_json({"calls": sw._required(task)})
    assert policy.reward(task, good) == 1.0
    assert policy.reward(task, canonical_json({"calls": ["open:/nope"]})) == 0.0
    assert policy.reward(task, "not json at all") == 0.0

    notes = " ".join(policy.notes).lower()
    assert "separate lm" in notes or "model" in notes
    assert "self_verify" in notes or policy.self_verify


def test_the_splits_are_wide_enough_for_the_workers_the_matrix_runs():
    """`run_port` refuses a run whose train split is under the worker count, so
    twelve tasks split 4/4/4 capped this port at four workers -- it failed
    outright at eight, which is what the matrix runs."""
    train, held_out, test = sw._split(0)
    assert len(train) == len(held_out) == len(test) == 16
    ids = [{t.id for t in g} for g in (train, held_out, test)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])


def test_held_out_pages_are_disjoint_so_only_a_placeholder_api_can_pass():
    """The generic key is where the reusable API lives precisely because a
    per-page entry learned on train cannot be retrieved for a held-out page."""
    train, held_out, _ = sw._split(0)
    train_pages = {t.meta["page"] for t in train}
    held_pages = {t.meta["page"] for t in held_out}
    assert held_pages - train_pages, "every held-out page was already seen"


# -- the site has to be learnable from what it says ---------------------------


def _through_the_agent(calls, task):
    """The real path: `_call_list` lowercases and strips, as a run does."""
    import json
    return sw.simulate(sw._call_list(json.dumps({"calls": calls}), "calls"), task)


def test_the_exact_tokens_the_site_only_gestures_at_are_not_required():
    """The message says *"the page hydrates before accepting input and confirms
    with a toast"* -- it names the concepts. Under string equality it still
    demanded `wait:hydration-complete` and `assert:saved-toast` exactly, so
    `wait:hydration`, `wait:page-hydrated`, `assert:toast` and
    `assert:success-toast` all did the right thing and were refused, as was
    `fill:timezone = UTC` for its spaces.
    """
    task = _task()
    required = sw._required(task)
    for wait in ("wait:hydration-complete", "wait:hydration", "wait:page-hydrated"):
        ok, _ = _through_the_agent([required[0], wait] + required[2:], task)
        assert ok, f"{wait!r} waits for the page and was refused"
    for check in ("assert:saved-toast", "assert:toast", "assert:success-toast"):
        ok, _ = _through_the_agent(required[:4] + [check], task)
        assert ok, f"{check!r} confirms the save and was refused"
    ok, _ = _through_the_agent(
        [required[0], required[1], "fill:timezone = UTC"] + required[3:], task)
    assert ok, "spaces around the equals sign are not a different action"


def test_the_task_derived_arguments_are_still_required():
    """Loosening the spelling of a token the site only hints at must not loosen
    the page, the field or the value, which the task states outright."""
    task = _task()
    required = sw._required(task)
    for wrong in ("fill:timezone=EST", "fill:language=UTC"):
        ok, _ = _through_the_agent(required[:2] + [wrong] + required[3:], task)
        assert not ok, f"{wrong!r} does not do what the task asked"
    ok, _ = _through_the_agent(["open:/settings/display"] + required[1:], task)
    assert not ok, "the wrong page was accepted"


def test_missing_misplaced_and_wrong_argument_are_three_messages():
    """Three failures, three repairs. Telling an agent that wrote a `fill` that
    no `fill` succeeded makes it write another one; telling it a *wrong value*
    is out of order makes it reorder. Both are the loop that kept Voyager at
    0.000 for three seeds."""
    task = _task()
    required = sw._required(task)

    _, absent = _through_the_agent(
        [required[0]] + required[2:], task)                 # no wait at all
    assert "never succeeded" in absent

    _, late = _through_the_agent(
        [required[0], required[2], required[1]] + required[3:], task)
    assert "after steps it has to follow" in late

    _, wrong = _through_the_agent(
        required[:2] + ["fill:timezone=EST"] + required[3:], task)
    assert "argument is not what the page acted on" in wrong


def test_the_messages_still_hand_over_nothing():
    task = _task()
    # From 1: `_call_list` refuses an empty list, so the zero-length prefix
    # cannot reach the site through the path a run uses.
    for prefix in range(1, len(sw._required(task))):
        _, message = _through_the_agent(sw._required(task)[:prefix], task)
        body = message.split("Available calls:")[0]
        for step in sw._required(task):
            assert step not in body, f"the feedback hands over {step!r}"
