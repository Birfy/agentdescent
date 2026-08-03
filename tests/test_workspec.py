"""Work has to be describable as data before it can be run anywhere else.

The wall is not the executor. Every factory in this package returns a closure and
so does every example in the docs, so a process pool fails on the first submit
regardless of how good the pool is. These tests pin the alternative: name the
callable, resolve it on the far side, and refuse anything that would make that
resolution a way in.
"""

import json
import pickle
import sys
import types

import pytest

from agentdescent import Task
from agentdescent.policies import SandboxSpec
from agentdescent.workspec import Ref, RefError, RolloutSpec, resolve_ref


TASK = Task(id="t1", prompt="what is 6*7?", meta={"gold": "42"})


# ---------------------------------------------------------------------------
# The premise
# ---------------------------------------------------------------------------


def test_the_things_a_caller_actually_passes_cannot_be_pickled():
    """Guard the premise. If this ever fails, `Ref` has lost its reason to exist.

    Data crosses fine; every callable does not, because they are all closures --
    which is why "just use ProcessPoolExecutor" is not a smaller version of this
    change but a different one that does not work."""
    from agentdescent import rewards
    from agentdescent.agents import echo
    from agentdescent.evolution import reflector

    pickle.dumps(TASK)                       # data: fine

    for name, obj in (("reward factory", rewards.last_number()),
                      ("reflector", reflector(echo())),
                      ("lambda", lambda rendered, task: "x")):
        # The *fact* is asserted, not the wording. CPython raises
        # `PicklingError` or `AttributeError` depending on the case, and 3.12
        # rephrased "Can't pickle local object" to "Can't get local object" --
        # an assertion on the sentence goes red on an upgrade and says nothing
        # about whether the premise still holds.
        with pytest.raises(Exception) as excinfo:
            pickle.dumps(obj)
        assert isinstance(excinfo.value, (pickle.PicklingError, AttributeError,
                                          TypeError)), (name, excinfo.value)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def test_a_resolved_factory_behaves_like_the_direct_call():
    from agentdescent import rewards

    direct = rewards.last_number()
    viaref = Ref("agentdescent.rewards:last_number").resolve()
    for output in ("the answer is 42", "no digits here", "7"):
        assert viaref(TASK, output) == direct(TASK, output)


def test_config_is_the_factories_keyword_arguments():
    from agentdescent import rewards

    task = Task(id="t", prompt="p", meta={"expected": "42"})
    direct = rewards.last_number(gold_key="expected")
    viaref = Ref("agentdescent.rewards:last_number",
                 {"gold_key": "expected"}).resolve()
    assert viaref(task, "42") == direct(task, "42") == 1.0


def test_call_false_takes_the_attribute_itself():
    ref = Ref("agentdescent.filetree:canonical", call=False)
    assert ref.resolve()({"a.txt": "x"}).strip()


def test_a_spec_survives_pickle_and_resolves_the_same_on_the_far_side():
    spec = RolloutSpec(rendered="rules", task=TASK,
                       run=Ref("agentdescent.agents:echo"),
                       reward=Ref("agentdescent.rewards:last_number"))
    back = pickle.loads(pickle.dumps(spec))
    assert back.task.id == TASK.id and back.rendered == "rules"
    _, reward = back.resolve()
    assert reward(TASK, "42") == 1.0


def test_a_lease_id_is_stable_across_the_wire():
    """Re-dispatch reuses it, which is how the merger tells a retry from a second
    opinion -- a lease id that changed in transit would put two cards for one
    task into the pool."""
    spec = RolloutSpec(rendered="r", task=TASK,
                       run=Ref("agentdescent.agents:echo"),
                       reward=Ref("agentdescent.rewards:last_number"))
    assert pickle.loads(pickle.dumps(spec)).lease_id == spec.lease_id
    assert spec.with_lease("fixed").lease_id == "fixed"


# ---------------------------------------------------------------------------
# The trust boundary
# ---------------------------------------------------------------------------


def test_a_target_outside_the_allowlist_is_refused():
    with pytest.raises(RefError, match="outside the allowed prefixes"):
        Ref("os:system").resolve()


def test_refusal_happens_before_the_import(tmp_path, monkeypatch):
    """Importing to find out it was not allowed has already run its top level.

    This is the difference between a check and a formality: a hostile target's
    side effects happen at import, not at call."""
    marker = tmp_path / "ran.txt"
    module_src = f"open({str(marker)!r}, 'w').write('imported')\nhandler = lambda: None\n"
    (tmp_path / "hostile_probe.py").write_text(module_src, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("hostile_probe", None)

    with pytest.raises(RefError):
        Ref("hostile_probe:handler").resolve()
    assert not marker.exists(), "the module was imported despite being refused"


def test_the_allowlist_can_be_widened_deliberately():
    mod = types.ModuleType("widened_probe")
    mod.make = lambda: (lambda task, output: 0.5)
    sys.modules["widened_probe"] = mod
    try:
        fn = resolve_ref(Ref("widened_probe:make"), allowed=("widened_probe",))
        assert fn(TASK, "x") == 0.5
    finally:
        del sys.modules["widened_probe"]


def test_config_rejects_anything_that_is_not_json():
    """An arbitrary object in `config` is a closure wearing a different hat."""
    with pytest.raises(RefError, match="only JSON"):
        Ref("agentdescent.rewards:last_number", {"fn": lambda: None})
    with pytest.raises(RefError, match="closure in disguise"):
        Ref("agentdescent.rewards:last_number", {"items": [object()]})
    Ref("agentdescent.rewards:last_number",
        {"a": 1, "b": [1, 2], "c": {"d": None}})        # allowed


def test_a_malformed_target_says_which_part_is_wrong():
    with pytest.raises(RefError, match="module:attribute"):
        Ref("agentdescent.rewards.last_number")


def test_failures_name_the_field_rather_than_raising_a_bare_import_error():
    with pytest.raises(RefError, match="has no attribute"):
        Ref("agentdescent.rewards:no_such_scorer").resolve()
    with pytest.raises(RefError, match="cannot import"):
        Ref("agentdescent.nonexistent:thing").resolve()
    with pytest.raises(RefError, match="call=False"):
        Ref("agentdescent.rewards:last_number", {"nope": 1}).resolve()


def test_a_spec_carries_no_secret_values():
    """A spec is pickled, logged, cached and journalled. One that held an API key
    would leak it into all four."""
    spec = RolloutSpec(
        rendered="rules", task=TASK,
        run=Ref("agentdescent.agents:echo"),
        reward=Ref("agentdescent.rewards:last_number"),
        sandbox=SandboxSpec(env_allowlist=("ANTHROPIC_API_KEY", "PATH")))
    blob = pickle.dumps(spec)
    assert b"ANTHROPIC_API_KEY" in blob            # the name is the whole point
    assert b"sk-" not in blob                      # a value would not be
    # and the sandbox half is JSON-clean, because it has to cross as JSON too
    from dataclasses import asdict
    json.dumps(asdict(spec.sandbox))


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def test_references_nest_so_composed_actors_can_cross():
    """`reflector(claude())` is the normal shape of an actor here.

    Without nesting, every composition would need a bespoke factory whose only
    job is to flatten its arguments into JSON -- one per combination, written by
    whoever hit the boundary first."""
    ref = Ref("agentdescent.evolution:reflector",
              {"complete": Ref("agentdescent.agents:echo")})
    back = pickle.loads(pickle.dumps(ref))
    assert callable(back.resolve())


def test_a_nested_reference_is_checked_like_any_other():
    """The allowlist is not a property of the outermost call."""
    ref = Ref("agentdescent.evolution:reflector", {"complete": Ref("os:getcwd")})
    with pytest.raises(RefError, match="outside the allowed prefixes"):
        ref.resolve()


def test_the_built_in_runners_are_already_addressable():
    """`code_runner` takes argv and strings, so it needs no adapter to cross --
    which is what makes the common directory-evolution case free."""
    import sys
    ref = Ref("agentdescent.runners:code_runner",
              {"entrypoint": [sys.executable, "main.py"]})
    pickle.loads(pickle.dumps(ref))
    assert callable(ref.resolve())
