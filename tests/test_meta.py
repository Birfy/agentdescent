"""agentdescent.meta: the decision slots of evolve() as the artifact. Offline."""

import json

import pytest

from agentdescent import Policies, Task, evolve
from agentdescent.meta import (SLOTS, MetaOutcome, ParamSlot, PrioritySelection,
                               PRIORITY_SEED, SourceSlot, auc, compile_priority,
                               evolve_problem, final_reward, meta_evolve,
                               meta_validate, priority_selection, rollouts_to,
                               slot_reflector, transfer_ratio)
from agentdescent.selection import Beam, FlatPuct, SingleHead


# -- outcomes and meta-rewards ------------------------------------------------


def test_meta_rewards_read_the_curve():
    outcome = MetaOutcome(curve=[0.2, 0.1, 0.5, 0.4], final=0.4, rollouts=4)
    assert outcome.best_so_far() == [0.2, 0.2, 0.5, 0.5]
    assert auc(outcome) == pytest.approx((0.2 + 0.2 + 0.5 + 0.5) / 4)
    assert final_reward(outcome) == 0.4
    assert rollouts_to(0.5)(outcome) == pytest.approx(1 / 3)
    assert rollouts_to(0.9)(outcome) == 0.0
    assert auc(MetaOutcome()) == 0.0
    assert MetaOutcome.from_json(outcome.to_json()).curve == outcome.curve


def test_outcome_from_an_inner_result():
    tasks = [Task(id=f"t{i}", prompt=f"item {i}") for i in range(8)]
    result = evolve(tasks, lambda t, o: 1.0 if "yes" in o else 0.0,
                    run=lambda r, t: "yes" if "yes" in r else "no",
                    propose=lambda r, t, o, w: "yes", rounds=2, n_workers=2)
    outcome = MetaOutcome.from_result(result)
    assert len(outcome.curve) == len(result.history)
    assert outcome.final == result.final_reward
    assert "outcomes" in outcome.detail


# -- specs --------------------------------------------------------------------


def test_param_slot_merges_by_parameter_and_refuses_out_of_bounds():
    spec = ParamSlot(FlatPuct, {"c_puct": 1.0, "prior_exponent": 0.0},
                     bounds={"c_puct": (0.0, 10.0), "prior_exponent": (0.0, 4.0)})
    state = spec.initial()
    diff = spec.to_diff(state, "c_puct: 2.5", "w1", 0, "a")
    assert diff.ops == {"c_puct": "2.5"}
    assert spec.to_diff(state, "c_puct: 99", "w2", 0, "a") is None
    assert spec.to_diff(state, "beam_width: 3", "w3", 0, "a") is None
    assert spec.to_diff(state, "c_puct: 1.0", "w4", 0, "a") is None     # unchanged
    assert spec.invalid_proposals == 3
    policy = spec.compile(spec.render({**state, "c_puct": "2.5"}))
    assert isinstance(policy, FlatPuct) and policy.c_puct == 2.5
    assert "c_puct" in spec.describe()


def test_source_slot_strips_fences_and_gates():
    def validate(text):
        if "bad" in text:
            raise ValueError("bad")
        return text.strip()

    spec = SourceSlot(initial_value="k = 1", validate=validate, build=lambda t: t.upper())
    assert spec.to_diff(spec.initial(), "```python\nk = 2\n```", "w", 0, "a").ops == {"value": "k = 2"}
    assert spec.to_diff(spec.initial(), "bad", "w", 0, "a") is None
    assert spec.invalid_proposals == 1
    assert spec.compile("k = 3") == "K = 3"


def test_the_priority_seed_is_flat_puct():
    from agentdescent.selection import Candidate, SelectionContext

    rows = tuple(Candidate("a", v, score=s, selected=n, parent=p)
                 for v, s, n, p in [(0, 0.1, 5, None), (1, 0.7, 2, 0), (2, 0.3, 1, 0),
                                    (3, 0.9, 1, 1)])
    ctx = SelectionContext(head=rows[0], candidates=rows, n_workers=3)
    ours = [c.version for c in PrioritySelection(PRIORITY_SEED).select(ctx, 3)]
    theirs = [c.version for c in FlatPuct(1.0).select(ctx, 3)]
    assert ours == theirs
    assert isinstance(priority_selection().compile(PRIORITY_SEED), PrioritySelection)


def test_the_priority_gate_refuses_a_rule_that_dies_at_the_root():
    with pytest.raises(ValueError, match="ZeroDivisionError"):
        compile_priority("def priority(rank, visits, total, prior, depth, n_nodes):\n"
                         "    return rank / visits\n")


# -- evolve_problem and meta_evolve -------------------------------------------


def _inner_tasks():
    return [Task(id=f"t{i}", prompt=f"item {i}") for i in range(10)]


def test_evolve_problem_installs_the_slot_and_refuses_machinery():
    problem = evolve_problem(_inner_tasks(), lambda t, o: 1.0 if "yes" in o else 0.0,
                             slot="selection",
                             run=lambda r, t: "yes" if "yes" in r else "no",
                             propose=lambda r, t, o, w: "yes",
                             rounds=2, n_workers=2)
    outcome = problem(SingleHead(), seed=0)
    assert outcome.curve and outcome.final == 1.0
    with pytest.raises(ValueError, match="not an evolvable slot"):
        evolve_problem(_inner_tasks(), lambda t, o: 0.0, slot="ledger")
    assert "ledger" not in SLOTS and "selection" in SLOTS


def _scripted_problem(value, seed):
    """A problem whose curve rises faster the larger `value.k` is."""
    k = getattr(value, "k", 1)
    curve = [min(1.0, 0.1 * (i + 1) * k) for i in range(6)]
    return MetaOutcome(curve=curve, final=curve[-1], rollouts=6)


def test_meta_evolve_evolves_a_param_slot_with_a_scripted_reflector():
    spec = ParamSlot(lambda k: Beam(int(k)), {"k": 1.0}, bounds={"k": (1, 8)})
    prompts = []

    def scripted(prompt):
        prompts.append(prompt)
        return "k: 3"

    result = meta_evolve({"toy": _scripted_problem}, slot="selection", spec=spec,
                         model=scripted, seeds=range(6), rounds=2, n_workers=2,
                         max_concurrency=2)
    assert result.error is None
    assert prompts and "numeric parameters" in prompts[0] and '"curve"' in prompts[0]
    evolved = spec.compile(result.rendered)
    assert isinstance(evolved, Beam) and evolved.k == 3
    assert result.final_reward > auc(_scripted_problem(Beam(1), 0))


def test_meta_evolve_refuses_a_machinery_slot_and_reserved_kwargs():
    spec = priority_selection()
    with pytest.raises(ValueError, match="not an evolvable slot"):
        meta_evolve([_scripted_problem], slot="verifier", spec=spec, model=lambda p: "")
    with pytest.raises(TypeError, match="sets strategy"):
        meta_evolve([_scripted_problem], slot="selection", spec=spec,
                    model=lambda p: "", strategy=spec)
    with pytest.raises(ValueError, match="propose= or model="):
        meta_evolve([_scripted_problem], slot="selection", spec=spec)


def test_meta_validate_pairs_by_seed_and_reads_the_transfer_ratio():
    spec = ParamSlot(lambda k: Beam(int(k)), {"k": 1.0})
    before, after = spec.render({"k": "1.0"}), spec.render({"k": "2.0"})
    report = meta_validate(spec, before, after,
                           {"src": _scripted_problem,
                            "tgt": lambda v, s: MetaOutcome(curve=[0.5] * 4, final=0.5)},
                           seeds=range(4))
    assert report["src"]["gain"] > 0 and report["src"]["wins"] == 4
    assert report["tgt"]["gain"] == 0.0
    assert transfer_ratio(report, "src", "tgt") == 0.0
    assert transfer_ratio({"a": {"gain": 0.0}, "b": {"gain": 0.1}}, "a", "b") is None


def test_slot_reflector_shows_the_spec_the_value_and_the_outcome():
    seen = {}
    spec = priority_selection()
    propose = slot_reflector(lambda p: seen.setdefault("p", p) and "", spec)
    task = Task(id="p0:1", prompt="p0 (seed 1)", meta={"problem": "p0", "seed": 1})
    propose(PRIORITY_SEED, task, MetaOutcome(curve=[0.1]).to_json(), 0.1)
    assert "SELECTION RULE" in seen["p"] and "def priority" in seen["p"]
    assert json.dumps([0.1]) in seen["p"]
