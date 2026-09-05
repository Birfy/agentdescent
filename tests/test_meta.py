"""agentdescent.meta: the decision slots of evolve() as the artifact. Offline."""

import json

import pytest

from agentdescent import Policies, Task, evolve
from agentdescent.meta import (SLOTS, SLOT_PROTOCOLS, MetaOutcome, ParamSlot, PrioritySelection,
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


# -- the general spec: any slot as class source --------------------------------


@pytest.mark.parametrize("slot", ["selection", "task_sampler", "staleness"])
def test_policy_source_seeds_satisfy_their_protocol(slot):
    from agentdescent.meta import SLOT_PROTOCOLS, policy_source

    spec = policy_source(slot)
    value = spec.compile(spec.render(spec.initial()))
    assert isinstance(value, SLOT_PROTOCOLS[slot])
    assert "def " in spec.describe() and slot in spec.describe()


@pytest.mark.parametrize("source, reason", [
    ("import os\nclass Policy:\n    def select(self, ctx, n): return [ctx.head]", "not allowed"),
    ("class Policy:\n    def select(self, ctx, n): return []", "1..n candidates"),
    ("class Policy:\n    def pick(self, keys, r): return keys[0]", "does not satisfy"),
    ("class Policy:\n    def select(self, ctx, n): return [ctx.head.__class__]", "dunder"),
    ("class Policy:\n    def select(self, ctx, n): open('x')", "forbidden call"),
    ("class Policy:\n    def select(self, ctx, n):\n        import os\n        return [ctx.head]",
     "not allowed"),
    ("class Policy:\n    def __init__(self, k): pass\n    def select(self, ctx, n): return [ctx.head]",
     "failed to build"),
    ("class Other:\n    def select(self, ctx, n): return [ctx.head]", "exactly one class"),
])
def test_the_general_gate_refuses(source, reason):
    from agentdescent.meta import compile_policy_source

    with pytest.raises(ValueError, match=reason):
        compile_policy_source("selection", source)


def test_a_method_level_import_from_the_allowlist_works():
    from agentdescent.meta import compile_policy_source

    policy = compile_policy_source("selection", (
        "class Policy:\n"
        "    def select(self, ctx, n):\n"
        "        import random\n"
        "        rng = random.Random(ctx.round)\n"
        "        return [rng.choice(list(ctx.candidates)) for _ in range(n)]\n"))
    from agentdescent.selection import Candidate, SelectionContext
    rows = (Candidate("a", 0), Candidate("a", 1, parent=0))
    assert len(policy.select(SelectionContext(head=rows[0], candidates=rows), 2)) == 2


def test_meta_evolve_over_class_source_for_the_selection_slot():
    from agentdescent.meta import policy_source

    spec = policy_source("selection")
    rewrite = """```python
class Policy:
    # greedy: always expand the best-scored candidate
    def select(self, ctx, n):
        rows = [c for c in ctx.candidates if c.score is not None] or [ctx.head]
        best = max(rows, key=lambda c: c.score if c.score is not None else -1.0)
        return [best] * n
```"""

    def problem(value, seed):
        # A curve that rewards a policy which picks the best-scored candidate.
        from agentdescent.selection import Candidate, SelectionContext
        rows = (Candidate("a", 0, score=0.2), Candidate("a", 1, score=0.9, parent=0))
        pick = value.select(SelectionContext(head=rows[0], candidates=rows), 1)[0]
        base = 0.9 if pick.version == 1 else 0.2
        return MetaOutcome(curve=[base] * 4, final=base)

    result = meta_evolve({"toy": problem}, slot="selection", spec=spec,
                         model=lambda prompt: rewrite, seeds=range(6), rounds=2,
                         n_workers=2, max_concurrency=2)
    assert result.error is None
    assert "greedy" in result.rendered
    assert result.final_reward == pytest.approx(0.9)
    assert isinstance(spec.compile(result.rendered), SLOT_PROTOCOLS["selection"])


@pytest.mark.parametrize("slot", SLOTS)
def test_every_slot_ships_a_seed_that_passes_its_own_gate(slot):
    from agentdescent.meta import policy_source, seed_source

    spec = policy_source(slot)
    assert spec.render(spec.initial()).strip() == seed_source(slot).strip()
    assert isinstance(spec.compile(seed_source(slot)), SLOT_PROTOCOLS[slot])


@pytest.mark.parametrize("slot", [s for s in SLOTS if s != "proposal"])
def test_every_seed_runs_inside_a_real_inner_evolve(slot):
    """The seeds are not only shaped right: the engine installs and honours them."""
    from agentdescent.meta import policy_source

    problem = evolve_problem(_inner_tasks(), lambda t, o: 1.0 if "yes" in o else 0.0,
                             slot=slot,
                             run=lambda r, t: "yes" if "yes" in r else "no",
                             propose=lambda r, t, o, w: "yes",
                             rounds=3, n_workers=2)
    spec = policy_source(slot)
    outcome = problem(spec.compile(spec.render(spec.initial())), seed=0)
    assert outcome.curve and outcome.detail["error"] is None
    assert outcome.final == 1.0


def test_the_proposal_seed_drives_an_inner_evolve_on_its_own():
    from agentdescent.meta import policy_source

    # The proposal policy replaces the actor's propose entirely, so the inner
    # run's only proposals are the seed's placeholder rule.
    seen = []
    problem = evolve_problem(_inner_tasks(), lambda t, o: 1.0 if "grader" in o else 0.0,
                             slot="proposal",
                             run=lambda r, t: seen.append(r) or ("grader" if "grader" in r else "no"),
                             propose=lambda r, t, o, w: "never used",
                             rounds=2, n_workers=2)
    spec = policy_source("proposal")
    outcome = problem(spec.compile(spec.render(spec.initial())), seed=0)
    assert outcome.detail["error"] is None and outcome.final == 1.0


@pytest.mark.parametrize("slot, source, reason", [
    ("conflict", "class Policy:\n    def resolve(self, artifact, cards):\n        return [], 0",
     "at least one"),
    ("conflict", "class Policy:\n    def resolve(self, artifact, cards):\n        return list(cards), 0",
     "contradicting"),
    ("fusion", "class Policy:\n    def select(self, artifact, diffs):\n"
               "        return Diff('x', 'smoke', {'zzz': '1'}), artifact, True", "invent keys"),
    ("acceptance", "class Policy:\n    def accept(self, ctx):\n        return True",
     "AcceptDecision"),
    ("promotion", "class Policy:\n    def observe(self, reports):\n        return ['smoke']",
     "Promotions"),
    ("proposal", "class Policy:\n    def propose(self, ctx):\n        return 'one string'",
     "sequence of strings"),
])
def test_the_merge_side_smokes_catch_shape_errors(slot, source, reason):
    from agentdescent.meta import compile_policy_source

    with pytest.raises(ValueError, match=reason):
        compile_policy_source(slot, source)


def test_accepts_answers_the_gate_without_side_effects():
    """A reporter must be able to ask "would this be taken?" without counting it.

    Written after a benchmark re-implemented the check, forgot that `to_diff`
    strips a code fence, and logged every accepted proposal as refused.
    """
    from agentdescent.meta import policy_source

    spec = policy_source("task_sampler")
    fenced = ("```python\nclass Policy:\n"
              "    def pick(self, keys, round_index): return keys[0]\n"
              "    def record(self, task_id, score): pass\n```")
    assert spec.accepts(fenced) == (True, "")
    ok, reason = spec.accepts("import os")
    assert ok is False and reason
    assert spec.accepts("") == (False, "policy source is empty or too long")
    assert spec.invalid_proposals == 0, "accepts() must not count anything"
    # ...and it agrees with the gate it reports on.
    assert spec.to_diff(spec.initial(), fenced, "w", 0, "a") is not None
    assert spec.to_diff(spec.initial(), "import os", "w", 0, "a") is None
    assert spec.invalid_proposals == 1


def test_outer_tasks_interleave_so_a_positional_split_sees_every_problem():
    """`evolve()` cuts train/held-out by position, so grouping by problem would
    train on one problem and gate on another -- measured, and the reason the
    order here is seed-major."""
    from agentdescent.meta import _outer_tasks

    tasks, named = _outer_tasks({"a": _scripted_problem, "b": _scripted_problem},
                                seeds=[0, 1, 2, 3])
    assert set(named) == {"a", "b"} and len(tasks) == 8
    assert [t.meta["problem"] for t in tasks] == ["a", "b"] * 4
    cut = len(tasks) // 2
    train = {t.meta["problem"] for t in tasks[:cut]}
    held_out = {t.meta["problem"] for t in tasks[cut:]}
    assert train == held_out == {"a", "b"}, "a positional split must see both problems"


# -- the task_sampler contract the engine actually imposes --------------------


STALE_KEY_SAMPLER = """class Policy:
    def __init__(self):
        self.counts = {}

    def pick(self, keys, round_index):
        untried = [k for k in keys if k not in self.counts]
        if untried:
            return untried[0]
        return max(self.counts, key=lambda k: self.counts[k])

    def record(self, task_id, score):
        self.counts[task_id] = self.counts.get(task_id, 0) + 1
"""

MUTATING_SAMPLER = """class Policy:
    def pick(self, keys, round_index):
        keys.sort()
        return keys[0]

    def record(self, task_id, score):
        pass
"""


@pytest.mark.parametrize("source, reason", [
    (STALE_KEY_SAMPLER, "not in the keys it was given"),
    (MUTATING_SAMPLER, "must not mutate"),
])
def test_the_sampler_smoke_walks_a_changing_shard(source, reason):
    """The engine hands each worker its own shard, so `keys` changes between
    calls -- and every sampler a live reflector proposed kept per-task state and
    returned a stale id, which the engine turns into a KeyError two rounds in.
    A fixed key list could not see it; the smoke test now walks the branches.
    """
    from agentdescent.meta import compile_policy_source

    with pytest.raises(ValueError, match=reason):
        compile_policy_source("task_sampler", source)


def test_the_shipped_samplers_pass_the_stricter_smoke():
    """A gate that rejects the engine's own policies would be wrong, not strict."""
    from agentdescent.meta import _smoke_task_sampler, compile_policy_source, seed_source
    from agentdescent.sampling import DifficultyWeighted, RoundRobin

    for policy in (RoundRobin(), DifficultyWeighted()):
        _smoke_task_sampler(policy)
    assert compile_policy_source("task_sampler", seed_source("task_sampler"))


def test_accepts_and_compile_answer_the_same_question():
    """`accepts` said yes and `compile` then raised on the same string, because
    only one of them stripped the code fence a proposal arrives in."""
    from agentdescent.meta import SLOT_PROTOCOLS, policy_source

    spec = policy_source("task_sampler")
    fenced = ("```python\nclass Policy:\n"
              "    def pick(self, keys, round_index): return keys[0]\n"
              "    def record(self, task_id, score): pass\n```")
    assert spec.accepts(fenced)[0] is True
    assert isinstance(spec.compile(fenced), SLOT_PROTOCOLS["task_sampler"])
    # ...and the unfenced form, which is what a rendered artifact looks like.
    assert isinstance(spec.compile(spec.render(spec.initial())),
                      SLOT_PROTOCOLS["task_sampler"])
