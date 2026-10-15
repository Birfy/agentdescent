"""Governance invariants have to stay right in the exact places a self-modifying
system is weakest.

A declared governance rule must be enforced before the expensive measurement
(not after, when the held-out sweep is already paid for); it must be enforced
even when the reflector proposes something subtly outside the rule; it must be
vacuous (zero overhead, zero counters) when nothing is declared; and a rule
that cannot run -- an unknown kind, a broken predicate -- must be a violation,
not a silent skip. Each of those is pinned here.
"""

import pytest

from agentdescent import Contract, Invariant, check_dependency_graph, check_invariants
from agentdescent.evolution import AppendRules, SingleSlot, Task, evolve
from agentdescent.evolvable import Diff
from agentdescent.workspec import Ref


def _artifact(state=None, blast_radius=0.2, contract=None, aid="artifact"):
    from agentdescent.evolution import EvolvingArtifact
    return EvolvingArtifact(aid, state or {}, blast_radius=blast_radius,
                            strategy=SingleSlot(), contract=contract)


def _diff(ops):
    return Diff(diff_id="t", target="artifact", ops=ops)


# -- built-in rules in isolation ----------------------------------------------


def test_mutable_keys_allows_the_declared_set_and_refuses_outsiders():
    base = _artifact(contract=Contract(invariants=(Invariant("mutable_keys", ("a", "b")),)))
    cand = _artifact()
    assert check_invariants(base, _diff({"a": "x"}), cand) == []
    assert check_invariants(base, _diff({"a": "x", "b": "y"}), cand) == []
    out = check_invariants(base, _diff({"c": "x"}), cand)
    assert len(out) == 1 and "outside the declared mutable set" in out[0]


def test_diff_size_bounded_caps_ops_per_commit():
    base = _artifact(contract=Contract(invariants=(Invariant("diff_size_bounded", (2,)),)))
    assert check_invariants(base, _diff({"a": "1", "b": "2"}), _artifact()) == []
    out = check_invariants(base, _diff({"a": "1", "b": "2", "c": "3"}), _artifact())
    assert len(out) == 1 and "caps it at 2" in out[0]


def test_diff_char_bounded_caps_value_size():
    base = _artifact(contract=Contract(invariants=(Invariant("diff_char_bounded", (10,)),)))
    assert check_invariants(base, _diff({"a": "12345"}), _artifact()) == []
    out = check_invariants(base, _diff({"a": "12345678901"}), _artifact())
    assert len(out) == 1 and "caps it at 10" in out[0]


def test_blast_radius_nondecreasing_refuses_a_shrink():
    base = _artifact(blast_radius=0.6, contract=Contract(
        invariants=(Invariant("blast_radius_nondecreasing"),)))
    grown = _artifact(blast_radius=0.7)
    same = _artifact(blast_radius=0.6)
    shrunk = _artifact(blast_radius=0.2)
    assert check_invariants(base, _diff({}), grown) == []
    assert check_invariants(base, _diff({}), same) == []
    out = check_invariants(base, _diff({}), shrunk)
    assert len(out) == 1 and "relax its own governance layer" in out[0]


def _dep_contract(depends_on=(), invariants=()):
    # `Contract.depends_on` is a multi-artifact field that has not landed on
    # main yet; the invariant machinery duck-types on it, so a lightweight
    # stand-in exercises the same paths without inventing the field.
    class _Fake:
        def __init__(self):
            self.input_schema = "any"
            self.output_schema = "any"
            self.side_effects = ()
            self.major = 1
            self.invariants = tuple(invariants)
            self.depends_on = tuple(depends_on)
    return _Fake()


def test_depends_no_self_refuses_a_self_reference():
    bad = _dep_contract(depends_on=("artifact",),
                        invariants=(Invariant("depends_no_self"),))
    base = _artifact(contract=bad)
    out = check_invariants(base, _diff({}), _artifact())
    assert len(out) == 1 and "depends_on itself" in out[0]
    ok = _artifact(contract=Contract(invariants=(Invariant("depends_no_self"),)))
    assert check_invariants(ok, _diff({}), _artifact()) == []


def test_an_unknown_kind_is_a_violation_not_a_silent_skip():
    base = _artifact(contract=Contract(invariants=(Invariant("no_such_rule"),)))
    out = check_invariants(base, _diff({"a": "x"}), _artifact())
    assert len(out) == 1 and "unknown invariant kind" in out[0]


def test_a_contract_with_no_invariants_is_the_untouched_path():
    assert check_invariants(_artifact(), _diff({"a": "x"}), _artifact()) == []


def test_multiple_invariants_collect_all_violations():
    base = _artifact(contract=Contract(invariants=(
        Invariant("mutable_keys", ("a",)),
        Invariant("diff_size_bounded", (1,)),
    )))
    out = check_invariants(base, _diff({"a": "x", "b": "y"}), _artifact())
    assert len(out) == 2


# -- custom predicates ----------------------------------------------------------


def _custom_predicate(base, diff, candidate):
    if "forbidden" in diff.ops:
        return ["custom says no"]
    return []


def test_a_resolved_predicate_is_the_rule(tmp_path):
    import agentdescent.invariants as inv
    inv._custom_for_tests = _custom_predicate   # a module attribute to resolve

    base = _artifact(contract=Contract(invariants=(
        Invariant("custom", ref=Ref("agentdescent.invariants:_custom_for_tests",
                                    call=False)),)))
    assert check_invariants(base, _diff({"a": "x"}), _artifact()) == []
    out = check_invariants(base, _diff({"forbidden": "x"}), _artifact())
    assert out == ["custom says no"]


def test_a_predicate_returning_a_string_counts_as_one_violation():
    import agentdescent.invariants as inv

    def p(base, diff, candidate):
        return "string violation"
    inv._custom_for_tests = p
    base = _artifact(contract=Contract(invariants=(
        Invariant("custom", ref=Ref("agentdescent.invariants:_custom_for_tests",
                                    call=False)),)))
    out = check_invariants(base, _diff({"forbidden": "x"}), _artifact())
    assert out == ["string violation"]


def test_a_predicate_that_raises_is_a_violation_not_a_crash():
    import agentdescent.invariants as inv

    def boom(base, diff, candidate):
        raise ValueError("predicate broke")
    inv._custom_for_tests = boom
    base = _artifact(contract=Contract(invariants=(
        Invariant("custom", ref=Ref("agentdescent.invariants:_custom_for_tests",
                                    call=False)),)))
    out = check_invariants(base, _diff({"a": "x"}), _artifact())
    assert len(out) == 1 and "raised" in out[0]


def test_an_unresolvable_ref_is_a_violation():
    base = _artifact(contract=Contract(invariants=(
        Invariant("custom", ref=Ref("agentdescent.invariants:_no_such_attr",
                                    call=False)),)))
    out = check_invariants(base, _diff({"a": "x"}), _artifact())
    assert len(out) == 1


# -- dependency graph (registration-time) ----------------------------------------


def test_dependency_graph_rejects_self_and_cycles():
    cyc = {
        "a": _dep_contract(depends_on=("b",)),
        "b": _dep_contract(depends_on=("c",)),
        "c": _dep_contract(depends_on=("a",)),
    }
    out = check_dependency_graph(cyc)
    assert any("cycle" in v for v in out)
    selfy = {"a": _dep_contract(depends_on=("a",))}
    out = check_dependency_graph(selfy)
    assert any("depends_on itself" in v for v in out)
    acyclic = {
        "a": _dep_contract(depends_on=("b",)),
        "b": _dep_contract(),
    }
    assert check_dependency_graph(acyclic) == []
    assert check_dependency_graph({"a": Contract()}) == []


def test_contract_round_trips_through_the_ledger_payload():
    # The merge gate enforces invariants, so they must survive the ledger
    # round-trip: an artifact whose invariants evaporate on the way out of
    # storage is a governance rule that silently stopped running.
    from agentdescent.invariants import contract_from_dict, contract_to_dict
    import agentdescent.invariants as inv
    inv._custom_for_tests = _custom_predicate
    contract = Contract(
        input_schema="task", output_schema="text", major=2,
        side_effects=("x",),
        invariants=(
            Invariant("mutable_keys", ("a", "b")),
            Invariant("diff_size_bounded", (3,)),
            Invariant("custom", ref=Ref("agentdescent.invariants:_custom_for_tests",
                                        call=False)),
        ),
    )
    payload = contract_to_dict(contract)
    back = contract_from_dict(payload)
    assert back.input_schema == "task"
    assert back.output_schema == "text"
    assert back.major == 2
    assert back.side_effects == ("x",)
    assert len(back.invariants) == 3
    assert back.invariants[0].kind == "mutable_keys"
    assert back.invariants[1].params == (3,)
    assert back.invariants[2].ref.target == "agentdescent.invariants:_custom_for_tests"
    # the restored contract enforces identically
    assert check_invariants(
        _artifact(contract=back), _diff({"b": "x"}), _artifact()) == []
    out = check_invariants(_artifact(contract=back), _diff({"c": "x"}), _artifact())
    assert len(out) == 1


def test_contract_without_invariants_reads_back_empty():
    from agentdescent.invariants import contract_from_dict, contract_to_dict
    payload = contract_to_dict(Contract())
    back = contract_from_dict(payload)
    assert back.invariants == ()


# -- the merge pipeline ----------------------------------------------------------


def _offline_actors():
    def run(rendered, task):
        return "answer" + (" 2026" if "year" in rendered else "")

    def reward(task, output):
        return 1.0 if "2026" in output else 0.0

    def propose(rendered, task, output, reward):
        return None if reward >= 0.999 else "always state the year"

    return run, reward, propose


def _tasks(n=8):
    return [Task(id=f"t{i}", prompt=f"item {i}") for i in range(n)]


def test_a_violating_diff_is_rejected_before_measurement(tmp_path):
    # SingleSlot writes one key ("value"); a contract that only permits another
    # key makes every proposal a policy violation. The run must complete and
    # report violates-invariant -- the violating diff is settled without a
    # held-out sweep.
    run, reward, propose = _offline_actors()
    contract = Contract(invariants=(Invariant("mutable_keys", ("other",)),))
    result = evolve(_tasks(), reward, run=run, propose=propose,
                    strategy=SingleSlot(initial_value="seed"), rounds=2,
                    n_workers=2, repo_path=str(tmp_path / "repo"),
                    contract=contract)
    assert result.invariant_rejections >= 1
    assert result.invariant_checked >= 1
    assert result.outcomes().get("violates-invariant", 0) >= 1
    assert result.outcomes().get("committed", 0) == 0


def test_a_satisfied_invariant_commits_normally(tmp_path):
    run, reward, propose = _offline_actors()
    contract = Contract(invariants=(Invariant("mutable_keys", ("value",)),))
    result = evolve(_tasks(), reward, run=run, propose=propose,
                    strategy=SingleSlot(initial_value="seed"), rounds=2,
                    n_workers=2, repo_path=str(tmp_path / "repo"),
                    contract=contract)
    assert result.invariant_checked >= 1
    assert result.invariant_rejections == 0
    assert result.outcomes().get("violates-invariant", 0) == 0
    assert result.outcomes().get("committed", 0) >= 1


def test_invariants_survive_commits_and_still_enforce_on_v2_base(tmp_path):
    """The governance rule must not evaporate after the first commit.

    The candidate every merge commits is produced by ``EvolvingArtifact.apply``;
    if it dropped the contract, the serialized head would store the default
    (invariant-free) contract, and every later diff would pass the gate unchecked
    while the run still counted ``invariant_checked``. The rule is a "freeze
    after v1": a diff on a base of version >= 2 is a violation. With the
    contract surviving the apply + store round-trip, exactly one commit lands
    (the first, on v1) and the later diffs are rejected on the v2+ base. The
    clue domain keeps proposing every round (each task needs its own clue), so
    the later diffs exist to be judged."""
    import agentdescent.invariants as inv

    def freeze_after_v1(base, diff, candidate):
        return [f"frozen after v1 (base v{base.version})"] if base.version >= 2 else []

    inv._custom_for_tests = freeze_after_v1

    def run(rendered, task):
        return "correct" if task.meta["h"] in rendered else "wrong"

    def reward(task, output):
        return 1.0 if output == "correct" else 0.0

    def propose(rendered, task, output, reward):
        return task.meta["h"] if task.meta["h"] not in rendered else None

    tasks = [Task(id=f"t{i}", prompt="p", meta={"h": f"H{i % 3}"})
             for i in range(9)]
    contract = Contract(invariants=(
        Invariant("custom", ref=Ref("agentdescent.invariants:_custom_for_tests",
                                    call=False)),))
    result = evolve(tasks, reward, run=run, propose=propose,
                    strategy=AppendRules(), rounds=6, n_workers=2,
                    repo_path=str(tmp_path / "repo"), contract=contract)
    outcomes = result.outcomes()
    assert outcomes.get("committed", 0) == 1, (
        "only the first commit (on v1) may land when the rule freezes v>=2; "
        f"got {outcomes}")
    assert outcomes.get("violates-invariant", 0) >= 1, (
        "the later commits must be rejected by the still-enforced rule; "
        f"got {outcomes}")
    assert result.invariant_checked >= 2, (
        "the gate must run on v2+ bases, not just the first; "
        f"checked={result.invariant_checked}")


def test_no_contract_means_no_invariant_accounting(tmp_path):
    run, reward, propose = _offline_actors()
    result = evolve(_tasks(), reward, run=run, propose=propose,
                    strategy=SingleSlot(initial_value="seed"), rounds=1,
                    n_workers=2, repo_path=str(tmp_path / "repo"))
    assert result.invariant_checked == 0
    assert result.invariant_rejections == 0
    assert result.outcomes().get("violates-invariant", 0) == 0


def test_a_self_referential_contract_is_refused_at_registration(tmp_path):
    run, reward, propose = _offline_actors()
    from agentdescent.governance import GovernanceError
    contract = _dep_contract(depends_on=("artifact",))
    with pytest.raises(GovernanceError):
        evolve(_tasks(), reward, run=run, propose=propose,
               strategy=SingleSlot(initial_value="seed"), rounds=1,
               n_workers=2, repo_path=str(tmp_path / "repo"),
               contract=contract)
