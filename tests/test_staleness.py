from agentdescent.staleness import (
    FullStaleness,
    GuardedStaleness,
    ReflectiveStaleness,
    StaleAction,
    get_policy,
)


def test_full_accepts_regardless_of_eta():
    p = FullStaleness()
    assert p.decide(eta=0, alpha=1, contract_breaking=False) is StaleAction.ACCEPT
    assert p.decide(eta=99, alpha=1, contract_breaking=False) is StaleAction.ACCEPT
    # but never blindly crosses a broken contract.
    assert p.decide(eta=1, alpha=1, contract_breaking=True) is StaleAction.DISCARD


def test_guarded_is_version_gated():
    p = GuardedStaleness()
    assert p.decide(eta=0, alpha=2, contract_breaking=False) is StaleAction.ACCEPT
    assert p.decide(eta=2, alpha=2, contract_breaking=False) is StaleAction.REBASE
    assert p.decide(eta=3, alpha=2, contract_breaking=False) is StaleAction.DISCARD
    assert p.decide(eta=1, alpha=5, contract_breaking=True) is StaleAction.DISCARD


def test_reflective_always_rebases_when_stale():
    p = ReflectiveStaleness()
    assert p.decide(eta=0, alpha=1, contract_breaking=False) is StaleAction.ACCEPT
    # ignores the alpha budget: even far-stale diffs get a reflective replay.
    assert p.decide(eta=50, alpha=1, contract_breaking=False) is StaleAction.REBASE
    assert p.decide(eta=2, alpha=1, contract_breaking=True) is StaleAction.DISCARD


def test_registry_lookup():
    assert isinstance(get_policy("full"), FullStaleness)
    assert isinstance(get_policy("guarded"), GuardedStaleness)
    assert isinstance(get_policy("reflective"), ReflectiveStaleness)


def test_unknown_policy_raises():
    try:
        get_policy("nope")
    except ValueError as e:
        assert "unknown staleness policy" in str(e)
    else:
        assert False, "expected ValueError"


# ---------------------------------------------------------------------------
# The REBASE branch's re-verification has to actually verify something
# ---------------------------------------------------------------------------


def test_an_evidence_card_with_nothing_to_score_says_so():
    """`REBASE` keeps a diff iff `before <= after`, and an empty task list scores
    0.0 on both sides -- so the check passes for every diff and the "cheap
    re-verification" the policy is named for becomes a no-op.

    `trajectory_refs` was annotated `List[str]`, so a domain that followed the
    annotation and stored ids arrives here with a non-empty list and nothing in
    it the engine can score. The annotation is fixed; this is the part that makes
    it visible while it is happening.
    """
    import warnings

    from agentdescent.evalcache import MemoryCache
    from agentdescent.evolution import EvolvingArtifact, _Runtime
    from agentdescent.evolvable import Diff, EvidenceCard

    runtime = _Runtime(run=lambda rendered, t: "x", reward=lambda t, o: 1.0,
                       cache=MemoryCache())
    art = EvolvingArtifact("a", {}, runtime=runtime)
    card = EvidenceCard(diff=Diff("d", "a", {"k": "v"}), base_version={"a": 1},
                        touched=["a"], trajectory_refs=["t0", "t1"])   # ids, not Tasks

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert art.evidence_eval(card) == 0.0
        art.evidence_eval(card)                      # ...and it is not repeated
    assert len(caught) == 1, f"expected exactly one warning, got {len(caught)}"
    assert "nothing to score" in str(caught[0].message)


def test_a_card_carrying_real_tasks_is_silent():
    """The warning must not fire on the ordinary path."""
    import warnings

    from agentdescent.evalcache import MemoryCache
    from agentdescent.evolution import EvolvingArtifact, Task, _Runtime
    from agentdescent.evolvable import Diff, EvidenceCard

    runtime = _Runtime(run=lambda rendered, t: "x", reward=lambda t, o: 1.0,
                       cache=MemoryCache())
    art = EvolvingArtifact("a", {}, runtime=runtime)
    card = EvidenceCard(diff=Diff("d", "a", {"k": "v"}), base_version={"a": 1},
                        touched=["a"], trajectory_refs=[Task(id="t0", prompt="q")])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert art.evidence_eval(card) == 1.0
    assert not caught, [str(w.message) for w in caught]


# ---------------------------------------------------------------------------
# `staleness_policy=` has to have something to decide
# ---------------------------------------------------------------------------


def _observed_etas(refresh_interval):
    """Run the synchronous engine and collect every eta the policy was asked about."""
    import warnings

    from agentdescent.evolution import AppendRules, Task, evolve

    seen = []

    class _Spy(GuardedStaleness):
        name = "guarded"

        def decide(self, eta, alpha, contract_breaking):
            action = super().decide(eta, alpha, contract_breaking)
            seen.append((eta, action))
            return action

    tasks = [Task(id=f"t{i}", prompt=f"q{i}", meta={"gold": str(i)}) for i in range(12)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = evolve(
            tasks, lambda t, o: 1.0 if o == t.meta["gold"] else 0.0,
            run=lambda rendered, t: t.meta["gold"] if t.id in rendered else "?",
            propose=lambda r, t, o, s: t.id, strategy=AppendRules(),
            rounds=10, n_workers=4, max_concurrency=4, held_out_frac=0.5,
            refresh_interval=refresh_interval, staleness_policy=_Spy())
    return seen, result


def test_the_default_synchronous_loop_has_no_staleness_to_decide():
    """Pinned as a property, not a defect: with `refresh_interval=1` every worker
    proposes against the round's fresh snapshot, so `eta` is 0 by construction.

    Worth a test because it is *surprising*: `evolve(staleness_policy=...)` is a
    documented knob, and on this path it could not change a single decision --
    Full, Guarded and Reflective were indistinguishable.
    """
    seen, _ = _observed_etas(refresh_interval=1)
    assert seen, "premise: the staleness filter ran at all"
    assert {eta for eta, _ in seen} == {0}
    assert {a for _, a in seen} == {StaleAction.ACCEPT}


def test_a_refresh_interval_makes_every_staleness_action_reachable():
    """Above 1, workers hold a staggered spread of versions, so their diffs
    arrive with a spread of eta and the policy actually decides."""
    seen, result = _observed_etas(refresh_interval=3)
    etas = {eta for eta, _ in seen}
    actions = {a for _, a in seen}
    assert max(etas) > 0, f"still no staleness on the synchronous path: {sorted(etas)}"
    assert actions == {StaleAction.ACCEPT, StaleAction.REBASE, StaleAction.DISCARD}, (
        f"not every branch of the policy is reachable: {sorted(a.name for a in actions)}")
    assert result.outcomes().get("committed", 0) > 0, (
        "staggering the snapshots stopped the run from converging")
