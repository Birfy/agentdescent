"""The contracts must describe the code, not the other way round.

The rule this file enforces: if an existing implementation fails one of these
Protocols, **the Protocol is wrong**. `agentdescent/policies.py` is a description
of what the engine already calls, so a mismatch means the description drifted --
which is exactly what happened to `docs/verifier.md`, where a missing
`learned_eval` turned into an `AttributeError` half an hour into a run.

So two of these tests re-derive the method sets by grepping the call sites. They
are ugly on purpose: a hand-maintained list of expected methods would drift in
precisely the same way the thing it is guarding did.
"""

import json
import re
from dataclasses import asdict
from pathlib import Path

import pytest

from agentdescent.aggregator import Aggregator, AggregatorConfig, AggregatorProtocol
from agentdescent.evolvable import Evolvable
from agentdescent.ledger import Ledger
from agentdescent.policies import (
    AcceptDecision, AcceptancePolicy, ConflictPolicy, FusionPolicy, LedgerProtocol,
    MergeContext, Policies, Promotion, PromotionPolicy, ProposalContext,
    ProposalPolicy, SandboxProvider, SandboxSpec, VerifierProtocol,
)
from agentdescent.sampling import DifficultyWeighted, RoundRobin
from agentdescent.scheduler import AuditScheduler
from agentdescent.staleness import get_policy
from agentdescent.verifier import ThreeLayerVerifier, VerifierBudget

SRC = Path(__file__).resolve().parent.parent / "agentdescent"


def _protocol_methods(proto) -> set:
    """The members a Protocol declares.

    `__protocol_attrs__` is 3.12+; this package supports 3.9, where reading the
    class namespace is the portable equivalent."""
    attrs = getattr(proto, "__protocol_attrs__", None)
    if attrs is None:
        attrs = set(vars(proto)) | set(getattr(proto, "__annotations__", {}))
    return {a for a in attrs if not a.startswith("_")}


def _called_on(attr: str, *files: str) -> set:
    """Every ``<attr>.<method>`` the given sources actually call."""
    found = set()
    for name in files:
        text = (SRC / name).read_text(encoding="utf-8")
        found |= set(re.findall(rf"\b(?:self\.|eng\.)?{attr}\.([a-z_]+)\b", text))
    return {m for m in found if not m.startswith("_")}


# ---------------------------------------------------------------------------
# The existing implementations satisfy the contracts
# ---------------------------------------------------------------------------


def _verifier():
    return ThreeLayerVerifier(eval_fn=lambda a, ts: 1.0, held_out=[1, 2, 3],
                              budget=VerifierBudget(oracle_calls_remaining=5))


def test_the_shipped_verifier_satisfies_the_contract():
    assert isinstance(_verifier(), VerifierProtocol)


def _ledger(tmp_path):
    return Ledger(str(tmp_path / "repo"),
                  serialize=lambda a: {"state": dict(getattr(a, "state", {}))},
                  deserialize=lambda aid, v, d: _Art())


def test_the_shipped_ledger_satisfies_the_contract(tmp_path):
    assert isinstance(_ledger(tmp_path), LedgerProtocol)


def test_the_shipped_aggregator_satisfies_the_contract(tmp_path):
    agg = Aggregator(_ledger(tmp_path), _verifier(), AuditScheduler(),
                     AggregatorConfig(), staleness_policy=get_policy("guarded"))
    assert isinstance(agg, AggregatorProtocol)


def test_the_shipped_samplers_satisfy_the_bundle_field():
    """`task_sampler` was already a Protocol; the bundle must not narrow it."""
    from agentdescent.sampling import TaskSampler
    assert isinstance(RoundRobin(), TaskSampler)
    assert isinstance(DifficultyWeighted(), TaskSampler)


# ---------------------------------------------------------------------------
# The contracts were derived from the call sites
# ---------------------------------------------------------------------------


def test_verifier_protocol_covers_every_call_the_aggregator_makes():
    """This is the `docs/verifier.md` regression, as a test.

    That page described three methods; the fourth is called from the merge path,
    so a verifier written against the page raised `AttributeError` mid-run."""
    called = _called_on("verifier", "aggregator.py")
    assert called, "grep found no verifier calls -- the pattern has rotted"
    missing = called - _protocol_methods(VerifierProtocol)
    assert not missing, f"VerifierProtocol omits methods the aggregator calls: {missing}"


def test_ledger_protocol_covers_the_engine_as_well_as_the_aggregator():
    """Deriving this contract from the aggregator alone is not enough.

    The aggregator calls four methods; `_build_engine` calls `register` before
    any merge happens, `_safe_log` calls `log` at the end of every run, and
    `_Engine.cleanup` calls `close`. A contract written from the merge path
    alone type-checks and then dies during setup."""
    called = _called_on("ledger", "aggregator.py", "evolution.py", "async_evolve.py")
    called -= {"md"}          # `docs/ledger.md` in a docstring, not a call
    assert {"register", "log", "close"} <= called, (
        "the engine's ledger calls have moved; re-derive the protocol")
    missing = called - _protocol_methods(LedgerProtocol)
    assert not missing, f"LedgerProtocol omits methods the engine calls: {missing}"


# ---------------------------------------------------------------------------
# The bundle
# ---------------------------------------------------------------------------


def test_an_empty_bundle_asks_for_nothing():
    assert Policies().is_empty()
    assert not Policies(task_sampler=RoundRobin()).is_empty()


def test_the_bundle_covers_every_piece_52_named():
    """#52 named six: task sampling, proposal generation, merge, staleness,
    acceptance, promotion. Merge is two decisions, so seven fields."""
    fields = set(Policies.__dataclass_fields__)
    assert {"task_sampler", "proposal", "conflict", "fusion", "staleness",
            "acceptance", "promotion"} <= fields


def test_an_explicit_argument_beats_a_bundle_default():
    """The legacy keyword arguments fold in as shortcuts; silently ignoring one
    would be the worst of the three possible behaviours."""
    bundle = Policies(task_sampler=RoundRobin())
    explicit = DifficultyWeighted()
    assert bundle.merged_with(task_sampler=explicit).task_sampler is explicit
    # None means "not specified", so it must not clear the bundle's value
    assert bundle.merged_with(task_sampler=None).task_sampler is bundle.task_sampler


def test_merging_an_unknown_field_raises():
    with pytest.raises(TypeError, match="no field"):
        Policies().merged_with(nonsense=1)


# ---------------------------------------------------------------------------
# MergeContext keeps the two evaluation layers apart
# ---------------------------------------------------------------------------


class _Art:
    id = "a"
    blast_radius = 0.2


def test_the_full_set_and_the_cheap_layer_are_separate_fields():
    """A regression guard that reads the cheap layer judges "quality dropped"
    from a sub-sample -- shipped once, hence two differently-named fields."""
    ctx = MergeContext(artifact=_Art(), candidate=_Art(), cards=[],
                       base_counts=(8.0, 10.0), cand_counts=(9.0, 10.0),
                       base_cheap=1.0, cand_cheap=0.0)
    assert ctx.base_counts != (ctx.base_cheap, ctx.base_cheap)
    # the commit gates read counts, where the candidate is better...
    assert ctx.cand_counts[0] > ctx.base_counts[0]
    # ...while the cheap layer says the opposite. Ranking may use it; gates may not.
    assert ctx.cand_cheap < ctx.base_cheap


def test_comparability_is_explicit():
    same = MergeContext(_Art(), _Art(), [], (1.0, 1.0), (1.0, 1.0))
    assert same.comparable()          # single machine: always true
    cross = MergeContext(_Art(), _Art(), [], (1.0, 1.0), (1.0, 1.0),
                         base_env="a", cand_env="b")
    assert not cross.comparable()


# ---------------------------------------------------------------------------
# SandboxSpec
# ---------------------------------------------------------------------------


def test_a_spec_survives_json():
    spec = SandboxSpec(image="py3.11", env_allowlist=("PATH", "HOME"), cpu=2.0)
    assert json.loads(json.dumps(asdict(spec)))["image"] == "py3.11"


def test_a_spec_carries_no_secrets():
    """`env_allowlist` holds names. A spec is pickled, logged, cached and
    journalled, so a spec that held values would leak them into all four."""
    spec = SandboxSpec(env_allowlist=("ANTHROPIC_API_KEY", "OPENAI_API_KEY"))
    blob = json.dumps(asdict(spec))
    assert "ANTHROPIC_API_KEY" in blob                 # the name is fine
    assert not re.search(r"sk-[A-Za-z0-9]", blob)      # a value would not be


def test_the_fingerprint_is_stable_and_sensitive():
    a = SandboxSpec(image="x", cpu=1.0, env_allowlist=("PATH",))
    b = SandboxSpec(cpu=1.0, image="x", env_allowlist=("PATH",))   # same, reordered
    assert a.fingerprint() == b.fingerprint() == a.fingerprint()
    assert a.fingerprint() != SandboxSpec(image="y", cpu=1.0,
                                          env_allowlist=("PATH",)).fingerprint()
    assert SandboxSpec().fingerprint() != SandboxSpec(reuse=True).fingerprint()


def test_reuse_defaults_off():
    """The frozen-file overlay assumes a clean workspace; a dirty one scores
    plausibly and wrongly."""
    assert SandboxSpec().reuse is False


# ---------------------------------------------------------------------------
# The protocols are usable: minimal implementations satisfy them
# ---------------------------------------------------------------------------


def test_a_minimal_implementation_of_each_algorithm_protocol_type_checks():
    class Proposal:
        def propose(self, ctx): return [ctx.task]

    class Conflict:
        def resolve(self, artifact, cards): return list(cards), 0

    class Fusion:
        def select(self, artifact, diffs): return diffs[0], artifact, False

    class Accept:
        def accept(self, ctx): return AcceptDecision(True, "committed")

    class Promote:
        def observe(self, reports): return [Promotion("a")]

    assert isinstance(Proposal(), ProposalPolicy)
    assert isinstance(Conflict(), ConflictPolicy)
    assert isinstance(Fusion(), FusionPolicy)
    assert isinstance(Accept(), AcceptancePolicy)
    assert isinstance(Promote(), PromotionPolicy)
    assert Policies(proposal=Proposal(), conflict=Conflict(), fusion=Fusion(),
                    acceptance=Accept(), promotion=Promote()) is not None


def test_a_sandbox_provider_needs_all_three_methods():
    class Half:
        def acquire(self, spec): ...
        def release(self, sandbox): ...

    class Whole(Half):
        def reap(self, older_than): return 0

    assert not isinstance(Half(), SandboxProvider)
    assert isinstance(Whole(), SandboxProvider)


def test_proposal_context_defaults_match_todays_single_proposal_call():
    ctx = ProposalContext(rendered="r", task=None, output="o", reward=1.0)
    assert (ctx.k, ctx.history, ctx.base_version) == (1, (), 0)


# ---------------------------------------------------------------------------
# The bundle reaches the engine, and refuses what it cannot yet honour
# ---------------------------------------------------------------------------


def _tasks(n=6):
    from agentdescent import Task
    return [Task(id=f"t{i}", prompt=f"q{i}", meta={"gold": str(i)}) for i in range(n)]


def _run_evolve(**kw):
    from agentdescent import AppendRules, evolve
    return evolve(_tasks(), lambda t, o: 1.0 if o == t.meta["gold"] else 0.0,
                  run=lambda r, t: t.meta["gold"] if t.id in r else "?",
                  propose=lambda r, t, o, s: t.id,
                  strategy=AppendRules(), rounds=3, n_workers=2,
                  held_out_frac=0.5, seed=0, **kw)


def _trace(result):
    return [(h.round, h.held_out_reward, h.committed, h.rejected, sorted(h.reasons.items()))
            for h in result.history]


def test_the_bundle_and_the_legacy_argument_produce_the_same_run():
    """Two spellings of one thing must not be two behaviours."""
    via_kwarg = _run_evolve(task_sampler=RoundRobin())
    via_bundle = _run_evolve(policies=Policies(task_sampler=RoundRobin()))
    assert _trace(via_kwarg) == _trace(via_bundle)


def test_an_empty_bundle_is_the_same_run_as_passing_nothing():
    assert _trace(_run_evolve()) == _trace(_run_evolve(policies=Policies()))


def test_a_replaced_acceptance_rule_actually_decides():
    """The seam has to be load-bearing, not decorative."""
    class NeverAccept:
        calls = 0

        def accept(self, ctx):
            NeverAccept.calls += 1
            return AcceptDecision(False, "below-threshold", "refused by policy")

    r = _run_evolve(policies=Policies(acceptance=NeverAccept()))
    assert NeverAccept.calls > 0, "the custom acceptance rule was never consulted"
    assert r.outcomes().get("committed", 0) == 0, "it refused and something committed"


def test_a_replaced_promotion_rule_actually_decides():
    class NeverPromote:
        def observe(self, reports): return []

    r = _run_evolve(policies=Policies(promotion=NeverPromote()))
    assert r is not None       # the run completes; nothing reaches stable


def test_a_field_with_no_implementation_still_raises():
    """The guard has to keep working for whatever is not wired yet -- being
    accepted and ignored is the one failure a caller cannot detect."""
    with pytest.raises(NotImplementedError, match="sandbox_provider"):
        _run_evolve(policies=Policies(sandbox_provider=object()))


def test_the_async_path_honours_the_same_bundle():
    """`sandbox_spec` became honoured when it started keying the eval cache, so
    the refusal is asserted against a field that still has no implementation."""
    from agentdescent import async_evolve
    with pytest.raises(NotImplementedError, match="sandbox_provider"):
        async_evolve(_tasks(), lambda t, o: 0.0, run=lambda r, t: "x",
                     propose=lambda r, t, o, s: None, max_iters=1,
                     policies=Policies(sandbox_provider=object()))


def test_the_async_path_refuses_an_executor_it_would_otherwise_ignore():
    """The barrier-free loop has no executor seam: its worker calls `eng.run`.

    Both engines shared one `_WIRED_POLICIES` tuple, so `executor` was declared
    supported for a loop that never reads it -- accepted and dropped, which is the
    single outcome `require_supported` exists to prevent. It matters more since a
    supplied executor started working under `evolve()`: flipping
    `asynchronous=True` would silently stop honouring it.
    """
    from agentdescent import async_evolve
    from agentdescent.executor import ThreadExecutor

    ex = ThreadExecutor(2, run=lambda r, t: "x", reward=lambda t, o: 0.0)
    try:
        with pytest.raises(NotImplementedError, match="executor"):
            async_evolve(_tasks(), lambda t, o: 0.0, run=lambda r, t: "x",
                         propose=lambda r, t, o, s: None, max_iters=1,
                         policies=Policies(executor=ex))
    finally:
        ex.shutdown()


def test_the_sync_path_still_honours_an_executor():
    """The other half: narrowing the async set must not narrow evolve()'s."""
    from agentdescent.executor import ThreadExecutor

    seen = []

    class Counting(ThreadExecutor):
        def rollout(self, spec):
            seen.append(spec.task.id)
            return super().rollout(spec)

    ex = Counting(2)
    try:
        _run_evolve(policies=Policies(executor=ex))
    finally:
        ex.shutdown()
    assert seen, "evolve() stopped routing rollouts through a supplied executor"


def test_a_multi_proposal_policy_is_refused_rather_than_truncated():
    """Keeping the first of k would make a sampling algorithm look like it ran
    when only a fraction of it did."""
    class ThreeAtATime:
        def propose(self, ctx): return ["a", "b", "c"]

    from agentdescent.evolution import ProposalContractError
    with pytest.raises(ProposalContractError, match="one proposal per rollout"):
        _run_evolve(policies=Policies(proposal=ThreeAtATime()))


def test_a_single_proposal_policy_is_equivalent_to_the_callable():
    from agentdescent.defaults import SingleProposal
    plain = _run_evolve()
    viapolicy = _run_evolve(policies=Policies(
        proposal=SingleProposal(lambda r, t, o, s: t.id)))
    assert _trace(plain) == _trace(viapolicy)


# ---------------------------------------------------------------------------
# Injected backends
# ---------------------------------------------------------------------------


def test_an_injected_verifier_is_called_exactly_as_the_built_in_one_is():
    """Injection must not quietly change *how much* the engine measures.

    A wrapper that adds or drops a gate call changes what the run costs while
    leaving every result identical -- invisible to a trace comparison, and the
    reason this counts calls rather than checking the outcome."""
    from agentdescent.verifier import ThreeLayerVerifier

    counts = {}

    class Counting:
        def __init__(self, inner): self.inner = inner
        def __getattr__(self, name):
            attr = getattr(self.inner, name)
            if not callable(attr):
                return attr
            def wrapped(*a, **kw):
                counts[name] = counts.get(name, 0) + 1
                return attr(*a, **kw)
            return wrapped

    seen = {}

    def factory(ledger, verifier, audit, config, policy):
        seen["verifier"] = verifier
        from agentdescent.aggregator import Aggregator
        return Aggregator(ledger, verifier, audit, config, staleness_policy=policy)

    inner = ThreeLayerVerifier(eval_fn=lambda a, ts: 1.0, held_out=[1, 2, 3])
    _run_evolve(policies=Policies(verifier=Counting(inner),
                                  aggregator_factory=factory))
    assert isinstance(seen.get("verifier"), Counting), "the engine built its own"
    assert set(counts) <= {"cheap_eval", "eval_counts", "oracle_eval",
                           "learned_eval", "budget", "held_out"}, counts


def test_an_in_memory_ledger_is_enough_to_finish_a_run(tmp_path):
    """`LedgerProtocol` has to be sufficient, not merely satisfied.

    This is the first thing that proves the protocol covers `register`, `log` and
    `close` -- the three the aggregator never calls and the engine always does."""
    from agentdescent.ledger import Ledger

    calls = []

    class Recording:
        def __init__(self, inner): self.inner = inner
        def register(self, artifact, branch=Ledger.DEV):
            calls.append("register")
            return self.inner.register(artifact, branch)
        def snapshot(self, branch=Ledger.DEV): return self.inner.snapshot(branch)
        def head_version(self, branch=Ledger.DEV): return self.inner.head_version(branch)
        def commit(self, *a, **kw): return self.inner.commit(*a, **kw)
        def promote_to_stable(self, aid): return self.inner.promote_to_stable(aid)
        def log(self, branch=Ledger.DEV, limit=20):
            calls.append("log")
            return self.inner.log(branch, limit)
        def close(self):
            calls.append("close")
            return self.inner.close()

    inner = Ledger(str(tmp_path / "repo"),
                   serialize=lambda a: {"state": dict(a.state),
                                        "blast_radius": a.blast_radius},
                   deserialize=lambda aid, v, d: None)
    # the engine deserialises with its own artifact type, so borrow the real one
    from agentdescent.evolution import EvolvingArtifact
    inner._deserialize = lambda aid, v, d: EvolvingArtifact(
        aid, d.get("state", {}), v, d.get("blast_radius", 0.2))

    r = _run_evolve(policies=Policies(ledger=Recording(inner)))
    assert r.final_reward >= 0.0
    assert "register" in calls, "the engine never registered through the protocol"
    assert "log" in calls, "the run summary skipped the ledger"
    # `close` is deliberately absent: the engine closes only the ledger it made
    # itself, the same rule that applies to a caller-supplied `repo_path`. An
    # injected ledger's lifetime belongs to whoever injected it.
    assert "close" not in calls, "the engine closed a ledger it does not own"
