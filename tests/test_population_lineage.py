"""PopulationAggregator must record Candidate.parent and subtree visits (#275).

FlatPuct / CostEfficient / MCTS all need a real tree under evolve(). Before
this fix the archive stored every candidate as a root and incremented only the
chosen node's selected count, so lineage-aware policies silently degenerated.
"""

from __future__ import annotations

import threading

from agentdescent.aggregator import Aggregator, AggregatorConfig
from agentdescent.evolvable import Diff, EvidenceCard
from agentdescent.evolution import AppendRules, EvolvingArtifact
from agentdescent.ledger import Ledger
from agentdescent.population import PopulationAggregator
from agentdescent.scheduler import AuditScheduler
from agentdescent.selection import Archive, Candidate, FlatPuct, SelectionContext
from agentdescent.verifier import ThreeLayerVerifier, VerifierBudget


def _ledger(tmp_path):
    lg = Ledger(str(tmp_path), lambda a: {"state": dict(a.state)},
                lambda aid, v, s: EvolvingArtifact(aid, s.get("state", {}), v,
                                                   0.2, None, AppendRules()))
    lg.register(EvolvingArtifact("a", {"k": "v0"}, 1, 0.2, None, AppendRules()))
    return lg


def _agg(tmp_path, selection):
    lg = _ledger(tmp_path)
    # Score by numeric suffix so each new "vN" strictly improves on the seed.
    v = ThreeLayerVerifier(
        eval_fn=lambda a, t: int(a.state.get("k", "v0").replace("v", "") or "0") / 10.0,
        held_out=[1, 2, 3], budget=VerifierBudget())
    return lg, PopulationAggregator(
        lg, v, AuditScheduler(), AggregatorConfig(batch_trigger=1),
        selection=selection, artifact_id="a")


def _propose(agg, lg, val: str):
    """Ingest one improving proposal against the current dev head."""
    snap = lg.snapshot("dev")
    ops = {"k": val}
    agg.ingest(EvidenceCard(
        diff=Diff(diff_id=f"c-{val}", target="a", ops=ops, author="w"),
        base_version={"a": snap.version["a"]},
        touched=["a"], trajectory_refs=[], branch=""))


def test_population_records_parent_across_selections(tmp_path):
    """≥3 selections: newly admitted candidates after a tip is set carry the
    parent version they were derived from. Fails on main (every parent is None).

    The first mutation lands before any selection tip exists, so it stays a
    root (same as the seed); every later admit on that branch is a child of
    the last selected tip.
    """

    class PreferNewest:
        def select(self, ctx, n):
            ranked = sorted(
                ctx.candidates,
                key=lambda c: c.score if c.score is not None else -1,
                reverse=True)
            return ranked[:n]

    lg, agg = _agg(tmp_path, PreferNewest())
    agg.step(n_workers=1)  # seed only — no selection yet

    # First mutation: admitted before any tip → root.
    _propose(agg, lg, "v1")
    agg.step(n_workers=1)
    v1 = next(e for e in agg._archive if e["state"].get("k") == "v1")
    assert v1.get("parent") is None, "pre-tip admit must stay a root"
    # PreferNewest selected v1 and set the tip.
    assert agg._branch_parent.get("dev") == int(v1["version"])

    parents_seen = []
    for val in ("v2", "v3", "v4"):
        tip_before = agg._branch_parent.get("dev")
        _propose(agg, lg, val)
        agg.step(n_workers=1)
        entry = next(e for e in agg._archive if e["state"].get("k") == val)
        assert entry.get("parent") == tip_before, (
            f"{val}: expected parent={tip_before}, got {entry.get('parent')}; "
            f"archive={[(e['state'].get('k'), e.get('parent'), e['version']) for e in agg._archive]}"
        )
        parents_seen.append(entry["parent"])

    assert len(parents_seen) >= 3
    versions = {int(e["version"]) for e in agg._archive}
    for p in parents_seen:
        assert int(p) in versions

    cands = agg._candidates()
    by_state = {dict(c.state).get("k"): c for c in cands}
    assert by_state["v0"].parent is None
    assert by_state["v1"].parent is None
    assert by_state["v2"].parent == by_state["v1"].version
    assert by_state["v3"].parent == by_state["v2"].version


def test_flatpuct_under_population_matches_hand_built_tree(tmp_path):
    """Subtree backprop is live: FlatPuct on archive candidates agrees with
    FlatPuct on an equivalent hand-built Candidate tree."""

    class SelectSeedThenNewest:
        def __init__(self):
            self.n = 0

        def select(self, ctx, n):
            self.n += 1
            if self.n == 1:
                seed = next(c for c in ctx.candidates
                            if dict(c.state).get("k") == "v0")
                return [seed]
            ranked = sorted(
                ctx.candidates,
                key=lambda c: c.score if c.score is not None else -1,
                reverse=True)
            return ranked[:n]

    lg, agg = _agg(tmp_path, SelectSeedThenNewest())
    agg.step(n_workers=1)
    _propose(agg, lg, "v1")
    agg.step(n_workers=1)  # admit v1 as root; select seed → tip=seed
    _propose(agg, lg, "v2")
    agg.step(n_workers=1)  # admit v2 as child of seed; select newest
    _propose(agg, lg, "v3")
    agg.step(n_workers=1)  # admit v3 as child of previous tip

    cands = agg._candidates()
    by_k = {dict(c.state).get("k"): c for c in cands}
    assert by_k["v2"].parent == by_k["v0"].version, (
        f"v2 should be child of seed, got parent={by_k['v2'].parent}"
    )

    hand = [
        Candidate(artifact_id="a", version=c.version, state=dict(c.state),
                  score=c.score, selected=c.selected, parent=c.parent)
        for c in cands
    ]
    pop_pick = FlatPuct(c_puct=1.0).select(
        SelectionContext(head=cands[0], candidates=tuple(cands),
                         round=0, n_workers=1), 1)
    hand_pick = FlatPuct(c_puct=1.0).select(
        SelectionContext(head=hand[0], candidates=tuple(hand),
                         round=0, n_workers=1), 1)
    assert [c.version for c in pop_pick] == [c.version for c in hand_pick]

    # Subtree property: bumping a leaf increments ancestors.
    before = {c.version: c.selected for c in agg._candidates()}
    leaf = by_k["v3"] if "v3" in by_k else max(
        agg._candidates(),
        key=lambda c: c.score if c.score is not None else -1)
    # Prefer a node that has a parent so the walk is non-trivial.
    chained = [c for c in agg._candidates() if c.parent is not None]
    assert chained, "expected at least one non-root candidate"
    leaf = max(chained, key=lambda c: c.score if c.score is not None else -1)
    agg._bump_selected(leaf.version)
    after = {int(e["version"]): int(e["selected"]) for e in agg._archive}
    assert after[leaf.version] == before[leaf.version] + 1
    parent = leaf.parent
    seen = {leaf.version}
    while parent is not None and parent not in seen:
        assert after[int(parent)] == before[int(parent)] + 1
        seen.add(int(parent))
        parent_entry = next(e for e in agg._archive
                            if int(e["version"]) == int(parent))
        parent = parent_entry.get("parent")


def test_population_checkpoint_roundtrips_parent():
    """checkpoint/restore keeps parent; a pre-#275 entry restores as a root."""
    agg = PopulationAggregator.__new__(PopulationAggregator)
    agg.selection = Archive(sampling="novelty")
    agg.population_artifact = "a"
    agg._archive = [
        {"state": {"k": "v0"}, "score": 0.5, "version": 1, "selected": 2,
         "parent": None},
        {"state": {"k": "v1"}, "score": 0.8, "version": 3, "selected": 1,
         "parent": 1},
    ]
    agg._keys = {"v0", "v1"}
    agg._archive_lock = threading.Lock()
    agg._selections = 4
    agg._branch_parent = {"dev": 3}
    agg._seen = set()
    agg.posteriors = {}
    agg.promoted_at = {}

    payload = agg.checkpoint()
    assert payload["archive"][1]["parent"] == 1
    assert payload["branch_parent"]["dev"] == 3

    fresh = PopulationAggregator.__new__(PopulationAggregator)
    fresh.selection = Archive(sampling="novelty")
    fresh.population_artifact = "a"
    fresh._archive = []
    fresh._keys = set()
    fresh._archive_lock = threading.Lock()
    fresh._selections = 0
    fresh._branch_parent = {}
    fresh._seen = set()
    fresh.posteriors = {}
    fresh.promoted_at = {}

    def _noop(self, state):
        self.posteriors = dict(state.get("posteriors") or {})
        self.promoted_at = dict(state.get("promoted_at") or {})
        seen = state.get("seen")
        if isinstance(seen, list):
            self._seen = set(seen)

    original = Aggregator.restore
    Aggregator.restore = _noop
    try:
        fresh.restore(payload)
    finally:
        Aggregator.restore = original

    assert fresh._archive[0]["parent"] is None
    assert fresh._archive[1]["parent"] == 1
    assert fresh._branch_parent == {"dev": 3}
    assert [c.parent for c in fresh._candidates()] == [None, 1]

    legacy = {
        "archive": [
            {"state": {"k": "old"}, "score": 0.1, "version": 9, "selected": 0},
        ],
        "keys": ["old"],
        "selections": 0,
        "posteriors": {},
        "promoted_at": {},
        "seen": [],
    }
    Aggregator.restore = _noop
    try:
        fresh.restore(legacy)
    finally:
        Aggregator.restore = original
    assert fresh._archive[0]["parent"] is None
    assert fresh._branch_parent == {}


def test_colliding_ledger_versions_on_two_forks_get_distinct_nodes(tmp_path):
    """Two forks from one snapshot can each commit a different state at the same
    ledger version. The archive must still give them distinct ids, or selecting
    one bumps (and parents) the other -- review feedback on #278."""
    lg, agg = _agg(tmp_path, Archive())
    seed = lg.snapshot("dev").get("a")
    agg._admit(seed, 1)
    left = EvolvingArtifact("a", {"k": "v2"}, 5, 0.2, None, AppendRules())
    right = EvolvingArtifact("a", {"k": "v3"}, 5, 0.2, None, AppendRules())
    agg._admit(left, 5, branch="head/0")
    agg._admit(right, 5, branch="head/1")

    cands = {c.state["k"]: c for c in agg._candidates()}
    assert len({c.version for c in cands.values()}) == 3
    assert cands["v2"].version == 5          # first claimant keeps the ledger id

    agg._bump_selected(cands["v3"].version)
    after = {c.state["k"]: c.selected for c in agg._candidates()}
    assert after == {"v0": 0, "v2": 0, "v3": 1}

    # A later child of the right fork must point at the right fork's node.
    agg._branch_parent["head/1"] = cands["v3"].version
    child = EvolvingArtifact("a", {"k": "v4"}, 6, 0.2, None, AppendRules())
    agg._admit(child, 6, branch="head/1")
    kid = next(c for c in agg._candidates() if c.state["k"] == "v4")
    assert kid.parent == cands["v3"].version != cands["v2"].version


def test_restore_gives_legacy_duplicate_versions_distinct_ids():
    """A pre-dedup multi-head checkpoint can hold two distinct fork states with
    one ledger version. restore() must give the later one a fresh id so that
    selecting either bumps only that candidate and its ancestors (#278)."""
    fresh = PopulationAggregator.__new__(PopulationAggregator)
    fresh.selection = Archive(sampling="novelty")
    fresh.population_artifact = "a"
    fresh._archive = []
    fresh._keys = set()
    fresh._archive_lock = threading.Lock()
    fresh._selections = 0
    fresh._branch_parent = {}
    fresh._seen = set()
    fresh.posteriors = {}
    fresh.promoted_at = {}
    legacy = {
        "archive": [
            {"state": {"k": "root"}, "score": 0.1, "version": 1, "selected": 0},
            {"state": {"k": "left"}, "score": 0.4, "version": 5, "selected": 0,
             "parent": 1},
            {"state": {"k": "right"}, "score": 0.6, "version": 5, "selected": 0,
             "parent": 1},
            {"state": {"k": "kid"}, "score": 0.7, "version": 6, "selected": 0,
             "parent": 5},
        ],
        "keys": ["root", "left", "right", "kid"],
        "selections": 0,
        "branch_parent": {"head/0": 5},
    }

    def _noop(self, state):
        pass

    original = Aggregator.restore
    Aggregator.restore = _noop
    try:
        fresh.restore(legacy)
    finally:
        Aggregator.restore = original

    cands = {c.state["k"]: c for c in fresh._candidates()}
    versions = [c.version for c in cands.values()]
    assert len(set(versions)) == len(versions) == 4
    # First claimant keeps the id; saved references resolve to it.
    assert cands["left"].version == 5
    assert cands["right"].version == 7
    assert cands["right"].parent == 1
    assert cands["kid"].parent == 5
    assert fresh._branch_parent == {"head/0": 5}

    fresh._bump_selected(cands["right"].version)
    after = {c.state["k"]: c.selected for c in fresh._candidates()}
    assert after == {"root": 1, "left": 0, "right": 1, "kid": 0}

    fresh._bump_selected(cands["left"].version)
    after = {c.state["k"]: c.selected for c in fresh._candidates()}
    assert after == {"root": 2, "left": 1, "right": 1, "kid": 0}
