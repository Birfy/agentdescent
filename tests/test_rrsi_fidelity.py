"""The RRSI fidelity closure has to pin the two claims the merge made:

1. the wiring now consumes the **pinned upstream vocabulary** (`components.py`:
   `K` / `K_STR` / `normalize` evidence rules), not the path-segment heuristic
   -- so `skills/x/SKILL.md` is a `skill` edit, `ToolRegistry(...)` is
   `client_tool`, and a pure prose edit is `prompt`;
2. the per-edit ledger is the pinned `history.History` (durable JSONL) when a
   path is given, with the in-memory twin pinned to the same summaries by a
   parity test -- so the twin cannot silently drift from the file backend.

Plus the novelty term actually firing on a genuinely novel structural component.
"""

import json

from agentdescent.policies import MergeContext
from agentdescent.evolvable import Diff, EvidenceCard

from examples.rrsi.components import normalize
from examples.rrsi.rrsi_policies import (
    RRsiAcceptance,
    RRsiConfig,
    RRsiState,
    _InMemoryHistory,
    upstream_components,
)
from examples.rrsi.history import History


def _diff(ops):
    return Diff(diff_id="d", target="artifact", ops=ops)


# -- the upstream vocabulary ----------------------------------------------------


def test_upstream_components_uses_the_pinned_vocabulary():
    # a FileTree harness's skills/ path is the upstream `skill` component
    assert upstream_components(_diff({"skills/x/SKILL.md": "use SkillRegistry"})) == ("skill",)
    # a tool registry edit is client_tool
    assert upstream_components(
        _diff({"tools/a.py": "ToolRegistry.register_tool('a')"})) == ("client_tool",)
    # memory syntax is memory
    assert upstream_components(_diff({"memory/y.md": "Memory().remember(...)"})) == ("memory",)
    # a pure prompt edit is prompt
    assert upstream_components(_diff({"value": "always answer with the total"})) == ("prompt",)


def test_upstream_components_matches_components_normalize():
    for ops in ({"skills/x/SKILL.md": "skill"}, {"value": "prose"},
                {"t.py": "ToolRegistry()"}):
        d = _diff(ops)
        expected = tuple(sorted({
            normalize(None, f"{k}\n{v}")
            for k, v in ops.items()}))
        assert upstream_components(d) == expected


# -- the History backends --------------------------------------------------------


def test_memory_and_file_history_agree(tmp_path):
    path = str(tmp_path / "h.jsonl")
    file_h = History(path)
    mem_h = _InMemoryHistory()
    args = dict(t=2, variant="v1",
                edits=[{"id": "e1", "component": "skill"},
                       {"id": "e2", "component": "memory"}],
                outcome="accepted", delta_S=0.1, delta_C=0.0, accepted=True,
                S=0.7, C=100, diff="diff")
    file_h.append_candidate(**args)
    mem_h.append_candidate(**args)
    assert file_h.incumbent_component_counts() == mem_h.incumbent_component_counts()
    assert file_h.yield_g(2, 5) == mem_h.yield_g(2, 5)
    assert file_h.prune_set(2, 5) == mem_h.prune_set(2, 5)
    assert file_h.render(10) == mem_h.render(10)


def test_file_history_survives_reopen_and_resume(tmp_path):
    path = str(tmp_path / "rrsi.jsonl")
    state = RRsiState(history_path=path)
    state.record_measurement(
        _diff({"skills/x/SKILL.md": "use SkillRegistry"}),
        delta_score=0.2, delta_cost=0.0, accepted=True)
    # a fresh process/state on the same path reads the same evidence
    resumed = RRsiState(history_path=path)
    assert len(resumed.history.records()) == 1
    assert resumed.history.records()[0]["component"] == "skill"
    assert resumed.history.attempted()["skill"] == 1


def test_the_checkpoint_keeps_the_search_state_and_the_history_stays_durable(tmp_path):
    path = str(tmp_path / "rrsi.jsonl")
    state = RRsiState(history_path=path)
    state.best_score = 0.8
    state.round = 4
    state.record_measurement(
        _diff({"tools/a.py": "ToolRegistry.register_tool('a')"}),
        delta_score=0.1, delta_cost=0.0, accepted=False)
    fresh = RRsiState(history_path=path)
    fresh.restore(state.checkpoint())
    assert fresh.best_score == 0.8 and fresh.round == 4
    assert len(fresh.history.records()) == 1
    assert fresh.history.records()[0]["component"] == "client_tool"


# -- novelty fires on a structural component --------------------------------------


def _context(state, *, base, cand, ops, costs=0):
    from agentdescent.evolution import EvolvingArtifact, SingleSlot
    art = EvolvingArtifact("artifact", {"value": "seed"}, strategy=SingleSlot())
    cand_art = EvolvingArtifact("artifact", {"value": "new"}, strategy=SingleSlot())
    cards = [EvidenceCard(diff=_diff({k: "x" for k in ops}), base_version={},
                          touched=["artifact"], trajectory_refs=[],
                          cost_tokens=costs)]
    return MergeContext(artifact=art, candidate=cand_art, cards=cards,
                        base_counts=base, cand_counts=cand, diff=cards[0].diff)


def test_novelty_carries_a_flat_candidate_inside_the_band():
    # dS = 0.02 <= band(0.05): within-band; a *novel structural* component
    # (skill, never accepted before) is what the w_n term counts.
    cfg = RRsiConfig(noise_band=0.05, w_s=1.0, w_c=1.0, w_n=1.0)
    state = RRsiState(cfg, base_score=0.5)
    policy = RRsiAcceptance(state)
    d = policy.accept(_context(
        state, base=(50, 50), cand=(52, 48),
        ops=("skills/x/SKILL.md",)))
    assert d.accept is True
    # the component recorded is the upstream `skill`, and it was novel
    assert state.history.records()[0]["component"] == "skill"
    assert state.history.incumbent_component_counts().get("skill", 0) == 1
