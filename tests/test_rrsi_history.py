"""Deterministic fixtures for the RRSI component vocabulary and edit history (#252)."""

from __future__ import annotations

import json
import math

import pytest

from examples.rrsi import selection
from examples.rrsi.components import (
    K, K_STR, classify_diff, has_evidence, normalize, novelty, text_only,
)
from examples.rrsi.history import History, exploration, stall_flag


def _cand(h, t, edits, delta_S, accepted=False, outcome=None, variant="v0"):
    h.append_candidate(t=t, variant=variant, edits=edits,
                       outcome=outcome or ("ACCEPTED" if accepted else "REJECTED"),
                       delta_S=delta_S, delta_C=0.0 if delta_S is not None else None,
                       accepted=accepted, S=0.5 if delta_S is not None else None,
                       C=1000.0 if delta_S is not None else None, diff="--- a\n+++ b\n")


def _edit(i, component, hypothesis="h"):
    return {"id": f"E{i}", "component": component, "hypothesis": hypothesis}


# ---------------------------------------------------------------------------
# components
# ---------------------------------------------------------------------------


def test_vocabulary_matches_the_pinned_upstream():
    assert K == ("prompt", "control_flow", "config", "output_plumbing",
                 "context_mgmt", "client_tool", "skill", "memory", "subagent")
    assert K_STR == ("client_tool", "skill", "memory", "subagent")
    assert set(K_STR) <= set(K)
    assert selection.STRUCTURAL_COMPONENTS == frozenset(K_STR)


def test_text_only_diffs_are_prompt_edits_whatever_the_words():
    diff = '--- a\n+++ b\n-    "Use the skills/ catalog"\n+    "Use the SkillRegistry"\n+    # note\n'
    assert text_only(diff)
    assert classify_diff(diff) == "prompt"
    assert not text_only("--- a\n+++ b\n")          # nothing changed is not text-only


def test_classify_uses_domain_signals_before_generic_ones():
    diff = "+reg = ToolRegistry()\n+cfg.retries = 3\n"
    assert classify_diff(diff) == "client_tool"
    assert classify_diff(diff, [("config", [r"cfg\."])]) == "config"
    assert classify_diff("+x = 1\n") == "prompt"


def test_normalize_keeps_a_declaration_only_with_evidence():
    skill_diff = "+load('skills/pdf/SKILL.md')\n"
    assert normalize("Skill", skill_diff) == "skill"
    # Declared skill, but the diff ships a memory store: reclassified.
    assert normalize("skill", "+m = Memory(path)\n") == "memory"
    # Unknown tag falls back to classification.
    assert normalize("bogus", skill_diff) == "skill"
    assert normalize(None, "+x = 1\n") == "prompt"


def test_a_text_only_diff_cannot_keep_a_structural_declaration():
    """Not upstream: prose that mentions ``SkillRegistry`` is still a prompt edit."""
    diff = '--- a\n+++ b\n-    "Use the catalog"\n+    "Use the SkillRegistry"\n'
    assert text_only(diff) and has_evidence("skill", diff)
    assert normalize("skill", diff) == "prompt"
    assert normalize("skill", "+reg = SkillRegistry()\n") == "skill"


def test_a_non_structural_declaration_needs_a_domain_signal():
    """Upstream quirk, pinned: only structural components have generic signals."""
    diff = "+cfg.retries = 3\n"
    assert not has_evidence("config", diff)
    assert normalize("config", diff) == "prompt"
    assert normalize("config", diff, [("config", [r"cfg\."])]) == "config"


def test_novelty_matches_selection_and_counts_untouched_structure_once():
    counts = {"skill": 1, "memory": 0}
    comps = ["skill", "memory", "memory", "prompt", "subagent"]
    assert novelty(comps, counts) == 2               # memory, subagent
    assert novelty(comps, counts) == selection._novelty(comps, counts)


# ---------------------------------------------------------------------------
# History: storage
# ---------------------------------------------------------------------------


def test_a_bundle_fans_out_one_record_per_edit_carrying_one_measurement(tmp_path):
    h = History(tmp_path / "h.jsonl")
    _cand(h, 1, [_edit(1, "skill"), _edit(2, "prompt")], 0.0312345678, accepted=True)
    recs = h.records()
    assert [r["edit_id"] for r in recs] == ["E1", "E2"]
    assert {r["delta_S"] for r in recs} == {0.031235}
    assert all(r["bundle"] == 2 and r["accepted"] for r in recs)
    assert h.has(1, "v0") and not h.has(1, "v1") and not h.has(2, "v0")


def test_no_edits_writes_one_placeholder_record(tmp_path):
    h = History(tmp_path / "h.jsonl")
    _cand(h, 0, [], None, outcome="CRITIC_REJECT")
    (rec,) = h.records()
    assert rec["edit_id"] == "C1" and rec["component"] is None and rec["bundle"] == 0
    assert rec["delta_S"] is None


def test_unmeasured_candidates_are_attempted_but_not_tried(tmp_path):
    h = History(tmp_path / "h.jsonl")
    _cand(h, 1, [_edit(1, "memory")], None, outcome="SMOKE_FAIL")
    _cand(h, 2, [_edit(2, "skill")], 0.02)
    assert h.tried() == {"skill"}
    att = h.attempted()
    assert att["memory"] == 1 and att["skill"] == 1 and set(att) == set(K)
    assert [r["component"] for r in h.measured()] == ["skill"]


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, "0.1", True])
def test_non_finite_or_non_numeric_measurements_are_rejected(tmp_path, bad):
    h = History(tmp_path / "h.jsonl")
    with pytest.raises(ValueError):
        _cand(h, 1, [_edit(1, "skill")], bad)
    assert h.records() == []


def test_a_torn_final_line_is_ignored_and_dropped_by_the_next_append(tmp_path):
    path = tmp_path / "h.jsonl"
    h = History(path)
    _cand(h, 1, [_edit(1, "skill")], 0.01)
    with open(path, "a") as fh:
        fh.write('{"t": 2, "edit_id": "E9", "comp')          # crash mid-append
    assert [r["edit_id"] for r in h.records()] == ["E1"]
    _cand(h, 2, [_edit(2, "prompt")], 0.0)
    assert [r["edit_id"] for r in h.records()] == ["E1", "E2"]
    assert all(json.loads(line) for line in path.read_text().splitlines())


def test_a_complete_but_unterminated_last_record_is_kept(tmp_path):
    path = tmp_path / "h.jsonl"
    path.write_text(json.dumps({"t": 1, "edit_id": "E1", "component": "skill",
                                "delta_S": 0.1}))
    h = History(path)
    assert len(h.records()) == 1
    _cand(h, 2, [_edit(2, "prompt")], 0.0)
    assert [r["edit_id"] for r in h.records()] == ["E1", "E2"]


def test_corruption_before_the_last_line_raises(tmp_path):
    path = tmp_path / "h.jsonl"
    path.write_text('{"t": 1}\nnot json\n{"t": 2}\n')
    with pytest.raises(ValueError, match=":2:"):
        History(path).records()


def test_a_torn_tail_cut_inside_a_multibyte_character_is_ignored(tmp_path):
    path = tmp_path / "h.jsonl"
    h = History(path)
    _cand(h, 1, [_edit(1, "skill", "技能")], 0.01)
    line = json.dumps({"t": 2, "hypothesis": "中文假设"}, ensure_ascii=False).encode()
    with open(path, "ab") as fh:
        fh.write(line[: line.index("中".encode()) + 1])        # cut mid-character
    assert [r["hypothesis"] for r in h.records()] == ["技能"]
    _cand(h, 3, [_edit(3, "prompt", "提示")], 0.0)
    assert [r["hypothesis"] for r in h.records()] == ["技能", "提示"]


@pytest.mark.parametrize("bad_line", [b"\xe4\xb8", b"[]", b"3"])
def test_invalid_utf8_or_non_object_before_the_last_line_raises(tmp_path, bad_line):
    path = tmp_path / "h.jsonl"
    path.write_bytes(b'{"t": 1}\n' + bad_line + b'\n{"t": 2}\n')
    with pytest.raises(ValueError, match=":2: malformed history record"):
        History(path).records()


def test_numpy_scalars_are_accepted_as_measurements(tmp_path):
    np = pytest.importorskip("numpy")
    h = History(tmp_path / "h.jsonl")
    h.append_candidate(1, "v", [_edit(1, "skill")], "ACCEPTED", np.float32(0.25),
                       np.int64(3), True, np.float32(0.5), np.float64(10.0), None)
    (rec,) = h.records()
    assert rec["delta_S"] == 0.25 and rec["delta_C"] == 3.0 and rec["S"] == 0.5
    with pytest.raises(ValueError):
        _cand(h, 2, [_edit(2, "skill")], np.float32("nan"))


def test_replace_round_keeps_the_file_mode(tmp_path):
    import os
    import stat
    path = tmp_path / "h.jsonl"
    h = History(path)
    _cand(h, 1, [_edit(1, "skill")], 0.01)
    os.chmod(path, 0o644)
    h.replace_round(1)
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o644


def test_replace_round_drops_that_rounds_edits_atomically(tmp_path):
    h = History(tmp_path / "h.jsonl")
    _cand(h, 1, [_edit(1, "skill")], 0.01)
    _cand(h, 2, [_edit(2, "prompt"), _edit(3, "memory")], -0.01)
    h.append({"t": 2, "note": "round summary"})                  # no edit_id: kept
    h.replace_round(2, keep=lambda r: r.get("component") == "memory")
    assert [(r.get("t"), r.get("edit_id")) for r in h.records()] == [
        (1, "E1"), (2, "E3"), (2, None)]
    assert [p.name for p in tmp_path.iterdir()] == ["h.jsonl"]   # no temp left behind


def test_a_missing_file_is_an_empty_history(tmp_path):
    h = History(tmp_path / "nope" / "h.jsonl")
    assert h.records() == [] and h.tried() == set() and h.prune_set(5, 3) == []


# ---------------------------------------------------------------------------
# History: yield, prune set, render
# ---------------------------------------------------------------------------


def _fixture(tmp_path):
    h = History(tmp_path / "h.jsonl")
    _cand(h, 1, [_edit(1, "skill", "pdf skill")], 0.05, accepted=True)
    _cand(h, 2, [_edit(2, "memory")], -0.02)
    _cand(h, 6, [_edit(3, "prompt")], 0.0, accepted=True)
    _cand(h, 7, [_edit(4, "skill")], -0.01)
    return h


def test_yield_is_the_recent_best_gain_and_minus_inf_outside_the_window(tmp_path):
    g = _fixture(tmp_path).yield_g(t=8, n_prune=3)
    assert g["skill"] == -0.01            # t=1 gain is outside the window
    assert g["prompt"] == 0.0
    assert g["memory"] == -math.inf       # tried, nothing recent


def test_prune_set_lists_non_positive_yield_with_its_accepted_machinery(tmp_path):
    h = _fixture(tmp_path)
    b = {e["component"]: e for e in h.prune_set(t=8, n_prune=3)}
    assert set(b) == {"memory", "prompt", "skill"}
    assert b["memory"]["recent_best_gain"] is None
    assert [e["edit_id"] for e in b["skill"]["accepted_edits_in_incumbent"]] == ["E1"]
    assert b["skill"]["recent_best_gain"] == -0.01
    # A wider window sees the t=1 gain: skill leaves B_t.
    assert "skill" not in {e["component"] for e in h.prune_set(t=8, n_prune=10)}
    assert h.incumbent_component_counts()["skill"] == 1


def test_render_caps_unmeasured_records_and_keeps_order(tmp_path):
    h = History(tmp_path / "h.jsonl")
    _cand(h, 0, [_edit(0, "skill")], 0.01)
    for t in range(1, 8):
        _cand(h, t, [_edit(t, "memory")], None, outcome="CRITIC_REJECT")
    view = h.render()
    assert [r["t"] for r in view] == [0, 4, 5, 6, 7]
    assert "diff" not in view[0] and "S" not in view[0]
    assert all("delta_S" not in r for r in view[1:])            # None keys dropped
    assert len(h.render(n=2)) == 2


# ---------------------------------------------------------------------------
# stall and exploration
# ---------------------------------------------------------------------------


def test_stall_flag():
    traj = [0.5, 0.51, 0.512, 0.515, 0.60]
    assert stall_flag(traj, t=1, w=2, delta=0.01) == 0          # fewer than w rounds
    assert stall_flag(traj, t=3, w=2, delta=0.01) == 1          # 0.515 - 0.51 <= 0.01
    assert stall_flag(traj, t=4, w=2, delta=0.01) == 0          # moved by 0.088
    assert stall_flag(traj, t=9, w=2, delta=0.01) == 0          # past the trajectory


def test_exploration_reserves_slots_only_when_stalled_and_something_is_untried():
    e = exploration(5, 1, {"prompt", "skill"}, m_draft=2)
    assert e["sigma"] == 1 and e["m_draft"] == 2
    assert e["untried"] == [c for c in K if c not in {"prompt", "skill"}]
    assert e["text"].startswith("STALL") and "RESERVED" in e["text"]
    assert exploration(5, 0, {"prompt"}, 2)["text"].startswith("Components not yet")
    assert exploration(5, 1, set(K), 2)["text"].startswith("Every component")
