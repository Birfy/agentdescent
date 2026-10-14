"""`FileTree`: the artifact is a directory, so a state key is a file path.

Two things are load-bearing here and neither exists for a text strategy: the
multi-file proposal protocol (`parse_edits`) and `frozen`, which is L0 expressed
in paths rather than in artifact ids.
"""

import pytest

from agentdescent.evolution import EvolvingArtifact, Task
from agentdescent.filetree import canonical
from agentdescent.treestrategy import FileTree, parse_edits, parse_proposal, tree_reflector


def _tree(**files):
    return FileTree(files or {"SKILL.md": "old"})


# ---------------------------------------------------------------------------
# the proposal protocol
# ---------------------------------------------------------------------------


def test_parse_edits_reads_the_documented_block():
    proposal = """Sure, here is the fix.
<EDITS>
{"rationale": "r", "edits": [{"path": "SKILL.md", "content": "new body"}]}
</EDITS>
Hope that helps."""
    assert parse_edits(proposal) == {"SKILL.md": "new body"}


def test_parse_edits_tolerates_a_fenced_block_or_a_bare_object():
    fenced = '```json\n{"edits": [{"path": "a.md", "content": "x"}]}\n```'
    bare = '{"edits": [{"path": "a.md", "content": "x"}]}'
    single = '{"path": "a.md", "content": "x"}'
    for text in (fenced, bare, single):
        assert parse_edits(text) == {"a.md": "x"}


def test_parse_edits_returns_nothing_rather_than_raising_on_junk():
    # a reflector that ignores the protocol is a quality problem the run absorbs
    # and counts, not a crash.
    for junk in ("", "no json here", "<EDITS>{broken</EDITS>", "[]", "{}", None, 17):
        assert parse_edits(junk) == {}


def test_parse_edits_drops_hostile_paths_and_keeps_the_rest():
    proposal = ('<EDITS>{"edits": [{"path": "../../etc/passwd", "content": "x"},'
                '{"path": "ok.md", "content": "y"}]}</EDITS>')
    assert parse_edits(proposal) == {"ok.md": "y"}


def test_parse_edits_understands_deletion():
    assert parse_edits('{"edits": [{"path": "old.md", "delete": true}]}') == {"old.md": None}


def test_edit_protocol_formats_without_leaking_braces():
    from agentdescent.treestrategy import EDIT_PROTOCOL

    text = EDIT_PROTOCOL.format(max_files=2, editable="**", frozen="tests/**")
    assert '"rationale"' in text and "{max_files}" not in text


# ---------------------------------------------------------------------------
# to_diff: what becomes a gradient
# ---------------------------------------------------------------------------


def _diff(strategy, state, proposal):
    return strategy.to_diff(state, proposal, "w0", 1, "art")


def _edits(*pairs):
    items = ", ".join('{"path": "%s", "content": "%s"}' % p for p in pairs)
    return "<EDITS>{\"edits\": [%s]}</EDITS>" % items


def test_a_changed_file_becomes_one_op():
    s = _tree(**{"SKILL.md": "old"})
    d = _diff(s, s.initial(), _edits(("SKILL.md", "new")))
    assert d.ops == {"SKILL.md": "new"}


def test_an_unchanged_file_is_not_a_diff():
    s = _tree(**{"SKILL.md": "same"})
    assert _diff(s, s.initial(), _edits(("SKILL.md", "same"))) is None


def test_frozen_paths_cannot_be_proposed():
    s = FileTree({"SKILL.md": "a", "tests/t.py": "assert real"}, frozen=["tests/**"])
    assert _diff(s, s.initial(), _edits(("tests/t.py", "assert True"))) is None
    # and a mixed proposal keeps only the writable half
    d = _diff(s, s.initial(), _edits(("tests/t.py", "assert True"), ("SKILL.md", "b")))
    assert d.ops == {"SKILL.md": "b"}


def test_non_editable_paths_are_ignored():
    s = FileTree({"SKILL.md": "a", "notes.txt": "n"}, editable=["*.md"])
    assert _diff(s, s.initial(), _edits(("notes.txt", "x"))) is None


def test_a_runaway_proposal_is_dropped_whole():
    s = FileTree({f"f{i}.md": "x" for i in range(5)}, max_files_per_diff=2)
    proposal = _edits(*[(f"f{i}.md", "y") for i in range(4)])
    assert _diff(s, s.initial(), proposal) is None


def test_oversized_content_never_reaches_the_aggregator():
    s = FileTree({"SKILL.md": "a"}, max_file_bytes=50)
    assert _diff(s, s.initial(), _edits(("SKILL.md", "x" * 200))) is None


def test_deletion_is_an_op_that_pops_the_key():
    s = FileTree({"SKILL.md": "a", "stale.md": "b"})
    d = _diff(s, s.initial(), '{"edits": [{"path": "stale.md", "delete": true}]}')
    assert d.ops == {"stale.md": None}
    art = EvolvingArtifact("art", s.initial(), strategy=s).apply(d)
    assert set(art.state) == {"SKILL.md"}          # popped, not stored as None


def test_deleting_a_file_that_is_not_there_is_not_a_diff():
    s = FileTree({"SKILL.md": "a"})
    assert _diff(s, s.initial(), '{"edits": [{"path": "ghost.md", "delete": true}]}') is None


def test_diff_ids_are_stable_for_the_same_edit():
    s = _tree()
    a = _diff(s, s.initial(), _edits(("SKILL.md", "new")))
    b = _diff(s, s.initial(), _edits(("SKILL.md", "new")))
    assert a.diff_id == b.diff_id    # identical proposals dedupe in the aggregator


# ---------------------------------------------------------------------------
# key space / tensor parallelism
# ---------------------------------------------------------------------------


def test_keys_are_the_editable_paths_plus_declared_future_ones():
    s = FileTree({"SKILL.md": "a", "tests/t.py": "b"}, frozen=["tests/**"],
                 planned_paths=["references/new.md"])
    assert s.keys() == ["SKILL.md", "references/new.md"]


def test_render_is_the_lossless_serialisation():
    s = _tree(**{"a.md": "1", "b.md": "2"})
    assert s.render(s.initial()) == canonical(s.initial())


# ---------------------------------------------------------------------------
# the reflector
# ---------------------------------------------------------------------------


def test_tree_reflector_shows_editable_content_and_the_protocol():
    seen = {}

    def fake(prompt):
        seen["prompt"] = prompt
        return "<EDITS>{\"edits\": [{\"path\": \"SKILL.md\", \"content\": \"v2\"}]}</EDITS>"

    s = FileTree({"SKILL.md": "current body", "tests/t.py": "secret"},
                 frozen=["tests/**"])
    propose = tree_reflector(fake, strategy=s)
    out = propose(s.render(s.initial()), Task("t1", "the question", {"gold": "42"}),
                  "wrong answer", 0.0)
    assert "current body" in seen["prompt"]        # the file it may edit
    assert "the question" in seen["prompt"] and "42" in seen["prompt"]
    assert "<EDITS>" in seen["prompt"]             # the protocol
    assert "secret" not in seen["prompt"]          # frozen file body is not shown
    assert parse_edits(out) == {"SKILL.md": "v2"}


def test_tree_reflector_bounds_what_it_puts_in_the_prompt():
    seen = {}

    def fake(prompt):
        seen["prompt"] = prompt
        return ""

    s = FileTree({f"f{i}.md": "y" * 5_000 for i in range(20)})
    tree_reflector(fake, strategy=s, max_context_chars=6_000)(
        s.render(s.initial()), Task("t", "q"), "out", 0.0)
    # the listing names every file, but only a couple of bodies are inlined.
    assert seen["prompt"].count("--- f") <= 2
    assert "f19.md" in seen["prompt"]              # still listed


# ---------------------------------------------------------------------------
# rationale: paid for in tokens, kept for lineage (#247)
# ---------------------------------------------------------------------------


def test_parse_proposal_returns_edits_and_rationale():
    block = """<EDITS>
{"rationale": "  fix the units   and retry  ",
 "edits": [{"path": "SKILL.md", "content": "new body"}]}
</EDITS>"""
    edits, rationale = parse_proposal(block)
    assert edits == {"SKILL.md": "new body"}
    assert rationale == "fix the units and retry"
    # wrappers: fenced and bare
    fenced = '```json\n{"rationale": "why", "edits": [{"path": "a.md", "content": "x"}]}\n```'
    bare = '{"rationale": "why", "edits": [{"path": "a.md", "content": "x"}]}'
    for text in (fenced, bare):
        e, r = parse_proposal(text)
        assert e == {"a.md": "x"} and r == "why"


def test_parse_proposal_non_string_or_missing_rationale_is_empty():
    assert parse_proposal('{"rationale": 17, "edits": [{"path": "a.md", "content": "x"}]}')[1] == ""
    assert parse_proposal('{"edits": [{"path": "a.md", "content": "x"}]}')[1] == ""
    assert parse_proposal("not json") == ({}, "")


def test_parse_edits_is_byte_for_byte_the_edits_half():
    """Existing callers of parse_edits must be unaffected (#247 acceptance)."""
    proposal = """Sure.
<EDITS>
{"rationale": "r", "edits": [{"path": "SKILL.md", "content": "new body"}]}
</EDITS>"""
    assert parse_edits(proposal) == parse_proposal(proposal)[0] == {"SKILL.md": "new body"}


def test_identical_edits_share_diff_id_even_with_different_rationales():
    s = _tree(**{"SKILL.md": "old"})
    a = _diff(s, s.initial(),
              '<EDITS>{"rationale": "first idea", "edits": '
              '[{"path": "SKILL.md", "content": "new"}]}</EDITS>')
    b = _diff(s, s.initial(),
              '<EDITS>{"rationale": "second idea", "edits": '
              '[{"path": "SKILL.md", "content": "new"}]}</EDITS>')
    assert a is not None and b is not None
    assert a.diff_id == b.diff_id
    assert a.rationale == "first idea" and b.rationale == "second idea"


def test_filetree_to_diff_carries_the_rationale():
    s = _tree(**{"SKILL.md": "old"})
    d = _diff(s, s.initial(),
              '<EDITS>{"rationale": "clarify the units", "edits": '
              '[{"path": "SKILL.md", "content": "new"}]}</EDITS>')
    assert d.rationale == "clarify the units"


def test_accepted_filetree_diff_puts_rationale_in_ledger_commit(tmp_path):
    """git log on the ledger must carry *why*, not just ids (#247)."""
    import subprocess
    from agentdescent.aggregator import merge_commit_message
    from agentdescent.evolution import EvolvingArtifact
    from agentdescent.ledger import Ledger

    s = FileTree({"SKILL.md": "old"})
    d = s.to_diff(
        s.initial(),
        '<EDITS>{"rationale": "units were wrong; check SI first",'
        ' "edits": [{"path": "SKILL.md", "content": "new"}]}</EDITS>',
        "w0", 1, "art")
    assert d is not None and d.rationale

    def serialize(a):
        return {"state": dict(a.state)}

    def deserialize(aid, v, payload):
        return EvolvingArtifact(aid, payload.get("state", {}), v)

    led = Ledger(str(tmp_path / "repo"), serialize, deserialize)
    art = EvolvingArtifact("art", s.initial(), strategy=s)
    led.register(art)
    base = led.head_version(Ledger.DEV)
    candidate = art.apply(d)
    msg = merge_commit_message(d, "art")
    led.commit(candidate, base, branch=Ledger.DEV, message=msg)
    body = subprocess.check_output(
        ["git", "-C", led.repo_path, "log", "-1", "--format=%B"],
        text=True)
    assert f"merge {d.diff_id} -> art" in body
    assert "units were wrong; check SI first" in body


def test_merge_commit_message_without_rationale_is_unchanged():
    from agentdescent.aggregator import merge_commit_message
    from agentdescent.evolvable import Diff
    d = Diff(diff_id="w0:abcd:1", target="art", ops={"k": "v"}, author="w0")
    assert merge_commit_message(d, "art") == "merge w0:abcd:1 -> art"


def test_fuse_diffs_concatenates_parent_rationales():
    from agentdescent.aggregator import fuse_diffs
    from agentdescent.evolvable import Diff
    a = Diff("a", "t", {"x": "1"}, author="w0", rationale="fix x")
    b = Diff("b", "t", {"y": "2"}, author="w1", rationale="add y")
    c = Diff("c", "t", {"z": "3"}, author="w2", rationale="fix x")  # dedupe
    fused = fuse_diffs([a, b, c])
    assert fused.rationale == "fix x; add y"


def test_rationale_strips_nul_and_lone_surrogates_so_git_commit_survives(tmp_path):
    """A reflector may emit an escaped NUL or lone surrogate in JSON; neither
    may reach ``git commit -m`` (embedded null byte / UnicodeEncodeError)."""
    import subprocess
    from agentdescent.aggregator import merge_commit_message
    from agentdescent.evolution import EvolvingArtifact
    from agentdescent.ledger import Ledger

    edits, rationale = parse_proposal(
        '{"rationale": "why\\u0000oops\\ud800 \\u0007bell",'
        ' "edits": [{"path": "SKILL.md", "content": "new"}]}')
    assert edits == {"SKILL.md": "new"}
    assert rationale == "why oops bell"

    s = FileTree({"SKILL.md": "old"})
    d = s.to_diff(
        s.initial(),
        '<EDITS>{"rationale": "nul\\u0000inside", "edits": '
        '[{"path": "SKILL.md", "content": "new"}]}</EDITS>',
        "w0", 1, "art")
    assert d is not None and "\x00" not in d.rationale

    # A hand-built Diff with raw control characters is normalized at commit time.
    from dataclasses import replace
    d = replace(d, rationale="raw\x00nul\ud800here")
    msg = merge_commit_message(d, "art")
    assert "\x00" not in msg and "\ud800" not in msg

    led = Ledger(str(tmp_path / "repo"), lambda a: {"state": dict(a.state)},
                 lambda aid, v, p: EvolvingArtifact(aid, p.get("state", {}), v))
    art = EvolvingArtifact("art", s.initial(), strategy=s)
    led.register(art)
    base = led.head_version(Ledger.DEV)
    led.commit(art.apply(d), base, branch=Ledger.DEV, message=msg)
    body = subprocess.check_output(
        ["git", "-C", led.repo_path, "log", "-1", "--format=%B"], text=True)
    assert "raw nulhere" in body


def test_parse_proposal_is_exported_from_the_package_root():
    import agentdescent
    assert agentdescent.parse_proposal is parse_proposal
    assert "parse_proposal" in agentdescent.__all__
