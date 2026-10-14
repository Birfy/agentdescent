"""Structural gate for ``kind="skill_dir"`` candidates (#246)."""

import json
import os
import pickle

import pytest

from agentdescent.evolvespec import EvolveSpec, compose
from agentdescent.skillspec import (
    SKILL_RULES,
    SkillTreeValidator,
    parse_skill_frontmatter,
    skill_tree_validator,
    validate_skill_tree,
)
from agentdescent.treestrategy import FileTree

from tests.test_evolvespec import _dir_spec

_VALID = "---\nname: pdf-audit\ndescription: Audit PDFs.\n---\nBody\n"


def _skill(name="pdf-audit", description="Audit PDFs.", extra=""):
    return {"SKILL.md": f"---\nname: {name}\ndescription: {description}\n{extra}---\nBody\n"}


def _edit(ft, state, edit):
    return ft.to_diff(state, "<EDITS>" + json.dumps({"edits": [edit]}) + "</EDITS>",
                      "w0", 0, "a")


# ---------------------------------------------------------------------------
# validate_skill_tree: one test per rule
# ---------------------------------------------------------------------------


def test_a_valid_skill_has_no_errors():
    assert validate_skill_tree({"SKILL.md": _VALID}) == []
    assert validate_skill_tree({"SKILL.md": _VALID}, root_name="pdf-audit") == []


def test_missing_skill_md():
    assert validate_skill_tree({"references/a.md": "x"}) == ["missing-skill-md"]


@pytest.mark.parametrize("text", [
    "Just a body\n",
    "---\nname: pdf-audit\ndescription: never closed\n",
    "\n---\nname: a\ndescription: b\n---\n",
])
def test_no_frontmatter(text):
    assert validate_skill_tree({"SKILL.md": text}) == ["no-frontmatter"]


def test_name_missing():
    assert validate_skill_tree({"SKILL.md": "---\ndescription: d\n---\n"}) == ["name-missing"]
    assert validate_skill_tree(_skill(name="")) == ["name-missing"]


@pytest.mark.parametrize("name", [
    "PDF-audit", "pdf_audit", "pdf audit", "pdf--audit", "-pdf", "pdf-",
    "a" * 65, "PDF_Audit!!",
])
def test_name_invalid(name):
    assert validate_skill_tree(_skill(name=name)) == ["name-invalid"]


@pytest.mark.parametrize("name", ["a", "pdf", "pdf-audit-2", "a" * 64])
def test_name_valid_shapes(name):
    assert validate_skill_tree(_skill(name=name)) == []


def test_name_mismatch_only_when_root_name_given():
    assert validate_skill_tree(_skill(name="other")) == []
    assert validate_skill_tree(_skill(name="other"), root_name="pdf-audit") == ["name-mismatch"]


def test_description_rules():
    assert validate_skill_tree(_skill(description="")) == ["description-missing"]
    assert validate_skill_tree(_skill(description="x" * 1024)) == []
    assert validate_skill_tree(_skill(description="x" * 1025)) == ["description-too-long"]


def test_compatibility_rule():
    assert validate_skill_tree(_skill(extra="compatibility: " + "y" * 500 + "\n")) == []
    assert validate_skill_tree(_skill(extra="compatibility: " + "y" * 501 + "\n")) == [
        "compatibility-too-long"]


def test_multiple_violations_are_all_reported():
    errs = validate_skill_tree(_skill(name="Bad_Name", description="x" * 2000))
    assert errs == ["name-invalid", "description-too-long"]
    assert all(e in SKILL_RULES for e in errs)


def test_frontmatter_parsing_handles_quotes_block_scalars_and_crlf():
    fm = parse_skill_frontmatter(
        "---\r\nname: \"pdf-audit\"\r\ndescription: >\r\n  Audit PDFs\r\n  carefully.\r\n"
        "metadata:\r\n  author: me\r\n---\r\nBody\r\n")
    assert fm["name"] == "pdf-audit"
    assert fm["description"] == "Audit PDFs carefully."
    assert parse_skill_frontmatter("no header") is None
    assert validate_skill_tree({"SKILL.md": "---\nname: 'pdf-audit'\ndescription: |\n"
                                            "  multi\n  line\n---\n"}) == []


# ---------------------------------------------------------------------------
# FileTree(validators=...)
# ---------------------------------------------------------------------------


def test_the_issue_repro_proposals_are_rejected_with_distinct_reasons():
    ft = FileTree({"SKILL.md": _VALID}, validators=[skill_tree_validator(root_name="pdf-audit")])
    st = ft.initial()
    bad = (
        {"path": "SKILL.md", "delete": True},
        {"path": "SKILL.md", "content": "Just a body, frontmatter gone\n"},
        {"path": "SKILL.md",
         "content": "---\nname: PDF_Audit!!\ndescription: " + "x" * 1500 + "\n---\nBody\n"},
    )
    for edit in bad:
        assert _edit(ft, st, edit) is None
    assert dict(ft.structural_rejections) == {
        "skill-missing-skill-md": 1, "skill-no-frontmatter": 1, "skill-name-invalid": 1}


def test_a_renamed_skill_is_rejected_as_a_mismatch():
    ft = FileTree({"SKILL.md": _VALID}, validators=[skill_tree_validator(root_name="pdf-audit")])
    content = _VALID.replace("name: pdf-audit", "name: pdf-auditor")
    assert _edit(ft, ft.initial(), {"path": "SKILL.md", "content": content}) is None
    assert ft.structural_rejections["skill-name-mismatch"] == 1


def test_valid_body_and_reference_edits_still_produce_diffs():
    tree = {"SKILL.md": _VALID, "references/units.md": "SI\n"}
    ft = FileTree(tree, validators=[skill_tree_validator(root_name="pdf-audit")])
    st = ft.initial()
    d = _edit(ft, st, {"path": "SKILL.md", "content": _VALID.replace("Body", "Better body")})
    assert d is not None and list(d.ops) == ["SKILL.md"]
    d = _edit(ft, st, {"path": "references/units.md", "content": "SI first\n"})
    assert d is not None and list(d.ops) == ["references/units.md"]
    d = _edit(ft, st, {"path": "references/units.md", "delete": True})
    assert d is not None and d.ops == {"references/units.md": None}
    assert not ft.structural_rejections


def test_a_filetree_without_validators_is_unchanged():
    ft = FileTree({"SKILL.md": _VALID})
    assert _edit(ft, ft.initial(), {"path": "SKILL.md", "delete": True}) is not None
    assert not ft.structural_rejections


def test_validators_and_counter_survive_pickling():
    ft = FileTree({"SKILL.md": _VALID}, validators=[SkillTreeValidator("pdf-audit")])
    assert _edit(ft, ft.initial(), {"path": "SKILL.md", "delete": True}) is None
    clone = pickle.loads(pickle.dumps(ft))
    assert clone.structural_rejections == ft.structural_rejections
    assert _edit(clone, clone.initial(), {"path": "SKILL.md", "delete": True}) is None
    assert clone.structural_rejections["skill-missing-skill-md"] == 2


# ---------------------------------------------------------------------------
# EvolveSpec wiring
# ---------------------------------------------------------------------------


def _write_skill_md(spec, text):
    with open(os.path.join(spec.target, "SKILL.md"), "w") as fh:
        fh.write(text)


def test_skill_dir_spec_wires_the_gate_with_the_initial_name(tmp_path):
    spec = _dir_spec(str(tmp_path))
    _write_skill_md(spec, "---\nname: shout-skill\ndescription: Shout things.\n---\nBody\n")
    comp = compose(spec)
    ft = comp.kwargs["strategy"]
    assert [repr(v) for v in ft.validators] == ["SkillTreeValidator(root_name='shout-skill')"]
    assert not any("structural gate" in n for n in comp.notes)
    assert _edit(ft, ft.initial(), {"path": "SKILL.md", "delete": True}) is None
    assert _edit(ft, ft.initial(), {"path": "rules.md", "content": "MODE: reverse\n"}) is not None


def test_a_nonconforming_initial_skill_still_runs_with_a_note(tmp_path):
    spec = _dir_spec(str(tmp_path))          # rules.md only, no SKILL.md
    comp = compose(spec)
    ft = comp.kwargs["strategy"]
    assert ft.validators == []
    assert any("missing-skill-md" in n and "structural gate" in n for n in comp.notes)
    assert _edit(ft, ft.initial(), {"path": "rules.md", "content": "MODE: reverse\n"}) is not None


def test_validate_skill_false_opts_out(tmp_path):
    spec = _dir_spec(str(tmp_path), validate_skill=False)
    _write_skill_md(spec, "---\nname: shout-skill\ndescription: Shout things.\n---\nBody\n")
    comp = compose(spec)
    assert comp.kwargs["strategy"].validators == []
    assert not any("structural gate" in n for n in comp.notes)
    assert spec.to_dict()["validate_skill"] is False
    assert EvolveSpec.from_dict(spec.to_dict()).validate_skill is False
