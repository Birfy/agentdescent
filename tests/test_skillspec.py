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


# ---------------------------------------------------------------------------
# Header shapes a YAML loader would refuse, and ones it would accept
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("line", [
    'description: "never closed',
    "description: 'never closed",
    "description: [a, b]",
    "description: {a: b}",
    "description: Use when: the user asks",
    "description: Use when:",
    'name: "pdf-audit',
])
def test_frontmatter_a_yaml_loader_would_refuse_is_malformed(line):
    key = line.split(":", 1)[0]
    lines = {"name": "name: pdf-audit", "description": "description: Audit PDFs."}
    lines[key] = line
    text = "---\n" + lines["name"] + "\n" + lines["description"] + "\n---\nBody\n"
    assert "frontmatter-malformed" in validate_skill_tree({"SKILL.md": text})


@pytest.mark.parametrize("text", [
    "---\nname: pdf-audit  # main\ndescription: Audit PDFs.\n---\n",
    '---\nname: pdf-audit\ndescription: "Use when: auditing"  # quoted\n---\n',
    "---\nname: pdf-audit\ndescription: >\n  Use when: folded is fine\n---\n",
    "\ufeff---\nname: pdf-audit\ndescription: Audit PDFs.\n---\n",
    "---\nname: pdf-audit\ndescription: Audit PDFs # and say why\n---\n",
])
def test_comments_bom_and_quoted_colons_are_accepted(text):
    assert validate_skill_tree({"SKILL.md": text}, root_name="pdf-audit") == []


def test_every_shipped_skill_md_still_validates():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    found = []
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "node_modules"]
        if "SKILL.md" in files:
            with open(os.path.join(base, "SKILL.md"), encoding="utf-8") as fh:
                found.append((base, validate_skill_tree({"SKILL.md": fh.read()})))
    assert [f for f in found if f[1]] == []


# ---------------------------------------------------------------------------
# The model-merge path (ReflectiveFusion) goes through the same gate
# ---------------------------------------------------------------------------


_SHOUT = "---\nname: shout-skill\ndescription: Shout things.\n---\n"


def _contested_skill_merge(tmp_path, merged_text):
    from agentdescent.evolution import EvolvingArtifact
    from agentdescent.fusion import ReflectiveFusion

    spec = _dir_spec(str(tmp_path))
    _write_skill_md(spec, _SHOUT + "Body\n")
    comp = compose(spec)
    assert isinstance(comp.kwargs["policies"].fusion, ReflectiveFusion)
    ft = comp.kwargs["strategy"]
    st = ft.initial()
    d1 = _edit(ft, st, {"path": "SKILL.md", "content": _SHOUT + "Body A\n"})
    d2 = _edit(ft, st, {"path": "SKILL.md", "content": _SHOUT + "Body B\n"})
    assert d1 is not None and d2 is not None
    fusion = ReflectiveFusion(lambda prompt: merged_text)
    art = EvolvingArtifact("a", state=st, strategy=ft)
    return ft, fusion, art, [d1, d2]


def test_a_synthesised_skill_md_that_drops_the_frontmatter_is_refused(tmp_path):
    ft, fusion, art, diffs = _contested_skill_merge(
        tmp_path, "Merged body: shout louder, no frontmatter")
    assert fusion._union_of(art, diffs) is None
    assert ft.structural_rejections["skill-no-frontmatter"] == 1


def test_a_refused_synthesis_falls_back_to_ranking_and_never_commits_it(tmp_path):
    ft, fusion, art, diffs = _contested_skill_merge(
        tmp_path, "Merged body: shout louder, no frontmatter")

    class _Verifier:
        def cheap_eval(self, candidate):
            return 1.0 if "Body B" in candidate.state.get("SKILL.md", "") else 0.5

    fusion.bind(_Verifier())
    chosen, applied, fused = fusion.select(art, diffs)
    assert fusion.trials[-1].reason == "synthesis-failed"
    assert chosen in diffs and not fused
    assert validate_skill_tree(applied.state, root_name="shout-skill") == []


def test_a_valid_synthesised_skill_md_still_merges(tmp_path):
    ft, fusion, art, diffs = _contested_skill_merge(tmp_path, _SHOUT + "Body A and B\n")
    union = fusion._union_of(art, diffs)
    assert union is not None and union.ops == {"SKILL.md": _SHOUT + "Body A and B"}
    assert not ft.structural_rejections


def test_check_candidate_reports_the_reason_and_counts_it():
    ft = FileTree({"SKILL.md": _VALID}, validators=[SkillTreeValidator("pdf-audit")])
    assert ft.check_candidate(ft.initial(), {"references/a.md": "x"}) is None
    assert ft.check_candidate(ft.initial(), {"SKILL.md": None}) == "skill-missing-skill-md"
    assert ft.structural_rejections == {"skill-missing-skill-md": 1}
    assert FileTree({"SKILL.md": "x"}).check_candidate({}, {"SKILL.md": None}) is None


def test_a_name_that_differs_from_the_target_directory_gets_a_note(tmp_path):
    spec = _dir_spec(str(tmp_path))
    dirname = os.path.basename(os.path.normpath(spec.target))
    _write_skill_md(spec, "---\nname: other-name\ndescription: Shout things.\n---\nBody\n")
    notes = compose(spec).notes
    assert any("'other-name'" in n and repr(dirname) in n for n in notes)
    _write_skill_md(spec, f"---\nname: {dirname}\ndescription: Shout things.\n---\nBody\n")
    if validate_skill_tree({"SKILL.md": f"---\nname: {dirname}\ndescription: d\n---\n"}) == []:
        assert not any("differs from the target directory" in n for n in compose(spec).notes)


# ---------------------------------------------------------------------------
# Scalar syntax and type: a header YAML would refuse, or load as non-text
# ---------------------------------------------------------------------------


_BAD_SCALARS = [
    "null", "~", "Null", "true", "yes", "off", "123", "1.5", ".inf", "2024-01-02",
    '"bad\\q"', '"\\xZZ"', "'it's'", '"a" b', "@x", "*alias", "!tag x", "- a",
]


@pytest.mark.parametrize("value", _BAD_SCALARS)
def test_non_string_or_unscannable_scalars_are_malformed(value):
    for key in ("name", "description"):
        lines = {"name": "name: pdf-audit", "description": "description: Audit PDFs."}
        lines[key] = f"{key}: {value}"
        text = "---\n" + lines["name"] + "\n" + lines["description"] + "\n---\nBody\n"
        assert "frontmatter-malformed" in validate_skill_tree({"SKILL.md": text}), (key, value)


@pytest.mark.parametrize("value, decoded", [
    ("Shout things. # Usage:", "Shout things."),
    ('"ok\\n"', "ok\n"),
    ('"caf\\u00e9"', "caf\u00e9"),
    ("'it''s'", "it's"),
    ('"a" # trailing comment', "a"),
    ("a:b", "a:b"),
    ("2024-1-1", "2024-1-1"),
])
def test_valid_string_scalars_are_accepted_and_decoded(value, decoded):
    text = f"---\nname: pdf-audit\ndescription: {value}\n---\nBody\n"
    assert validate_skill_tree({"SKILL.md": text}) == []
    assert parse_skill_frontmatter(text)["description"] == decoded


def test_a_bare_comment_value_is_an_empty_description():
    text = "---\nname: pdf-audit\ndescription: # todo\n---\n"
    assert validate_skill_tree({"SKILL.md": text}) == ["description-missing"]


@pytest.mark.parametrize("value", _BAD_SCALARS + [
    "Shout things. # Usage:", '"ok\\n"', "'it''s'", '"a" # c', "a:b", "plain text",
    "Use when:", "a: b", "[a]", '"open',
])
def test_agrees_with_pyyaml_on_the_description_scalar(value):
    yaml = pytest.importorskip("yaml")
    text = f"---\nname: pdf-audit\ndescription: {value}\n---\n"
    try:
        loaded = yaml.safe_load(text.split("---")[1])["description"]
        yaml_ok = isinstance(loaded, str) and bool(loaded.strip())
    except yaml.YAMLError:
        yaml_ok = False
    assert (validate_skill_tree({"SKILL.md": text}) == []) is yaml_ok


@pytest.mark.parametrize("header", [
    "---\nname: shout-skill\ndescription: null\n---\n",
    '---\nname: shout-skill\ndescription: "bad\\q"\n---\n',
])
def test_invalid_scalars_are_refused_in_diffs_and_synthesised_merges(tmp_path, header):
    ft, fusion, art, diffs = _contested_skill_merge(tmp_path, header + "Merged body\n")
    assert fusion._union_of(art, diffs) is None
    assert _edit(ft, ft.initial(), {"path": "SKILL.md", "content": header + "Body\n"}) is None
    assert ft.structural_rejections == {"skill-frontmatter-malformed": 2}


def test_a_commented_initial_skill_keeps_its_gate(tmp_path):
    spec = _dir_spec(str(tmp_path))
    _write_skill_md(spec, "---\nname: shout-skill\ndescription: Shout things. # Usage:\n---\nBody\n")
    comp = compose(spec)
    ft = comp.kwargs["strategy"]
    assert [repr(v) for v in ft.validators] == ["SkillTreeValidator(root_name='shout-skill')"]
    assert not any("structural gate" in n for n in comp.notes)
    assert _edit(ft, ft.initial(), {"path": "SKILL.md", "delete": True}) is None
