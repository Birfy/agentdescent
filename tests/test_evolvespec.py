"""An `evolve()` call as data: does the spec compose the call the quickstarts write?

Offline throughout. The agent is the same tiny subprocess program
`test_dir_evolution.py` uses, so the directory kinds really do stage a workspace
and read the candidate off disk; the reflector is a stub that proposes the fix.
"""

import json
import os
import re
import sys

import pytest

from agentdescent import (
    AggregatorConfig, EvolveSpec, Policies, SpecError, Task, compose, load_spec,
)
from agentdescent.advantage import AdvantageAcceptance, AdvantageConflict
from agentdescent.evolvespec import (
    KIND_ROWS, KINDS, SHORT_REFS, build_policies, estimate, to_ref,
)
from agentdescent.evolution import SingleSlot
from agentdescent.fusion import KeepContradictions, ReflectiveFusion
from agentdescent.governance import HARNESS_BLAST_RADIUS, SKILL_BLAST_RADIUS
from agentdescent.sampling import DifficultyWeighted
from agentdescent.selection import Beam
from agentdescent.staleness import FullStaleness
from agentdescent.treestrategy import FileTree
from agentdescent.workspec import Ref

# ---------------------------------------------------------------------------
# fixtures: a skill directory, a dataset, a stub agent and reflector
# ---------------------------------------------------------------------------

_SKILL = "shout-skill"
_AGENT_SRC = (
    "import os,sys;"
    "q=sys.argv[1];"
    f"p=os.path.join('.claude','skills','{_SKILL}','rules.md');"
    "t=open(p).read() if os.path.exists(p) else '';"
    "print(q[::-1] if 'MODE: reverse' in t else q)"
)
_WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
          "hotel", "india", "juliet", "kilo", "lima"]


def stub_reflect(prompt: str) -> str:
    """Module-level so a spec can name it: proposes MODE: reverse."""
    m = re.search(r"--- rules\.md ---\n(.*?)\nTASK THE AGENT WAS GIVEN:", prompt, re.DOTALL)
    body = (m.group(1) if m else "").strip()
    new = body.replace("MODE: forward", "MODE: reverse") or "MODE: reverse"
    return "<EDITS>" + json.dumps({"rationale": "reverse", "edits": [
        {"path": "rules.md", "content": new + "\n"}]}) + "</EDITS>"


def text_reflect(prompt: str) -> str:
    return "Always reply with the word reversed."


def _skill_dir(tmp):
    path = os.path.join(tmp, _SKILL)
    os.makedirs(path)
    with open(os.path.join(path, "rules.md"), "w") as fh:
        fh.write("MODE: forward\n")
    return path


def _rows_file(tmp, ext=".jsonl"):
    rows = [{"prompt": w, "gold": w[::-1]} for w in _WORDS]
    path = os.path.join(tmp, "cases" + ext)
    with open(path, "w") as fh:
        if ext == ".jsonl":
            fh.write("\n".join(json.dumps(r) for r in rows) + "\n")
        elif ext == ".json":
            json.dump(rows, fh)
        else:
            fh.write("prompt,gold\n" + "\n".join(f"{r['prompt']},{r['gold']}" for r in rows) + "\n")
    return path


def _agent_ref():
    return {"ref": "cli_agent", "command": [sys.executable, "-c", _AGENT_SRC]}


def _dir_spec(tmp, **extra):
    d = dict(kind="skill_dir", target=_skill_dir(tmp),
             data={"path": _rows_file(tmp), "prompt": "prompt", "gold": "gold"},
             score="exact", agent=_agent_ref(),
             reflect={"ref": "tests.test_evolvespec:stub_reflect", "call": False},
             allow=["tests."], prompt_template="{prompt}",   # the stub agent echoes verbatim
             evolve={"rounds": 3, "n_workers": 2, "seed": 0})
    d.update(extra)
    return EvolveSpec.from_dict(d)


# ---------------------------------------------------------------------------
# the table and the refs
# ---------------------------------------------------------------------------


def test_every_kind_has_a_row_in_the_composition_table():
    assert set(KIND_ROWS) == set(KINDS)
    for row in KIND_ROWS.values():
        assert set(row) == {"strategy", "run", "propose", "reward", "blast_radius"}


def test_every_short_ref_resolves_to_a_public_factory():
    for name, target in SHORT_REFS.items():
        assert ":" in target
        Ref(target, call=False).resolve()   # imports and finds the attribute; does not call


def test_to_ref_accepts_the_three_spellings_and_nests():
    assert to_ref("claude_code", where="x") == Ref("agentdescent.agents:claude_code")
    assert to_ref("pkg.mod:fn", where="x") == Ref("pkg.mod:fn")
    r = to_ref({"ref": "reflective_merge", "complete": {"ref": "echo"}}, where="x")
    assert r.target == "agentdescent.fusion:reflective_merge"
    assert r.config["complete"] == Ref("agentdescent.agents:echo")
    assert to_ref({"ref": "pkg.mod:fn", "call": False}, where="x").call is False
    with pytest.raises(SpecError, match="not a known short name"):
        to_ref("nonsense", where="agent")
    with pytest.raises(SpecError):
        to_ref(42, where="agent")


def test_refs_outside_the_allowlist_are_refused_before_import():
    spec = EvolveSpec(kind="text", target="hi", data={"inline": [{"prompt": "q", "gold": "a"}]},
                      agent="os:system")
    with pytest.raises(SpecError, match="outside the allowed prefixes"):
        compose(spec)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_from_dict_rejects_unknown_fields_and_missing_required():
    with pytest.raises(SpecError, match="unknown spec field"):
        EvolveSpec.from_dict({"kind": "text", "target": "x", "data": {}, "colour": 1})
    with pytest.raises(SpecError, match="needs 'data'"):
        EvolveSpec.from_dict({"kind": "text", "target": "x"})


def test_round_trips_through_json(tmp_path):
    spec = _dir_spec(str(tmp_path), frozen=["tests/**"])
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec.to_dict()))
    again = load_spec(str(path))
    assert again.to_dict() == spec.to_dict()
    assert isinstance(again.to_dict()["frozen"], list)


@pytest.mark.parametrize("bad, match", [
    (dict(kind="nope"), "kind must be one of"),
    (dict(agent=None), "agent is required"),
    (dict(score="fuzzy"), "unknown score"),
    (dict(data={"inline": [{"prompt": "q"}], "path": "x"}), "exactly one of"),
    (dict(policies={"colour": "Beam"}), "unknown slot"),
    (dict(agg_config={"colour": 1}), "unknown field"),
    (dict(policies={"reflective_merge": {"ref": "reflective_merge", "complete": "echo"},
                    "fusion": "DefaultFusion"}), "do not also set them"),
])
def test_compose_names_the_field_that_is_wrong(tmp_path, bad, match):
    spec = _dir_spec(str(tmp_path), **bad)
    with pytest.raises(SpecError, match=match):
        compose(spec)


def test_a_plain_completion_cannot_evolve_a_directory(tmp_path):
    spec = _dir_spec(str(tmp_path), agent="echo")
    with pytest.raises(SpecError, match="WorkspaceAgent"):
        compose(spec)


def test_text_template_must_carry_both_slots():
    spec = EvolveSpec(kind="text", target="x", data={"inline": [{"prompt": "q", "gold": "a"}]},
                      agent="echo", template="{skill}")
    with pytest.raises(SpecError, match="template"):
        compose(spec)


# ---------------------------------------------------------------------------
# composition: the quickstart call, field for field
# ---------------------------------------------------------------------------


def test_skill_dir_composes_the_directory_quickstart(tmp_path):
    comp = compose(_dir_spec(str(tmp_path)))
    k = comp.kwargs
    assert isinstance(k["strategy"], FileTree)
    assert k["blast_radius"] == SKILL_BLAST_RADIUS
    assert k["self_verify"] is False and k["cheap_eval_tasks"] == 4
    assert k["n_workers"] == 2 and k["max_concurrency"] == 2
    assert k["held_out_frac"] == 0.3 and k["rounds"] == 3
    assert k["artifact_id"] == _SKILL
    assert isinstance(k["agg_config"], AggregatorConfig)
    assert k["agg_config"].batch_trigger == 2
    assert comp.tree == {"rules.md": "MODE: forward\n"}
    assert len(comp.tasks) == len(_WORDS)
    assert "policies" not in k                     # the empty bundle is the shipped run


def test_agent_dir_is_the_same_call_at_the_harness_layer(tmp_path):
    comp = compose(_dir_spec(str(tmp_path), kind="agent_dir"))
    assert comp.kwargs["blast_radius"] == HARNESS_BLAST_RADIUS


def test_agent_code_gates_the_reward_and_freezes_the_tests(tmp_path):
    from agentdescent.runners import TEST_FAILURE_MARKER

    spec = _dir_spec(str(tmp_path), kind="agent_code", entrypoint=["python", "main.py"])
    comp = compose(spec)
    assert comp.kwargs["blast_radius"] == HARNESS_BLAST_RADIUS
    assert list(comp.kwargs["strategy"].frozen) == ["tests/**", "conftest.py"]
    task = Task(id="t", prompt="q", meta={"gold": "q"})
    assert comp.reward(task, TEST_FAILURE_MARKER + " (tests)") == 0.0
    assert comp.reward(task, "q") == 1.0


def test_agent_code_needs_an_entrypoint(tmp_path):
    with pytest.raises(SpecError, match="entrypoint"):
        compose(_dir_spec(str(tmp_path), kind="agent_code"))


def test_text_composes_the_dataset_quickstart():
    spec = EvolveSpec(kind="text", target="You are a helpful assistant.",
                      data={"inline": [{"q": f"item {i}", "a": str(i)} for i in range(20)],
                            "prompt": "q", "gold": "a"},
                      score="exact", agent="echo")
    comp = compose(spec)
    k = comp.kwargs
    assert isinstance(k["strategy"], SingleSlot)
    assert k["strategy"].initial_value == "You are a helpful assistant."
    assert k["blast_radius"] == SKILL_BLAST_RADIUS
    assert k["rounds"] == 8 and k["patience"] == 3 and k["target_reward"] == 0.98
    assert k["n_workers"] == 8 and k["max_concurrency"] == 8    # min(8, 14 train tasks)
    assert "self_verify" not in k                               # text keeps the engine default
    # the run puts the skill in front of the question exactly as the quickstart lambda does
    out = k["run"]("SKILL", Task(id="t", prompt="Q?"))
    assert out == "SKILL\n\nQ?"


def test_text_target_may_be_a_file(tmp_path):
    p = tmp_path / "prompt.md"
    p.write_text("Be terse.")
    spec = EvolveSpec(kind="text", target=str(p), agent="echo",
                      data={"inline": [{"prompt": "q", "gold": "a"}] * 6})
    assert compose(spec).kwargs["strategy"].initial_value == "Be terse."


def test_spec_evolve_block_overrides_kind_defaults_and_caller_overrides_win(tmp_path):
    spec = _dir_spec(str(tmp_path), evolve={"rounds": 2, "self_verify": True, "n_workers": 3})
    comp = compose(spec, rounds=9)
    assert comp.kwargs["rounds"] == 9
    assert comp.kwargs["self_verify"] is True
    assert comp.kwargs["n_workers"] == 3


def test_hooks_reach_evolve(tmp_path):
    from agentdescent.agents import Usage

    def on_round(info):
        pass

    u = Usage()
    comp = compose(_dir_spec(str(tmp_path)), usage=u, on_round=on_round,
                   repo_path=str(tmp_path / "ledger"))
    assert comp.kwargs["usage"] is u
    assert comp.kwargs["on_round"] is on_round
    assert comp.kwargs["repo_path"] == str(tmp_path / "ledger")


# ---------------------------------------------------------------------------
# data forms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ext", [".jsonl", ".json", ".csv"])
def test_local_data_files(tmp_path, ext):
    spec = _dir_spec(str(tmp_path), data={"path": _rows_file(str(tmp_path), ext)})
    comp = compose(spec)
    assert [t.prompt for t in comp.tasks] == _WORDS
    assert comp.tasks[0].meta["gold"] == "ahpla"


def test_inline_rows_and_fixtures_column(tmp_path):
    rows = [{"prompt": w, "gold": w, "fixtures": {"data.csv": "x"}} for w in _WORDS]
    comp = compose(_dir_spec(str(tmp_path), data={"inline": rows}))
    assert comp.tasks[0].meta["fixtures"] == {"data.csv": "x"}


def test_missing_data_file_is_a_spec_error(tmp_path):
    with pytest.raises(SpecError, match="does not exist"):
        compose(_dir_spec(str(tmp_path), data={"path": "/nowhere/cases.jsonl"}))


# ---------------------------------------------------------------------------
# score forms
# ---------------------------------------------------------------------------


def test_cmd_scorer_gets_the_task_on_stdin(tmp_path):
    spec = _dir_spec(str(tmp_path), score={"cmd": [
        sys.executable, "-c",
        "import sys,json;d=json.load(sys.stdin);print(1 if d['meta']['gold']==d['output'] else 0)"]})
    comp = compose(spec)
    t = Task(id="t", prompt="q", meta={"gold": "abc"})
    assert comp.reward(t, "abc") == 1.0 and comp.reward(t, "x") == 0.0


def test_ref_scorer(tmp_path):
    spec = _dir_spec(str(tmp_path), score={"ref": "agentdescent.rewards:contains"})
    t = Task(id="t", prompt="q", meta={"gold": "b"})
    assert compose(spec).reward(t, "abc") == 1.0


# ---------------------------------------------------------------------------
# policies
# ---------------------------------------------------------------------------


def test_policies_block_builds_the_bundle_from_json_scalars(tmp_path):
    spec = _dir_spec(str(tmp_path), policies={
        "selection": {"ref": "Beam", "k": 3},
        "task_sampler": "DifficultyWeighted",
        "acceptance": {"ref": "AdvantageAcceptance", "strength": 0.5},
        "conflict": "AdvantageConflict",
        "staleness": "full",
    })
    pol = build_policies(spec)
    assert isinstance(pol, Policies)
    assert isinstance(pol.selection, Beam) and pol.selection.k == 3
    assert isinstance(pol.task_sampler, DifficultyWeighted)
    assert isinstance(pol.acceptance, AdvantageAcceptance) and pol.acceptance.strength == 0.5
    assert isinstance(pol.conflict, AdvantageConflict)
    assert isinstance(pol.staleness, FullStaleness)
    assert compose(spec).kwargs["policies"] is not None


def test_reflective_merge_fills_the_pair(tmp_path):
    spec = _dir_spec(str(tmp_path), policies={
        "reflective_merge": {"ref": "reflective_merge", "complete": {"ref": "echo"}}})
    pol = build_policies(spec)
    assert isinstance(pol.fusion, ReflectiveFusion)
    assert isinstance(pol.conflict, KeepContradictions)


def test_unknown_staleness_name_is_a_spec_error(tmp_path):
    with pytest.raises(SpecError, match="staleness"):
        build_policies(_dir_spec(str(tmp_path), policies={"staleness": "sideways"}))


def test_agg_config_holds_the_numbers(tmp_path):
    comp = compose(_dir_spec(str(tmp_path), agg_config={"base_delta": 0.3}))
    assert comp.kwargs["agg_config"].base_delta == 0.3
    assert comp.kwargs["agg_config"].batch_trigger == 2      # kind default kept


# ---------------------------------------------------------------------------
# estimate, and the whole thing running
# ---------------------------------------------------------------------------


def test_estimate_counts_and_declares_its_assumptions(tmp_path):
    comp = compose(_dir_spec(str(tmp_path)))
    est = estimate(comp)
    assert est["rounds"] == 3 and est["n_workers"] == 2
    assert est["agent_calls_upper_bound"] == sum(est["calls_per_round"].values()) * 3
    assert est["usd_upper_bound"] is None and any("price" in a for a in est["assumptions"])
    assert estimate(comp, usd_per_call=0.1)["usd_upper_bound"] > 0


def test_a_skill_dir_spec_runs_end_to_end_offline(tmp_path):
    comp = compose(_dir_spec(str(tmp_path)))
    result = comp.run()
    assert result.error is None, result.error
    assert result.final_reward == 1.0
    assert "MODE: reverse" in result.state["rules.md"]
    assert result.outcomes().get("committed", 0) >= 1


def test_a_text_spec_runs_end_to_end_offline():
    # make_model() reverses the word when the skill says so; the reflector proposes exactly that
    rows = [{"prompt": w, "gold": w[::-1]} for w in _WORDS]
    spec = EvolveSpec(kind="text", target="Be helpful.", data={"inline": rows}, score="exact",
                      agent="tests.test_evolvespec:make_model",
                      reflect={"ref": "tests.test_evolvespec:text_reflect", "call": False},
                      allow=["tests."],
                      evolve={"rounds": 3, "n_workers": 2, "seed": 0})
    result = compose(spec).run()
    assert result.error is None, result.error
    assert result.final_reward == 1.0
    assert "reversed" in result.rendered


def make_model():
    def model(prompt):
        skill, _, q = prompt.partition("\n\n")
        return q[::-1] if "reversed" in skill else q
    return model


# ---------------------------------------------------------------------------
# relative paths: resolved once, where the spec was read
# ---------------------------------------------------------------------------


def test_absolutise_resolves_target_data_and_a_cmd_grader(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "skill").mkdir()
    (tmp_path / "cases.jsonl").write_text('{"prompt": "q", "gold": "a"}\n')
    (tmp_path / "grade.sh").write_text("#!/bin/sh\necho 1\n")
    spec = EvolveSpec(kind="skill_dir", target="./skill", agent="echo",
                      data={"path": "cases.jsonl"}, score={"cmd": "./grade.sh --strict"})
    out = spec.absolutise()
    assert out.target == str(tmp_path / "skill")
    assert out.data["path"] == str(tmp_path / "cases.jsonl")
    assert out.score["cmd"] == [str(tmp_path / "grade.sh"), "--strict"]
    assert spec.target == "./skill"                      # the original is untouched
    # a grader that is a program on PATH is left alone
    assert EvolveSpec(kind="skill_dir", target="./skill", agent="echo", data={"inline": [{}]},
                      score={"cmd": "grep -q x"}).absolutise().score["cmd"] == "grep -q x"


def test_absolutise_leaves_a_text_target_that_is_not_a_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = EvolveSpec(kind="text", target="You are a helpful assistant.", agent="echo",
                      data={"inline": [{"prompt": "q", "gold": "a"}]})
    assert spec.absolutise().target == "You are a helpful assistant."
    (tmp_path / "p.md").write_text("Be terse.")
    got = EvolveSpec(kind="text", target="p.md", agent="echo",
                     data={"inline": [{"prompt": "q", "gold": "a"}]}).absolutise()
    assert got.target == str(tmp_path / "p.md")


def test_load_spec_absolutises_by_default(tmp_path, monkeypatch):
    spec = _dir_spec(str(tmp_path))
    d = spec.to_dict()
    d["data"] = {**d["data"], "path": os.path.relpath(d["data"]["path"], str(tmp_path))}
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(d))
    monkeypatch.chdir(tmp_path)
    assert os.path.isabs(load_spec("spec.json").data["path"])
    assert not os.path.isabs(load_spec("spec.json", absolutise=False).data["path"])
