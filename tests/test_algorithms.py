"""The algorithm registry and ``EvolveSpec.algorithm`` (#384)."""

from __future__ import annotations

import json
import subprocess
import sys
import warnings

import pytest

from agentdescent import algorithms, cli
from agentdescent.algorithms import (Algorithm, AlgorithmError, Overrides, Param,
                                     resolve_params)
from agentdescent.algorithms.rrsi.rrsi_policies import (RRsiAcceptance, RRsiCritic,
                                                         RRsiProposalPolicy)
from agentdescent.evolvespec import EvolveSpec, SpecError, compose
from agentdescent.selection import Archive

ROWS = [{"prompt": f"what is {i}", "gold": f"zzz{i}"} for i in range(8)]


def _spec(**kw):
    base = dict(kind="text", target="Be helpful.", data={"inline": ROWS}, agent="echo",
                evolve={"rounds": 2, "n_workers": 2, "seed": 0})
    base.update(kw)
    return EvolveSpec(**base)


def _write(tmp_path, spec):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec.to_dict()))
    return str(path)


# -- the registry ----------------------------------------------------------------


def test_builtin_algorithms_are_registered_with_declared_fidelity():
    assert algorithms.names() == ["ace", "archive_selection", "evoskill", "gepa", "rrsi", "skillopt"]
    for alg in algorithms.listing():
        assert alg.fidelity in algorithms.FIDELITY
        assert alg.summary and alg.kinds


def test_listing_filters_by_kind():
    assert [a.name for a in algorithms.listing("text")] == ["ace", "archive_selection", "gepa", "rrsi", "skillopt"]
    assert [a.name for a in algorithms.listing("skill_dir")] == ["archive_selection"]
    assert [a.name for a in algorithms.listing("agent_dir")] == ["archive_selection", "evoskill"]
    assert algorithms.listing("policy_slot") == []


def test_get_unknown_names_the_available_ones():
    with pytest.raises(AlgorithmError, match=r"unknown algorithm 'nope'.*rrsi"):
        algorithms.get("nope")


def test_duplicate_registration_is_refused():
    with pytest.raises(ValueError, match="already registered"):
        algorithms.register(algorithms.get("rrsi"))


@pytest.mark.parametrize("params, message", [
    ({"nope": 1}, r"no parameter\(s\) \['nope'\].*b_max"),
    ({"b_max": "3"}, r"algorithm\.params\.b_max: expected int, got '3'"),
    ({"b_max": True}, r"expected int"),
    ({"b_max": 2.5}, r"expected int"),
    ({"b_max": 0}, r"must be >= 1"),
    ({"noise_band": 2}, r"noise_band: must be <= 1"),
    ({"noise_band": -0.1}, r"noise_band: must be >= 0"),
    ({"protected": "t3"}, r"protected: expected list\[str\]"),
    ({"protected": [1]}, r"expected list\[str\]"),
])
def test_resolve_params_rejects_what_it_cannot_honour(params, message):
    with pytest.raises(AlgorithmError, match=message):
        resolve_params(algorithms.get("rrsi"), params)


def test_resolve_params_fills_defaults_and_widens_int_to_float():
    out = resolve_params(algorithms.get("rrsi"), {"w_s": 2, "b_max": 3})
    assert out["w_s"] == 2.0 and isinstance(out["w_s"], float)
    assert out["b_max"] == 3 and out["T"] == 20 and out["protected"] == []


def test_a_zero_temperature_is_refused_by_the_schema_not_deep_in_build():
    with pytest.raises(AlgorithmError, match=r"temperature: must be >="):
        resolve_params(algorithms.get("archive_selection"), {"temperature": 0})


def test_rrsi_with_no_protected_strings_warns_that_the_screen_is_vacuous():
    note = " ".join(compose(_spec(algorithm="rrsi")).notes)
    assert "`protected` is empty" in note and "never reject" in note
    quiet = compose(_spec(algorithm={"name": "rrsi", "params": {"protected": ["zzz1"]}}))
    assert "protected` is empty" not in " ".join(quiet.notes)


def test_a_failed_builtin_load_keeps_its_original_error(monkeypatch):
    monkeypatch.setattr(algorithms, "_loaded", False)
    monkeypatch.setattr(algorithms, "_load_error", None)
    monkeypatch.delattr(algorithms, "builtin", raising=False)
    monkeypatch.setitem(sys.modules, "agentdescent.algorithms.builtin", None)
    with pytest.raises(ImportError) as first:
        algorithms.names()
    with pytest.raises(ImportError) as second:
        algorithms.names()
    assert second.value is first.value          # not "already registered"


def test_choices_are_enforced():
    with pytest.raises(AlgorithmError, match=r"sampling: must be one of \[.*'best'"):
        resolve_params(algorithms.get("archive_selection"), {"sampling": "greedy"})


def test_the_registry_works_from_a_wheel_without_examples():
    """``examples/`` is not shipped; list and show must not need it."""
    code = ("import sys; sys.modules['examples'] = None\n"
            "from agentdescent.cli import main\n"
            "assert main(['algorithms', 'list']) == 0\n"
            "assert main(['algorithms', 'show', 'rrsi']) == 0\n")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert "rrsi" in done.stdout and "archive_selection" in done.stdout


def test_the_examples_rrsi_modules_are_aliases_of_the_packaged_ones():
    import agentdescent.algorithms.rrsi.rrsi_policies as moved
    import examples.rrsi.rrsi_policies as alias
    from examples.rrsi import selection as alias_selection
    from agentdescent.algorithms.rrsi import selection

    assert alias is moved and alias_selection is selection


# -- EvolveSpec.algorithm --------------------------------------------------------


def test_the_spec_round_trips_an_algorithm():
    spec = _spec(algorithm={"name": "rrsi", "params": {"b_max": 3}})
    again = EvolveSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert again.algorithm == {"name": "rrsi", "params": {"b_max": 3}}
    assert EvolveSpec.from_dict(_spec(algorithm="rrsi").to_dict()).algorithm == "rrsi"


@pytest.mark.parametrize("spec_kw, message", [
    (dict(algorithm="nope"), r"algorithm: unknown algorithm 'nope'; available: \['ace', 'archive_selection', 'evoskill', 'gepa', 'rrsi', 'skillopt'\]"),
    (dict(algorithm={"name": "rrsi", "params": {"nope": 1}}),
     r"algorithm\.params: 'rrsi' has no parameter\(s\) \['nope'\]"),
    (dict(algorithm={"name": "rrsi", "params": {"b_max": "x"}}),
     r"algorithm\.params\.b_max: expected int"),
    (dict(algorithm={"name": "rrsi", "params": {"b_min": 4, "b_max": 2}}),
     r"b_min \(4\) must not exceed b_max \(2\)"),
    (dict(algorithm={"params": {}}), r"algorithm: expected a name or"),
    (dict(algorithm={"name": "rrsi", "param": {}}), r"unexpected key\(s\) \['param'\]"),
    (dict(algorithm={"name": "rrsi", "params": [1]}), r"algorithm\.params: expected an object"),
    (dict(algorithm={"name": "rrsi", "params": []}), r"algorithm\.params: expected an object"),
    (dict(algorithm={"name": "rrsi", "params": ""}), r"algorithm\.params: expected an object"),
    (dict(algorithm={"name": "rrsi", "params": 0}), r"algorithm\.params: expected an object"),
    (dict(algorithm={"name": "rrsi", "params": False}), r"algorithm\.params: expected an object"),
    (dict(algorithm=3), r"algorithm: expected a name or"),
])
def test_a_bad_algorithm_fails_in_compose_naming_the_field(spec_kw, message):
    with pytest.raises(SpecError, match=message):
        compose(_spec(**spec_kw))


def test_null_algorithm_params_means_defaults():
    compose(_spec(algorithm={"name": "rrsi", "params": None}))


def test_noise_band_at_the_bound_is_accepted():
    assert resolve_params(algorithms.get("rrsi"), {"noise_band": 1})["noise_band"] == 1.0


def test_an_algorithm_that_does_not_apply_to_the_kind_names_the_ones_that_do(tmp_path):
    (tmp_path / "SKILL.md").write_text("---\nname: s\ndescription: d\n---\nbody\n")
    spec = _spec(kind="skill_dir", target=str(tmp_path), agent="opencode",
                 algorithm="rrsi")
    with pytest.raises(SpecError,
                       match=r"'rrsi' applies to kind \['text'\], not 'skill_dir'; "
                             r"for kind 'skill_dir': \['archive_selection'\]"):
        compose(spec)


def test_policy_slot_refuses_an_algorithm():
    spec = EvolveSpec(kind="policy_slot", target="selection", data={}, agent="echo",
                      algorithm="rrsi")
    with pytest.raises(SpecError, match=r"algorithm: not available for kind='policy_slot'"):
        compose(spec)


def test_rrsi_composes_into_the_three_policies_and_keeps_the_default_merge():
    comp = compose(_spec(algorithm={"name": "rrsi", "params": {"protected": ["zzz3"]}}))
    pol = comp.kwargs["policies"]
    assert isinstance(pol.screen, RRsiCritic)
    assert isinstance(pol.proposal, RRsiProposalPolicy)
    assert isinstance(pol.acceptance, RRsiAcceptance)
    assert pol.conflict is not None and pol.fusion is not None
    assert comp.algorithm["name"] == "rrsi" and comp.algorithm["params"]["b_max"] == 5


def test_the_specs_own_policy_overrides_the_algorithms_and_plan_says_so():
    spec = _spec(algorithm="archive_selection", policies={"selection": "Beam"})
    comp = compose(spec)
    assert type(comp.kwargs["policies"].selection).__name__ == "Beam"
    assert comp.algorithm["overridden_by_spec"] == ["policies.selection"]
    assert compose(_spec(algorithm="archive_selection")).algorithm["overridden_by_spec"] == []


def test_an_algorithms_defaults_sit_between_the_kind_and_the_spec():
    calls = []

    def build(ctx, p):
        calls.append(p)
        return Overrides(knobs={"rounds": 9, "patience": 7},
                         agg_config={"batch_trigger": 5, "max_wait_rounds": 4},
                         notes=["hello"])

    algorithms.register(Algorithm(
        name="_probe", summary="s", kinds=("text",), fidelity="preset", params=(
            Param("x", "int", 1),), build=build), replace=True)
    try:
        comp = compose(_spec(algorithm={"name": "_probe", "params": {"x": 2}},
                             evolve={"rounds": 3, "n_workers": 2, "seed": 0},
                             agg_config={"max_wait_rounds": 1}))
    finally:
        algorithms._REGISTRY.pop("_probe")
    k = comp.kwargs
    assert calls == [{"x": 2}]
    assert k["rounds"] == 3          # spec.evolve beats the algorithm
    assert k["patience"] == 7        # the algorithm beats the kind's default (3)
    assert k["agg_config"].batch_trigger == 5      # the algorithm
    assert k["agg_config"].max_wait_rounds == 1    # the spec beats it
    assert "hello" in comp.notes
    assert comp.algorithm["overridden_by_spec"] == ["agg_config.max_wait_rounds",
                                                    "evolve.rounds"]


def test_a_directory_kind_composes_with_a_selection_algorithm(tmp_path):
    (tmp_path / "SKILL.md").write_text("---\nname: s\ndescription: d\n---\nbody\n")
    spec = EvolveSpec(kind="skill_dir", target=str(tmp_path), agent="opencode",
                      data={"inline": ROWS}, algorithm="archive_selection")
    assert isinstance(compose(spec).kwargs["policies"].selection, Archive)


def test_no_algorithm_changes_nothing():
    comp = compose(_spec())
    assert comp.algorithm is None
    assert comp.kwargs["policies"].screen is None


# -- plan ------------------------------------------------------------------------


def test_plan_echoes_the_resolved_algorithm(tmp_path, capsys):
    path = _write(tmp_path, _spec(algorithm={"name": "rrsi", "params": {"b_max": 3}}))
    assert cli.main(["--json", "plan", path]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True
    alg = out["algorithm"]
    assert alg["name"] == "rrsi" and alg["fidelity"] == "mechanism_microport"
    assert alg["params"]["b_max"] == 3 and alg["params"]["T"] == 20
    assert "extra_calls" in alg and alg["overridden_by_spec"] == []


def test_plan_without_an_algorithm_has_no_algorithm_key(tmp_path, capsys):
    assert cli.main(["--json", "plan", _write(tmp_path, _spec())]) == 0
    assert "algorithm" not in json.loads(capsys.readouterr().out)


def test_plan_fails_closed_for_a_bad_algorithm(tmp_path, capsys):
    path = _write(tmp_path, _spec(algorithm={"name": "rrsi", "params": {"b_max": "x"}}))
    assert cli.main(["--json", "plan", path]) == 2
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is False and "algorithm.params.b_max" in out["error"]


# -- running ---------------------------------------------------------------------


def _run(spec, tmp_path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = compose(spec, repo_path=str(tmp_path / "repo")).run()
    stopped = [str(w.message) for w in caught if "stopped early" in str(w.message)]
    assert not stopped, stopped
    return result


def test_rrsi_runs_an_offline_evolve_round(tmp_path):
    """The screen and the acceptance floor are live in the loop, not just composed."""
    result = _run(_spec(algorithm="rrsi"), tmp_path)
    assert result.critic_checked >= 1
    assert sum(result.outcomes().values()) >= 1


def test_rrsi_screens_a_leaking_candidate_before_measurement(tmp_path):
    """``protected`` reaches the screen: the echo reflector reproduces the task
    prompt, which carries the protected string, so it is refused unmeasured."""
    spec = _spec(algorithm={"name": "rrsi", "params": {"protected": ["what is"]}})
    result = _run(spec, tmp_path)
    assert result.critic_rejected >= 1
    assert result.outcomes().get("critic-rejected", 0) >= 1


def test_archive_selection_runs_an_offline_evolve_round(tmp_path):
    spec = _spec(algorithm={"name": "archive_selection", "params": {"seed": 1}},
                 evolve={"rounds": 3, "n_workers": 2, "seed": 0})
    result = _run(spec, tmp_path)
    assert result.outcomes().get("population-select", 0) >= 1


# -- the CLI ---------------------------------------------------------------------


def test_cli_list_text_and_json(capsys):
    assert cli.main(["algorithms", "list"]) == 0
    text = capsys.readouterr().out
    assert "rrsi" in text and "mechanism_microport" in text and "archive_selection" in text
    for argv in (["--json", "algorithms", "list"], ["algorithms", "list", "--json"]):
        assert cli.main(argv) == 0
        rows = json.loads(capsys.readouterr().out)
        assert [r["name"] for r in rows] == ["ace", "archive_selection", "evoskill", "gepa", "rrsi", "skillopt"]
        rrsi = next(r for r in rows if r["name"] == "rrsi")
        assert {p["name"] for p in rrsi["params"]} >= {"b_max", "protected"}


def test_cli_list_filters_by_kind_and_rejects_an_unknown_one(capsys):
    assert cli.main(["algorithms", "list", "--kind", "plugin", "--json"]) == 0
    assert [r["name"] for r in json.loads(capsys.readouterr().out)] == ["archive_selection"]
    assert cli.main(["algorithms", "list", "--kind", "bogus"]) == 2
    assert "--kind must be one of" in capsys.readouterr().err


def test_cli_show_prints_the_card_and_fails_for_an_unknown_name(capsys):
    assert cli.main(["algorithms", "show", "rrsi"]) == 0
    text = capsys.readouterr().out
    assert "mechanism_microport" in text and "b_max: int = 5" in text and "docs/algo-rrsi.md" in text
    assert cli.main(["algorithms", "show", "rrsi", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["name"] == "rrsi"
    assert cli.main(["algorithms", "show", "nope"]) == 2
    assert "unknown algorithm 'nope'" in capsys.readouterr().err


def test_ace_registers_the_grow_and_refine_playbook():
    alg = algorithms.get("ace")
    assert alg.kinds == ("text",) and alg.fidelity == "mechanism_microport"
    assert {p.name for p in alg.params} == {"title", "dedup_threshold"}
    comp = compose(_spec(algorithm={"name": "ace",
                                    "params": {"dedup_threshold": 0.7}}))
    from agentdescent.algorithms.ace import ACEPlaybook, GrowAndRefine
    assert isinstance(comp.kwargs["strategy"], ACEPlaybook)
    assert comp.kwargs["strategy"].dedup_threshold == 0.7
    assert isinstance(comp.kwargs["policies"].acceptance, GrowAndRefine)
def test_gepa_registers_the_pareto_optimizer():
    alg = algorithms.get("gepa")
    assert alg.kinds == ("text",) and alg.fidelity == "mechanism_microport"
    assert {p.name for p in alg.params} == {"seed", "eval_concurrency"}
    comp = compose(_spec(algorithm={"name": "gepa", "params": {"seed": 3}}))
    assert "aggregator_factory" in comp.kwargs
    factory = comp.kwargs["aggregator_factory"]
    assert callable(factory) and factory.holder == {}


def test_gepa_spec_run_returns_the_best_average_candidate(tmp_path):
    spec = _spec(algorithm="gepa",
                 evolve={"rounds": 3, "n_workers": 2, "seed": 0})
    comp = compose(spec, repo_path=str(tmp_path / "repo"))
    holder = comp.kwargs["aggregator_factory"].holder
    result = comp.run()
    agg = holder["agg"]
    assert agg.states
    # finalize() leaves GEPA's best-average candidate on dev, not the last
    # win-frequency sample the step loop committed.
    assert result.state == agg.best_state
    assert "gepa-select" in result.outcomes()
# -- the plugin surface: init, CLI overrides, MCP, docs (#384 slice 3) ----------


def test_init_writes_the_algorithm_and_only_the_params_given(tmp_path, capsys):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("be terse\n")
    assert cli.main(["init", str(prompt), "--out", str(tmp_path / "spec.json"),
                     "--algorithm", "rrsi", "--param", 'protected=["secret"]',
                     "--model", "m"]) == 0
    spec = json.loads((tmp_path / "spec.json").read_text())
    assert spec["algorithm"] == {"name": "rrsi", "params": {"protected": ["secret"]}}
    assert "wrote" in capsys.readouterr().out


def test_init_refuses_an_unknown_algorithm_or_wrong_kind(tmp_path, capsys):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("be terse\n")
    assert cli.main(["init", str(prompt), "--out", str(tmp_path / "a.json"),
                     "--algorithm", "nope"]) == 2
    assert "unknown algorithm" in capsys.readouterr().err
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: s\ndescription: d\n---\nbody\n")
    assert cli.main(["init", str(skill), "--out", str(tmp_path / "b.json"),
                     "--algorithm", "rrsi", "--agent", "opencode"]) == 2
    assert "applies to kind" in capsys.readouterr().err


def test_init_refuses_params_without_an_algorithm(tmp_path, capsys):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("be terse\n")
    assert cli.main(["init", str(prompt), "--out", str(tmp_path / "a.json"),
                     "--param", "b_max=3"]) == 2
    assert "--param needs --algorithm" in capsys.readouterr().err


def test_plan_can_override_the_spec_algorithm(tmp_path, capsys):
    path = _write(tmp_path, _spec())
    assert cli.main(["--json", "plan", path, "--algorithm", "rrsi",
                     "--param", "b_max=3"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["algorithm"]["name"] == "rrsi"
    assert out["algorithm"]["params"]["b_max"] == 3
    assert json.loads(open(path).read()).get("algorithm") is None   # not written


def test_plan_override_fails_closed(tmp_path, capsys):
    path = _write(tmp_path, _spec())
    assert cli.main(["--json", "plan", path, "--algorithm", "nope"]) == 2
    assert "unknown algorithm" in json.loads(capsys.readouterr().out)["error"]


def test_mcp_lists_and_describes_the_registry():
    from agentdescent.mcp import Tools

    t = Tools(None)
    listed = t.list_algorithms()
    assert [a["name"] for a in listed["algorithms"]] == algorithms.names()
    assert [a["name"] for a in t.list_algorithms("skill_dir")["algorithms"]] == \
        ["archive_selection"]
    assert [a["name"] for a in t.list_algorithms("agent_dir")["algorithms"]] == \
        ["archive_selection", "evoskill"]
    assert "unknown kind" in t.list_algorithms("nope")["error"]
    card = t.describe_algorithm("rrsi")
    assert card["name"] == "rrsi" and any(p["name"] == "b_max" for p in card["params"])
    assert "unknown algorithm" in t.describe_algorithm("nope")["error"]


def test_the_docs_catalog_is_generated_from_the_registry():
    from tools import gen_algorithms_docs as gen

    current = open(gen.OUT, encoding="utf-8").read()
    assert current == gen.render()
    for name in algorithms.names():
        assert f"## `{name}`" in current


def test_a_same_name_override_merges_params(tmp_path, capsys):
    path = _write(tmp_path, _spec(algorithm={"name": "rrsi", "params": {"b_max": 3}}))
    assert cli.main(["--json", "plan", path, "--algorithm", "rrsi",
                     "--param", "b_min=2"]) == 0
    alg = json.loads(capsys.readouterr().out)["algorithm"]
    assert alg["params"]["b_max"] == 3 and alg["params"]["b_min"] == 2


def test_param_without_flag_tweaks_the_specs_algorithm(tmp_path, capsys):
    path = _write(tmp_path, _spec(algorithm={"name": "rrsi", "params": {"b_max": 3}}))
    assert cli.main(["--json", "plan", path, "--param", "b_min=2"]) == 0
    alg = json.loads(capsys.readouterr().out)["algorithm"]
    assert alg["params"]["b_max"] == 3 and alg["params"]["b_min"] == 2


def test_mcp_explains_that_policy_slot_has_no_algorithms():
    from agentdescent.mcp import Tools

    out = Tools(None).list_algorithms("policy_slot")
    assert out["algorithms"] == [] and "no algorithm applies" in out["note"]


def test_evoskill_registers_the_skill_library_frontier(tmp_path):
    (tmp_path / "AGENT.md").write_text("# agent\n")
    spec = EvolveSpec(kind="agent_dir", target=str(tmp_path),
                      data={"inline": ROWS}, agent="opencode",
                      editable=["SKILL.md"], frozen=["skip/**"],
                      algorithm={"name": "evoskill",
                                 "params": {"max_frontier": 3}})
    comp = compose(spec)
    from agentdescent.algorithms.evoskill import SkillLibraryTree
    strategy = comp.kwargs["strategy"]
    assert isinstance(strategy, SkillLibraryTree)
    assert strategy.editable == ("SKILL.md",)
    assert strategy.frozen == ("skip/**",)
    assert not strategy.writable("skills/new/SKILL.md")   # guardrail honoured
    assert callable(comp.kwargs["aggregator_factory"])
    assert comp.algorithm["params"]["max_frontier"] == 3


def test_evoskill_make_propose_offers_a_skill_path():
    from agentdescent.algorithms.evoskill import (EvoSkillContext, Frontier,
                                                   make_propose)
    from agentdescent.filetree import canonical

    ctx = EvoSkillContext(docs={}, frontier=Frontier(), batch_size=1)

    def complete(prompt):
        return "create careful" if "Skill Proposer" in prompt else "Read the task."

    propose = make_propose(ctx, complete)
    class T:
        prompt = "q"
        meta = {"gold": "a"}
    assert propose(canonical({"SKILL.md": "seed"}), T(), "wrong", 0.0) == \
        "careful :: Read the task."


def test_evoskill_runs_an_offline_evolve_round(tmp_path):
    from tests.test_evolvespec import _dir_spec

    spec = _dir_spec(
        str(tmp_path), kind="agent_dir",
        reflect={"ref": "tests.test_algorithms:_evoskill_reflect", "call": False},
        algorithm={"name": "evoskill", "params": {"batch_size": 1}},
        evolve={"rounds": 2, "n_workers": 1, "seed": 0})
    comp = compose(spec, repo_path=str(tmp_path / "repo"))
    holder = comp.kwargs["aggregator_factory"].holder
    result = comp.run()
    agg = holder["agg"]
    assert result.state is not None
    assert agg.ctx.frontier.members            # seeded, even if nothing was admitted
    assert any(
        any(path.startswith("skills/") and path.endswith("SKILL.md")
            for path in state)
        for state, _ in agg.ctx.frontier.members
    ), "the frontier never saw an induced skills/<name>/SKILL.md"


def _evoskill_reflect(prompt):
    if "Skill Proposer" in prompt:
        return "create careful\nreason"
    return "Read the task carefully."


def test_skillopt_registers_the_strict_gate_optimizer():
    alg = algorithms.get("skillopt")
    assert alg.kinds == ("text",) and alg.fidelity == "mechanism_microport"
    assert {p.name for p in alg.params} == {"max_lr", "min_lr", "total_steps", "mode"}
    comp = compose(_spec(algorithm={"name": "skillopt",
                                    "params": {"max_lr": 3, "min_lr": 1}}))
    from agentdescent.algorithms.skillopt import SkillDocStrategy
    assert isinstance(comp.kwargs["strategy"], SkillDocStrategy)
    assert comp.kwargs["strategy"].seed
    assert callable(comp.kwargs["aggregator_factory"])
    assert isinstance(comp.kwargs["initial_state"], dict)


def test_skillopt_runs_an_offline_evolve_round(tmp_path):
    result = _run(_spec(algorithm="skillopt"), tmp_path)
    assert sum(result.outcomes().values()) >= 1
    assert "unknown" not in result.outcomes()
