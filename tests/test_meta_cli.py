"""The meta stack on the command line: portfolio, exam, audit.

The self-evolution machinery is operator-facing through three verbs:
`portfolio` ranks the archive's layers and reallocates the next budget window,
`exam` re-judges adopted rules with an independent oracle (the domain comes
from a `module:attribute` builder), and `audit` traces one adoption through its
evidence rings. The fixtures pin the surfaces that make these deployable:
machine-readable `--json` output, clean exit codes, the archive default under
$AGENTDESCENT_HOME, and a bad domain ref failing cleanly rather than
tracebacking.
"""

import json
import os

import pytest

from agentdescent import cli
from agentdescent import (
    ConsolidationStore,
    EvolutionResult,
    ParamSlot,
    SingleSlot,
    Task,
    final_reward,
    problem_descriptor,
)
from agentdescent.evolution import RoundInfo
from agentdescent.sampling import DifficultyWeighted

from tests.test_evolvespec import _WORDS  # noqa: F401  (shared fixture import)


# -- the domain builder a CLI `exam` resolves via --domain ----------------------


def _cap(rendered):
    import re
    m = re.search(r"cap:(\d+)", rendered)
    return int(m.group(1)) if m else 0


def _run(rendered, task):
    return str(task.meta["gold"]) if task.meta.get("a", 0) <= _cap(rendered) else "?"


def _reward(task, output):
    return 1.0 if str(output).strip() == str(task.meta["gold"]).strip() else 0.0


def _propose(rendered, task, output, reward):
    if reward >= 0.999:
        return "cap:0"
    a = task.meta.get("a", 0)
    if a > _cap(rendered):
        return f"cap:{a}"
    return None


def _tasks_for(record):
    order = [1, 2, 3, 4, 8, 5, 6, 7]
    return [Task(id=f"t{i}", prompt=f"item {i}",
                 meta={"gold": str(2 * a), "a": a}) for i, a in enumerate(order)]


def _sampler_slot():
    return ParamSlot(factory=lambda c: DifficultyWeighted(c=c),
                     params={"c": 0.2}, bounds={"c": (0.05, 1.0)})


def build_domain():
    return {"oracle": final_reward, "tasks_for": _tasks_for, "run": _run,
            "reward": _reward, "propose": _propose,
            "strategy": SingleSlot(initial_value="cap:0"), "spec": _sampler_slot()}


# -- fixtures -------------------------------------------------------------------


def _archive(tmp_path, *, n=1, at="archive"):
    store = ConsolidationStore(str(tmp_path / at))
    for i in range(n):
        hist = [RoundInfo(round=r, held_out_reward=0.8, n_items=1,
                          committed=1, rejected=0) for r in range(5)]
        store.archive(
            EvolutionResult(state={"c": "0.4"}, rendered="# Policy parameters\nc: 0.4",
                            final_reward=0.8, history=hist, ledger_log=[]),
            run_id=f"run-{i}", strategy="ParamSlot",
            config={"dataset": "x"}, artifact_id="meta-rule",
            problem=problem_descriptor("task_sampler", rounds=5, n_workers=1))
    return str(tmp_path / at)


def _json(capsys):
    return json.loads(capsys.readouterr().out)


# -- portfolio ------------------------------------------------------------------


def test_portfolio_reports_and_exits_zero(tmp_path, capsys):
    archive = _archive(tmp_path, n=2)
    code = cli.main(["--json", "portfolio", archive])
    assert code == 0
    out = _json(capsys)
    assert out["archive"] == archive
    assert "reallocated" in out
    assert len(out["layers"]) == 1


def test_portfolio_defaults_to_agentdescent_home(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AGENTDESCENT_HOME", str(tmp_path / "home"))
    _archive(tmp_path, n=1, at="home/archive")
    code = cli.main(["--json", "portfolio"])
    assert code == 0
    out = _json(capsys)
    assert out["archive"] == str(tmp_path / "home" / "archive")


def test_portfolio_empty_archive_exits_one(tmp_path, capsys):
    (tmp_path / "empty").mkdir()
    code = cli.main(["--json", "portfolio", str(tmp_path / "empty")])
    assert code == 1
    assert "no rule records" in capsys.readouterr().err


# -- a missing archive is an error, not a fresh empty one (#502) ----------------


@pytest.mark.parametrize("argv", [
    ["portfolio", "{missing}"],
    ["exam", "{missing}", "--domain", "tests.test_meta_cli:build_domain",
     "--slot", "task_sampler"],
    ["audit", "run-0", "--archive", "{missing}"],
])
def test_missing_archive_fails_without_creating_it(tmp_path, capsys, argv):
    missing = tmp_path / "typo"
    code = cli.main([x.format(missing=missing) for x in argv])
    assert code == 2
    err = capsys.readouterr().err.strip()
    assert err.startswith(f"error: archive {str(missing)!r} does not exist")
    assert "continuous_meta_run" in err
    assert "evolve / plugin runs do not write" in err
    assert not missing.exists()


@pytest.mark.parametrize("argv", [
    ["safety", "--archive", "{empty}"],
    ["portfolio", "{empty}"],
    ["audit", "run-0", "--archive", "{empty}"],
])
def test_readonly_verbs_do_not_create_runs_under_empty_archive(tmp_path, argv):
    """An existing but empty archive must stay empty after a read (#512)."""
    empty = tmp_path / "empty"
    empty.mkdir()
    before = sorted(p.relative_to(empty) for p in empty.rglob("*"))
    code = cli.main([x.format(empty=empty) for x in argv])
    assert code in (1, 2)  # empty portfolio -> 1; audit unknown -> 2; safety caution/warn path
    after = sorted(p.relative_to(empty) for p in empty.rglob("*"))
    assert after == before == []


@pytest.mark.parametrize("flag", ["--meta-audit", "--inner-audit"])
def test_audit_missing_store_file_fails(tmp_path, capsys, flag):
    archive = _archive(tmp_path)
    missing = str(tmp_path / "nope.jsonl")
    code = cli.main(["audit", "run-0", "--archive", archive, flag, missing])
    assert code == 2
    assert capsys.readouterr().err.strip() == (
        f"error: {flag} store {missing!r} does not exist")
    assert not os.path.exists(missing)


# -- exam -----------------------------------------------------------------------


def test_exam_runs_and_stamps_rules(tmp_path, capsys):
    archive = _archive(tmp_path)
    code = cli.main(["--json", "exam", archive,
                     "--domain", "tests.test_meta_cli:build_domain",
                     "--slot", "task_sampler"])
    assert code == 0
    out = _json(capsys)
    assert out["archive"] == archive
    assert len(out["examined"]) == 1
    assert out["examined"][0]["run_id"] == "run-0"
    assert out["examined"][0]["verdict"] in ("passed", "flagged")
    # the stamp landed on the record
    store = ConsolidationStore(archive)
    assert store.find("run-0").exam in ("passed", "flagged")


def test_exam_bad_domain_ref_fails_cleanly(tmp_path):
    archive = _archive(tmp_path)
    code = cli.main(["--json", "exam", archive,
                     "--domain", "no.such.module:thing", "--slot", "task_sampler"])
    assert code != 0


def test_exam_config_flag_is_json_parsed_into_a_dict(tmp_path, capsys):
    # two rules on the same slot, different configs and rewards: the parsed
    # --config dict must rank its matching rule first even though the other has
    # the higher reward -- a raw string would never match either fingerprint.
    store = ConsolidationStore(str(tmp_path / "archive"))
    for run_id, cfg, reward in [("run-a", {"dataset": "x"}, 0.8),
                                ("run-b", {"dataset": "y"}, 0.9)]:
        hist = [RoundInfo(round=r, held_out_reward=0.8, n_items=1,
                          committed=1, rejected=0) for r in range(5)]
        store.archive(
            EvolutionResult(state={"c": "0.4"},
                            rendered="# Policy parameters\nc: 0.4",
                            final_reward=reward, history=hist, ledger_log=[]),
            run_id=run_id, strategy="ParamSlot", config=cfg,
            artifact_id="meta-rule",
            problem=problem_descriptor("task_sampler", rounds=5, n_workers=1))
    archive = str(tmp_path / "archive")
    code = cli.main(["--json", "exam", archive,
                     "--domain", "tests.test_meta_cli:build_domain",
                     "--slot", "task_sampler", "--config", '{"dataset": "x"}'])
    assert code == 0
    examined = _json(capsys)["examined"]
    assert [e["run_id"] for e in examined] == ["run-a", "run-b"]
    code = cli.main(["--json", "exam", archive,
                     "--domain", "tests.test_meta_cli:build_domain",
                     "--slot", "task_sampler", "--config", '{"dataset": "y"}'])
    assert code == 0
    examined = _json(capsys)["examined"]
    assert [e["run_id"] for e in examined] == ["run-b", "run-a"]


def test_exam_bad_config_json_fails_cleanly(tmp_path):
    archive = _archive(tmp_path)
    code = cli.main(["--json", "exam", archive,
                     "--domain", "tests.test_meta_cli:build_domain",
                     "--slot", "task_sampler", "--config", "not json"])
    assert code != 0


def test_exam_non_dict_config_fails_cleanly(tmp_path):
    archive = _archive(tmp_path)
    code = cli.main(["--json", "exam", archive,
                     "--domain", "tests.test_meta_cli:build_domain",
                     "--slot", "task_sampler", "--config", "[1, 2]"])
    assert code != 0


# -- audit ----------------------------------------------------------------------


def test_audit_traces_rings(tmp_path, capsys):
    archive = _archive(tmp_path)
    code = cli.main(["--json", "audit", "run-0", "--archive", archive])
    assert code == 0
    out = _json(capsys)
    assert out["run_id"] == "run-0"
    assert out["verified"] is False
    layers = [l["layer"] for l in out["links"]]
    assert {"governance", "decision", "meta-gain", "self-exam",
            "inner-reward", "harness"} <= set(layers)
    assert all(l["status"] in ("verified", "flagged", "unverified")
               for l in out["links"])


def test_audit_unknown_run_fails(tmp_path, capsys):
    code = cli.main(["--json", "audit", "nope", "--archive", str(tmp_path)])
    # Same exit code and unquoted wording as show/cancel/resume on an unknown
    # run, not the quoted repr of a KeyError (#481).
    assert code == 2
    err = capsys.readouterr().err.strip()
    assert err == "error: no archived run 'nope' to audit"


def test_audit_non_callable_harness_fails_cleanly(tmp_path):
    archive = _archive(tmp_path)
    # build_domain resolves to a dict-returning function? no -- it is a function,
    # so pass a ref that resolves to a non-callable object (a module attribute
    # that is a dict) and expect a clean failure.
    code = cli.main(["--json", "audit", "run-0", "--archive", archive,
                     "--harness", "sys:path"])
    assert code != 0


def test_audit_wrong_shaped_harness_is_flagged_not_crashed(tmp_path, capsys):
    # builtins:len is callable but returns an int, not the (name, score) pair
    # chain_audit expects; it must produce a flagged link, not a raw traceback
    archive = _archive(tmp_path)
    code = cli.main(["--json", "audit", "run-0", "--archive", archive,
                     "--harness", "builtins:len"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    harness = [l for l in out["links"] if l["layer"] == "harness"][0]
    assert harness["status"] == "flagged"
