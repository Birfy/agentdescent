"""Turning the audit on from a spec, so the CLI and the MCP server can do it.

Everything in `agentdescent.audit` was reachable only from Python. A run started
by `agentdescent evolve spec.json` or by the MCP `start` tool could not create an
audit store -- which left the seven `audit_*` tools able to read a file that
path could not produce.

`test_the_store_lands_where_the_audit_tools_will_look` is the one that closes
that loop.
"""

import itertools
import json
import os
import pathlib

import pytest

from agentdescent import runstore
from agentdescent.audit import Audit, AuditedReward
from agentdescent.audit.gate import RectifiedAcceptance
from agentdescent.cli import plan_payload, status_payload
from agentdescent.evolvespec import EvolveSpec, SpecError, compose

from tests.test_evolvespec import _dir_spec, stub_reflect  # noqa: F401

ORACLE = "agentdescent.rewards:exact_match"


_counter = itertools.count()


def _spec(tmp, audit=None, **extra):
    """`_dir_spec` builds a skill directory, so each call needs its own base."""
    base = pathlib.Path(tmp) / f"case{next(_counter)}"
    base.mkdir()
    if audit is not None:
        extra["audit"] = audit
    return _dir_spec(base, **extra)


# -- the loop that was open --------------------------------------------------

def test_the_store_lands_where_the_audit_tools_will_look(tmp_path):
    """Beside the ledger, derived from the run id rather than configured.

    A run that writes its audit somewhere only the caller knows is a run whose
    audit nobody reads -- and `audit_status` takes a path, not a run id.
    """
    ledger = str(tmp_path / "run" / "ledger")
    comp = compose(_spec(tmp_path, {"oracle": ORACLE}), repo_path=ledger)
    assert comp.audit.store.path == str(tmp_path / "run" / "audit.jsonl")


def test_the_run_directory_names_it_the_same_way(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTDESCENT_HOME", str(tmp_path / "home"))
    rd = runstore.create({"kind": "text"}, store=str(tmp_path / "runs"))
    assert rd.audit_path == os.path.join(rd.path, "audit.jsonl")
    assert os.path.dirname(rd.ledger_path) == os.path.dirname(rd.audit_path)


def test_status_names_the_store_only_once_it_exists(tmp_path, monkeypatch):
    """A path to a file that was never written invites `audit_status` to report
    an empty store as an answer."""
    monkeypatch.setenv("AGENTDESCENT_HOME", str(tmp_path / "home"))
    store = str(tmp_path / "runs")
    rd = runstore.create({"kind": "text"}, store=store)

    assert "audit_store" not in status_payload(rd.run_id, store=store)
    with open(rd.audit_path, "w", encoding="utf-8") as fh:
        fh.write("")
    assert status_payload(rd.run_id, store=store)["audit_store"] == rd.audit_path


# -- what a spec gets --------------------------------------------------------

def test_the_reward_and_the_run_the_loop_gets_are_the_taps(tmp_path):
    comp = compose(_spec(tmp_path, {"oracle": ORACLE}))
    assert isinstance(comp.audit, Audit)
    assert isinstance(comp.reward, AuditedReward)
    assert comp.reward is comp.audit.reward
    assert comp.kwargs["run"] is comp.audit.run


def test_no_audit_block_changes_nothing(tmp_path):
    comp = compose(_spec(tmp_path))
    assert comp.audit is None
    assert not isinstance(comp.reward, AuditedReward)


def test_enabled_defaults_to_false(tmp_path):
    """Collecting records is free and changes nothing; correcting the gate
    changes what commits, and a spec that merely names an oracle has not asked
    for that."""
    comp = compose(_spec(tmp_path, {"oracle": ORACLE}))
    assert not comp.audit.enabled
    assert any("collects and corrects nothing" in n for n in comp.notes)

    on = compose(_spec(tmp_path, {"oracle": ORACLE, "enabled": True}))
    assert on.audit.enabled
    assert not any("corrects nothing" in n for n in on.notes)


def test_the_acceptance_policy_a_spec_named_is_wrapped_not_replaced(tmp_path):
    """The audit adds a term. Replacing the rule would silently drop whatever
    the spec asked for."""
    spec = _spec(tmp_path, {"oracle": ORACLE},
                 policies={"acceptance": {
                     "ref": "agentdescent.advantage:AdvantageAcceptance"}})
    comp = compose(spec)
    gate = comp.kwargs["policies"].acceptance
    assert isinstance(gate, RectifiedAcceptance)
    assert type(gate.inner).__name__ == "AdvantageAcceptance"


def test_without_a_named_policy_the_wrapper_keeps_the_shipped_gate(tmp_path):
    comp = compose(_spec(tmp_path, {"oracle": ORACLE}))
    gate = comp.kwargs["policies"].acceptance
    assert type(gate.inner).__name__ == "DefaultAcceptance"


def test_the_settings_reach_the_tap(tmp_path):
    comp = compose(_spec(tmp_path, {
        "oracle": ORACLE, "sample_rate": 0.25, "calibration_fraction": 0.9,
        "draw_by": "output", "seed": 11}))
    reward = comp.audit.reward
    assert reward.sample_rate == 0.25
    assert reward.calibration_fraction == 0.9
    assert reward.draw_by == "output" and reward.seed == 11


def test_an_explicit_store_path_wins_and_is_absolutised(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = _spec(tmp_path, {"oracle": ORACLE, "store": "audits/mine.jsonl"})
    comp = compose(spec.absolutise(), repo_path=str(tmp_path / "run" / "ledger"))
    assert comp.audit.store.path == str(tmp_path / "audits" / "mine.jsonl")


def test_no_oracle_still_records_the_questions(tmp_path):
    comp = compose(_spec(tmp_path, {"sample_rate": 1.0}))
    assert comp.audit is not None
    assert type(comp.audit.reward.oracle).__name__ == "NullOracle"


# -- the errors --------------------------------------------------------------

def test_a_mistyped_key_is_an_error_with_the_keys_beside_it(tmp_path):
    """Otherwise `sample-rate` is a setting that silently did nothing and the
    audit ran at the default."""
    with pytest.raises(SpecError, match="unknown key"):
        compose(_spec(tmp_path, {"oracle": ORACLE, "sample-rate": 0.5}))


def test_the_oracle_goes_through_the_specs_own_allowlist(tmp_path):
    """The audit is not a reason to widen the trust boundary."""
    with pytest.raises(SpecError, match="outside the allowed prefixes"):
        compose(_spec(tmp_path, {"oracle": "os:getcwd"}, allow=[]))


def test_a_bad_setting_fails_here_rather_than_in_round_one(tmp_path):
    with pytest.raises(SpecError, match="draw_by"):
        compose(_spec(tmp_path, {"oracle": ORACLE, "draw_by": "unit"}))
    with pytest.raises(SpecError, match="sample_rate"):
        compose(_spec(tmp_path, {"oracle": ORACLE, "sample_rate": 2.0}))


# -- what a person sees before running ---------------------------------------

def test_plan_reports_the_audit_before_anything_runs(tmp_path):
    payload = plan_payload(_spec(tmp_path, {"oracle": ORACLE, "sample_rate": 0.3}))
    audit = payload["audit"]
    assert audit["enabled"] is False and audit["sample_rate"] == 0.3
    assert audit["oracle"] == "GoldAnswer"
    assert audit["draw_by"] == "task"
    assert len(audit["verifier_version"]) > 8


def test_plan_says_nothing_about_an_audit_that_was_not_asked_for(tmp_path):
    assert "audit" not in plan_payload(_spec(tmp_path))


def test_the_spec_round_trips_through_json(tmp_path):
    spec = _spec(tmp_path, {"oracle": ORACLE, "sample_rate": 0.3})
    again = EvolveSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert again.audit == spec.audit
