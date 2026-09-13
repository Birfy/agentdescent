"""The audit's verbs over a path, which is the only handle a resolver has.

Truth may take days. The process that dispatched a record is gone by the time a
wet-lab result or a human review comes back, so everything here takes a JSONL
path and returns JSON. `test_a_second_result_for_the_same_unit_is_refused` is
the one that matters: replaying a batch file is what people do when a job
half-failed, and a silent overwrite would make the calibration set depend on how
many times that happened.
"""

import uuid

import numpy as np
import pytest

from agentdescent.audit import AuditRecord, AuditStore, Purpose
from agentdescent.audit.service import (OUTPUT_PREVIEW, audit_drift,
                                        audit_pending, audit_recompute,
                                        audit_rescan, audit_resolve,
                                        audit_scorecard, audit_status,
                                        pick_version)
from agentdescent.mcp import TOOL_DESCRIPTIONS, Tools


def _store(path, *, n=120, version="v1", bias=0.3, base=0.5, seed=0,
           n_unlab=400, sig="a"):
    rng = np.random.default_rng(seed)
    store = AuditStore(str(path))
    for i in range(n):
        y = 1.0 if rng.random() < base else 0.0
        f = 1.0 if (y == 0.0 and rng.random() < bias) else y
        rec = store.append(AuditRecord(
            record_id=uuid.uuid4().hex, task_id=f"{version}-t{i}",
            artifact_signature=f"{sig}{i % 4}", output=f"answer {i}",
            verifier_version=version, verifier_score=f, inclusion_prob=1.0,
            purpose=Purpose.CALIBRATION, dispatched_at=1000.0 + i))
        store.resolve(rec.record_id, y, at=1001.0 + i)
    for _ in range(n_unlab):
        y = 1.0 if rng.random() < base else 0.0
        store.observe_unlabelled(
            version, "all", 1.0 if (y == 0.0 and rng.random() < bias) else y)
    store.flush()
    return store


@pytest.fixture
def path(tmp_path):
    p = tmp_path / "audit.jsonl"
    _store(p)
    return str(p)


# -- status ------------------------------------------------------------------

def test_status_leads_with_both_halves_of_the_error():
    """`delta_hat` is the bias and `resid_sd` is the spread, and a caller that
    reads only the first has the smaller of the two terms."""
    import tempfile, os

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.jsonl")
        _store(p)
        got = audit_status(p)
    rect = got["rectification"]
    assert 0.05 < rect["delta_hat"] < 0.35, (
        "a generous verifier, recovered -- the estimator's accuracy is "
        "test_audit_calibrator's job, not this layer's")
    assert rect["resid_sd"] > 1.5 * rect["delta_hat"], (
        "the spread is the larger fact and has to be in the payload")
    assert got["calibration_labels"] == 120 and got["unlabelled"] == 400


def test_a_typo_in_the_path_does_not_read_as_an_unbiased_verifier(tmp_path):
    """`AuditStore` treats an absent file as an empty store, which is right for
    a run about to write one and wrong for a question about one."""
    got = audit_status(str(tmp_path / "nope.jsonl"))
    assert "error" in got and "nope.jsonl" in got["error"]
    assert "does not exist" in got["error"]
    assert "rectification" not in got


def test_an_empty_store_says_so(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    assert "no records" in audit_status(str(p))["error"]


def test_the_version_it_picked_is_always_named(tmp_path):
    """A store can hold several, and answering about the wrong one silently is
    the failure this package exists to prevent, committed by its own report."""
    p = tmp_path / "two.jsonl"
    _store(p, version="old", n=40)
    _store(p, version="new", n=120, seed=1)
    got = audit_status(str(p))
    assert got["version"] == "new", "the busiest, when none is named"
    assert set(got["versions"]) == {"old", "new"}
    assert audit_status(str(p), version="old")["version"] == "old"


def test_an_unknown_version_lists_the_ones_that_exist(path):
    got = audit_status(path, version="ghost")
    assert "ghost" in got["error"] and got["versions"] == ["v1"]


def test_pick_version_orders_by_how_much_it_rests_on(tmp_path):
    p = tmp_path / "two.jsonl"
    _store(p, version="small", n=20)
    _store(p, version="big", n=90, seed=2)
    chosen, versions = pick_version(AuditStore(str(p)))
    assert chosen == "big" and versions == ["big", "small"]


# -- pending -----------------------------------------------------------------

def _pending(path, output="x", record_id="p1"):
    store = AuditStore(path)
    store.append(AuditRecord(
        record_id=record_id, task_id="tz", artifact_signature="a0",
        output=output, verifier_version="v1", verifier_score=1.0,
        inclusion_prob=0.5, purpose=Purpose.CALIBRATION, dispatched_at=9000.0))
    return store


def test_pending_truncates_the_outputs_and_says_that_it_did(path):
    _pending(path, output="y" * (OUTPUT_PREVIEW * 3))
    got = audit_pending(path)
    assert got["pending"] == 1
    row = got["records"][0]
    assert len(row["output"]) == OUTPUT_PREVIEW and row["output_truncated"]
    assert got["truncated_outputs_at"] == OUTPUT_PREVIEW


def test_pending_reports_the_total_beside_what_it_returned(path):
    for i in range(10):
        _pending(path, record_id=f"p{i}")
    got = audit_pending(path, limit=3)
    assert got["pending"] == 10 and got["returned"] == 3


def test_a_queue_with_nothing_in_it_is_not_an_error(path):
    got = audit_pending(path)
    assert got["pending"] == 0 and got["records"] == []


# -- resolve -----------------------------------------------------------------

def test_a_resolution_reports_the_residual_it_just_created(path):
    _pending(path)
    got = audit_resolve(path, "p1", 0.0)
    assert got["ok"] and got["residual"] == pytest.approx(1.0)
    assert got["pending"] == 0
    assert "not recomputed" in got["note"]


def test_a_second_result_for_the_same_unit_is_refused(path):
    """Replaying a batch file is what people do when a job half-failed."""
    _pending(path)
    assert audit_resolve(path, "p1", 0.0)["ok"]
    again = audit_resolve(path, "p1", 1.0)
    assert not again["ok"]
    assert "refusing to overwrite" in again["error"]
    assert again["oracle_score"] == 0.0, "and it says what is already there"
    assert audit_status(path)["observed"]["n"] == 121, "the first result stands"


def test_a_record_that_does_not_exist_reads_differently_from_one_already_done(path):
    _pending(path)
    audit_resolve(path, "p1", 0.0)
    missing = audit_resolve(path, "ghost", 0.0)
    done = audit_resolve(path, "p1", 0.0)
    assert "no record" in missing["error"]
    assert "already resolved" in done["error"]


# -- recompute ---------------------------------------------------------------

def test_recompute_reflects_labels_added_since(path):
    before = audit_status(path)["rectification"]["n"]
    _pending(path)
    audit_resolve(path, "p1", 0.0)
    assert audit_recompute(path)["rectification"]["n"] == before + 1


# -- the reports -------------------------------------------------------------

def test_the_scorecard_comes_back_as_rows_and_as_prose(path):
    got = audit_scorecard(path)
    names = [m["name"] for m in got["metrics"]]
    assert names[0] == "sigma", "the row that decides comes first"
    assert "delta_hat" in names
    assert next(m for m in got["metrics"] if m["name"] == "delta_hat")["goal"] \
        == "WATCH"
    assert "Verifier scorecard" in got["markdown"]


def test_the_scorecard_can_compare_two_versions(tmp_path):
    p = tmp_path / "two.jsonl"
    _store(p, version="v1", n=120, bias=0.4)
    _store(p, version="v2", n=120, bias=0.1, seed=3)
    got = audit_scorecard(str(p), version="v2", previous="v1")
    assert got["previous_version"] == "v1"
    sigma = next(m for m in got["metrics"] if m["name"] == "sigma")
    assert sigma["verdict"] == "better" and got["ship"]


def test_an_unknown_previous_version_is_named(path):
    got = audit_scorecard(path, previous="ghost")
    assert "ghost" in got["error"]


def test_a_cost_ratio_can_block_from_here(path):
    assert not audit_scorecard(path, verifier_seconds=9.0,
                               oracle_seconds=10.0)["ship"]
    assert audit_scorecard(path, verifier_seconds=0.1,
                           oracle_seconds=10.0)["ship"]


# -- rescan, and the trust boundary ------------------------------------------

def test_rescan_refuses_a_module_outside_the_allowed_prefixes(path):
    """Resolving a reference runs whatever it imports.

    So the calling model cannot name a module and have it imported; widening the
    list is the operator's decision.
    """
    got = audit_rescan(path, "os:getcwd")
    assert "refusing to resolve" in got["error"] and "os" in got["error"]


def test_rescan_names_the_attribute_it_could_not_find(path):
    got = audit_rescan(path, "agentdescent.audit.service:no_such_thing")
    assert "no attribute" in got["error"]


def test_a_bare_dotted_path_is_refused_as_ambiguous(path):
    assert "error" in audit_rescan(path, "agentdescent.audit.service")


def test_rescan_runs_a_widened_reference_and_reports_what_moved(path):
    got = audit_rescan(path, "tests.test_audit_service:harsher",
                       allow=["tests."])
    assert "error" not in got
    assert got["n"] == 120 and got["agreement"] < 1.0
    assert got["mean_shift"] < 0.0
    assert got["sigma_after"] != got["sigma_before"]
    assert "over-counts comparisons" in got["note"]


def test_a_scorer_that_raises_is_reported_as_the_callers_code(path):
    got = audit_rescan(path, "tests.test_audit_service:explodes",
                       allow=["tests."])
    assert "raised ValueError" in got["error"]
    assert "tests.test_audit_service:explodes" in got["error"]


# -- drift -------------------------------------------------------------------

def test_drift_charts_one_point_per_version_and_says_that_is_weak(tmp_path):
    p = tmp_path / "many.jsonl"
    for i, bias in enumerate([0.1, 0.2, 0.3]):
        _store(p, version=f"v{i}", n=120, bias=bias, seed=i)
    got = audit_drift(str(p))
    assert got["charted"] == 3
    assert got["versions"] == ["v0", "v1", "v2"]
    assert "not per generation" in got["note"]


def test_drift_on_an_empty_store_is_an_error(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    assert "error" in audit_drift(str(p))


# -- the MCP surface ---------------------------------------------------------

def test_every_audit_tool_is_reachable_through_tools(path):
    t = Tools()
    assert t.audit_status(path)["version"] == "v1"
    assert t.audit_pending(path)["pending"] == 0
    assert "no record" in t.audit_resolve(path, "ghost", 1.0)["error"]
    assert t.audit_recompute(path)["version"] == "v1"
    assert "metrics" in t.audit_scorecard(path)
    assert "error" in t.audit_rescan(path, "os:getcwd")
    assert t.audit_drift(path)["charted"] == 1


def test_the_audit_tool_descriptions_warn_the_model_about_the_two_traps():
    resolve = TOOL_DESCRIPTIONS["audit_resolve"]
    assert "REFUSES to overwrite" in resolve

    card = TOOL_DESCRIPTIONS["audit_scorecard"]
    assert "Do NOT" in card and "delta_hat" in card, (
        "a model told to watch the bias will recommend exactly the change that "
        "makes the verifier worse")

    rescan_desc = TOOL_DESCRIPTIONS["audit_rescan"]
    assert "RUNS the module" in rescan_desc and "do not widen it" in rescan_desc


# -- references used by the rescan tests -------------------------------------

def harsher(record, context):
    """Rejects every second answer the shipped verifier accepted."""
    if record.verifier_score > 0.0 and int(record.task_id.split("t")[-1]) % 2:
        return 0.0
    return record.verifier_score


def explodes(record, context):
    raise ValueError("this scorer is broken")
