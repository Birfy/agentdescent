"""Six recorded vectors: the estimator must still say exactly what it said.

Coverage tests answer "is it right"; these answer "did it change". Both are
needed and neither substitutes for the other -- coverage moves a point or two
under any small change, so it cannot separate a refactor from a regression,
while these are exact to 1e-12 and cannot tell you whether the behaviour they
pin is *correct*. The coverage suite makes that claim; this one freezes it.

    python -m tools.gen_audit_ppi_golden        # regenerate, deliberately
"""

import json
import os
import subprocess
import sys

import pytest

from audit_ppi_cases import CASES, build
from agentdescent.audit.ppi import ppi_mean_stratified

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(ROOT, "tests", "fixtures", "ppi_golden.json")


@pytest.fixture(scope="module")
def golden():
    with open(FIXTURE, encoding="utf-8") as fh:
        return json.load(fh)


def test_every_case_is_recorded(golden):
    """A case added without regenerating guards nothing, and says nothing."""
    assert set(golden) == set(CASES)


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_estimator_still_says_what_it_said(golden, name):
    strata, truth = build(**CASES[name])
    r = ppi_mean_stratified(strata, seed=CASES[name]["seed"])
    want = golden[name]

    assert truth == pytest.approx(want["truth"], abs=1e-12)
    assert r.theta == pytest.approx(want["theta"], abs=1e-12)
    assert list(r.ci) == pytest.approx(want["ci"], abs=1e-12)
    assert r.se == pytest.approx(want["se"], abs=1e-12)
    assert r.df == pytest.approx(want["df"], abs=1e-9)
    assert r.lambda_ == pytest.approx(want["lambda"], abs=1e-12)
    assert r.gain_factor == pytest.approx(want["gain_factor"], abs=1e-9)
    assert r.n == want["n"] and r.n_unlab == want["n_unlab"]
    assert list(r.warnings) == want["warnings"]
    for stratum, cell in want["per_stratum"].items():
        for key, value in cell.items():
            assert r.per_stratum[stratum][key] == pytest.approx(value, abs=1e-12)


def test_the_recorded_estimates_are_close_to_the_truth_they_estimate(golden):
    """A guard on the fixture itself.

    Vectors recorded from a broken estimator would still match it forever. This
    is the cheapest independent check available: on every case the recorded
    estimate must land near the population mean it was computed from, and the
    recorded interval must contain it.
    """
    for name, want in golden.items():
        assert abs(want["theta"] - want["truth"]) < 0.12, name
        lo, hi = want["ci"]
        assert lo <= want["truth"] <= hi, f"{name}: {want['ci']} misses {want['truth']}"


def test_the_cases_span_the_regimes_they_claim_to(golden):
    """Six vectors in the middle of the space would agree with anything."""
    lams = {n: w["lambda"] for n, w in golden.items()}
    assert lams["useless_verifier"] < 0.2, "the useless case must collapse lam"
    assert lams["perfect_verifier"] > 0.8, "the perfect case must saturate lam"
    assert lams["no_unlabelled"] == 0.0, "nothing to borrow means lam is 0"
    assert golden["perfect_verifier"]["gain_factor"] > 3.0
    assert golden["no_unlabelled"]["gain_factor"] == pytest.approx(1.0)
    assert any("MIN_N_DOMINANT" in w
               for w in golden["thin_dominant_stratum"]["warnings"])


def test_the_generator_check_mode_agrees_with_the_fixture():
    """`--check` is what CI would run; it must pass on a clean tree."""
    proc = subprocess.run([sys.executable, "-m", "tools.gen_audit_ppi_golden",
                           "--check"], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
