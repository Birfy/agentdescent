"""The merge decision sequence must not move while it is being taken apart.

`_process` runs seven things in order -- trust region, staleness, conflict,
fusion, audit, acceptance, commit -- and what breaks when they are pulled out
into separate objects is rarely any one of them. It is the *interaction*: which
runs first, whether a card dropped as contradictory still counts toward the
denominator the next step divides by, whether the candidate handed to the
acceptance gate is the fused one or the winner of the tournament.

No unit test of an individual policy covers that. This does, by comparing the
whole sequence against a fixture generated before the extraction started.

If a change here goes red, the fixture is right and the change is wrong, unless
you can say exactly which decision should have moved and why. Regenerating it to
make a diff go away converts the guard into a record of whatever happened last.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from goldens import WORKLOADS, run_workload, trace_of      # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "merge_trace.json"


@pytest.fixture(scope="module")
def expected():
    with open(FIXTURE, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.mark.parametrize("name", sorted(WORKLOADS))
def test_the_merge_sequence_is_unchanged(name, expected):
    got = trace_of(*run_workload(name))
    assert got == expected[name], (
        f"workload {name!r} decides differently than the recorded run. If that is "
        "intended, say which decision changed and why before regenerating "
        "tests/fixtures/merge_trace.json.")


def test_the_workloads_actually_reach_the_decisions_they_guard(expected):
    """A trace that never fuses and never conflicts would pass any refactor of
    the code that fuses and resolves conflicts."""
    merges = [m for trace in expected.values() for m in trace["merges"]]
    assert sum(m["fused"] for m in merges) > 0, "fusion never ran"
    assert sum(m["conflicts_dropped"] for m in merges) > 0, "conflict resolution never ran"
    categories = {m["category"] for m in merges}
    assert {"committed", "below-threshold"} <= categories, (
        f"acceptance only ever went one way: {categories}")
    assert any(m["committed_version"] for m in merges), "nothing was ever committed"


def test_the_fixture_carries_nothing_machine_specific():
    """A fixture holding a `$TMPDIR` path, a hostname or a duration goes red in
    CI for reasons unrelated to what it guards -- and the response to that is
    always to weaken it."""
    raw = FIXTURE.read_text(encoding="utf-8")
    for needle in ("/tmp", "/var/folders", "elapsed", "wallclock", "seconds",
                   os.path.expanduser("~")):
        assert needle not in raw, f"fixture embeds {needle!r}"


def test_the_trace_survives_a_different_environment(tmp_path):
    """Same decisions with `HOME`, `TMPDIR` and the working directory moved.

    `evolve()` makes its scratch ledger with `mkdtemp`, so a run is full of
    paths that differ every time; none of them may reach a decision."""
    repo = Path(__file__).resolve().parent.parent
    home = tmp_path / "home"
    tmp = tmp_path / "tmp"
    home.mkdir()
    tmp.mkdir()
    env = dict(os.environ, HOME=str(home), TMPDIR=str(tmp),
               PYTHONPATH=f"{repo}{os.pathsep}{repo / 'tests'}",
               PYTHONWARNINGS="ignore")
    snippet = (
        "import json;"
        "from goldens import run_workload, trace_of;"
        "print(json.dumps(trace_of(*run_workload('slot')), sort_keys=True))"
    )
    out = subprocess.run([sys.executable, "-c", snippet], cwd=str(tmp_path),
                         env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    got = json.loads(out.stdout)
    with open(FIXTURE, encoding="utf-8") as fh:
        assert got == json.loads(json.dumps(json.load(fh)["slot"], sort_keys=True))
