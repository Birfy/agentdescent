"""Draw the 2F1 stress set and record its reference values.

    python -m tools.gen_hyp2f1_stress            # write the suite
    python -m tools.gen_hyp2f1_stress --check    # fail if the file has drifted

The output, ``examples/era/data/hyp2f1_stress.json``, is **committed**. The
example that consumes it therefore needs no mpmath, no network and no
regeneration: anyone can rerun the benchmark offline and get the same points and
the same references. This script exists so the references are auditable rather
than asserted -- it is the only place they are produced, and it is short enough
to read in one sitting.

Three properties are what make the resulting numbers worth reporting.

**The distribution is declared, not tuned.** ``a, b, c ~ U(-30, 30)`` and
``z ~ U(-40, 0.999)``, fixed before anything was measured and unchanged since.
A stress set drawn *after* looking at where an implementation fails would
measure the drawing, not the implementation.

**The reference is independent of everything being scored.** Values come from
mpmath, an arbitrary-precision library that shares no code with SciPy and none
with any candidate program. Nothing in the sandbox can reach it.

**The reference is checked against itself.** Every point is evaluated twice, at
30 and at 60 decimal digits, and is *discarded* unless the two agree to 25
digits. That is what makes "SciPy got 2 digits here" a statement about SciPy: a
point where the reference is not itself converged never enters the suite.

Degenerate points are dropped rather than scored: ``c`` at or near a
non-positive integer (the function has a pole), and values below 1e-250 in
magnitude (a relative error against them is not a measurement).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


OUTPUT = Path(__file__).resolve().parents[1] / "examples" / "era" / "data" / "hyp2f1_stress.json"

#: Declared up front; see the module docstring.
PARAMETER_RANGE = 30.0
Z_LOW, Z_HIGH = -40.0, 0.999
SHARDS = 12
#: 250 a shard, so the acceptance gate sees 1000 points rather than 80. That
#: number is not a preference, it is what the measurement needs: per-point
#: correct digits have a standard deviation of 3.39 on this suite -- the
#: outcome is close to bimodal, a program either handles a region or scores
#: near zero there -- so an 80-point mean carries a standard error of 0.38
#: digits. A 48-expansion run moved that gate by 0.56, which is 1.5 SE: inside
#: the noise. Resolving a +0.2 digit improvement at two standard errors needs
#: about 1100 points, and pairing barely helps (the paired difference has an SD
#: of 3.05, because when a program changes a point it changes it by ten digits,
#: not by a tenth). Evaluation is 0.7% of a run's wall clock, so the points are
#: nearly free; only this generator pays, once.
POINTS_PER_SHARD = 250
SEED = 20260819

#: Digits kept in the stored reference. Twice what a float64 result can use, so
#: the scoring never sees the reference's own truncation.
REFERENCE_DIGITS = 25
LOW_DPS, HIGH_DPS = 30, 60
MIN_MAGNITUDE = "1e-250"


def draw(rng: random.Random) -> Dict[str, float]:
    return {
        "a": rng.uniform(-PARAMETER_RANGE, PARAMETER_RANGE),
        "b": rng.uniform(-PARAMETER_RANGE, PARAMETER_RANGE),
        "c": rng.uniform(-PARAMETER_RANGE, PARAMETER_RANGE),
        "z": rng.uniform(Z_LOW, Z_HIGH),
    }


def build_shard(index: int, rejected: Optional[Dict[str, int]] = None) -> List[Dict[str, Any]]:
    """Draw one shard, and only that shard.

    Seeded per shard rather than from one stream across all of them, so a test
    can redraw shard 0 alone and compare against the committed file. Verifying
    the whole file means the full generation -- twelve minutes of
    arbitrary-precision arithmetic -- and a check nobody runs is not a check.
    A string seed hashes with SHA-512 rather than ``hash()``, so the draw does
    not move with ``PYTHONHASHSEED``.
    """
    import mpmath as mp  # generator-only dependency, deliberately not the example's

    counts = rejected if rejected is not None else {}
    rng = random.Random(f"era-hyp2f1:{SEED}:{index}")
    shard: List[Dict[str, Any]] = []
    while len(shard) < POINTS_PER_SHARD:
        point = draw(rng)
        c = point["c"]
        if abs(c - round(c)) < 1e-3 and round(c) <= 0:
            counts["pole"] = counts.get("pole", 0) + 1
            continue
        try:
            mp.mp.dps = LOW_DPS
            low = mp.hyp2f1(point["a"], point["b"], point["c"], point["z"])
            mp.mp.dps = HIGH_DPS
            high = mp.hyp2f1(point["a"], point["b"], point["c"], point["z"])
        except Exception:
            counts["failed"] = counts.get("failed", 0) + 1
            continue
        if not mp.isfinite(high) or abs(high) < mp.mpf(MIN_MAGNITUDE):
            counts["tiny"] = counts.get("tiny", 0) + 1
            continue
        if abs(mp.mpf(low) - high) / abs(high) > mp.mpf(f"1e-{REFERENCE_DIGITS}"):
            counts["unstable"] = counts.get("unstable", 0) + 1
            continue
        shard.append({
            # `repr` of a float round-trips exactly, so the point a candidate is
            # handed is bit-for-bit the point the reference was computed at.
            "a": repr(point["a"]), "b": repr(point["b"]),
            "c": repr(point["c"]), "z": repr(point["z"]),
            "value": mp.nstr(high, REFERENCE_DIGITS, strip_zeros=False),
        })
    return shard


def build() -> Dict[str, Any]:
    import mpmath as mp  # for the version recorded below

    rejected = {"pole": 0, "tiny": 0, "unstable": 0, "failed": 0}
    shards = [build_shard(index, rejected) for index in range(SHARDS)]

    return {
        "task": "Gauss hypergeometric 2F1(a, b; c; z), real parameters, float64",
        "reference": {
            "library": "mpmath",
            "version": mp.__version__,
            "precisions_dps": [LOW_DPS, HIGH_DPS],
            "kept_when_they_agree_to_digits": REFERENCE_DIGITS,
            "stored_digits": REFERENCE_DIGITS,
        },
        "distribution": {
            "a": f"U(-{PARAMETER_RANGE}, {PARAMETER_RANGE})",
            "b": f"U(-{PARAMETER_RANGE}, {PARAMETER_RANGE})",
            "c": f"U(-{PARAMETER_RANGE}, {PARAMETER_RANGE})",
            "z": f"U({Z_LOW}, {Z_HIGH})",
            "rejected": ("c within 1e-3 of a non-positive integer; |value| < "
                         f"{MIN_MAGNITUDE}; the two precisions disagreeing"),
        },
        "seed": SEED,
        "rejected_counts": rejected,
        "shards": shards,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="exit 1 if the committed file differs from a fresh draw")
    args = parser.parse_args(argv)

    payload = build()
    text = json.dumps(payload, indent=1, sort_keys=True) + "\n"
    if args.check:
        if not OUTPUT.exists():
            print(f"{OUTPUT} is missing", file=sys.stderr)
            return 1
        current = OUTPUT.read_text(encoding="utf-8")
        if current != text:
            print(f"{OUTPUT} is out of date; rerun python -m tools.gen_hyp2f1_stress",
                  file=sys.stderr)
            return 1
        print(f"{OUTPUT} is up to date")
        return 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(text, encoding="utf-8")
    counts = payload["rejected_counts"]
    print(f"wrote {OUTPUT}: {SHARDS} shards x {POINTS_PER_SHARD} points, "
          f"rejected {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
