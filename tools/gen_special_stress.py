"""Draw a stress set for one SciPy entry point and record its reference values.

    python -m tools.gen_special_stress --target pbdv
    python -m tools.gen_special_stress --target hyperu --check
    python -m tools.gen_special_stress --all

The outputs, ``examples/era/data/<target>_stress.json``, are **committed**, so
the examples that consume them need no mpmath, no network and no regeneration.
This is :mod:`tools.gen_hyp2f1_stress` generalised to the targets that
:mod:`tools.scan_numeric_precision` found -- same discipline, one table instead
of one hard-coded function.

The three properties that make the resulting numbers reportable are the ones
the 2F1 generator already states, and they are not weakened here:

**The distribution is declared, not tuned.** Each target's range below is
*character for character the range the sweep declared* before any of these
functions had been measured. The sweep found where SciPy fails; the stress set
does not then narrow onto that region. A suite drawn around known failures
would measure the drawing, and the honest number -- "what does this function do
over the range a user might plausibly call it on" -- would be lost.

**The reference is independent of everything being scored.** Values come from
mpmath, which shares no code with SciPy and none with any candidate program.
Nothing inside the sandbox can reach it.

**The reference is checked against itself.** Every point is evaluated at 30 and
at 60 decimal digits and *discarded* unless the two agree to 25. That is what
makes "SciPy got zero digits here" a statement about SciPy rather than about
mpmath: a point where the reference has not converged never enters the suite.

Degenerate points are dropped rather than scored -- poles, and values whose
magnitude is below 1e-250 or above 1e250, where a float64 result is bounded by
the format rather than by the algorithm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "era" / "data"

SHARDS = 12
#: 250 a shard, so the acceptance gate sees 2000 points and 1000 are held back.
#: Sized from the sweep, not from taste: per-point correct digits have a
#: standard deviation of 4.78 on `pbdv` and 2.83 on `hyperu` -- these outcomes
#: are close to bimodal, a program either handles a region or scores zero
#: there -- so a 2000-point gate carries a standard error of 0.11 digits and a
#: 1000-point held-back set 0.15. An 80-point gate, which is what a smaller
#: suite would give, could not tell a half-digit gain from noise.
POINTS_PER_SHARD = 250
SEED = 20260825

REFERENCE_DIGITS = 25
LOW_DPS, HIGH_DPS = 30, 60
MIN_MAGNITUDE = "1e-250"
MAX_MAGNITUDE = "1e250"


def _logu(rng: random.Random, low: float, high: float) -> float:
    return math.exp(rng.uniform(math.log(low), math.log(high)))


@dataclass(frozen=True)
class Target:
    """One SciPy entry point, its declared input range, and its reference."""

    key: str
    task: str
    #: The function a candidate program must define.
    entrypoint: str
    #: Ordered parameter names -- the call signature, and the JSON keys.
    params: Tuple[str, ...]
    #: Human-readable declared distribution, recorded in the file.
    distribution: Dict[str, str]
    #: ``rng -> {name: float}``.
    draw: Callable[[random.Random], Dict[str, float]]
    #: ``point -> mpf``, under whatever precision mpmath is set to.
    reference: Callable[[Dict[str, float], Any], Any]
    #: What the sweep measured for the baseline, recorded for the record.
    baseline_scan: Dict[str, Any]
    #: ``point -> bool``; True drops the point before any evaluation.
    reject: Optional[Callable[[Dict[str, float]], bool]] = None
    reject_note: str = ""


TARGETS: Tuple[Target, ...] = (
    Target(
        key="pbdv",
        task=("Parabolic cylinder function D_v(x), real order and argument, "
              "float64"),
        entrypoint="pbdv",
        params=("v", "x"),
        distribution={"v": "U(-20.0, 20.0)", "x": "U(-30.0, 30.0)"},
        draw=lambda r: {"v": r.uniform(-20.0, 20.0), "x": r.uniform(-30.0, 30.0)},
        reference=lambda p, mp: mp.pcfd(p["v"], p["x"]),
        baseline_scan={
            "baseline": "scipy.special.pbdv(v, x)[0]",
            "mean_digits": 11.67,
            "frac_below_8_digits": 0.178,
            "frac_below_1_digit": 0.122,
            "note": ("the sweep's worst non-2F1 target: on v >~ 15 with x < 0 "
                     "SciPy returns values wrong by 20 orders of magnitude, and "
                     "with the wrong sign"),
        },
    ),
    Target(
        key="pbvv",
        task=("Parabolic cylinder function V_v(x), real order and argument, "
              "float64"),
        entrypoint="pbvv",
        params=("v", "x"),
        distribution={"v": "U(-20.0, 20.0)", "x": "U(-30.0, 30.0)"},
        draw=lambda r: {"v": r.uniform(-20.0, 20.0), "x": r.uniform(-30.0, 30.0)},
        # SciPy indexes V by the same v as its D_v; mpmath indexes by DLMF's
        # `a`, and U(a, z) = D_{-a-1/2}(z), so a = -v - 1/2. Verified against
        # SciPy on points where both are reliable before being used to judge
        # points where one is not.
        reference=lambda p, mp: mp.pcfv(-p["v"] - 0.5, p["x"]),
        baseline_scan={
            "baseline": "scipy.special.pbvv(v, x)[0]",
            "mean_digits": 11.87,
            "frac_below_8_digits": 0.167,
            "frac_below_1_digit": 0.094,
            "note": ("pbdv's sibling and the same disease: on v >~ 9 with x < 0 "
                     "SciPy returns values wrong by up to eleven orders of "
                     "magnitude, and with the wrong sign"),
        },
    ),
    Target(
        key="hyperu",
        task=("Confluent hypergeometric function of the second kind U(a, b, x), "
              "real parameters, x > 0, float64"),
        entrypoint="hyperu",
        params=("a", "b", "x"),
        distribution={"a": "U(-20.0, 20.0)", "b": "U(-20.0, 20.0)",
                      "x": "logU(1e-3, 100.0)"},
        draw=lambda r: {"a": r.uniform(-20.0, 20.0), "b": r.uniform(-20.0, 20.0),
                        "x": _logu(r, 1e-3, 100.0)},
        reference=lambda p, mp: mp.hyperu(p["a"], p["b"], p["x"]),
        baseline_scan={
            "baseline": "scipy.special.hyperu(a, b, x)",
            "mean_digits": 14.36,
            "frac_below_8_digits": 0.030,
            "frac_below_1_digit": 0.028,
            "note": ("SciPy returns nan -- not an inaccurate value, no value at "
                     "all -- on about 3% of the range, concentrated on a < 0; "
                     "the function is finite and well-conditioned there"),
        },
    ),
)

TARGETS_BY_KEY = {target.key: target for target in TARGETS}


def output_for(target: Target) -> Path:
    return DATA_DIR / f"{target.key}_stress.json"


def build_shard(target: Target, index: int,
                rejected: Optional[Dict[str, int]] = None) -> List[Dict[str, Any]]:
    """Draw one shard, and only that shard.

    Seeded per shard rather than from one stream across all of them, so a test
    can redraw shard 0 alone and compare it against the committed file --
    verifying the whole file means the full generation, and a check nobody runs
    is not a check. The seed is a string hashed by :mod:`random` itself, so the
    draw does not move with ``PYTHONHASHSEED``.
    """
    import mpmath as mp  # generator-only dependency, deliberately not the example's

    counts = rejected if rejected is not None else {}
    rng = random.Random(f"era-{target.key}:{SEED}:{index}")
    shard: List[Dict[str, Any]] = []
    while len(shard) < POINTS_PER_SHARD:
        point = target.draw(rng)
        if target.reject is not None and target.reject(point):
            counts["pole"] = counts.get("pole", 0) + 1
            continue
        try:
            mp.mp.dps = LOW_DPS
            low = target.reference(point, mp)
            mp.mp.dps = HIGH_DPS
            high = target.reference(point, mp)
        except Exception:  # noqa: BLE001 - NoConvergence and friends are a reject
            counts["failed"] = counts.get("failed", 0) + 1
            continue
        finally:
            mp.mp.dps = LOW_DPS
        try:
            low, high = mp.mpf(low.real), mp.mpf(high.real)
            if not mp.isfinite(high):
                counts["tiny"] = counts.get("tiny", 0) + 1
                continue
            magnitude = abs(high)
            if magnitude < mp.mpf(MIN_MAGNITUDE) or magnitude > mp.mpf(MAX_MAGNITUDE):
                counts["tiny"] = counts.get("tiny", 0) + 1
                continue
            if abs(low - high) / magnitude > mp.mpf(f"1e-{REFERENCE_DIGITS}"):
                counts["unstable"] = counts.get("unstable", 0) + 1
                continue
        except Exception:  # noqa: BLE001
            counts["failed"] = counts.get("failed", 0) + 1
            continue
        row = {
            # `repr` of a float round-trips exactly, so the point a candidate is
            # handed is bit-for-bit the point the reference was computed at.
            name: repr(point[name]) for name in target.params
        }
        row["value"] = mp.nstr(high, REFERENCE_DIGITS, strip_zeros=False)
        shard.append(row)
    return shard


def build(target: Target) -> Dict[str, Any]:
    import mpmath as mp  # for the version recorded below

    rejected = {"pole": 0, "tiny": 0, "unstable": 0, "failed": 0}
    shards = [build_shard(target, index, rejected) for index in range(SHARDS)]
    distribution = dict(target.distribution)
    distribution["rejected"] = (
        (target.reject_note + "; " if target.reject_note else "")
        + f"|value| outside [{MIN_MAGNITUDE}, {MAX_MAGNITUDE}]; "
        "the two precisions disagreeing"
    )
    return {
        "task": target.task,
        "target": target.baseline_scan["baseline"],
        "entrypoint": target.entrypoint,
        "params": list(target.params),
        "baseline_scan": target.baseline_scan,
        "reference": {
            "library": "mpmath",
            "version": mp.__version__,
            "precisions_dps": [LOW_DPS, HIGH_DPS],
            "kept_when_they_agree_to_digits": REFERENCE_DIGITS,
            "stored_digits": REFERENCE_DIGITS,
        },
        "distribution": distribution,
        "seed": SEED,
        "rejected_counts": rejected,
        "shards": shards,
    }


def write_or_check(target: Target, *, check: bool) -> int:
    payload = build(target)
    text = json.dumps(payload, indent=1, sort_keys=True) + "\n"
    path = output_for(target)
    if check:
        if not path.exists():
            print(f"{path} is missing", file=sys.stderr)
            return 1
        if path.read_text(encoding="utf-8") != text:
            print(f"{path} is out of date; rerun "
                  f"python -m tools.gen_special_stress --target {target.key}",
                  file=sys.stderr)
            return 1
        print(f"{path} is up to date")
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    print(f"wrote {path}: {SHARDS} shards x {POINTS_PER_SHARD} points, "
          f"rejected {payload['rejected_counts']}, sha256:{digest}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", choices=sorted(TARGETS_BY_KEY),
                        help="which stress set to draw")
    parser.add_argument("--all", action="store_true", help="every target")
    parser.add_argument("--check", action="store_true",
                        help="exit 1 if a committed file differs from a fresh draw")
    args = parser.parse_args(argv)
    if not args.target and not args.all:
        parser.error("give --target KEY or --all")
    chosen = TARGETS if args.all else (TARGETS_BY_KEY[args.target],)
    return max(write_or_check(target, check=args.check) for target in chosen)


if __name__ == "__main__":
    raise SystemExit(main())
