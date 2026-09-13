"""Record the PPI estimator's output on six fixed inputs.

    python -m tools.gen_audit_ppi_golden          # write the fixture
    python -m tools.gen_audit_ppi_golden --check  # fail if the code has drifted

**Run this before changing `ppi.py`, never after.** A fixture written after a
change records the change and agrees with it forever, which is the one way a
golden file is worse than no golden file at all.

What it guards that the coverage tests do not: coverage is a property of 400
replications and moves by a point or two under any small change, so it cannot
tell a refactor from a regression. These vectors are pinned to 1e-9 relative --
tight enough that any change worth catching trips it, loose enough to survive a
numpy build summing a reduction in a different order. A refactor that preserves
behaviour leaves them alone; one that does not fails immediately and says which
case and which field.

Regenerating is legitimate when the numerical behaviour is **meant** to change --
a better `lam`, a different fold scheme. Then the diff on this file is the
review: every number that moved, in one place, next to the change that moved it.
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "tests"))

from audit_ppi_cases import CASES, build                    # noqa: E402
from agentdescent.audit.ppi import ppi_mean_stratified      # noqa: E402

FIXTURE = os.path.join(os.path.dirname(__file__), os.pardir,
                       "tests", "fixtures", "ppi_golden.json")


def record() -> dict:
    """Every case, run at a pinned seed, as plain JSON."""
    out = {}
    for name, params in CASES.items():
        strata, truth = build(**params)
        r = ppi_mean_stratified(strata, seed=params["seed"])
        out[name] = {
            "truth": truth,
            "theta": r.theta,
            "ci": list(r.ci),
            "se": r.se,
            "df": r.df,
            "lambda": r.lambda_,
            "gain_factor": r.gain_factor,
            "n": r.n,
            "n_unlab": r.n_unlab,
            # Recorded in full: a caller stores the estimate and the reason it
            # is weak together, so a change in the wording is a change in the
            # record and should be reviewed as one.
            "warnings": list(r.warnings),
            "per_stratum": {
                k: {kk: vv for kk, vv in cell.items()}
                for k, cell in r.per_stratum.items()
            },
        }
    return out


#: How far a recorded number may move before the drift is real. A refactor that
#: preserves behaviour reproduces these to the last bit on one machine, but not
#: across machines: `df` is a ratio of sums of squares over numpy reductions, and
#: numpy is free to sum them in a different order on a different build. Two runs
#: that agree to 1e-9 relative differ by nothing anyone can act on, while a
#: change worth catching -- a different `lam`, a different fold scheme -- moves
#: these by 1e-3 or more. Bit-equality here made CI contradict itself across the
#: matrix on one unchanged tree: green on 3.11, red on 3.12 by 7e-15 of `df`.
REL_TOL = 1e-9
ABS_TOL = 1e-12


def differences(fresh: dict, stored: dict, path: str = "") -> list:
    """Every field where the two disagree beyond the tolerance, by name.

    The old check compared serialised JSON, so it could say *that* the vectors
    moved but never *which* -- and the diff is the whole point of a golden file.
    """
    out = []
    if isinstance(fresh, dict) and isinstance(stored, dict):
        for key in sorted(set(fresh) | set(stored)):
            if key not in stored:
                out.append(f"{path}/{key}: added ({fresh[key]!r})")
            elif key not in fresh:
                out.append(f"{path}/{key}: removed ({stored[key]!r})")
            else:
                out += differences(fresh[key], stored[key], f"{path}/{key}")
    elif isinstance(fresh, list) and isinstance(stored, list):
        if len(fresh) != len(stored):
            out.append(f"{path}: length {len(stored)} -> {len(fresh)}")
        else:
            for i, (a, b) in enumerate(zip(fresh, stored)):
                out += differences(a, b, f"{path}[{i}]")
    elif isinstance(fresh, (int, float)) and isinstance(stored, (int, float)) \
            and not isinstance(fresh, bool) and not isinstance(stored, bool):
        if not math.isclose(fresh, stored, rel_tol=REL_TOL, abs_tol=ABS_TOL):
            out.append(f"{path}: {stored!r} -> {fresh!r}")
    elif fresh != stored:
        out.append(f"{path}: {stored!r} -> {fresh!r}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if the recorded vectors no longer match")
    args = ap.parse_args(argv)

    fresh = record()
    if args.check:
        if not os.path.exists(FIXTURE):
            print(f"{FIXTURE} does not exist -- run without --check", file=sys.stderr)
            return 1
        with open(FIXTURE, encoding="utf-8") as fh:
            stored = json.load(fh)
        moved = differences(fresh, stored)
        if moved:
            print("ppi golden vectors have drifted; re-run "
                  "`python -m tools.gen_audit_ppi_golden` if the change was "
                  "intended, and review the diff.", file=sys.stderr)
            for line in moved:
                print(f"  {line}", file=sys.stderr)
            return 1
        print(f"{len(fresh)} vectors match")
        return 0

    os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
    with open(FIXTURE, "w", encoding="utf-8") as fh:
        json.dump(fresh, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"wrote {FIXTURE} ({len(fresh)} vectors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
