"""Record the PPI estimator's output on six fixed inputs.

    python -m tools.gen_audit_ppi_golden          # write the fixture
    python -m tools.gen_audit_ppi_golden --check  # fail if the code has drifted

**Run this before changing `ppi.py`, never after.** A fixture written after a
change records the change and agrees with it forever, which is the one way a
golden file is worse than no golden file at all.

What it guards that the coverage tests do not: coverage is a property of 400
replications and moves by a point or two under any small change, so it cannot
tell a refactor from a regression. These vectors are exact to 1e-12. A refactor
that preserves behaviour leaves them alone; one that does not fails immediately
and says which case and which field.

Regenerating is legitimate when the numerical behaviour is **meant** to change --
a better `lam`, a different fold scheme. Then the diff on this file is the
review: every number that moved, in one place, next to the change that moved it.
"""

import argparse
import json
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
        if json.dumps(stored, sort_keys=True) != json.dumps(fresh, sort_keys=True):
            print("ppi golden vectors have drifted; re-run "
                  "`python -m tools.gen_audit_ppi_golden` if the change was "
                  "intended, and review the diff.", file=sys.stderr)
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
