"""Record the merge decision sequence, to compare a refactor against.

    python -m tools.gen_merge_golden          # write the fixture
    python -m tools.gen_merge_golden --check  # fail if the code has drifted

**Generate this before touching the code it guards.** A fixture written after a
refactor records the refactor's behaviour and agrees with it forever, which is
the one way this file can be worse than useless.

What it captures is the *interaction* between the merge steps -- whether
staleness runs before the trust region, whether cards dropped as contradictory
still count toward a denominator -- and interactions are what unit tests for the
individual steps do not cover. That is the whole reason for extracting them one
at a time and re-comparing after each.
"""

import argparse
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "tests"))

from goldens import WORKLOADS, run_workload, trace_of      # noqa: E402

FIXTURE = os.path.join(os.path.dirname(__file__), os.pardir,
                       "tests", "fixtures", "merge_trace.json")


def build() -> dict:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    return {name: trace_of(*run_workload(name)) for name in sorted(WORKLOADS)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="compare instead of writing; non-zero exit if different")
    args = ap.parse_args()

    traces = build()
    if args.check:
        with open(FIXTURE, encoding="utf-8") as fh:
            expected = json.load(fh)
        if traces == expected:
            print(f"{os.path.relpath(FIXTURE)} is up to date")
            return 0
        print("merge trace has drifted from the fixture", file=sys.stderr)
        for name in sorted(traces):
            if traces.get(name) != expected.get(name):
                print(f"  workload {name!r} differs", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
    with open(FIXTURE, "w", encoding="utf-8") as fh:
        json.dump(traces, fh, indent=2, sort_keys=True)
        fh.write("\n")
    total = sum(len(t["merges"]) for t in traces.values())
    print(f"wrote {os.path.relpath(FIXTURE)} "
          f"({len(traces)} workloads, {total} merge decisions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
