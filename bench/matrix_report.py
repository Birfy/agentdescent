"""Render the parallelisation matrix as the table issue #73 asks for.

Reads the cells `bench/matrix_run.py` wrote and emits one row per algorithm:
serial against the parallel and async arms, the speedup between them, the
held-out quality delta, and **what each arm changed about the published
algorithm**. That last column is not written here and not written by hand --
it is :data:`bench.matrix_run.SEMANTICS`, and a row with no entry there is
refused rather than rendered blank. A speedup table whose semantics column is
empty is a claim that nothing changed, made by omission.

    python -m bench.matrix_report --json bench/results/matrix.json

Three things this refuses to do quietly, because each of them turns a number
into a different number wearing the same name:

* **Report a speedup across unequal budgets.** The arms are only comparable
  because `--budget-rollouts` pinned them; where the cells recorded what they
  actually spent (`engine_rollouts`) and the arms disagree by more than the
  synchronous path's `N-1` barrier overshoot, the row says so instead of
  dividing.
* **Report a speedup off raw wall-clock when an arm hit bad endpoint weather.**
  `wall_seconds - failure_seconds` is the basis where the port reported the
  failure time; a retry storm in one arm is not a property of its scheduler.
* **Print a single seed as a result.** Fewer than three seeds renders the
  number with the count attached, so a pilot cannot be read as a measurement.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
from typing import Dict, Iterable, List, Optional, Sequence

from bench.matrix_run import ROWS, SEMANTICS

ARMS = ("serial", "parallel", "async")
#: Seeds below this cannot support a spread; the same bar as everywhere else.
MIN_SEEDS = 3


def _cells_by_arm(cells: Sequence[dict], row: str) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {arm: [] for arm in ARMS}
    for cell in cells:
        if cell.get("row") == row and cell.get("arm") in out and "error" not in cell:
            out[cell["arm"]].append(cell)
    return out


def _net_wall(cell: dict) -> Optional[float]:
    """Wall-clock net of time lost inside failed calls, when the port reported it.

    A cell whose endpoint spent ten minutes in retry backoff did not take that
    long to *evolve* anything, and the arm that was unlucky is not the arm with
    the worse scheduler.
    """
    wall = cell.get("wall_seconds")
    if wall is None:
        return None
    return max(0.0, wall - float(cell.get("failure_seconds", 0.0)))


def _median(values: Iterable[Optional[float]]) -> Optional[float]:
    rows = [v for v in values if v is not None]
    return statistics.median(rows) if rows else None


def _spread(values: Iterable[Optional[float]], digits: int = 2) -> str:
    """Median with the min-max range beside it -- never a bare point estimate.

    A single seed has no range to show, so it shows none; the row carries a
    dagger and a caveat line instead, which says the same thing once rather
    than in every cell.
    """
    rows = sorted(v for v in values if v is not None)
    if not rows:
        return "—"
    med = statistics.median(rows)
    if len(rows) == 1:
        return f"{med:.{digits}f}"
    return f"{med:.{digits}f} [{rows[0]:.{digits}f}–{rows[-1]:.{digits}f}]"


def _budget_note(arms: Dict[str, List[dict]], width: int) -> str:
    """Empty when the arms really did spend the same budget, a note when not.

    The pin is checked against what the cells measured rather than trusted
    because a flag was passed. The synchronous path checks at the round barrier,
    so a `width`-worker arm may legitimately overshoot by up to `width - 1`; the
    async path has no barrier and overshoots on its own schedule. A mismatch no
    longer withholds the speedup -- :func:`_seconds_per_rollout` divides it out
    -- but it is still worth saying which cells needed dividing.
    """
    spent = {arm: _median(c.get("engine_rollouts") for c in cells)
             for arm, cells in arms.items() if cells}
    known = {arm: v for arm, v in spent.items() if v is not None}
    if len(known) < 2:
        return ""
    low, high = min(known.values()), max(known.values())
    if high - low <= max(0, width - 1):
        return ""
    return (f"rollouts actually spent differ across arms "
            f"({', '.join(f'{a}={v:.0f}' for a, v in sorted(known.items()))})")


def _seconds_per_rollout(cell: dict) -> Optional[float]:
    """Wall-clock per rollout actually completed -- the unit a speedup divides.

    Raw wall-clock only compares arms that spent the same budget, and they do
    not: the sync path overshoots its budget at the round barrier by up to
    `n_workers - 1`, and the async path, having no barrier, overshoots on its
    own. Measured on the GEPA pilot the async arm completed **19** rollouts
    against the control's 16 -- so a raw wall-clock ratio there is a rate
    compared against a different amount of work, and it happens to *understate*
    the wider arm, since that arm did more.

    Dividing by the rollouts each cell actually completed makes the two sides
    the same unit: seconds per rollout. `engine_rollouts` is the engine's own
    count, not the requested budget, which is why the ports print it.
    """
    wall = _net_wall(cell)
    done = cell.get("engine_rollouts")
    if wall is None or not done:
        return None
    return wall / float(done)


def _speedup(arms: Dict[str, List[dict]], arm: str, blocked: bool = False) -> str:
    base = _median(_seconds_per_rollout(c) for c in arms["serial"])
    other = _median(_seconds_per_rollout(c) for c in arms[arm])
    if not base or not other:
        # No `engine_rollouts` on one side: cells written before the ports
        # printed it. Fall back to raw wall-clock rather than to nothing, and
        # mark it, because that ratio carries whatever budget gap the arms had.
        base = _median(_net_wall(c) for c in arms["serial"])
        other = _median(_net_wall(c) for c in arms[arm])
        if not base or not other:
            return "—"
        return f"{base / other:.2f}× ‡"
    return f"{base / other:.2f}×"


def _stale(arms: Dict[str, List[dict]]) -> str:
    """The async arm's stale rate, numerator with denominator or not at all."""
    considered = [c.get("stale_considered") for c in arms["async"]]
    discarded = [c.get("stale_discarded") for c in arms["async"]]
    total_c = sum(v for v in considered if v is not None)
    total_d = sum(v for v in discarded if v is not None)
    if not any(v is not None for v in considered):
        return "not recorded"
    if not total_c:
        return "0 considered"
    return f"{total_d}/{total_c} ({total_d / total_c:.0%})"


def _quality(arms: Dict[str, List[dict]], arm: str) -> str:
    """Held-out score for an arm, and its delta against the serial control."""
    base = _median(c.get("test") for c in arms["serial"])
    here = _median(c.get("test") for c in arms[arm])
    if here is None:
        return "—"
    shown = _spread((c.get("test") for c in arms[arm]), digits=3)
    if base is None:
        return shown
    return f"{shown} (Δ {here - base:+.3f})"


def _seed_note(arms: Dict[str, List[dict]]) -> str:
    counts = {arm: len(cells) for arm, cells in arms.items() if cells}
    if not counts:
        return ""
    fewest = min(counts.values())
    if fewest >= MIN_SEEDS:
        return ""
    return (f"{fewest} seed{'s' if fewest != 1 else ''} — below the {MIN_SEEDS} "
            f"this repository requires before a number is a result")


def render(payload: dict) -> str:
    """The issue-73 table, plus the notes that keep each row readable."""
    cells = payload.get("cells", [])
    width = int(payload.get("width", 8) or 8)
    budget = payload.get("budget_rollouts")
    measured = {c.get("row") for c in cells}

    out: List[str] = []
    out.append("# The parallelisation matrix — measured")
    out.append("")
    out.append(f"Generated by `bench/matrix_report.py` from "
               f"`{payload.get('_source', 'the cells file')}`. "
               f"Budget: **{budget} rollouts pinned on every arm**; "
               f"parallel and async width {width}; "
               f"async lag budget {payload.get('async_ratio', '?')}; "
               f"model `{payload.get('model')}` via `{payload.get('provider')}`.")
    out.append("")
    out.append("Speedup is seconds per completed rollout, from wall-clock **net "
               "of time lost inside failed calls** — so one arm's retry storm is "
               "not reported as the other arm's scheduler, and an arm that "
               "overran its budget is not credited with the extra work. Quality "
               "is the held-out test score: median across seeds, with the range, "
               "and the delta against the serial control. A `‡` marks a row "
               "whose cells predate `engine_rollouts` and fall back to raw "
               "wall-clock.")
    out.append("")
    out.append("| Algorithm | Dataset | Serial | Parallel N=" + str(width)
               + " | Async N=" + str(width) + " | Speedup sync / async "
               "| Held-out Δ sync / async | Stale (async) |")
    out.append("|---|---|---|---|---|---|---|---|")

    notes: List[str] = []
    for row in ROWS:
        name = row["name"]
        if name not in SEMANTICS:
            raise SystemExit(
                f"row {name!r} has no entry in bench.matrix_run.SEMANTICS. The "
                f"semantics column is not optional: a speedup published beside a "
                f"blank one claims by omission that the algorithm was untouched.")
        arms = _cells_by_arm(cells, name)
        if name not in measured:
            out.append(f"| {name} | {row['dataset']} | — | — | — | — | — | — |")
            continue
        walls = {arm: _spread((_net_wall(c) for c in arms[arm]), digits=0)
                 for arm in ARMS}
        # The dagger travels with the row so a reader scanning only the table
        # cannot mistake a pilot for a measurement.
        label = name + (" †" if _seed_note(arms) else "")
        out.append(
            f"| {label} | {row['dataset']} "
            f"| {walls['serial']} s | {walls['parallel']} s | {walls['async']} s "
            f"| {_speedup(arms, 'parallel')} / {_speedup(arms, 'async')} "
            f"| {_quality(arms, 'parallel')} / {_quality(arms, 'async')} "
            f"| {_stale(arms)} |")
        for marker, text in (("", _budget_note(arms, width)),
                             ("† ", _seed_note(arms))):
            if text:
                notes.append(f"* {marker}**{name}** — {text}")

    if notes:
        out += ["", "## Caveats the cells themselves raised", ""] + notes

    out += ["", "## What each arm changed about the published algorithm", "",
            "The column issue #73 says the table is worthless without. Source: "
            "`bench.matrix_run.SEMANTICS`, declared beside the rows themselves "
            "rather than written up afterwards.", "",
            "| Algorithm | Serial (the control) | Parallel | Async |",
            "|---|---|---|---|"]
    for row in ROWS:
        sem = SEMANTICS[row["name"]]
        out.append(f"| {row['name']} | {sem['serial']} | {sem['parallel']} "
                   f"| {sem['async']} |")

    out += ["", "A row reading anything other than *rollout scheduling and merge "
            "timing only* is either a bug or a finding, and either way its "
            "speedup and quality numbers have to be read through it. "
            "**The quality column is allowed to go down** — a table of all-green "
            "more likely means the control arm was not really the control than "
            "that asynchrony is free.", ""]
    return "\n".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--json", default="bench/results/matrix.json")
    p.add_argument("--out", default="", help="write here instead of stdout")
    args = p.parse_args(argv)

    path = pathlib.Path(args.json)
    if not path.exists():
        raise SystemExit(f"no cells at {path} -- run `python -m bench.matrix_run` first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["_source"] = str(path)
    text = render(payload)
    if args.out:
        pathlib.Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
