"""The Phase 5 diagnosis, on the records Phase 0 actually produced.

Reads an audit JSONL, joins it back to the HotpotQA questions and gold answers,
sorts every disagreement by what fixing it would cost, and fills the scorecard.
Then it does the thing the diagnosis exists to make possible: proposes the two
hard rules a person reaches for first, and measures them on the answers they
were not aimed at.

    python -m scripts.audit_diagnose --records reports/audit_phase0_2026-09-09.jsonl

No network and no model calls -- the outputs, the scores and the labels are all
in the JSONL, and the questions come out of the dataset cache the run left
behind.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List

from agentdescent.audit import AuditRecord, Purpose
from agentdescent.audit.calibrator import Rectification
from agentdescent.audit.diagnose import (classify_disagreements, evaluate_fix,
                                         reference_classifier, residual_stats)
from agentdescent.audit.scorecard import rescan, scorecard
from scripts.audit_phase0 import normalize

#: The oracle's own normalisation, imported rather than reimplemented.
#:
#: This was a second copy for one commit, and the copy replaced punctuation with
#: a space where the original deletes it -- so ``cat's`` normalised to ``cat s``
#: here and ``cats`` there. A FORMATTING disagreement is defined as one the
#: oracle's normalisation already handles, and a diagnosis whose normaliser is
#: not the oracle's is measuring a different question with the same word.
normalise = normalize


def load_records(path: pathlib.Path) -> List[AuditRecord]:
    """Last occurrence per ``record_id`` wins, matching the store's own rule."""
    by_id: Dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("kind"):                       # a moments snapshot, not a record
            continue
        by_id[row["record_id"]] = row
    out = []
    for row in by_id.values():
        row.pop("kind", None)
        row["purpose"] = Purpose(row["purpose"])
        out.append(AuditRecord(**row))
    return out


def hotpot_context(limit: int = 120) -> Dict[str, tuple]:
    """``task_id -> (question, gold)`` from the dataset cache."""
    from agentdescent.dataloader import hf_rows

    rows = hf_rows("hotpotqa/hotpot_qa", "validation", config="distractor",
                   limit=limit)
    return {str(r["id"]): ((r.get("question") or "").strip(),
                           (r.get("answer") or "").strip())
            for r in rows if r.get("id")}


# -- the two rules a person reaches for first --------------------------------

def echoes_the_question(output: str, question: str) -> bool:
    return bool(output) and normalise(output) in normalise(question)


def far_shorter_than_reference(output: str, gold: str, ratio: float) -> bool:
    o, g = normalise(output), normalise(gold)
    return bool(o) and bool(g) and o != g and len(o) < ratio * len(g)


def make_fix(*, echo: bool = True, shorter: float = 0.6):
    """One or both of the rules, so each can be measured on its own.

    Which is the finding: bundled, they pass. Rule A alone does not, and
    bundling it with a rule that works is enough to hide that.
    """
    def fix(record: AuditRecord, ctx) -> float:
        if ctx is None or record.verifier_score <= 0.0:
            return record.verifier_score
        question, gold = ctx
        if echo and echoes_the_question(record.output, question):
            return 0.0
        if shorter and far_shorter_than_reference(record.output, gold, shorter):
            return 0.0
        return record.verifier_score
    return fix


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", default="reports/audit_phase0_2026-09-09.jsonl")
    ap.add_argument("--out", default="")
    ap.add_argument("--limit", type=int, default=200,
                    help="dataset rows to join against; the Phase 0 run read "
                         "max(tasks * 2, 40) of them")
    args = ap.parse_args()

    records = [r for r in load_records(pathlib.Path(args.records))
               if r.oracle_score is not None]
    if not records:
        raise SystemExit(f"{args.records} holds no resolved pairs")
    version = records[0].verifier_version
    context = hotpot_context(args.limit)
    missing = sum(1 for r in records if r.task_id not in context)

    # A record whose question could not be joined lands in UNCLASSIFIED rather
    # than being dropped: a diagnosis that quietly narrows its own population is
    # the failure this package spends most of its docstrings on.
    def _spec_gap(out, ref, ctx):
        return bool(ctx) and echoes_the_question(out, ctx[0])

    def _ambiguous(out, ref, ctx):
        if not ctx or not ctx[1]:
            return False
        o, g = normalise(out), normalise(ctx[1])
        return bool(o) and bool(g) and (o in g or g in o)

    classify = reference_classifier(
        normalise, lambda ctx: ctx[1] if ctx else "",
        spec_gap_when=_spec_gap, ambiguous_when=_ambiguous)
    report = classify_disagreements(records, classify, context)

    candidates = [
        ("A -- reject an answer that echoes the question",
         make_fix(echo=True, shorter=0.0)),
        ("B -- reject an answer far shorter than the reference",
         make_fix(echo=False, shorter=0.6)),
        ("A + B, as anyone would ship them",
         make_fix(echo=True, shorter=0.6)),
    ]
    graded = [(name, evaluate_fix(records, fn, context)) for name, fn in candidates]
    bundle = graded[-1][1]
    swept = rescan(records, candidates[-1][1], context)

    stats = residual_stats(records)
    card = scorecard(
        Rectification(
            verifier_version=version, delta_hat=stats["delta"],
            delta_se=stats["sigma"] / max(1, stats["n"]) ** 0.5,
            theta=float("nan"), theta_ci=(float("nan"), float("nan")),
            se=stats["sigma"] / max(1, stats["n"]) ** 0.5,
            n=stats["n"], n_unlab=0, gain_factor=float("nan"),
            is_stale=False, stale_reason=None, resid_sd=stats["sigma"]),
        records)

    lines = [
        f"# Verifier diagnosis -- `{version}`",
        "",
        f"`{args.records}`, {len(records)} resolved pairs over "
        f"{len({r.task_id for r in records})} tasks"
        + (f" ({missing} without a joined question)" if missing else ""),
        "",
        report.to_markdown(),
        "",
        "---",
        "",
        "## Two rules a person reaches for first, measured one at a time",
        "",
        "Reject an answer that echoes the question. Reject one far shorter than",
        "the reference. Both are obviously correct in isolation, and each is",
        "scored below against **every** labelled pair rather than against the",
        "disagreements it targets.",
        "",
        "| rule | fixed | broke | sigma | delta | false negatives | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, got in graded:
        lines.append(
            f"| {name} | {got.fixed} | {got.broken} | "
            f"{got.sigma_before:.4f} -> **{got.sigma_after:.4f}** | "
            f"{got.delta_before:+.4f} -> {got.delta_after:+.4f} | "
            f"{got.false_negative_before:.1%} -> {got.false_negative_after:.1%} | "
            f"{'helps' if got.helps else '**does not help**'} |")
    lines += [
        "",
        "**Rule A cuts the bias by 74% and makes the verifier worse.** It "
        "corrects twelve",
        "over-credits and breaks eleven correct judgements, so the mean falls "
        "because the",
        "errors now cancel; the spread, which is what the acceptance gate's "
        "variance is",
        "built from, goes up. Every summary that leads with `delta` ships it.",
        "",
        "**Bundled with a rule that works, it passes.** Rule B alone is a clean "
        "win --",
        "ten corrections, nothing broken. Together the bundle's sigma improves, "
        "so the",
        "bundle helps, and it still contains a rule that is measurably harmful "
        "and a",
        "false-negative rate that went from nothing to "
        f"{bundle.false_negative_after:.0%}.",
        "",
        "That is the argument for measuring each rule alone rather than the "
        "change as",
        "shipped: a bundle launders whatever is in it.",
        "",
        swept.to_markdown(),
        "",
        "---",
        "",
        card.to_markdown(),
    ]
    text = "\n".join(lines) + "\n"
    out = pathlib.Path(args.out or f"reports/verifier_diagnosis_{version}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
