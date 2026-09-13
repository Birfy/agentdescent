"""Can a prompt fix the judge, and can you tell that from noise?

Phase 0 on BBH found something `delta_hat` cannot express: on label-shaped
answers the judge stopped *discriminating* rather than merely becoming generous.
It said right on 7 of 20 labelled-answer units where even a lenient oracle --
compare the option labels alone, forgiving every formatting difference the judge
is told to forgive -- says wrong.

The hypothesis is structural rather than about this model. The shipped prompt
says "ignore differences in wording, formatting, extra context and completeness
-- grade the meaning only". When the reference answer *is* a bare label, `(B)`,
that instruction tells the judge to ignore the only content there is.

This tests it, and the whole point is that testing it is nearly free. The
expensive half -- producing the outputs and buying their ground truth -- is
already paid. A candidate judge is evaluated by re-scoring stored outputs.

Three things about the design, each of which was a way to get an unreadable
answer:

**One clause changes, not one prompt.** A bundle launders whatever is in it, and
a prompt is a bundle by nature. The `labelled` arm adds a single sentence to the
shipped template and changes nothing else, so a difference is attributable.

**The control is re-run, not read out of the store.** Otherwise a sigma that
moves by 0.02 is indistinguishable from the judge disagreeing with itself. The
control arm re-scores the same outputs with the same prompt, and the gap between
it and the stored scores is the noise floor every other arm has to clear. There
is no point reporting an improvement smaller than that.

**Both workloads, always.** A fix aimed at label-shaped answers has to be
measured on the free-text answers it was not aimed at -- that is
`evaluate_fix`'s discipline, one level up. HotpotQA has no option labels at all,
so the `labelled` clause should be inert there; if it is not, the clause is
doing something other than what it says.

    python -m scripts.audit_judge_repair --model deepseek-v4-flash
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from agentdescent.agents import Usage, anthropic_compatible
from agentdescent.audit import AuditRecord, Purpose
from agentdescent.audit.diagnose import evaluate_fix, residual_stats
from agentdescent.evolution import Task

from scripts.audit_workloads import resolved_records
from scripts.audit_phase0 import (_JUDGE_TMPL, label_agreement,
                                  task_index)

#: The one sentence the `labelled` arm adds, and nothing else. It names the case
#: the shipped instruction is degenerate on rather than rewriting the
#: instruction: a reference that *is* a label has no meaning to grade apart from
#: which option it names.
_LABEL_CLAUSE = (
    "One exception: when the reference answer is a multiple-choice label such as "
    "(A) or (C), the candidate is correct only if it chooses that same option -- "
    "the label is the answer, not its formatting."
)

ARMS: Dict[str, str] = {
    "control": _JUDGE_TMPL,
    "labelled": _JUDGE_TMPL.replace(
        "\n\nReply with exactly one word:",
        "\n\n" + _LABEL_CLAUSE + "\n\nReply with exactly one word:"),
}

#: workload -> the records that run wrote. The tasks come from
#: :func:`~scripts.audit_phase0.task_index`, which reads a *window* of the
#: dataset rather than drawing a sample -- rebuilding the context with a loader
#: means guessing the ``n`` and ``seed`` of a past run, and guessing wrong loses
#: half the records silently.
RECORDS = {
    "hotpot": "reports/audit_phase0_2026-09-09.jsonl",
    "bbh": "reports/audit_phase0_bbh_2026-09-10.jsonl",
}


def judge_for(template: str, complete) -> Any:
    """A judge that scores a *stored* output against its task."""
    def score(record: AuditRecord, task: Optional[Task]) -> float:
        if task is None or not (record.output or "").strip():
            return record.verifier_score
        reply = complete(template.format(
            question=task.prompt, gold=task.meta["gold"],
            candidate=record.output[:2000]))
        head = (reply or "").strip().upper()
        if head.startswith("YES"):
            return 1.0
        if head.startswith("NO"):
            return 0.0
        return 1.0 if "YES" in head else 0.0
    return score


def _rescored(records, judge, context) -> List[AuditRecord]:
    """The same records carrying the candidate judge's scores.

    Materialised rather than computed twice: `evaluate_fix` and
    `label_agreement` both need them, and paying for the judge twice to avoid
    one list would be an odd saving.
    """
    from dataclasses import replace

    out = []
    for rec in records:
        out.append(replace(rec, verifier_score=float(
            judge(rec, context.get(rec.task_id)))))
    return out


def run_arm(name: str, template: str, data, complete,
            floors: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """Score one candidate judge on every workload.

    ``floors`` is how many units the **control** arm flipped, per workload. An
    LLM judge re-run on the same outputs with the same prompt does not give the
    same scores, and those flips move ``sigma`` by themselves -- so a candidate
    that moves fewer units than the control has not been shown to do anything.
    Without this the control arm's own verdict can come back "helps", which is
    the reading the floor exists to refuse.
    """
    judge = judge_for(template, complete)
    per_workload = {}
    for workload, (records, context) in data.items():
        rescored = _rescored(records, judge, context)
        by_id = {r.record_id: r.verifier_score for r in rescored}
        got = evaluate_fix(records, lambda rec, ctx: by_id[rec.record_id],
                           context,
                           noise_floor=(floors or {}).get(workload, 0))
        stamp = label_agreement(rescored, list(context.values()))
        # Self-consistency against the scores the original run stored. For the
        # control arm this *is* the noise floor; for the others it is confounded
        # with the change, which is why only the control's is reported as one.
        flipped = sum(1 for r in records
                      if by_id[r.record_id] != r.verifier_score)
        per_workload[workload] = {
            "n": got.n_pairs,
            "sigma_before": got.sigma_before, "sigma_after": got.sigma_after,
            "delta_before": got.delta_before, "delta_after": got.delta_after,
            "disagree_before": got.disagree_before,
            "disagree_after": got.disagree_after,
            "fixed": got.fixed, "broke": got.broken,
            "fn_before": got.false_negative_before,
            "fn_after": got.false_negative_after,
            "helps": got.helps,
            "changed": got.changed, "noise_floor": got.noise_floor,
            "above_the_noise": got.above_the_noise,
            "flipped_vs_stored": flipped,
            "stamp": stamp,
        }
    return {"arm": name, "template": template, "by_workload": per_workload}


def markdown(arms: List[Dict[str, Any]], args, usage: Usage,
             elapsed: float) -> str:
    control = next((a for a in arms if a["arm"] == "control"), None)
    rows = [
        f"# Judge repair -- can a prompt fix it, and can you tell? "
        f"({time.strftime('%Y-%m-%d')})",
        "",
        "Phase 0 on BBH found the judge had stopped *discriminating* on "
        "label-shaped answers, not merely become generous -- a failure "
        "`delta_hat` cannot tell from the benign one. The hypothesis is that "
        '"ignore formatting, grade the meaning" is degenerate when the '
        "reference answer *is* a bare label.",
        "",
        "| | |",
        "|---|---|",
        f"| model | {args.model} |",
        f"| arms | {', '.join(a['arm'] for a in arms)} |",
        f"| evaluation | re-scoring **stored** outputs; no rollouts |",
        f"| model calls | {usage.calls} "
        f"({usage.prompt_tokens}+{usage.completion_tokens} tokens) |",
        f"| wall clock | {elapsed:.0f}s |",
        "",
    ]

    if control is not None:
        rows += ["## The noise floor", "",
                 "The control arm re-scores the same outputs with the **same** "
                 "prompt. Its disagreement with the stored scores is the judge "
                 "disagreeing with itself, and no other arm's improvement means "
                 "anything below it.", "",
                 "| workload | n | flips vs the stored run | as a rate | "
                 "`sigma` of the *same* prompt |",
                 "|---|---|---|---|---|"]
        improved = []
        for workload, cell in sorted(control["by_workload"].items()):
            moved = ("**improved**" if cell["sigma_after"] < cell["sigma_before"]
                     else "worsened" if cell["sigma_after"] > cell["sigma_before"]
                     else "unchanged")
            if cell["sigma_after"] < cell["sigma_before"]:
                improved.append(workload)
            rows.append(f"| `{workload}` | {cell['n']} | "
                        f"{cell['flipped_vs_stored']} | "
                        f"{cell['flipped_vs_stored'] / max(1, cell['n']):.1%} | "
                        f"{cell['sigma_before']:.4f} -> "
                        f"{cell['sigma_after']:.4f} ({moved}) |")
        rows.append("")
        if improved:
            rows += [
                "!!! danger \"The control improved on itself\"",
                f"    On `{'`, `'.join(improved)}` the **unchanged prompt** "
                f"scored a smaller residual than the run that produced the "
                f"stored scores. Nothing was fixed; the judge answered "
                f"differently.",
                "",
                "    This is not a display artifact, it is the finding. A "
                "`sigma` that fell is not evidence on its own when the verifier "
                "is stochastic, and the control arm is the one row that can "
                "demonstrate that -- it has no floor to be read against, because "
                "it *is* the floor.",
                "",
                "    Read every other row's `helps` as \"cleared this\", and "
                "read this row as how little that means.",
                "",
            ]

    rows += ["## Every arm, on both workloads", "",
             "`sigma` is the target. `delta` is reported and never scored: a "
             "mean goes to zero when errors cancel.", "",
             "`helps` reads the residual **and** the noise floor: an arm that "
             "moved no more units than re-running the same prompt does has not "
             "been shown to do anything.", "",
             "| arm | workload | n | `sigma` | `delta` | disagree | fixed | "
             "broke | moved | floor | helps |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in arms:
        for workload, c in sorted(arm["by_workload"].items()):
            rows.append(
                f"| `{arm['arm']}` | `{workload}` | {c['n']} | "
                f"{c['sigma_before']:.4f} -> **{c['sigma_after']:.4f}** | "
                f"{c['delta_before']:+.4f} -> {c['delta_after']:+.4f} | "
                f"{c['disagree_before']:.3f} -> {c['disagree_after']:.3f} | "
                f"{c['fixed']} | {c['broke']} | {c['changed']} | "
                + ("-- |" if arm["arm"] == "control" else f"{c['noise_floor']} | ")
                + ("*is the floor* |" if arm["arm"] == "control"
                   else "yes |" if c["helps"] else "**no** |"))

    rows += ["", "## Does it still rubber-stamp?", "",
             "Forgive every formatting difference the judge is *told* to "
             "forgive -- compare option labels alone -- and ask whether it "
             "still says yes where that says no. A merely generous judge scores "
             "near zero here.", "",
             "| arm | labelled units | lenient-correct | judge says right | "
             "rubber-stamped |", "|---|---|---|---|---|"]
    for arm in arms:
        for workload, c in sorted(arm["by_workload"].items()):
            stamp = c["stamp"]
            if not stamp:
                continue
            rows.append(
                f"| `{arm['arm']}` (`{workload}`) | {stamp['n']} | "
                f"{stamp['lenient_correct']} | {stamp['judge_says_right']} | "
                f"**{stamp['rubber_stamped']}/{stamp['n']} = "
                f"{stamp['rate']:.0%}** |")

    rows += ["", "## The one clause", "", "```", _LABEL_CLAUSE, "```", "",
             "One sentence, added to the shipped template, nothing else "
             "changed. A prompt is a bundle by nature and a bundle launders "
             "whatever is in it, so a rewrite would have produced a number "
             "nobody could attribute.", "",
             "## Reproduce", "", "```bash",
             f"python -m scripts.audit_judge_repair --model {args.model}",
             "```"]
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--arms", default="control,labelled")
    ap.add_argument("--workloads", default="hotpot,bbh")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap records per workload; 0 uses all of them")
    ap.add_argument("--index-rows", type=int, default=400,
                    help="dataset rows to index when re-joining records to "
                         "their gold answers")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--from-json", default=None,
                    help="re-render the report from a previous run's JSON "
                         "instead of spending 400 model calls to change a "
                         "sentence")
    args = ap.parse_args()

    if args.from_json:
        payload = json.loads(pathlib.Path(args.from_json).read_text())
        arms = payload["arms"]
        usage = Usage()
        usage.calls = payload.get("calls", 0)
        text = markdown(arms, args, usage, payload.get("elapsed", 0.0))
        out = pathlib.Path(args.out or args.from_json).with_suffix(".md")
        out.write_text(text + "\n", encoding="utf-8")
        print(text)
        print(f"\nre-rendered {out} from {args.from_json}", file=sys.stderr)
        return

    data: Dict[str, Tuple[List[AuditRecord], Dict[str, Task]]] = {}
    for name in args.workloads.split(","):
        records = resolved_records(RECORDS[name])
        if args.limit:
            records = records[:args.limit]
        context = task_index(name, rows=args.index_rows)
        missing = sum(1 for r in records if r.task_id not in context)
        if missing:
            print(f"warning: {missing}/{len(records)} {name} records could not "
                  f"be joined to a task; they keep their stored score",
                  file=sys.stderr)
        data[name] = (records, context)
        print(f"{name}: {len(records)} records, {len(context)} tasks",
              file=sys.stderr)

    usage = Usage()
    complete = anthropic_compatible(
        args.model, max_tokens=args.max_tokens, usage=usage,
        timeout=args.timeout, thinking={"type": "disabled"})

    started = time.time()
    arms: List[Dict[str, Any]] = []
    floors: Dict[str, int] = {}
    # The control first, always: every other arm is read against the number it
    # produces, and an arm scored before the floor exists is scored against 0.
    names = sorted(args.arms.split(","), key=lambda n: n != "control")
    for name in names:
        print(f"arm {name} ...", file=sys.stderr)
        arm = run_arm(name, ARMS[name], data, complete, floors)
        if name == "control":
            floors = {w: c["flipped_vs_stored"]
                      for w, c in arm["by_workload"].items()}
            print(f"  noise floor: {floors}", file=sys.stderr)
        arms.append(arm)
    elapsed = time.time() - started

    payload = {"model": args.model, "elapsed": elapsed, "floors": floors,
               "calls": usage.calls, "arms": [
                   {"arm": a["arm"], "by_workload": a["by_workload"]}
                   for a in arms]}
    json_out = pathlib.Path(
        (args.out or f"reports/audit_judge_repair_"
                     f"{time.strftime('%Y-%m-%d')}.md")).with_suffix(".json")
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2, default=str) + "\n",
                        encoding="utf-8")

    text = markdown(arms, args, usage, elapsed)
    out = pathlib.Path(args.out or
                       f"reports/audit_judge_repair_{time.strftime('%Y-%m-%d')}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\nwrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
