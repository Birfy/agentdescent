#!/usr/bin/env python3
"""Phase 0 of the sparse-audit plan: is the verifier's bias worth correcting?

This is a **kill test**. It runs before any of the calibration machinery is
built and it is allowed to say "do not build it": if the bias between the cheap
verifier and ground truth is not distinguishable from zero, or is small next to
the sampling noise the acceptance gate already carries, then a correction layer
is machinery that changes no decision.

## What changed from the plan in IMPLEMENTATION.md

The original Phase 0 was a *replay* over the Ledger: sample past accepted diffs,
send them to an oracle, and estimate the bias after the fact. That cannot be run
against this repository's history. `ThreeLayerVerifier.full_eval` (called
`oracle_eval` before 0.6) is the caller's own `eval_fn` on the whole held-out
set -- the same call `eval_counts` makes -- so no run ever recorded an
independent second opinion, and there is nothing in the Ledger to replay against.
See issue #179 §1.1.

So Phase 0 is a **fresh measurement on a workload that genuinely has two
sources**, which is what the plan's own closing section recommends anyway:

    verifier (cheap, biased)  an LLM judge scoring the answer for semantic
                              equivalence against the reference
    oracle   (ground truth)   normalized exact match against the same reference

On HotpotQA, exact match against the annotated answer is the benchmark's own
definition of correct, and the judge is the proxy a loop would optimise instead.
They disagree in a *direction*: the judge accepts paraphrases, extra words, and
restated context that exact match refuses, so `f - Y >= 0` structurally and the
loop that optimises `f` is free to drift towards answers only `f` likes.

Note the inversion this setup carries, and it is convenient rather than
misleading: here the **oracle is free** (string comparison) and the **verifier is
expensive** (a model call). That is the reverse of the situation the audit layer
is built for. It costs nothing to the estimate -- `Delta` is a property of the
pair, not of what they cost -- and it means Phase 0 can audit every unit instead
of a sample, so the interval is as tight as the run allows.

## Usage

    export ANTHROPIC_BASE_URL=https://your-gateway/anthropic
    export ANTHROPIC_API_KEY=...
    python3 scripts/audit_phase0.py --tasks 40 --rounds 5 --model deepseek-v4-flash

Writes `reports/audit_phase0_<date>.md`. `--dry-run` uses a deterministic
offline stand-in for the judge so the harness itself can be exercised without
spending anything.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import string
import sys
import time
from collections import defaultdict
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentdescent import (AppendRules, AuditedReward, AuditStore, GoldAnswer,  # noqa: E402
                          RenderTap, Task, Usage, anthropic_compatible, evolve)
from agentdescent.audit import Purpose, residual_bias  # noqa: E402
from agentdescent.evolution import LLMAgent  # noqa: E402

# ---------------------------------------------------------------------------
# The two scorers
# ---------------------------------------------------------------------------

_ARTICLES = {"a", "an", "the"}


def normalize(text: str) -> str:
    """The SQuAD/HotpotQA normalisation: lowercase, strip punctuation and articles."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    return " ".join(w for w in text.split() if w not in _ARTICLES)


def exact_match(task: Task, output: str) -> float:
    """**The oracle.** The benchmark's own definition of a correct answer."""
    return 1.0 if normalize(output) == normalize(task.meta["gold"]) else 0.0


_JUDGE_TMPL = """You are grading an answer to a question.

Question: {question}
Reference answer: {gold}
Candidate answer: {candidate}

Does the candidate convey the same answer as the reference? Ignore differences in
wording, formatting, extra context, and completeness of phrasing -- grade the
meaning only.

Reply with exactly one word: YES or NO."""


def llm_judge(complete, usage_note: Dict[str, int]):
    """**The verifier.** A model asked whether the answer *means* the right thing.

    This is the proxy. It is not adversarial and it is not badly written -- it is
    a reasonable judge, which is the point: the bias this measures is what an
    ordinary, sensible LLM judge costs you, not what a broken one does.
    """
    def score(task: Task, output: str) -> float:
        if not (output or "").strip():
            return 0.0
        reply = complete(_JUDGE_TMPL.format(
            question=task.prompt, gold=task.meta["gold"], candidate=output[:2000]))
        head = (reply or "").strip().upper()
        if head.startswith("YES"):
            return 1.0
        if head.startswith("NO"):
            return 0.0
        # An unparseable reply is not a 0: scoring it as "wrong" would blame the
        # artifact for the judge's failure to answer the question it was asked.
        usage_note["unparsed"] = usage_note.get("unparsed", 0) + 1
        return 1.0 if "YES" in head else 0.0

    return score


def offline_judge(generosity: float = 0.35, seed: int = 0):
    """A deterministic stand-in for `--dry-run`, biased in a known *direction*.

    Symmetric noise would make a broken estimator look fine -- it is unbiased on
    balanced binary outcomes, which is the fourth pitfall IMPLEMENTATION.md
    records. So this only ever scores *up*: a wrong answer is forgiven with
    probability `generosity`, a right one is never marked down, and the true
    `Delta` is therefore `generosity * P(wrong)` and known in advance.
    """
    def score(task: Task, output: str) -> float:
        truth = exact_match(task, output)
        if truth == 1.0:
            return 1.0
        rng = random.Random(f"{seed}:{task.id}:{output}")
        return 1.0 if rng.random() < generosity else 0.0

    return score


# ---------------------------------------------------------------------------
# The workload
# ---------------------------------------------------------------------------

def hotpot_tasks(n: int, *, seed: int = 0) -> List[Task]:
    """HotpotQA validation questions, answerable without the distractor context.

    The questions are multi-hop and the context is not given to the solver: the
    point is a workload where a capable model is *often but not always* right and
    its near-misses are real, not a benchmark score.
    """
    from agentdescent.dataloader import hf_rows

    rows = hf_rows("hotpotqa/hotpot_qa", "validation", config="distractor",
                   limit=max(n * 2, 40))
    rng = random.Random(seed)
    rng.shuffle(rows)
    tasks: List[Task] = []
    for row in rows:
        answer = (row.get("answer") or "").strip()
        question = (row.get("question") or "").strip()
        if not answer or not question:
            continue
        tasks.append(Task(id=str(row["id"]), prompt=question,
                          meta={"gold": answer, "expected": answer}))
        if len(tasks) >= n:
            break
    return tasks


# ---------------------------------------------------------------------------
# The statistic
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def run(args) -> Dict:
    tasks = hotpot_tasks(args.tasks, seed=args.seed)
    usage = Usage()
    notes: Dict[str, int] = {}

    if args.dry_run:
        judge = offline_judge(generosity=args.dry_generosity, seed=args.seed)
        solve = None
        judge_label = f"offline stand-in (generosity={args.dry_generosity})"
    else:
        # Two clients, and the split is deliberate rather than a saving.
        #
        # The **solver** thinks: HotpotQA is multi-hop, and a solver that cannot
        # reason produces garbage rather than near-misses -- and near-misses are
        # the entire phenomenon under study. The **judge** does not: it is
        # answering YES/NO about semantic equivalence, which is the shape of a
        # judge anyone would actually deploy at scale, and a judge run cheaply is
        # exactly the proxy this experiment is about. Making the judge as capable
        # as the solver would measure a bias nobody would ever pay for.
        solve = anthropic_compatible(args.model, max_tokens=args.max_tokens,
                                     usage=usage, timeout=args.timeout)
        judge_kwargs = {} if args.judge_thinking else {"thinking": {"type": "disabled"}}
        judge_complete = anthropic_compatible(
            args.model, max_tokens=args.judge_max_tokens, usage=usage,
            timeout=args.timeout, **judge_kwargs)
        judge = llm_judge(judge_complete, notes)
        judge_label = (f"LLM judge ({args.model}, "
                       f"thinking={'on' if args.judge_thinking else 'off'})")

    store = AuditStore(args.store)
    # An `AuditStore` accumulates across runs **by design** -- it is the durable
    # place labels land, sometimes weeks after the run that asked for them. That
    # is right for the store and wrong for a report about one run: pointed at a
    # file an earlier run wrote, the analysis silently mixed 78 old pairs into
    # 113 new ones and reported n=191 next to "units seen: 113". Remember what
    # was already there and subtract it, rather than forbidding an existing file:
    # appending is the behaviour the store is for.
    preexisting = {r.record_id for r in store.all()}
    audited = AuditedReward(
        judge,
        oracle=GoldAnswer(exact_match),
        store=store,
        sample_rate=args.sample_rate,
        calibration_fraction=args.calibration_fraction,
        seed=args.seed,
        version_extra={"judge": judge_label, "template": _JUDGE_TMPL},
    )

    if args.dry_run:
        # A deterministic solver whose answers are right, near-miss or wrong in
        # roughly realistic proportions -- enough for the harness to have pairs.
        def _run(rendered: str, task: Task) -> str:
            rng = random.Random(f"{args.seed}:{task.id}:{len(rendered)}")
            gold = task.meta["gold"]
            roll = rng.random()
            if roll < 0.35 + 0.25 * min(1.0, len(rendered) / 400.0):
                return gold
            if roll < 0.75:
                return f"The answer is {gold}."      # near-miss: judge yes, EM no
            return "unknown"
        kwargs = dict(run=RenderTap(_run), propose=lambda r, t, o, s: None)
    else:
        agent = LLMAgent(solve)
        kwargs = dict(run=RenderTap(lambda rendered, t: agent.solve(rendered, t)),
                      propose=agent.propose)

    # Capture the verifier the engine builds, so the fixed cheap subset it drew
    # can be read back for the §1.2 split. Going through the factory rather than
    # constructing a verifier here keeps `evolve`'s own held-out split -- which
    # decides *which* tasks the subset is drawn from -- as the single source.
    captured: Dict[str, object] = {}

    def _factory(ledger, verifier, audit, config, policy):
        from agentdescent.aggregator import Aggregator
        captured["verifier"] = verifier
        return Aggregator(ledger, verifier, audit, config, staleness_policy=policy)

    t0 = time.time()
    result = evolve(
        tasks, audited,
        aggregator_factory=_factory,
        strategy=AppendRules(),
        rounds=args.rounds, n_workers=args.workers,
        eval_concurrency=args.concurrency,
        held_out_frac=args.held_out_frac,
        cheap_eval_tasks=args.cheap_eval_tasks,
        fusion_tournament=args.tournament,
        self_verify=False, seed=args.seed, usage=usage,
        verbose=args.verbose,
        **kwargs)
    elapsed = time.time() - t0

    verifier = captured.get("verifier")
    cheap_ids = set()
    if verifier is not None:
        try:
            k = getattr(verifier, "rule_subset", args.cheap_eval_tasks)
            cheap_ids = {getattr(t, "id", None) for t in verifier._subset(k)}
            cheap_ids.discard(None)
        except Exception:  # noqa: BLE001 - a custom verifier need not have one
            cheap_ids = set()

    return {
        "result": result, "audited": audited, "store": store, "usage": usage,
        "tasks": tasks, "elapsed": elapsed, "notes": notes,
        "judge_label": judge_label, "cheap_ids": cheap_ids,
        "preexisting": preexisting,
        "held_out_ids": {getattr(t, "id", None) for t in getattr(verifier, "held_out", [])},
    }


def analyse(bundle: Dict, args) -> Dict:
    store: AuditStore = bundle["store"]
    audited: AuditedReward = bundle["audited"]
    prior = bundle.get("preexisting") or set()
    recs = [r for r in store.all() if r.resolved and r.record_id not in prior]
    cal = [r for r in recs if r.purpose is Purpose.CALIBRATION]

    all_stats = residual_bias(recs, seed=args.seed)
    cal_stats = residual_bias(cal, seed=args.seed)

    # -- decision relevance -------------------------------------------------
    # The acceptance gate's own noise: a Beta posterior over `n_held_out`
    # binary trials has sd ~ sqrt(p(1-p)/n). A bias far below that is invisible
    # to the gate no matter how carefully it is estimated.
    n_held = max(1, int(round(len(bundle["tasks"]) * args.held_out_frac)))
    p = all_stats.get("f_mean", 0.5) if all_stats["n"] else 0.5
    gate_sd = (p * (1 - p) / n_held) ** 0.5

    # -- the fixed cheap subset (issue #179 §1.2) ---------------------------
    # `ThreeLayerVerifier._subset` draws k held-out tasks once and reuses them,
    # so ranking overfits a *fixed* sample that the acceptance measurement then
    # includes. If that matters, artifacts should score higher on those k tasks
    # than on the rest. Computed from the audit records, so it costs no calls.
    per_artifact = defaultdict(lambda: ([], []))
    cheap_ids = bundle.get("cheap_ids") or set()
    held_ids = bundle.get("held_out_ids") or set()
    for r in recs:
        if not r.artifact_signature or (held_ids and r.task_id not in held_ids):
            # Training tasks are never in the cheap subset's population, so
            # counting them as "outside" would compare held-out against train
            # and report the split between those two as selection pressure.
            continue
        inside, outside = per_artifact[r.artifact_signature]
        (inside if r.task_id in cheap_ids else outside).append(r.verifier_score)
    gaps = [statistics.fmean(a) - statistics.fmean(b)
            for a, b in per_artifact.values() if len(a) >= 2 and len(b) >= 2]

    return {
        "all": all_stats, "calibration": cal_stats,
        "gate_sd": gate_sd, "n_held": n_held,
        "audit_limited": (all_stats.get("se", 0.0) ** 2 > gate_sd ** 2
                          if all_stats["n"] else None),
        "subset_gaps": gaps,
        "subset_gap_mean": statistics.fmean(gaps) if gaps else None,
        "seen": audited.seen, "audited_n": audited.audited,
        "carried": len(prior),
    }


def verdict(an: Dict) -> Tuple[str, str]:
    """The gate, as issue #179 states it, plus the decision-relevance half."""
    s = an["all"]
    if s["n"] < 20:
        return ("INCONCLUSIVE",
                f"only {s['n']} resolved pairs -- too few to call. Re-run larger.")
    lo, hi = s["ci_clustered"]          # the honest one; see `_bootstrap_means`
    if lo <= 0.0 <= hi:
        return ("STOP",
                "the clustered 95% CI for the verifier's bias contains 0. On this workload "
                "the calibration layer would correct a bias that is not "
                "distinguishable from zero: build Phase 1 (recording) only.")
    ratio = abs(s["delta"]) / an["gate_sd"] if an["gate_sd"] else float("inf")
    if ratio < 1.0:
        return ("RECORD-ONLY",
                f"the bias is real but small next to the acceptance gate's own "
                f"noise (|delta|/sd = {ratio:.2f} < 1). Correcting it would move "
                f"decisions less than resampling the held-out set does.")
    return ("PROCEED",
            f"the bias is real and larger than the gate's sampling noise "
            f"(|delta|/sd = {ratio:.2f}). A correction changes decisions.")


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def report(bundle: Dict, an: Dict, args) -> str:
    s, c = an["all"], an["calibration"]
    call, why = verdict(an)
    u: Usage = bundle["usage"]
    res = bundle["result"]

    def fmt(x, n=4):
        return "n/a" if x is None or x != x else f"{x:.{n}f}"

    lines = [
        f"# Sparse audit -- Phase 0 report ({date.today().isoformat()})",
        "",
        f"**Verdict: {call}.** {why}",
        "",
        "## What was measured",
        "",
        "| | |",
        "|---|---|",
        f"| workload | HotpotQA validation, {len(bundle['tasks'])} questions |",
        f"| verifier `f` (cheap, biased) | {bundle['judge_label']} |",
        "| oracle `Y` (ground truth) | normalized exact match against the reference |",
        f"| `verifier_version` | `{bundle['audited'].verifier_version}` |",
        f"| loop | {args.rounds} rounds x {args.workers} workers, "
        f"held_out_frac={args.held_out_frac}, tournament={args.tournament} |",
        f"| sampling | i.i.d., inclusion probability {args.sample_rate} |",
        f"| records | `{os.path.relpath(args.store, os.getcwd())}` |",
        f"| wall clock | {bundle['elapsed']:.0f}s |",
        f"| model calls | {u.calls} ({u.prompt_tokens}+{u.completion_tokens} tokens) |",
        f"| seed | {args.seed} |",
        "",
        "## The bias",
        "",
        "`Delta = E[f - Y]`. Positive means the judge scores answers **higher** "
        "than exact match does -- the direction that lets a loop accept changes "
        "ground truth says improved nothing.",
        "",
        "| pool | n | tasks | `Delta_hat` | 95% CI (unit) | 95% CI (clustered) "
        "| mean `f` | mean `Y` | disagree |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, st in (("all resolved", s), ("calibration only", c)):
        if st["n"]:
            lines.append(
                f"| {name} | {st['n']} | {st['n_tasks']} | {fmt(st['delta'])} | "
                f"[{fmt(st['ci'][0])}, {fmt(st['ci'][1])}] | "
                f"[{fmt(st['ci_clustered'][0])}, {fmt(st['ci_clustered'][1])}] | "
                f"{fmt(st['f_mean'], 3)} | {fmt(st['y_mean'], 3)} | "
                f"{fmt(st['disagree'], 3)} |")
        else:
            lines.append(f"| {name} | 0 | -- | -- | -- | -- | -- | -- | -- |")

    lines += [
        "",
        f"Units seen by the tap: {an['seen']}; audited: {an['audited_n']}; "
        f"resolved and analysed: {s['n']}."
        + (f" ({an['carried']} earlier records in the store were excluded.)"
           if an["carried"] else ""),
        "",
        "The estimator is the Hajek (inclusion-probability-weighted) mean with a "
        "percentile bootstrap. It is **not** the PPI estimator Phase 3 needs: "
        "with the labels this cheap there is no unlabelled mass to borrow "
        "strength from, and Phase 0 only has to decide whether the bias exists "
        "and matters.",
        "",
        "**Read the clustered interval.** A run scores the same task again for "
        "every artifact version, so the audited units are not independent draws "
        "and the unit bootstrap is too narrow. Resampling tasks is the honest "
        "one; the gap between the two columns is the size of that dependence.",
        "",
        "## Does it move a decision?",
        "",
        f"The acceptance gate reads a Beta posterior over {an['n_held']} held-out "
        f"tasks, whose own sd at the observed rate is **{fmt(an['gate_sd'])}**.",
        "",
        f"- `|Delta_hat| / gate sd` = "
        f"{fmt(abs(s['delta']) / an['gate_sd'], 2) if s['n'] and an['gate_sd'] else 'n/a'}",
        f"- audit-limited (`SE(Delta)^2 > var_p`): **{an['audit_limited']}** -- when "
        "true, buying more in-loop evaluation cannot improve the criterion and "
        "the budget belongs on oracle labels instead.",
        "",
        "## The fixed cheap subset (issue #179 §1.2)",
        "",
        "`ThreeLayerVerifier._subset` draws `cheap_eval_tasks` held-out items "
        "**once** and reuses them, and the acceptance measurement includes them. "
        "Ranking can therefore overfit a fixed sample the gate then reads. If it "
        "does, artifacts score higher inside that subset than outside it.",
        "",
    ]
    if an["subset_gap_mean"] is None:
        lines.append("Not measurable on this run: too few audited units per "
                     "artifact to split. Raise `--tasks` or `--sample-rate`.")
    else:
        lines.append(
            f"Mean `score(inside cheap subset) - score(outside)` over "
            f"{len(an['subset_gaps'])} artifacts: **{fmt(an['subset_gap_mean'])}**.")
        lines.append("")
        if args.tournament:
            lines.append(
                "Ranking **was** on for this run, so the cheap layer really did "
                "choose which candidate went forward. A *positive* gap would be "
                "the contamination: candidates picked for scoring well on those "
                "few tasks, then measured again on a set that contains them. A "
                "negative one says the subset happens to hold harder tasks and "
                "selection did not overcome that.")
        else:
            lines.append(
                "Ranking was **off** (`fusion_tournament=False`, the default), so "
                "nothing selected on the cheap layer at all. Whatever gap appears "
                "here is workload variation -- which is what makes it the "
                "baseline the `--tournament` arm has to be read against.")
        lines.append("")
        lines.append(
            f"Resolution: {args.cheap_eval_tasks} tasks inside against "
            f"{an['n_held'] - args.cheap_eval_tasks} outside, over "
            f"{len(an['subset_gaps'])} artifacts. That can see a large "
            "contamination and cannot resolve a small one; read a null here as "
            "\"no evidence of\", not \"none\".")

    lines += [
        "",
        "## Final artifact",
        "",
        "```",
        json.dumps(getattr(res, "state", {}) or {}, indent=2, ensure_ascii=False)[:1500],
        "```",
        "",
        f"held-out reward (as the loop measured it, i.e. through `f`): "
        f"{fmt(getattr(res, 'final_reward', float('nan')), 3)}",
        "",
        "## Reproduce",
        "",
        "```bash",
        f"python3 scripts/audit_phase0.py --tasks {args.tasks} --rounds {args.rounds} "
        f"--workers {args.workers} --seed {args.seed}"
        + (" --dry-run" if args.dry_run else f" --model {args.model}"),
        "```",
    ]
    if bundle["notes"]:
        lines += ["", f"Harness notes: {bundle['notes']}"]
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tasks", type=int, default=40)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--held-out-frac", type=float, default=0.4)
    ap.add_argument("--cheap-eval-tasks", type=int, default=4)
    ap.add_argument("--tournament", action="store_true",
                    help="rank candidates on the cheap layer (off by default, "
                         "matching AggregatorConfig.fusion_tournament)")
    ap.add_argument("--sample-rate", type=float, default=1.0,
                    help="inclusion probability; 1.0 because here the oracle is "
                         "the free half")
    ap.add_argument("--calibration-fraction", type=float, default=0.7)
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--judge-max-tokens", type=int, default=16)
    ap.add_argument("--judge-thinking", action="store_true",
                    help="let the judge reason before grading (off by default: a "
                         "judge you would deploy at scale does not)")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="offline stand-in for the judge, with a known injected bias")
    ap.add_argument("--dry-generosity", type=float, default=0.35)
    ap.add_argument("--verbose", action="store_true",
                    help="print each round as it lands -- a run this long should not\n"
                         "be indistinguishable from a hung one")
    ap.add_argument("--out", default=None)
    ap.add_argument("--store", default=None,
                    help="JSONL for the audit records. Defaults beside the\n"
                         "report, so a run can be re-analysed without re-\n"
                         "buying every model call.")
    args = ap.parse_args(argv)

    stamp = date.today().isoformat()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    suffix = "_tournament" if args.tournament else ""
    if args.store is None:
        args.store = os.path.join(root, "reports", f"audit_phase0_{stamp}{suffix}.jsonl")
    os.makedirs(os.path.dirname(args.store), exist_ok=True)

    bundle = run(args)
    an = analyse(bundle, args)

    text = report(bundle, an, args)
    out = args.out or os.path.join(root, "reports",
                                   f"audit_phase0_{stamp}{suffix}.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
