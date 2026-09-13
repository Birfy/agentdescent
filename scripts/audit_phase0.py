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

Writes `reports/audit_phase0_[<workload>_]<date>.md`. `--dry-run` uses a deterministic
offline stand-in for the judge so the harness itself can be exercised without
spending anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import random
import re
import statistics
import string
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Callable, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentdescent import (AppendRules, AuditedReward, AuditStore, GoldAnswer,  # noqa: E402
                          RenderTap, Task, Usage, anthropic_compatible, evolve)
from scripts.audit_workloads import (BBH_SUBTASKS, CODE_TIMEOUT,  # noqa: E402,F401
                                     WORKLOADS, Workload, exact_match,
                                     extract_code, final_number, normalize,
                                     number_match, read_verdict,
                                     refuse_to_overwrite,
                                     run_tests, task_index, tests_pass)
from agentdescent.audit import Purpose, residual_bias  # noqa: E402
from agentdescent.evolution import LLMAgent  # noqa: E402

# ---------------------------------------------------------------------------
# The two scorers
# ---------------------------------------------------------------------------

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
        def _unparsed() -> None:
            usage_note["unparsed"] = usage_note.get("unparsed", 0) + 1

        return read_verdict(reply, on_unparsed=_unparsed)

    return score


def offline_judge(generosity: float = 0.35, seed: int = 0, oracle=exact_match):
    """A deterministic stand-in for `--dry-run`, biased in a known *direction*.

    Symmetric noise would make a broken estimator look fine -- it is unbiased on
    balanced binary outcomes, which is the fourth pitfall IMPLEMENTATION.md
    records. So this only ever scores *up*: a wrong answer is forgiven with
    probability `generosity`, a right one is never marked down, and the true
    `Delta` is therefore `generosity * P(wrong)` and known in advance.

    ``oracle`` is the workload's own, not a default: with `exact_match` hard-coded
    the dry run on GSM8K would call every ``$18`` wrong and then forgive it at
    `generosity`, so the known `Delta` -- the one thing this stand-in exists to
    make known -- would be a number about the wrong oracle.
    """
    def score(task: Task, output: str) -> float:
        truth = oracle(task, output)
        if truth == 1.0:
            return 1.0
        rng = random.Random(f"{seed}:{task.id}:{output}")
        return 1.0 if rng.random() < generosity else 0.0

    return score


# ---------------------------------------------------------------------------
# The workload
# ---------------------------------------------------------------------------

_OPTION = re.compile(r"\s*\(([A-Za-z])\)")


def _option_label(text: str) -> Optional[str]:
    """The leading ``(B)`` of an answer, if it has one."""
    m = _OPTION.match(text or "")
    return m.group(1).upper() if m else None


def label_agreement(records, tasks) -> Optional[Dict[str, float]]:
    """On multiple-choice answers, does the judge track even a *lenient* oracle?

    Exact match refuses ``(B) Numerical Values`` against a gold of ``(B)``, and a
    judge accepting that is the benign story this experiment was built to
    measure. So this asks the harder question: forgive every formatting
    difference the judge is *told* to forgive -- compare option labels alone --
    and see whether the judge still says yes where that says no.

    A judge that is merely generous scores near zero here. One that has stopped
    discriminating scores high, and the two are indistinguishable in ``Delta``.

    ``None`` when the workload has no labelled answers, which is most of them.
    """
    gold = {t.id: t.meta.get("gold", "") for t in tasks}
    n = stamped = lenient_right = judged_right = 0
    for rec in records:
        if rec.oracle_score is None:
            continue
        want = _option_label(gold.get(rec.task_id, ""))
        if want is None:
            continue
        n += 1
        got = _option_label(rec.output)
        lenient_right += int(got == want)
        judged_right += int(rec.verifier_score > 0)
        stamped += int(rec.verifier_score > 0 and got != want)
    if not n:
        return None
    return {"n": n, "lenient_correct": lenient_right,
            "judge_says_right": judged_right, "rubber_stamped": stamped,
            "rate": stamped / n}


def by_subtask(records, tasks) -> Dict[str, Dict[str, float]]:
    """Residual per ``meta['subtask']``, for a workload that has them.

    The residual is mostly a property of the *shape* of the answer, so a
    workload drawn across shapes has to be reported across them or the headline
    number is an average of two different phenomena.
    """
    group = {t.id: t.meta.get("subtask") for t in tasks}
    buckets: Dict[str, List[float]] = {}
    for rec in records:
        name = group.get(rec.task_id)
        if name is None or rec.oracle_score is None:
            continue
        buckets.setdefault(name, []).append(rec.residual)
    out = {}
    for name, resid in sorted(buckets.items()):
        out[name] = {
            "n": len(resid),
            "delta": statistics.fmean(resid),
            "sigma": statistics.stdev(resid) if len(resid) > 1 else 0.0,
            "disagree": sum(1 for x in resid if x) / len(resid),
        }
    return out


# ---------------------------------------------------------------------------
# The statistic
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def run(args) -> Dict:
    workload = WORKLOADS[args.workload]
    tasks = workload.sample(args.tasks, seed=args.seed)
    oracle = workload.oracle
    usage = Usage()
    notes: Dict[str, int] = {}

    if args.dry_run:
        judge = offline_judge(generosity=args.dry_generosity, seed=args.seed,
                              oracle=oracle)
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
        oracle=GoldAnswer(oracle),
        store=store,
        sample_rate=args.sample_rate,
        calibration_fraction=args.calibration_fraction,
        seed=args.seed,
        version_extra={"judge": judge_label, "template": _JUDGE_TMPL},
    )

    if args.dry_run:
        # A deterministic solver whose answers are right, near-miss or wrong in
        # roughly realistic proportions -- enough for the harness to have pairs.
        #
        # The near-miss has to be a near-miss *for this workload's oracle*.
        # `The answer is 18.` is one for exact match and is simply **correct**
        # for `number_match`, so the shared string would have given the GSM8K
        # dry run no disagreements at all -- a harness rehearsal that exercises
        # none of the paths it exists to rehearse.
        near_miss, wrong = workload.near_miss, workload.wrong

        def _run(rendered: str, task: Task) -> str:
            rng = random.Random(f"{args.seed}:{task.id}:{len(rendered)}")
            gold = task.meta["gold"]
            roll = rng.random()
            if roll < 0.35 + 0.25 * min(1.0, len(rendered) / 400.0):
                return gold
            if roll < 0.75:
                return near_miss(gold)
            return wrong[rng.randrange(len(wrong))](gold, rng)
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
        # A deadline, because this run talks to a remote endpoint and `evolve`'s
        # own docstring names the failure: `round_timeout=None` waits forever,
        # "which is what you want when every rollout is bounded -- but a single
        # hung rollout then stalls the run, because the aggregator is a
        # barrier". A rollout here is *not* reliably bounded, and the cost of
        # finding that out was a GSM-Hard run that sat at zero CPU for 15
        # minutes with three worker threads blocked on one lock, holding 58
        # resolved records it would never analyse or report.
        round_timeout=args.round_timeout, max_seconds=args.max_seconds,
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

def _shape_section(bundle: Dict, args) -> List[str]:
    """Where the residual actually came from, when the workload has shapes.

    Empty for a workload without ``meta['subtask']``, which keeps this out of
    the HotpotQA report rather than printing a table of one row.
    """
    # This run's records only. The store accumulates across runs by design, and
    # a per-shape table that mixed an earlier run's answers into this one's
    # would attribute another workload's residual to these subtasks.
    prior = bundle.get("preexisting") or set()
    records = [r for r in bundle["store"].all()
               if r.resolved and r.record_id not in prior]
    tasks = bundle["tasks"]
    split = by_subtask(records, tasks)
    if len(split) < 2:
        return []
    rows = ["## Where the residual came from", "",
            "The residual is mostly a property of the *shape* of the answer, so "
            "a workload drawn across shapes has to be reported across them -- "
            "otherwise the headline number is an average of two different "
            "phenomena.",
            "",
            "| subtask | n | `Delta` | `sigma` | disagree |",
            "|---|---|---|---|---|"]
    for name, cell in sorted(split.items(), key=lambda kv: -kv[1]["delta"]):
        rows.append(f"| `{name}` | {cell['n']} | {cell['delta']:+.4f} | "
                    f"{cell['sigma']:.4f} | {cell['disagree']:.3f} |")

    stamp = label_agreement(records, tasks)
    if stamp:
        rows += [
            "",
            "### Generous, or not discriminating?",
            "",
            "Exact match refuses `(B) Numerical Values` against a gold of `(B)`, "
            "and a judge accepting that is the benign story this experiment was "
            "built to measure. So: forgive every formatting difference the judge "
            "is *told* to forgive -- compare the option labels alone -- and ask "
            "whether it still says yes where that says no.",
            "",
            f"- labelled-answer units: **{stamp['n']}**",
            f"- correct by the lenient label oracle: **{stamp['lenient_correct']}**",
            f"- the judge called right: **{stamp['judge_says_right']}**",
            f"- **rubber-stamped** (judge said right, lenient oracle says wrong): "
            f"**{stamp['rubber_stamped']}/{stamp['n']} = {stamp['rate']:.0%}**",
            "",
        ]
        if stamp["rate"] > 0.25:
            rows.append(
                "A judge that is merely *generous* scores near zero here. This "
                "one has stopped discriminating on label-shaped answers -- a "
                "different failure from the one `Delta` describes, and "
                "indistinguishable from it in `Delta`.")
        else:
            rows.append(
                "Near zero: the disagreement is formatting, which is the benign "
                "case and the one a normalisation can fix.")
        rows.append("")
    return rows


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
        f"| workload | {WORKLOADS[args.workload].label}, "
        f"{len(bundle['tasks'])} questions |",
        f"| verifier `f` (cheap, biased) | {bundle['judge_label']} |",
        f"| oracle `Y` (ground truth) | "
        f"{WORKLOADS[args.workload].oracle_label} |",
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
        *_shape_section(bundle, args),
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
        f"python3 scripts/audit_phase0.py --workload {args.workload} "
        f"--tasks {args.tasks} --rounds {args.rounds} "
        f"--workers {args.workers} --seed {args.seed}"
        + (" --tournament" if args.tournament else "")
        + (" --dry-run" if args.dry_run else f" --model {args.model}"),
        "```",
    ]
    if bundle["notes"]:
        lines += ["", f"Harness notes: {bundle['notes']}"]
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workload", choices=sorted(WORKLOADS), default="hotpot",
                    help="hotpot: multi-hop free-text answers. bbh: BIG-Bench "
                         "Hard, sampled across subtasks so the measurement is "
                         "not one answer shape. gsm8k: arithmetic word "
                         "problems, where the output carries a derivation the "
                         "judge can be seduced by and the other two do not -- "
                         "though a capable solver gets 98% of them right, so "
                         "gsm_hard, the same problems with large numbers, is "
                         "the one that leaves the judge something to be wrong "
                         "about.")
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
    ap.add_argument("--round-timeout", type=float, default=600.0,
                    help="seconds a round waits for its workers before "
                         "abandoning the slow ones. The default is a number "
                         "rather than `None` on purpose: the run is analysed "
                         "and reported from whatever resolved, and a run that "
                         "hangs reports nothing at all. Pass 0 to wait forever.")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="wall-clock budget for the whole loop. Off by "
                         "default; the round timeout is the one that matters, "
                         "because the aggregator is a barrier.")
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

    if args.round_timeout is not None and args.round_timeout <= 0:
        args.round_timeout = None          # an explicit "wait forever"

    stamp = date.today().isoformat()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # The workload belongs in the name. Without it two workloads run on the same
    # day write the same report path -- the second overwrites the first -- and,
    # worse, **append into the same store**. The store accumulating across runs
    # is by design and right; a JSONL holding two workloads' records is not,
    # because the gold answers are joined per workload and half the file cannot
    # be joined at all.
    suffix = f"_{args.workload}" if args.workload != "hotpot" else ""
    suffix += "_tournament" if args.tournament else ""
    if args.store is None:
        args.store = os.path.join(root, "reports", f"audit_phase0_{stamp}{suffix}.jsonl")
    os.makedirs(os.path.dirname(args.store), exist_ok=True)

    if not args.out:
        refuse_to_overwrite(pathlib.Path(
            os.path.join(root, "reports", f"audit_phase0_{stamp}{suffix}.md")))

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
