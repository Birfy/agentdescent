#!/usr/bin/env python3
"""Rung 5: let the oracle evolve the judge, instead of a person guessing clauses.

Rungs 1-4 of the ladder all end with a person writing a rule or a sentence and
the harness scoring it. `scripts/audit_judge_repair.py` is the top of that:
**one** clause, hand-written from a diagnosis, measured against a control. It
worked -- `sigma` 0.4738 -> 0.4461 on BBH, rubber-stamping 30% -> 20% -- and it
does not scale, because the next clause also has to be thought of.

This rung hands that search to `evolve()`. The artifact is the judge's rubric,
the reward is agreement with ground truth, and the labels come out of the
audit's **improvement** pool, which exists for exactly this.

Five things it refuses to do, each of which is a way to get a number that looks
like progress and is not:

**It will not run on a saturated pool.** Good-Turing over the improvement pool
estimates `P(the next label shows an error mode nobody has seen)`. Across
HotpotQA and BBH that had fallen to **0.0169**: nine and four disagreements
respectively, every one of them a shape already understood. Evolving a prompt
against thirteen known errors is fitting noise with extra steps, and the honest
answer is a new *workload* rather than more labels -- which is why
`scripts/audit_phase0.py` grew a GSM8K arm before this file existed. `--min-unseen`
is the gate and it defaults to 0.25.

**It will not train and test on the same task.** The two pools are split by
purpose, and purpose is drawn per *unit*: a run scores the same task again for
every artifact version, so one task can have units in both pools. Splitting on
purpose alone would put the same question on both sides. The evaluation set here
is the calibration pool **minus every task the training set touched**, and the
report says how many that dropped.

**It will not let "always NO" win.** The reward is agreement with the oracle, so
a rubric that rejects everything scores `P(the answer was wrong)`. On the
HotpotQA improvement pool -- the 55 labels this rung would actually train on --
that is **0.836, exactly the real judge's agreement rate on the same units**. It
does not have to beat the judge to be found: a search that ties the incumbent
while being trivially simpler is a search that has gone nowhere and cannot tell.
The training set is down-sampled to equal numbers of right and wrong answers,
which puts that rubric at 0.5, and the scorecard blocks on false-negative rate
besides. Two guards, because the first is a property of a sample and the second
of the decision.

**It will not read a `sigma` that fell as evidence.** An LLM judge re-run on the
same outputs with the same rubric does not give the same scores. On BBH the
**unchanged** prompt scored a smaller residual than the run it was copied from.
So the control arm here is a re-run of the starting rubric, its flips are the
floor, and a candidate that moves fewer units than that has not been shown to do
anything.

**It will not hand back a judge, only a scorecard.** A verifier is the
instrument every other number in a run is measured with, so swapping it
invalidates the run's history in a way nothing in the run can see. The output is
`scorecard()` plus `rescan()` -- what would have flipped -- and a person decides.

    python -m scripts.audit_evolve_judge --workload gsm8k \\
        --records reports/audit_phase0_gsm8k.jsonl --model deepseek-v4-flash

`--dry-run` uses an offline judge whose behaviour depends on the rubric in a
known way, so the whole path can be rehearsed without a key.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, replace
from datetime import date
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentdescent import Task, Usage, anthropic_compatible, evolve  # noqa: E402
from agentdescent.audit import AuditRecord, Purpose  # noqa: E402
from agentdescent.audit.calibrator import Rectification  # noqa: E402
from agentdescent.audit.coverage import (coverage_of,  # noqa: E402
                                         unseen_mass_overall)
from agentdescent.audit.diagnose import evaluate_fix, residual_stats  # noqa: E402
from agentdescent.audit.scorecard import rescan, scorecard  # noqa: E402
from agentdescent.evolution import LLMAgent  # noqa: E402
from agentdescent.strategies import AppendRules  # noqa: E402
from scripts.audit_modes import (ERROR_MODES, context_for,  # noqa: E402
                                 resolved_records)
from scripts.audit_phase0 import _JUDGE_TMPL  # noqa: E402

#: The last line of the shipped judge template, and the seam the rubric is
#: inserted at. Grading rules have to arrive *before* the instruction that says
#: what to output, or the model is being told how to answer and then given more
#: to think about.
_REPLY_LINE = "Reply with exactly one word: YES or NO."

#: The starting rubric: empty. Not a seeded clause, because a seed is a guess,
#: and the whole argument for this rung is that the guessing is what does not
#: scale. An empty `AppendRules` renders `# Playbook\n(empty)`, which
#: `judging_prompt` drops entirely -- see there.
_TITLE = "# Grading rules"


def judging_prompt(rubric: str, question: str, gold: str, candidate: str) -> str:
    """The shipped template, with the evolved rubric spliced in before the reply line.

    **The frame is fixed and only the rubric evolves.** `evolve()` optimises what
    it is given, and given the whole prompt it would be free to delete
    `{candidate}` -- a judge that never sees the answer agrees with the oracle
    on whatever the majority class is, which is a real optimum and a useless
    one. The placeholders are not the loop's to edit.

    An empty rubric returns the shipped template **byte-identical**, which is
    what makes the control arm a control rather than a near-copy with two extra
    newlines in it.
    """
    base = _JUDGE_TMPL.format(question=question, gold=gold,
                              candidate=candidate[:2000])
    body = _rubric_body(rubric)
    if not body:
        return base
    head, sep, tail = base.partition(_REPLY_LINE)
    if not sep:                       # the template changed under us
        raise SystemExit("the shipped judge template no longer ends with the "
                         "reply line this script splices against")
    return f"{head}{body}\n\n{sep}{tail}"


def _rubric_body(rubric: str) -> str:
    """The rubric's rules, or ``""`` when it has none.

    `AppendRules.render({})` is the string ``# Playbook\\n(empty)``, and pasting
    that into a prompt would tell the judge it has an empty playbook -- which is
    an instruction, not the absence of one.
    """
    lines = [ln for ln in (rubric or "").splitlines()
             if ln.strip().startswith("- ")]
    return f"{_TITLE}\n" + "\n".join(lines) if lines else ""


# ---------------------------------------------------------------------------
# Is the pool worth training on?
# ---------------------------------------------------------------------------

def score_band(record: AuditRecord) -> str:
    """A key computable at dispatch, before any label exists."""
    return "high" if record.verifier_score >= 0.5 else "low"


#: Examples of one error mode below which a rubric clause cannot be *shown* to
#: work. Derived rather than chosen: a clause that fixes all `k` of them has a
#: one-sided p of `0.5 ** k` under a coin-flip null, and `k = 5` is the smallest
#: that clears 0.05 (0.031; `k = 4` gives 0.062). Nothing about a particular
#: pool went into it.
MIN_EXAMPLES = 5


@dataclass(frozen=True)
class Pool:
    """Whether the improvement pool is worth evolving a rubric against.

    **Two ways to be worth it, and the gate needs both questions asked.**

    `unseen` is Good--Turing: the chance the next label shows an error mode
    nobody has seen. High means keep labelling -- there is variety left to find.

    `largest` is the biggest single mode. That is *sufficiency*, and it is the
    question variety cannot answer: a mode function has a bounded range, so
    after enough labels `unseen` goes to zero whatever the pool holds. MBPP's
    pool reached `unseen = 0.02` while holding six examples of one mode --
    plenty to write a clause against, and refused by a variety test alone.
    """

    unseen: float
    largest: int
    largest_mode: Optional[str]
    bands: Dict[str, Any]
    min_unseen: float
    min_examples: int

    @property
    def worth_training(self) -> bool:
        return self.unseen >= self.min_unseen or self.largest >= self.min_examples

    @property
    def reason(self) -> str:
        if self.unseen >= self.min_unseen:
            return (f"P(new error mode) = {self.unseen:.4f}, at or above "
                    f"{self.min_unseen}: there is variety left to learn from")
        if self.largest >= self.min_examples:
            return (f"{self.largest} examples of `{self.largest_mode}`, at or "
                    f"above {self.min_examples}: enough of one mode to write a "
                    f"clause against and show it worked")
        return (f"P(new error mode) = {self.unseen:.4f} (below "
                f"{self.min_unseen}) and the largest mode has {self.largest} "
                f"example(s) (below {self.min_examples}). Neither varied enough "
                f"to keep learning from nor concentrated enough to fix.")


def assess(records: Sequence[AuditRecord], context, workload: str, *,
           min_unseen: float = 0.25,
           min_examples: int = MIN_EXAMPLES) -> Pool:
    """Measure the improvement pool both ways."""
    mode = ERROR_MODES[workload]
    coverage = coverage_of(records, score_band,
                           lambda r: mode(r, context.get(r.task_id)))
    counts = Counter(
        m for m in (mode(r, context.get(r.task_id)) for r in records
                    if r.oracle_score is not None
                    and r.verifier_score != r.oracle_score) if m)
    top, n = counts.most_common(1)[0] if counts else (None, 0)
    return Pool(unseen=unseen_mass_overall(coverage), largest=n,
                largest_mode=top, min_unseen=min_unseen,
                min_examples=min_examples,
                bands={k: {"labels": c.labels, "modes": c.modes,
                           "singletons": c.singletons, "unseen": c.unseen}
                       for k, c in coverage.items()})


# ---------------------------------------------------------------------------
# The training set
# ---------------------------------------------------------------------------

def balanced(records: Sequence[AuditRecord], seed: int = 0
             ) -> Tuple[List[AuditRecord], Dict[str, int]]:
    """Equal numbers of right and wrong answers, by the **oracle's** reckoning.

    Without this the reward has a degenerate optimum. Agreement with the oracle
    is maximised at `max(P(right), P(wrong))` by a rubric that ignores the
    candidate and always says the majority -- and on an audit pool the majority
    is "wrong", so the rubric `evolve()` would converge on is "reject
    everything". Balanced, that scores 0.5 and loses to anything that reads the
    answer.

    Down-sampling rather than weighting because `evolve()`'s reward is per task
    and its held-out split is a count: a weight would have to be smuggled
    through the reward, where it would also silently reweight the *held-out*
    measurement the acceptance gate reads.
    """
    right = [r for r in records if r.oracle_score == 1.0]
    wrong = [r for r in records if r.oracle_score == 0.0]
    keep = min(len(right), len(wrong))
    rng = random.Random(seed)
    rng.shuffle(right)
    rng.shuffle(wrong)
    out = right[:keep] + wrong[:keep]
    rng.shuffle(out)
    return out, {"right": len(right), "wrong": len(wrong), "kept_each": keep}


def as_tasks(records: Sequence[AuditRecord], context) -> List[Task]:
    """One judging decision per task, carrying the truth in ``meta``.

    ``Task.id`` is the **record** id: a task id would collide across the
    versions of an artifact that answered the same question, and `evolve()`
    dedupes its held-out split by task id.
    """
    out = []
    for rec in records:
        ctx = context.get(rec.task_id)
        if ctx is None:
            continue
        out.append(Task(
            id=rec.record_id,
            prompt=ctx[0],
            meta={"gold": ctx[1], "candidate": rec.output,
                  "oracle": rec.oracle_score, "task_id": rec.task_id,
                  "stored_verifier": rec.verifier_score}))
    return out


def disjoint_heldout(calibration: Sequence[AuditRecord],
                     trained_on: Sequence[AuditRecord]
                     ) -> Tuple[List[AuditRecord], int]:
    """Calibration records on tasks the training set never touched.

    The two pools are split by **purpose**, and purpose is drawn per unit while
    inclusion is drawn per task -- so the same question can have one unit in each
    pool, and a purpose-only split would train and test on it. Returns the kept
    records and how many were dropped, because a drop count of "most of them" is
    itself the finding.
    """
    seen = {r.task_id for r in trained_on}
    kept = [r for r in calibration if r.task_id not in seen]
    return kept, len(calibration) - len(kept)


# ---------------------------------------------------------------------------
# The judge, live and offline
# ---------------------------------------------------------------------------

def verdict(reply: str) -> float:
    head = (reply or "").strip().upper()
    if head.startswith("YES"):
        return 1.0
    if head.startswith("NO"):
        return 0.0
    return 1.0 if "YES" in head else 0.0


#: ``(rubric, question, gold, candidate, task_id) -> the judge's reply``.
#:
#: ``task_id`` is there for the offline stand-in alone -- a live judge has no
#: use for it. It is in the signature rather than threaded past it because the
#: stand-in has to reproduce the *exact* draw that wrote the stored scores, and
#: that draw is seeded from the task id. Matching only the generosity leaves two
#: independent coin flips, which put the dry run's noise floor at 21 of 59 units
#: and made the success path unreachable in rehearsal.
Ask = Callable[[str, str, str, str, str], str]


def live_judge(complete) -> Ask:
    def ask(rubric: str, question: str, gold: str, candidate: str,
            task_id: str) -> str:
        del task_id
        return complete(judging_prompt(rubric, question, gold, candidate))
    return ask


def offline_judge(seed: int = 0, flip_rate: float = 0.06,
                  generosity: float = 0.35) -> Ask:
    """A stand-in whose behaviour depends on the rubric, for `--dry-run`.

    Three properties it has to have, or the rehearsal proves nothing:

    * **A rubric can help it.** It is generous about a wrong number unless the
      rubric mentions one; that gives `evolve()` something findable, so the path
      through proposal, acceptance and the scorecard actually executes.
    * **Empty-rubric, it reproduces the judge that wrote the stored scores** --
      the same generosity *and the same draw*, seeded from ``(seed, task_id,
      output)`` exactly as `scripts.audit_phase0.offline_judge` seeds it. Equal
      generosity alone is not enough and looks like it is: two independent coin
      flips at p=0.35 disagree on 46% of the wrong answers, which put the
      rehearsal's noise floor at 21 units of 59 and made the success path
      unreachable. The floor has to come from the flip rate below and nothing
      else.
    * **It is not deterministic.** A fixed fraction of its answers flip on a
      re-run, seeded from the candidate rather than a shared stream. Without
      that the floor is exactly 0 and the one guard this script inherited from
      the repair experiment is never exercised. 0.06 rather than 0.03 for that
      reason alone: at 0.03 a 59-unit rehearsal draws no flips at all about a
      sixth of the time, and a floor of zero rehearses nothing.
    """
    def ask(rubric: str, question: str, gold: str, candidate: str,
            task_id: str) -> str:
        from scripts.audit_phase0 import final_number

        body = _rubric_body(rubric).lower()
        want, got = final_number(gold), final_number(candidate)
        correct = want is not None and got is not None and abs(want - got) < 1e-6
        if "number" in body or "quantity" in body:
            said = correct
        else:
            # The same expression `audit_phase0.offline_judge` seeds with.
            forgive = random.Random(f"{seed}:{task_id}:{candidate}")
            said = correct or forgive.random() < generosity
        noise = random.Random(f"{seed}:{task_id}:{candidate}:noise")
        if noise.random() < flip_rate:
            said = not said
        return "YES" if said else "NO"
    return ask


def rescore(records: Sequence[AuditRecord], ask, rubric: str, context
            ) -> Dict[str, float]:
    """``record_id -> the candidate judge's score`` over stored outputs."""
    out: Dict[str, float] = {}
    for rec in records:
        ctx = context.get(rec.task_id)
        if ctx is None or not (rec.output or "").strip():
            out[rec.record_id] = rec.verifier_score
            continue
        out[rec.record_id] = verdict(
            ask(rubric, ctx[0], ctx[1], rec.output, rec.task_id))
    return out


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def evolve_rubric(tasks: Sequence[Task], ask, args, usage: Usage
                  ) -> Tuple[str, Any]:
    """Run the loop. Returns the winning rubric and the `EvolutionResult`."""
    def run_one(rendered: str, task: Task) -> str:
        return ask(rendered, task.prompt, task.meta["gold"],
                   task.meta["candidate"], task.meta["task_id"])

    def reward(task: Task, output: str) -> float:
        """Agreement with ground truth on this one judging decision.

        Not `1 - |f - Y|` over a set and not the residual's mean: a mean goes to
        zero when errors cancel, and a rubric that learns to over-accept as
        often as it over-rejects would score perfectly while being worse. Per
        decision, agreement *is* the residual's magnitude, and it cannot cancel.
        """
        return 1.0 if verdict(output) == task.meta["oracle"] else 0.0

    if args.dry_run:
        propose = _offline_propose
    else:
        agent = LLMAgent(anthropic_compatible(
            args.model, max_tokens=args.max_tokens, usage=usage,
            timeout=args.timeout))

        def propose(rendered: str, task: Task, output: str,
                    score: float) -> Optional[str]:
            if score == 1.0:
                return None           # nothing to learn from a correct call
            said = "correct" if verdict(output) == 1.0 else "incorrect"
            truth = "correct" if task.meta["oracle"] == 1.0 else "incorrect"
            return agent.propose(rendered, replace_prompt(task, said, truth),
                                 output, score)

    result = evolve(
        list(tasks), reward,
        run=run_one, propose=propose,
        artifact_id="judge_rubric",
        strategy=AppendRules(title=_TITLE),
        rounds=args.rounds, n_workers=args.workers,
        held_out_frac=args.held_out_frac, seed=args.seed,
        usage=usage, verbose=args.verbose)
    return result.rendered, result


def replace_prompt(task: Task, said: str, truth: str) -> Task:
    """The task, restated as the grading failure it was.

    The reflector is being asked to improve a *rubric*, and handed the original
    question it proposes an answering rule -- "state the number plainly" --
    which is advice for the solver and noise in a judge's prompt. It has to be
    told what the judge did and what was true.
    """
    return Task(
        id=task.id,
        prompt=("A grader was asked whether a candidate answer matches a "
                f"reference.\n\nQuestion: {task.prompt}\nReference: "
                f"{task.meta['gold']}\nCandidate: {task.meta['candidate'][:600]}"
                f"\n\nThe grader said {said}. The truth is that it is {truth}. "
                "Propose one grading rule that would have prevented this "
                "mistake and would not change any correct grading."),
        meta={})


def _offline_propose(rendered: str, task: Task, output: str,
                     score: float) -> Optional[str]:
    """A deterministic proposer for `--dry-run`.

    It proposes the one clause the offline judge responds to, and only on a
    disagreement -- enough for the acceptance path to have a real diff to
    accept, and no claim beyond that.
    """
    if score == 1.0:
        return None
    return ("Compare the final number as a quantity; the candidate is correct "
            "only if that number matches the reference.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", required=True,
                    help="the audit JSONL a Phase 0 run wrote")
    ap.add_argument("--workload", default="gsm8k",
                    choices=sorted(ERROR_MODES),
                    help="which workload wrote those records; picks the error "
                         "modes and the gold answers")
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--min-unseen", type=float, default=0.25,
                    help="a pool still finding new error modes at this rate is "
                         "worth training on. The pool that first blocked this "
                         "rung measured 0.0169")
    ap.add_argument("--min-examples", type=int, default=MIN_EXAMPLES,
                    help="...and so is one holding this many examples of a "
                         "single mode, however little variety is left. A clause "
                         "fixing all k has a one-sided p of 0.5**k, so 5 is the "
                         "smallest k that clears 0.05. The pool is refused only "
                         "when it fails both tests")
    ap.add_argument("--rounds", type=int, default=10,
                    help="a proposal is only requested on a rollout the judge "
                         "got wrong, so a loop sized by the usual 4-5 rounds "
                         "can finish having asked for none at all")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--held-out-frac", type=float, default=0.4)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--limit", type=int, default=400,
                    help="dataset rows to join against for gold answers")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--force", action="store_true",
                    help="train anyway on a saturated pool. Records the "
                         "override in the report; it does not remove it")
    args = ap.parse_args()

    records = resolved_records(args.records)
    if not records:
        raise SystemExit(f"{args.records} holds no resolved pairs")
    context = context_for(args.workload, limit=args.limit)
    improvement = [r for r in records if r.purpose is Purpose.IMPROVEMENT]
    calibration = [r for r in records if r.purpose is Purpose.CALIBRATION]
    if not improvement:
        raise SystemExit(
            "no improvement-pool labels. The pool is chosen at audit time by "
            "`calibration_fraction`; a run that set it to 1.0 kept none")

    pool = assess(improvement, context, args.workload,
                  min_unseen=args.min_unseen, min_examples=args.min_examples)
    if not pool.worth_training and not args.force:
        raise SystemExit(
            f"{len(improvement)} improvement labels. {pool.reason}\n"
            "A rubric evolved against this pool would be fitted to the errors "
            "that happen to be in hand.\n"
            "The fix is usually a workload where *grading* is hard rather than "
            "more labels on one where it is not -- see "
            "`scripts/audit_phase0.py --workload`. Pass --force to override.")

    train, balance = balanced(improvement, seed=args.seed)
    tasks = as_tasks(train, context)
    if not tasks:
        raise SystemExit("no training task could be joined to a gold answer; "
                         "check --workload and --limit")
    held, dropped = disjoint_heldout(calibration, train)

    usage = Usage()
    ask = (offline_judge(seed=args.seed) if args.dry_run
           else live_judge(anthropic_compatible(
               args.model, max_tokens=64, usage=usage, timeout=args.timeout,
               thinking={"type": "disabled"})))

    t0 = time.time()
    rubric, result = evolve_rubric(tasks, ask, args, usage)

    # The control arm: the STARTING rubric, re-run. Its disagreement with the
    # stored scores is the judge disagreeing with itself, and no candidate's
    # improvement means anything below it.
    control = rescore(held, ask, "", context)
    floor = sum(1 for r in held if control[r.record_id] != r.verifier_score)
    candidate = rescore(held, ask, rubric, context)

    got = evaluate_fix(held, lambda rec, ctx: candidate[rec.record_id],
                       context, noise_floor=floor)
    swept = rescan(held, lambda rec, ctx: candidate[rec.record_id], context)
    stats = residual_stats([replace(r, verifier_score=candidate[r.record_id])
                            for r in held])
    card = scorecard(
        Rectification(
            verifier_version=f"{records[0].verifier_version}+rubric",
            delta_hat=stats["delta"],
            delta_se=stats["sigma"] / max(1, stats["n"]) ** 0.5,
            theta=float("nan"), theta_ci=(float("nan"), float("nan")),
            se=stats["sigma"] / max(1, stats["n"]) ** 0.5,
            n=stats["n"], n_unlab=0, gain_factor=float("nan"),
            is_stale=False, stale_reason=None, resid_sd=stats["sigma"]),
        [replace(r, verifier_score=candidate[r.record_id]) for r in held],
        previous_records=list(held), rescan_report=swept)

    payload = {
        "workload": args.workload, "records": args.records,
        "unseen": pool.unseen, "bands": pool.bands,
        "min_unseen": args.min_unseen, "min_examples": args.min_examples,
        "largest_mode": pool.largest_mode, "largest": pool.largest,
        "pool_reason": pool.reason,
        "forced": bool(args.force and not pool.worth_training),
        "balance": balance, "n_train": len(tasks),
        "n_calibration": len(calibration), "n_heldout": len(held),
        "dropped_for_task_overlap": dropped,
        "rubric": rubric, "noise_floor": floor,
        "sigma_before": got.sigma_before, "sigma_after": got.sigma_after,
        "delta_before": got.delta_before, "delta_after": got.delta_after,
        "fixed": got.fixed, "broke": got.broken, "changed": got.changed,
        "helps": got.helps, "above_the_noise": got.above_the_noise,
        "fn_before": got.false_negative_before,
        "fn_after": got.false_negative_after,
        "flip_rate": swept.flip_rate, "passes": card.ship,
        "blockers": list(card.blockers),
        "seconds": round(time.time() - t0, 1),
        "calls": usage.calls, "dry_run": bool(args.dry_run),
    }
    out = pathlib.Path(args.out or
                       f"reports/audit_evolve_judge_{args.workload}_"
                       f"{date.today().isoformat()}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown(payload, card), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(payload, indent=2),
                                        encoding="utf-8")
    print(f"\nwrote {out}")


def markdown(p: Dict[str, Any], card) -> str:
    verdict_line = (
        "**The rubric clears the bar.**" if p["passes"] and p["above_the_noise"]
        else "**Not shown to help.**")
    if not p["above_the_noise"]:
        verdict_line += (f" It moved {p['changed']} units against a noise floor "
                         f"of {p['noise_floor']} -- re-running the starting "
                         "rubric moves about as many.")
    elif not p["passes"]:
        verdict_line += " " + "; ".join(p["blockers"])
    lines = [
        f"# Evolving the judge's rubric -- {p['workload']} "
        f"({date.today().isoformat()})",
        "",
        verdict_line,
        "",
        "| | |",
        "|---|---|",
        f"| records | `{p['records']}` |",
        f"| P(new error mode) | {p['unseen']:.4f} "
        f"(worth training at {p['min_unseen']}) |",
        f"| largest error mode | **{p['largest']}** x "
        f"`{p['largest_mode']}` (worth training at {p['min_examples']}) |",
        f"| training units | {p['n_train']} "
        f"({p['balance']['kept_each']} right + {p['balance']['kept_each']} "
        f"wrong, from {p['balance']['right']}/{p['balance']['wrong']}) |",
        f"| held-out units | {p['n_heldout']} of {p['n_calibration']} "
        f"calibration labels |",
        f"| dropped for task overlap | {p['dropped_for_task_overlap']} |",
        f"| model calls | {p['calls']} |",
        f"| wall clock | {p['seconds']}s |",
        "",
        "## The rubric it found",
        "",
        "```",
        p["rubric"].strip() or "(empty -- nothing was accepted)",
        "```",
        "",
        "## On the held-out calibration labels",
        "",
        "`sigma` is the target. `delta` is reported and never scored: a mean "
        "goes to zero when errors cancel.",
        "",
        "| | before | after |",
        "|---|---|---|",
        f"| `sigma` | {p['sigma_before']:.4f} | **{p['sigma_after']:.4f}** |",
        f"| `delta` | {p['delta_before']:+.4f} | {p['delta_after']:+.4f} |",
        f"| false negatives | {p['fn_before']:.1%} | {p['fn_after']:.1%} |",
        "",
        f"Fixed {p['fixed']}, broke {p['broke']}, moved {p['changed']} units "
        f"against a noise floor of **{p['noise_floor']}** -- the flips a re-run "
        "of the *starting* rubric produced on the same outputs.",
        "",
        "## Scorecard",
        "",
        card.to_markdown(),
    ]
    if p["forced"]:
        lines.insert(3, f"> Trained under `--force` on a pool the gate "
                        f"refused: {p['pool_reason']}\n")
    if p["dry_run"]:
        lines.insert(3, "> `--dry-run`: the judge is an offline stand-in. This "
                        "exercises the path and measures nothing.\n")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
