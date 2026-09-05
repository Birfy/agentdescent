"""Evolve one of `evolve()`'s decision slots, on a real dataset, and validate the transfer.

An inner problem here is a whole **inner `evolve()`** -- a run that evolves
one instruction against a slice of a benchmark -- and the outer artifact is a
`Policies` field that decides something about how that run is conducted.
Two axes, both registries in this file:

* **which benchmark** the inner run works on (:data:`BENCHMARKS`): the slice,
  the scorer, the wrapper the instruction meets a question through, and the
  instruction every inner run starts from;
* **which slot** evolves (:data:`SLOT_NOTES`): `task_sampler` decides which
  task each rollout spends, `acceptance` decides whether a candidate
  instruction commits. Both change the inner run's curve, which is what the
  meta-reward reads.

None of it needs a sandbox, numpy or a container -- only model calls, which is
why this exists beside the AlgoTune and Harbor scripts.

The transfer question is the same for every cell: a rule evolved on a few
slices of one benchmark is validated on slices it never saw *and* on another
benchmark, and the run reports the gain on each plus the ratio between them.

Sizing, before running it. The outer gate scores every candidate on the outer
**held-out** tasks, and an outer task is a whole inner run. The gate is also
the thing a narrow configuration breaks: measured, an effect worth +0.034 mean
across four windows was invisible to a gate holding out two of them, and under
L1 a tie is an oracle veto -- so `--train-windows` buys resolution, not just
data. The number that carries a claim is the paired validation, not the outer
`final_reward`.

    python -m bench.metasearch_slots --dry-run
    python -m bench.metasearch_slots --slot acceptance --benchmark aime --yes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from agentdescent.agents import Usage, with_retries
from agentdescent.dataloader import select_hard
from agentdescent.evalcache import FileCache
from agentdescent.evolution import Task, reflector
from agentdescent.meta import (MetaReward, Problem, auc, evolve_problem, final_reward,
                               meta_evolve, meta_validate, policy_source, rollouts_to,
                               slot_reflector)
from agentdescent.policies import Policies
from agentdescent.rewards import scorer as make_scorer
from agentdescent.strategies import SingleSlot

from examples._common import add_standard_args, completion_for, confirm, worker_count
from examples._measure import usage_dict


@dataclass(frozen=True)
class Benchmark:
    """One dataset in the four terms an inner run needs.

    ``template`` is not decoration. `examples._gsmhard_domain`'s ablation puts
    0.08-0.23 on the wrapper alone, so changing it changes the headroom the
    experiment measures; each one here is recorded with the baseline it was
    measured at rather than chosen freshly.
    """

    name: str
    load: Callable[[int], List[dict]]
    #: A name in :data:`agentdescent.rewards.SCORERS`.
    scorer: str
    template: str
    seed_instruction: str
    #: What the seed instruction scores here, measured; for choosing a target
    #: and for noticing a saturated benchmark before it is used as one.
    measured_baseline: str = ""


#: `examples._gsmhard_domain`'s own framing. Measured on 24 GSM-Hard items with
#: `deepseek-v4-flash` at temperature 0: this wrapper 0.500, the unlabelled
#: `"{skill}\n\nProblem:\n{prompt}"` 0.583 -- so this one, for the headroom.
LABELLED = "System instruction:\n{skill}\n\nUser problem:\n{prompt}"

MATH_SEED = "Solve the math word problem. Return only the final answer."
COMP_SEED = ("Solve the competition mathematics problem. The answer is an "
             "integer; return only that integer.")
QA_SEED = ("Answer the question using the passages provided. Return only the "
           "answer, as few words as possible.")

DEFAULT_OUTPUT = Path("bench/results/metasearch-slots.json")


def _gsm8k_rows(limit: int) -> List[dict]:
    from agentdescent.dataloader import hf_rows

    rows = hf_rows("openai/gsm8k", "test", config="main", limit=limit)
    return [{"question": r["question"], "answer": r["answer"]} for r in rows]


def _gsmhard_rows(limit: int) -> List[dict]:
    from examples._gsmhard_domain import load_rows

    return [{"question": r["input"], "answer": str(r["target"])} for r in load_rows()[:limit]]


def _aime_rows(limit: int) -> List[dict]:
    """Every AIME problem 1983-2024. Integer answers, so `last_number` grades it.

    Chosen over MATH-500 for the reason `examples._gsmhard_domain` gives for
    rejecting it: MATH-500's answers are LaTeX and comparing them needs a
    symbolic comparator written here, which would put the grader in this
    repository's hands. An AIME answer is an integer in [0, 999].
    """
    from agentdescent.dataloader import hf_rows

    rows = hf_rows("gneubig/aime-1983-2024", "train", limit=limit)
    return [{"question": r["Question"], "answer": str(r["Answer"])} for r in rows]


def _hotpotqa_rows(limit: int) -> List[dict]:
    """HotpotQA distractor: multi-hop QA, so the answer is words rather than a
    number and the scorer is `contains`. A different modality from the maths
    sets, which is the point of having it -- a sampler that only works on
    arithmetic is not a sampler."""
    from agentdescent.dataloader import hf_rows

    rows = hf_rows("hotpotqa/hotpot_qa", "validation", config="distractor", limit=limit)
    out = []
    for row in rows:
        context = row.get("context") or {}
        titles = context.get("title") or []
        sentences = context.get("sentences") or []
        passages = "\n".join(f"[{t}] {''.join(s)}" for t, s in zip(titles, sentences))
        out.append({"question": f"{passages}\n\nQuestion: {row['question']}",
                    "answer": row["answer"]})
    return out


BENCHMARKS: Dict[str, Benchmark] = {
    "gsmhard": Benchmark("gsmhard", _gsmhard_rows, "last_number", LABELLED, MATH_SEED,
                         "0.500 on 24 items (deepseek-v4-flash, temperature 0)"),
    "gsm8k": Benchmark("gsm8k", _gsm8k_rows, "last_number", LABELLED, MATH_SEED,
                       "1.000 on the windows used -- SATURATED, use --hard-other"),
    "aime": Benchmark("aime", _aime_rows, "last_number", LABELLED, COMP_SEED,
                      "measured by the run plan"),
    "hotpotqa": Benchmark("hotpotqa", _hotpotqa_rows, "contains", LABELLED, QA_SEED,
                          "measured by the run plan"),
}


def windows(benchmark: str, count: int, size: int, *, seed: int = 0,
            pool: int = 400, rows: Optional[Sequence[dict]] = None) -> Dict[str, List[Task]]:
    """``count`` disjoint slices of ``size`` tasks each, named ``<benchmark>-<i>``.

    Disjoint by construction: one shuffle of the pool, then consecutive slices.
    Two windows therefore never share a question, which is what lets one be
    evolved on while another validates."""
    rows = list(rows) if rows is not None else BENCHMARKS[benchmark].load(max(pool, count * size))
    order = list(range(len(rows)))
    random.Random(seed * 7919 + 13).shuffle(order)
    if count * size > len(order):
        raise ValueError(f"{benchmark}: {count} x {size} tasks requested, pool holds {len(order)}")
    out: Dict[str, List[Task]] = {}
    for index in range(count):
        chunk = order[index * size:(index + 1) * size]
        out[f"{benchmark}-{index}"] = [
            Task(id=f"{benchmark}:{i}", prompt=str(rows[i]["question"]).strip(),
                 meta={"gold": rows[i]["answer"]})
            for i in chunk
        ]
    return out


def hard_rows(rows: Sequence[dict], complete: Callable[[str], str], *,
              keep: int, pool: int, benchmark: Optional[Benchmark] = None) -> List[dict]:
    """The rows the seed instruction gets **wrong** -- headroom from a saturated set.

    Measured, and the reason this exists: on two GSM8K windows the seed
    instruction scored **1.000**, so the only move available to an evolved
    sampler was down, and the -0.100 that run reported was a property of the
    validation set rather than of the sampler. `select_hard` is the repository's
    own lever for exactly this.

    Its docstring warns that one noisy baseline pass selects the *unlucky*
    answers, which then regress to the mean. That warning does not bite here and
    it is worth saying why: `cached_completion` makes the baseline a function of
    the prompt, so an item scored wrong once is scored wrong every time, and
    `passes=1` is the whole measurement rather than a sample of it.

    **A hard subset is a different benchmark.** Numbers from it are not
    comparable with numbers from the full split, and the run plan says which
    was used.
    """
    benchmark = benchmark or BENCHMARKS["gsm8k"]
    score = make_scorer(benchmark.scorer)
    scanned = list(rows)[:pool]

    def wrong(row: dict) -> float:
        task = Task(id="probe", prompt=str(row["question"]).strip(),
                    meta={"gold": row["answer"]})
        return score(task, complete(benchmark.template.format(
            skill=benchmark.seed_instruction, prompt=task.prompt)))

    # `min_items=1`, deliberately: `select_hard` tops a short result up with the
    # items the baseline **solved**, which is the saturation this call exists to
    # remove. Refusing loudly is the only useful answer when the pool is too easy.
    kept = select_hard(scanned, wrong, keep=keep, min_items=1, passes=1)
    if len(kept) < keep or len(kept) == len(scanned):
        raise ValueError(
            f"only {len(kept)} of {len(scanned)} rows are hard for the seed "
            f"instruction, {keep} needed -- raise --hard-pool"
            + (" (nothing failed at all, so the pool is fully solved)"
               if len(kept) == len(scanned) else ""))
    return kept

# ---------------------------------------------------------------------------
# Determinism, and what the engine asks of each slot
# ---------------------------------------------------------------------------


#: What the engine imposes on each slot beyond its Protocol signature, stated
#: for the reflector. Not hints about *which* rule to write -- they are the
#: calling conventions any implementer would be told, and every proposal the
#: first live runs produced violated the sampler one.
SLOT_NOTES: Dict[str, str] = {
    "task_sampler": """How the engine calls this, which is stricter than the signature:

- `keys` is ONE WORKER'S SHARD for this round, not the whole task set, and it
  CHANGES between calls. The id you return is looked up in that round's tasks,
  so it must come from the `keys` you were just handed -- never from a set you
  remembered earlier. Returning a remembered id raises KeyError and kills the run.
- You may keep state across calls (a dict of task_id -> score is normal), but
  use it only to CHOOSE AMONG the current `keys`.
- Do not mutate `keys`; it belongs to the caller.
- `record(task_id, score)` arrives after the rollout you picked, so a task's
  score is known only after it has been spent at least once. `score >= 1.0`
  means the artifact already solves that task -- and the engine asks for no
  proposal from a rollout that passed, so a pick that lands on a solved task
  buys nothing.""",
    "acceptance": """How the engine calls this, and what the numbers mean:

- `ctx.base_counts` and `ctx.cand_counts` are `(successes, failures)` over the
  WHOLE held-out set -- not `(successes, trials)`. `MergeContext.rate(counts)`
  turns either into a rate; dividing by the second element gives a number that
  is wrong and plausible.
- `ctx.base_cheap` / `ctx.cand_cheap` are a SUB-SAMPLE. Ranking may use them.
  Deciding to commit may not: a gate that reads the cheap layer judges a
  regression from a handful of tasks.
- Return `AcceptDecision(accept, category, detail, p_improve, observed_delta)`.
  `category` is a stable bucket -- "committed" when you accept, and something
  like "below-threshold" when you do not -- because it is counted across rounds.
- You are the last gate before a commit, and this artifact is L1: an oracle has
  already vetoed anything that does not strictly beat the base on ground truth,
  so accepting everything does not make more things commit. Being too strict
  commits nothing and the run learns nothing; being too loose commits noise.""",
}


def cached_completion(complete: Callable[[str], str], directory: str, *,
                      key_extra: str = "") -> Callable[[str], str]:
    """``prompt -> text``, memoised on disk. Makes an inner run reproducible.

    A :class:`~agentdescent.meta.Problem` is documented as
    ``(value, seed) -> MetaOutcome``, and the paired comparison
    :func:`~agentdescent.meta.meta_validate` makes rests on that: score the seed
    rule and the evolved rule on the same problem and seed, and the difference
    is the rule. The engine's ``eval_cache`` memoises the *gate*, and nothing
    memoised the **rollouts** -- so two runs of the same sampler could take
    different proposals and land on different instructions.

    Measured, and the reason this exists: validating the seed rule against
    **itself**, byte for byte, reported a gain of -0.0625 on one 20-task
    GSM-Hard window (one seed better, one worse). A noise floor that size sits
    on top of any effect the outer loop could find, and under L1 governance
    every tie is an oracle veto -- which is exactly what the first two live runs
    produced: three sweeps, `{'oracle-rejected': 3}`, nothing committed.

    Deterministic sampling (``temperature=0``) is necessary and was not
    sufficient: the endpoint returned the same text for the same prompt most of
    the time, not always. Caching the call is what closes it.

    The key covers the prompt and whatever ``key_extra`` names about the
    request (model, temperature, token budget) -- change any of those and the
    old entries are simply not found rather than silently reused.
    """
    os.makedirs(directory, exist_ok=True)
    lock = threading.Lock()
    stats = {"hits": 0, "misses": 0}

    def complete_cached(prompt: str) -> str:
        digest = hashlib.sha256(f"{key_extra}\0{prompt}".encode("utf-8")).hexdigest()
        path = os.path.join(directory, f"{digest}.json")
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as handle:
                    with lock:
                        stats["hits"] += 1
                    return json.load(handle)["response"]
            except (OSError, ValueError, KeyError):
                pass          # a half-written entry is a miss, not a failure
        response = complete(prompt)
        with lock:
            stats["misses"] += 1
        # Written whole then renamed: a concurrent reader never sees a partial
        # entry, which is the failure the `except` above would otherwise absorb
        # silently on every call.
        fd, tmp = tempfile.mkstemp(dir=directory, suffix=".partial")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"response": response}, handle)
        os.replace(tmp, path)
        return response

    complete_cached.stats = stats          # type: ignore[attr-defined]
    return complete_cached


# ---------------------------------------------------------------------------
# The inner problem: one whole evolve() with the candidate rule installed
# ---------------------------------------------------------------------------


def inner_problem(tasks: Sequence[Task], complete: Callable[[str], str], *,
                  benchmark: Benchmark, slot: str = "task_sampler",
                  rounds: int = 4, workers: int = 1, held_out_frac: float = 0.4,
                  cache_dir: str = "", usage: Optional[Usage] = None) -> Problem:
    """``(slot value, seed) -> MetaOutcome``: an inner instruction-evolution run.

    ``workers=1`` on purpose: a sampler learns from ``record(task_id, score)``
    between picks, and with concurrent rollouts half the picks are made before
    any of their scores come back. The serial inner loop is the setting the
    slot is actually about, and it is also the cheapest.

    The gate's evaluations are memoised on disk when ``cache_dir`` is given.
    An instruction scored on a split once is scored once for the whole
    experiment -- and the seed instruction is scored by every single inner run,
    so this is most of the saving.
    """
    base = Policies(eval_cache=FileCache(cache_dir)) if cache_dir else Policies()
    return evolve_problem(
        list(tasks), make_scorer(benchmark.scorer),
        slot=slot, base=base,
        run=lambda rendered, task: complete(
            benchmark.template.format(skill=rendered, prompt=task.prompt)),
        propose=reflector(complete),
        strategy=SingleSlot(initial_value=benchmark.seed_instruction),
        rounds=rounds, n_workers=workers, max_concurrency=workers,
        held_out_frac=held_out_frac, eval_concurrency=8, self_verify=False,
        usage=usage,
    )


# ---------------------------------------------------------------------------
# The experiment
# ---------------------------------------------------------------------------


def build_problems(complete: Callable[[str], str], *, source: str, other: str,
                   train_windows: int, unseen_windows: int, other_windows: int,
                   size: int, data_seed: int, inner: Dict[str, Any],
                   usage: Optional[Usage] = None, hard_other: bool = False,
                   hard_pool: int = 400,
                   ) -> Tuple[Dict[str, Problem], Dict[str, Problem], Dict[str, List[str]]]:
    """Three groups of inner problems: evolved on, unseen, other benchmark.

    The first two are disjoint windows of the **same** benchmark, so a gain on
    the second is generalisation within a distribution; the third is the other
    benchmark, which is the transfer that matters. Reported apart because they
    are different claims.
    """
    same = windows(source, train_windows + unseen_windows, size, seed=data_seed)
    names = list(same)
    groups = {"train": names[:train_windows], "unseen": names[train_windows:]}
    every = dict(same)
    if other_windows:
        rows = None
        if hard_other:
            # Only the rows the seed instruction fails, so the group has room to
            # move in both directions rather than only down.
            rows = hard_rows(BENCHMARKS[other].load(hard_pool), complete,
                             keep=other_windows * size, pool=hard_pool,
                             benchmark=BENCHMARKS[other])
        cross = windows(other, other_windows, size, seed=data_seed, rows=rows)
        groups["other"] = list(cross)
        every.update(cross)
    else:
        groups["other"] = []
    built = {name: inner_problem(tasks, complete, usage=usage,
                                 benchmark=BENCHMARKS[name.rsplit("-", 1)[0]], **inner)
             for name, tasks in every.items()}
    train = {n: built[n] for n in groups["train"]}
    validate = {n: built[n] for n in groups["unseen"] + groups["other"]}
    return train, validate, groups


#: The meta-rewards this script offers, and the one thing to know about each.
META_REWARDS: Dict[str, Callable[[float], MetaReward]] = {
    # Mean best-so-far: what a decision rule controls in general, and the
    # library default. It reads high on this domain (0.875 on some 20-task
    # windows), and a tie under L1 governance is an oracle veto -- but the
    # first two live runs' `{'oracle-rejected': 3}` was **not** that: it was
    # the inner run not being reproducible, which `cached_completion` fixes.
    # With the run deterministic a tie means the two samplers behaved the same,
    # which is a finding rather than an artefact.
    "auc": lambda target: auc,
    # The inner run's own final score. Least sensitive to *speed*, which is the
    # thing a sampler actually changes.
    "final": lambda target: final_reward,
    # Time to quality: 1/(1+sweeps until the inner curve first reaches `target`).
    # Spread over {1, 0.5, 0.33, 0.25, 0}, so a rule that finds the informative
    # task one sweep earlier is visibly better -- but only if `target` sits above
    # what the inner run scores at sweep 0. Measured: at 0.75 on 20-task windows
    # the outer held-out read a flat 1.000, because an 8-task inner held-out set
    # puts several windows' seed instruction at 6/8 before anything is learned.
    # Pick the target against the inner baseline, not against the benchmark's.
    "time-to-quality": rollouts_to,
}


def meta_reward_for(name: str, target: float) -> MetaReward:
    if name not in META_REWARDS:
        raise SystemExit(f"unknown --meta-reward {name!r}; choose from {sorted(META_REWARDS)}")
    return META_REWARDS[name](target)


def recording_reflector(complete: Callable[[str], str], spec: Any
                        ) -> Tuple[Callable[..., Optional[str]], List[Dict[str, Any]]]:
    """The slot reflector, plus a log of every proposal it made.

    A run that commits nothing is a legitimate result and an illegible one
    unless it can say *what it tried*: the first four live runs here reported
    `{'oracle-rejected': 3}` and could not distinguish "the reflector proposed
    three samplers that genuinely did not beat round-robin" from "it proposed
    three that did not compile". Each entry records the raw text, whether the
    gate accepted it, and the reason when it did not.
    """
    proposals: List[Dict[str, Any]] = []
    inner = slot_reflector(complete, spec)
    lock = threading.Lock()

    def propose(rendered: str, task: Any, output: str, reward: float) -> Optional[str]:
        text = inner(rendered, task, output, reward)
        # `spec.accepts` rather than a second copy of the gate: the copy this
        # replaced forgot that `to_diff` strips a code fence, so every accepted
        # proposal was logged as refused.
        accepted, reason = spec.accepts(text or "")
        with lock:
            proposals.append({"source": text, "accepted_by_gate": accepted,
                              "reason": reason, "on_task": getattr(task, "id", "")})
        return text

    return propose, proposals

def progress(label: str) -> Callable[[Any], None]:
    """One line per outer sweep. A run that reports nothing until its summary
    cannot be told from a stalled one, and an outer sweep here is minutes of
    inner searches -- the same reason `examples._method_runner` prints one."""

    def on_round(info: Any) -> None:
        print(f"[{label} sweep {info.round}] held_out={info.held_out_reward:.3f} "
              f"committed={info.committed} rejected={info.rejected} "
              f"reasons={info.reasons} elapsed={info.elapsed_s:.0f}s "
              f"rollouts={info.rollouts}", flush=True)

    return on_round


def run_experiment(complete: Callable[[str], str], *, train: Dict[str, Problem],
                   validate: Dict[str, Problem], groups: Dict[str, List[str]],
                   seeds: Sequence[int], validate_seeds: Sequence[int], rounds: int,
                   workers: int, outer_seed: int = 0,
                   usage: Optional[Usage] = None, max_seconds: Optional[float] = None,
                   max_rollouts: Optional[int] = None,
                   eval_concurrency: Optional[int] = None,
                   meta_reward: Optional[MetaReward] = None,
                   slot: str = "task_sampler") -> Dict[str, Any]:
    """Evolve the sampler on ``train``, then score seed vs evolved on everything.

    ``meta_reward`` is what an inner run is worth; ``None`` is
    :func:`~agentdescent.meta.auc`. See :func:`meta_reward_for` for why the
    default is not always the right one here.
    """
    if set(train) & set(validate):
        raise ValueError(f"train and validate share problems: {sorted(set(train) & set(validate))}")
    if set(seeds) & set(validate_seeds):
        raise ValueError("validation seeds must not overlap the outer run's seeds")
    spec = policy_source(slot, notes=SLOT_NOTES[slot])
    seed_rule = spec.render(spec.initial())
    propose, proposals = recording_reflector(complete, spec)
    started = time.monotonic()
    result = meta_evolve(train, slot=slot, spec=spec,
                         propose=propose, seeds=list(seeds),
                         rounds=rounds, n_workers=workers, max_concurrency=workers,
                         held_out_frac=0.5,
                         eval_concurrency=eval_concurrency or max(1, workers),
                         max_seconds=max_seconds, max_rollouts=max_rollouts,
                         meta_reward=meta_reward, seed=outer_seed, usage=usage,
                         on_round=progress('gsm'))
    outer_seconds = time.monotonic() - started
    report = meta_validate(spec, seed_rule, result.rendered, {**train, **validate},
                           seeds=list(validate_seeds), meta_reward=meta_reward)
    by_group: Dict[str, Dict[str, float]] = {}
    for group, names in groups.items():
        rows = [report[n] for n in names if n in report]
        if not rows:
            continue
        by_group[group] = {
            "problems": [n for n in names if n in report],
            "seed_rule": sum(r["before"] for r in rows) / len(rows),
            "evolved_rule": sum(r["after"] for r in rows) / len(rows),
            "gain": sum(r["gain"] for r in rows) / len(rows),
            "wins": sum(r["wins"] for r in rows),
            "losses": sum(r["losses"] for r in rows),
        }
    train_gain = by_group.get("train", {}).get("gain", 0.0)

    def ratio(group: str) -> Optional[float]:
        if group not in by_group or abs(train_gain) < 1e-9:
            return None
        return by_group[group]["gain"] / train_gain

    return {
        "seed_source": seed_rule,
        "evolved_source": result.rendered,
        "outer": {"final_reward": result.final_reward, "rollouts": result.rollouts,
                  "outcomes": result.outcomes(), "stop_reason": result.stop_reason,
                  "error": result.error, "seconds": outer_seconds, "rounds": rounds,
                  "workers": workers, "seeds": list(seeds),
                  # What the search actually tried, so a run that commits
                  # nothing still says why.
                  "proposals": proposals,
                  "proposals_rejected_by_gate": sum(
                      1 for p in proposals if not p["accepted_by_gate"]),
                  "invalid_proposals": int(getattr(spec, "invalid_proposals", 0)),
                  "history": [{"round": h.round, "held_out_reward": h.held_out_reward,
                               "committed": h.committed, "rejected": h.rejected,
                               "reasons": h.reasons} for h in result.history]},
        "validation": report,
        "by_group": by_group,
        "transfer_ratio": {"unseen": ratio("unseen"), "other": ratio("other")},
        "validate_seeds": list(validate_seeds),
    }


def format_report(payload: Dict[str, Any]) -> str:
    group_of = {n: g for g, row in payload["by_group"].items() for n in row["problems"]}
    lines = [f"{'problem':<14} {'group':<8} {'seed':>7} {'evolved':>8} {'gain':>7} {'sd':>6} {'w/l':>5}"]
    for name, row in payload["validation"].items():
        lines.append(f"{name:<14} {group_of.get(name, '?'):<8} {row['before']:>7.3f} "
                     f"{row['after']:>8.3f} {row['gain']:>+7.3f} {row['gain_sd']:>6.3f} "
                     f"{row['wins']:>2}/{row['losses']}")
    lines.append("")
    for group, row in payload["by_group"].items():
        lines.append(f"{group:<14} {'mean':<8} {row['seed_rule']:>7.3f} "
                     f"{row['evolved_rule']:>8.3f} {row['gain']:>+7.3f} {'':>6} "
                     f"{row['wins']:>2}/{row['losses']}")
    for group, value in payload["transfer_ratio"].items():
        lines.append(f"transfer ratio ({group} gain / train gain): "
                     + ("n/a (no train gain)" if value is None else f"{value:.2f}"))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_standard_args(parser, model_default="deepseek-v4-flash", max_seconds_default=3600.0,
                      eval_concurrency_default=None, include_val_cap=False)
    parser.set_defaults(provider="openai", async_ratio=1)
    parser.add_argument("--slot", default="task_sampler", choices=sorted(SLOT_NOTES),
                        help="which decision of the inner run evolves")
    parser.add_argument("--source", "--benchmark", dest="source", default="gsmhard",
                        choices=sorted(BENCHMARKS),
                        help="the benchmark the rule is evolved on")
    parser.add_argument("--other", default="gsm8k", choices=sorted(BENCHMARKS),
                        help="the benchmark it is validated on but never evolved on")
    parser.add_argument("--train-windows", type=int, default=2)
    parser.add_argument("--unseen-windows", type=int, default=2)
    parser.add_argument("--other-windows", type=int, default=2)
    parser.add_argument("--hard-other", action="store_true",
                        help=("keep only the rows of --other that the seed "
                              "instruction gets wrong. A saturated group can only "
                              "move down: measured, the seed rule scored 1.000 on "
                              "both plain GSM8K windows. A hard subset is a "
                              "different benchmark -- say which you used"))
    parser.add_argument("--hard-pool", type=int, default=400,
                        help="rows of --other to scan for the hard subset")
    parser.add_argument("--window-size", type=int, default=20, help="tasks per inner problem")
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=2, help="inner seeds per train problem")
    parser.add_argument("--validate-seeds", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=3, help="outer rounds")
    parser.add_argument("--workers", type=int, default=2, help="outer workers")
    parser.add_argument("--inner-rounds", type=int, default=5)
    parser.add_argument("--inner-workers", type=int, default=1,
                        help=("workers inside one inner run. 1 is the setting the "
                              "sequential slots are about -- a sampler learns from "
                              "`record` between picks, and with concurrent rollouts "
                              "half the picks are made before any score comes back. "
                              "Raise it to make `conflict`/`fusion` live at all: "
                              "with one worker a merge has one diff, so there is "
                              "nothing to resolve or fuse"))
    parser.add_argument("--meta-reward", default="auc", choices=sorted(META_REWARDS),
                        help=("what one inner run is worth to the outer loop; see "
                              "META_REWARDS for what each one saturates on"))
    parser.add_argument("--completion-cache", default="",
                        help=("memoise prompt -> text here, which is what makes an "
                              "inner run reproducible and so a paired comparison "
                              "meaningful. Defaults to <--eval-cache>/completions"))
    parser.add_argument("--quality-target", type=float, default=0.75,
                        help=("the bar --meta-reward time-to-quality measures the "
                              "time to. Must sit above the seed instruction's own "
                              "score or every run scores 1.0"))
    parser.add_argument("--inner-held-out-frac", type=float, default=0.4)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--api-timeout", type=float, default=180.0)
    parser.add_argument("--thinking", choices=("disabled", "enabled", "default"),
                        default="disabled",
                        help=("reasoning tokens. Disabled by default: the solver is "
                              "asked for one number and the reflector for one "
                              "instruction, and a reasoning preamble is most of the "
                              "latency. A run that changes this is not comparable "
                              "with one that did not"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.asynchronous or args.pipelined_gate:
        # An outer rollout here is a whole inner `evolve()`, and the barrier-free
        # runtime's staleness is defined over artifact versions the merger moved
        # -- honouring it would need a second design, not a flag. Refused rather
        # than accepted and ignored, which is how a port grows a switch nobody
        # reads.
        raise SystemExit("--async / --pipelined-gate are not supported by the "
                         "meta loop; the outer runtime is the synchronous one")
    workers = worker_count(args, args.workers)
    # The completion cache is what makes an inner run a function of the sampler
    # (see `cached_completion`); it lives beside the gate cache and is on
    # whenever that is, because a paired comparison without it measures noise.
    completion_cache = (args.completion_cache
                        or (os.path.join(args.eval_cache, "completions")
                            if args.eval_cache else ""))
    seeds = list(range(args.seeds))
    validate_seeds = list(range(1_000, 1_000 + args.validate_seeds))
    inner = {"rounds": args.inner_rounds, "workers": args.inner_workers,
             "slot": args.slot,
             "held_out_frac": args.inner_held_out_frac,
             # `--eval-cache` is the shared flag; here it memoises the *inner*
             # gate, where the seed instruction is re-scored by every inner run.
             "cache_dir": args.eval_cache}
    outer_tasks = args.train_windows * len(seeds)
    validations = 2 * (args.train_windows + args.unseen_windows + args.other_windows) * len(validate_seeds)
    print(f"Algorithm : meta_evolve over the engine's `{args.slot}` slot (policy_source)")
    bench_src = BENCHMARKS[args.source]
    print(f"Inner     : one whole evolve() per rollout -- {args.inner_rounds} sweeps, "
          f"{args.inner_workers} worker(s), {args.window_size} tasks split "
          f"{1 - args.inner_held_out_frac:.0%}/{args.inner_held_out_frac:.0%}, "
          f"instruction evolution scored by `{bench_src.scorer}`")
    print(f"Baseline  : {bench_src.name} seed instruction -- {bench_src.measured_baseline}")
    print(f"Train     : {args.train_windows} x {args.source} window(s) x {len(seeds)} seed(s) "
          f"= {outer_tasks} outer tasks")
    print(f"Validate  : {args.unseen_windows} unseen {args.source} + {args.other_windows} "
          f"{args.other}{' (HARD subset -- a different benchmark)' if args.hard_other else ''} "
          f"window(s), {len(validate_seeds)} fresh seed(s) -> {validations} inner runs")
    print(f"Reward    : {args.meta_reward}"
          + (f" @ {args.quality_target}" if args.meta_reward == "time-to-quality" else "")
          + "  (L1: a candidate that only ties the seed on the outer held-out set "
            "is vetoed by the oracle, so a saturated reward commits nothing)")
    print(f"Outer     : rounds={args.rounds} workers={workers} blast_radius=0.6 (L1)"
          + (f" budget={args.budget_rollouts} rollouts" if args.budget_rollouts else "")
          + f" max_seconds={args.max_seconds:.0f}")
    print(f"Model     : {args.provider}/{args.model} temperature={args.temperature} "
          f"thinking={args.thinking}")
    print("Caches    : gate=" + (args.eval_cache or "off") + "  completions="
          + (completion_cache or "OFF -- inner runs will not be reproducible, "
                                 "so a paired gain measures noise"))
    if args.dry_run:
        print("[dry-run] no API call and no dataset fetch.")
        return 0
    if not confirm(args):
        return 0
    usage = Usage()
    options: Dict[str, Any] = {}
    if args.thinking != "default":
        options["thinking"] = {"type": args.thinking}
    complete = with_retries(
        completion_for(args, usage=usage, max_tokens=args.max_tokens,
                       timeout=args.api_timeout, temperature=args.temperature,
                       retries=1, **options),
        attempts=4, backoff=3.0)
    if completion_cache:
        complete = cached_completion(
            complete, completion_cache,
            key_extra=f"{args.model}|{args.temperature}|{args.max_tokens}|{args.thinking}")
    train, validate, groups = build_problems(
        complete, source=args.source, other=args.other,
        train_windows=args.train_windows, unseen_windows=args.unseen_windows,
        other_windows=args.other_windows, size=args.window_size,
        data_seed=args.data_seed, inner=inner, usage=usage,
        hard_other=args.hard_other, hard_pool=args.hard_pool)
    started = time.monotonic()
    payload = run_experiment(complete, train=train, validate=validate, groups=groups,
                             seeds=seeds, validate_seeds=validate_seeds,
                             rounds=args.rounds, workers=workers, outer_seed=args.seed,
                             usage=usage, max_seconds=args.max_seconds,
                             max_rollouts=args.budget_rollouts or None,
                             eval_concurrency=args.eval_concurrency,
                             meta_reward=meta_reward_for(args.meta_reward,
                                                         args.quality_target),
                             slot=args.slot)
    payload["config"] = {**inner, "source": args.source, "other": args.other,
                         "window_size": args.window_size, "data_seed": args.data_seed,
                         "model": args.model, "provider": args.provider,
                         "temperature": args.temperature, "thinking": args.thinking,
                         "hard_other": args.hard_other,
                         "meta_reward": args.meta_reward,
                         "quality_target": args.quality_target,
                         "slot": args.slot,
                         "template": BENCHMARKS[args.source].template,
                         "seed_instruction": BENCHMARKS[args.source].seed_instruction,
                         "outer_seed": args.seed}
    # `usage_dict` rather than a fourth hand-rolled copy: three of them in this
    # repository used `input_tokens`/`output_tokens`, which `Usage` does not
    # have (they are the Anthropic SDK's names), and the AttributeError landed
    # on the last line of an hour-long run -- after every measurement was taken
    # and before any of it was written.
    payload["usage"] = {**usage_dict(usage), "wall_seconds": time.monotonic() - started,
                        "completion_cache": getattr(complete, "stats", None)}
    outer = payload["outer"]
    print(f"[proposals] {len(outer['proposals'])} made, "
          f"{outer['proposals_rejected_by_gate']} refused by the gate, "
          f"{outer['invalid_proposals']} produced no diff")
    print("[evolved sampler]\n" + payload["evolved_source"])
    print(format_report(payload))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    print(f"[result saved] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
