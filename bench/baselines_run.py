"""Run the equal-budget arms on a real dataset and print the table.

    python -m bench.baselines_run --dataset hotpotqa --budget-rollouts 96 \
        --width 4 --seeds 0,1,2 --provider openai --model GLM-5.2 --yes

Three arms -- `serial`, `best_of_n_fork`, `merge_of_n` -- over one
:class:`~agentdescent.baselines.Workload`, so only the execution shape varies.
See `agentdescent/baselines.py` for why this exists: every efficiency number in
this repository is a throughput speedup, and throughput cannot distinguish
merging from sampling-and-selecting because fork-and-select is parallel too.

**The default aggregator, deliberately.** The datasets come from the GEPA and ACE
ports, but their custom optimizers do not: GEPA's Pareto selection and ACE's
grow-and-refine are *search* strategies, and running them here would leave the
comparison unable to say whether a difference came from merging or from the
search. The thing under test is the merge, so the merge is the only thing that
changes. That also means the numbers here are not comparable with those ports'
own results, and are not meant to be.

**The third split.** `evolve()` optimises against its own held-out split and stops
when that split says so, and the fork arm *selects* on it -- so reporting it
would be reporting a training score twice over. `test_eval` scores `ds.test`,
which no gate in any arm ever sees.

Cost is the reason the defaults are small. Three arms x three seeds is nine runs,
and the fork arm is N runs by itself, so `--width 4 --seeds 0,1,2` is 33 runs of
`--budget-rollouts` rollouts each. Print `--plan` first.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from typing import Callable, List, Optional, Sequence

from agentdescent import SingleSlot, Usage
from agentdescent.dataloader import hf_rows
from agentdescent.evolution import Task
from agentdescent.baselines import (
    ArmResult, Budget, Workload, best_of_n_fork, compare, merge_of_n, serial,
    to_markdown,
)
from dataclasses import replace

from agentdescent.evolution import EvolutionResult
from agentdescent.fusion import reflective_merge
from agentdescent.policies import Policies
from examples._common import completion_for, confirm, score_tasks


def _hotpotqa(fetch: int, seed: int, completion, *, self_verify: bool = True,
              eval_concurrency: int = 8,
              val_cap: Optional[int] = None) -> Workload:
    """GEPA's dataset and actor; the engine's own optimizer."""
    from examples.gepa import gepa_prompt_evolution as gepa

    ds = gepa.load_dataset(fetch, seed=seed)
    reward = gepa.make_reward()
    agent = gepa.gepa_agent(completion)

    def test_eval(result: EvolutionResult) -> float:
        return gepa.evaluate(agent, result.state.get("instruction", ""), ds.test,
                             reward)

    tasks, val_frac = _capped(ds.trainval, ds.val_frac, val_cap)
    return Workload(
        tasks=tasks, reward=reward, test_eval=test_eval, agent=agent,
        strategy=gepa.InstructionSlot(),
        evolve_kwargs={
            "initial_state": {"instruction": gepa._SEED_INSTRUCTION},
            "artifact_id": "gepa_prompt", "blast_radius": 0.2,
            "held_out_frac": val_frac, "rounds": 10_000,
            "self_verify": self_verify, "eval_concurrency": eval_concurrency,
        })


def _capped(trainval, val_frac: float, val_cap: Optional[int]):
    """Trim the gate's split without touching train or test.

    The loaders split at fixed ratios, so `--pool` moves all three at once --
    and the two that matter pull in opposite directions. **val** is what a run
    costs: the gate sweeps it once per candidate, so it multiplies by every
    proposal. **test** is what a run can resolve: it is scored twice, at the end.
    Chasing a 20-task test split by raising `--pool` buys a 113-task gate nobody
    asked for.

    `evolve()` splits by position -- the last `held_out_frac` of the sequence --
    so capping is a slice of the tail plus a recomputed fraction. Train keeps
    everything the cap drops rather than discarding it.

    The gate gets noisier, which is a real cost and not a free one: acceptance
    decisions rest on fewer tasks. It is the same size of gate HotpotQA has been
    running with throughout.
    """
    if not val_cap:
        return list(trainval), val_frac
    n = len(trainval)
    n_val = max(1, round(n * val_frac))
    if n_val <= val_cap:
        return list(trainval), val_frac
    train, val = list(trainval[:n - n_val]), list(trainval[n - n_val:])
    kept, spare = val[:val_cap], val[val_cap:]
    tasks = train + spare + kept          # the cap's leftovers become train
    return tasks, len(kept) / len(tasks)


def _finer(pool: int, top_k: int, seed: int, completion, *,
           self_verify: bool = True, eval_concurrency: int = 8,
           val_cap: Optional[int] = None) -> Workload:
    """ACE's dataset and actor; the engine's own optimizer."""
    from examples.ace import ace_context_evolution as ace

    ds = ace.load_dataset(pool, top_k, seed=seed)
    reward = ace.make_reward()
    agent = ace.ace_agent(completion)

    def test_eval(result: EvolutionResult) -> float:
        return ace.evaluate(agent, result.rendered, ds.test, reward)

    tasks, val_frac = _capped(ds.trainval, ds.val_frac, val_cap)
    return Workload(
        tasks=tasks, reward=reward, test_eval=test_eval, agent=agent,
        strategy=ace.ACEPlaybook(),
        evolve_kwargs={
            "artifact_id": "ace_playbook", "blast_radius": 0.2,
            "held_out_frac": val_frac, "rounds": 10_000,
            "self_verify": self_verify, "eval_concurrency": eval_concurrency,
        })


def _gsm8k(fetch: int, seed: int, completion, *, self_verify: bool = True,
           eval_concurrency: int = 8,
           val_cap: Optional[int] = None) -> Workload:
    """GSM8K: a grade-school word problem in, a number out.

    The simplest real QA shape available, and the fastest. A HotpotQA prompt
    carries ten context paragraphs and cost ~38 seconds a call on a reasoning
    model; a GSM8K prompt is two sentences. Nothing else about the comparison
    changes -- one evolving instruction, exact-match on the final number.

    The artifact is an `InstructionSlot`, so this is a **one-key** workload: every
    pair of worker proposals contradicts, and `merge_of_n` cannot fuse anything
    without `--reflective-merge`. That is deliberate; it is the shape the merge
    question is being asked on.
    """
    from agentdescent.dataloader import split_dataset
    from agentdescent.evolution import LLMAgent

    rows = hf_rows("openai/gsm8k", "train", config="main", limit=fetch)
    tasks = [Task(id=str(i), prompt=r["question"],
                  meta={"gold": r["answer"].rsplit("####", 1)[-1].strip()})
             for i, r in enumerate(rows)]
    ds = split_dataset(tasks, ratios=(0.5, 0.25, 0.25), seed=seed, name="GSM8K")

    def reward(task, output: str) -> float:
        found = re.findall(r"-?[\d,]*\.?\d+", output.replace(",", ""))
        gold = task.meta["gold"].replace(",", "")
        return 1.0 if found and found[-1].rstrip(".") == gold else 0.0

    agent = LLMAgent(
        completion,
        solve_template="{artifact}\n\nProblem: {prompt}\n\n"
                       "End with a line `Answer: <number>`.",
        propose_template=(
            "You are improving the instruction given to a maths solver. "
            "It scored {reward:.2f} on this problem.\n\n"
            "Current instruction:\n{artifact}\n\n"
            "Problem:\n{prompt}\n\n"
            "Its answer (wrong or low-scoring):\n{output}\n\n"
            "Write an improved, general instruction. 2-3 sentences. "
            "Output only the instruction."),
    )

    def test_eval(result: EvolutionResult) -> float:
        return score_tasks(agent.solve, result.state.get("instruction", ""),
                           ds.test, reward)

    tasks_, val_frac = _capped(ds.trainval, ds.val_frac, val_cap)
    return Workload(
        tasks=tasks_, reward=reward, test_eval=test_eval, agent=agent,
        strategy=SingleSlot(key="instruction"),
        evolve_kwargs={
            "initial_state": {"instruction": "Solve the problem."},
            "artifact_id": "gsm8k_prompt", "blast_radius": 0.2,
            "held_out_frac": val_frac, "rounds": 10_000,
            "self_verify": self_verify, "eval_concurrency": eval_concurrency,
        })


def _bbh(fetch: int, seed: int, completion, *, task: str = "dyck_languages",
         self_verify: bool = True, eval_concurrency: int = 8,
         val_cap: Optional[int] = None) -> Workload:
    """BIG-Bench Hard: a short prompt that a strong model still gets wrong.

    The two properties an experiment here needs pull against each other on every
    other dataset tried. HotpotQA is hard but carries ten context paragraphs, so
    a call costs ~38 seconds. GSM8K and FiNER have short prompts and a reasoning
    model is already at 0.967 and 0.900 on them -- nothing left to measure.

    BBH was built to be the tasks large models fail, and several of its subtasks
    are a single line: `word_sorting` is 65 characters, `dyck_languages` 99.
    Short *and* unsolved is what makes it the right shape here, and `--bbh-task`
    is the difficulty dial -- no context to lengthen, no label space to widen.

    `InstructionSlot`, so this is a one-key workload: every pair of proposals
    contradicts, and `merge_of_n` cannot fuse without `--reflective-merge`.
    """
    from agentdescent.dataloader import split_dataset
    from agentdescent.evolution import LLMAgent

    rows = hf_rows("lukaemon/bbh", "test", config=task, limit=fetch)
    tasks = [Task(id=str(i), prompt=r["input"],
                  meta={"gold": str(r["target"]).strip()})
             for i, r in enumerate(rows)]
    ds = split_dataset(tasks, ratios=(0.5, 0.25, 0.25), seed=seed, name=f"BBH/{task}")

    def reward(t, output: str) -> float:
        # The last non-empty line, so a model that reasons before answering is
        # scored on its answer rather than on its working.
        lines = [ln.strip() for ln in str(output).splitlines() if ln.strip()]
        got = lines[-1] if lines else ""
        got = got.split("Answer:")[-1].strip().strip(".").strip()
        return 1.0 if got.lower() == t.meta["gold"].lower() else 0.0

    agent = LLMAgent(
        completion,
        solve_template="{artifact}\n\n{prompt}\n\n"
                       "End with a line containing only the final answer.",
        propose_template=(
            "You are improving the instruction given to a solver. It scored "
            "{reward:.2f} on this problem.\n\n"
            "Current instruction:\n{artifact}\n\n"
            "Problem:\n{prompt}\n\n"
            "Its answer (wrong or low-scoring):\n{output}\n\n"
            "Write an improved, general instruction. 2-3 sentences. "
            "Output only the instruction."),
    )

    def test_eval(result: EvolutionResult) -> float:
        return score_tasks(agent.solve, result.state.get("instruction", ""),
                           ds.test, reward)

    tasks_, val_frac = _capped(ds.trainval, ds.val_frac, val_cap)
    return Workload(
        tasks=tasks_, reward=reward, test_eval=test_eval, agent=agent,
        strategy=SingleSlot(key="instruction"),
        evolve_kwargs={
            "initial_state": {"instruction": "Solve the problem."},
            "artifact_id": "bbh_prompt", "blast_radius": 0.2,
            "held_out_frac": val_frac, "rounds": 10_000,
            "self_verify": self_verify, "eval_concurrency": eval_concurrency,
        })


def _bbeh(fetch: int, seed: int, completion, *, task: str = "disambiguation qa",
          self_verify: bool = True, eval_concurrency: int = 8,
          val_cap: Optional[int] = None) -> Workload:
    """BIG-Bench **Extra** Hard: short prompts a strong model still fails.

    Every other dataset tried here failed one of the two requirements. HotpotQA
    is hard and carries ten context paragraphs, so a call costs ~38 seconds.
    GSM8K and FiNER have short prompts and this model is already at 0.967 and
    0.900 -- nothing left to measure. BBH looked promising and turned out to be a
    measurement bug: a seed score of 0.267 on `word_sorting` was the *scorer*
    rejecting correct answers over comma-vs-space, and one commit that fixed the
    formatting took dev to 1.000 with eight rollouts left unspent.

    BBEH exists because BBH saturated -- Google built it by replacing each BBH
    task with a harder one probing the same capability, and reports 9.8% for the
    best general-purpose model and 44.8% for the best reasoning-specialised one.
    `disambiguation qa` is its shortest subtask at ~750 characters, and it is
    multiple-choice, so the answer is a label like ``(A)``.

    **The scorer is a label match, deliberately.** Free-text comparison is what
    produced the `word_sorting` mirage: a model that answers correctly in a
    different shape scores zero, and the run then optimises formatting while the
    numbers look like reasoning. A label cannot be right in two formats.
    """
    from agentdescent.dataloader import split_dataset
    from agentdescent.evolution import LLMAgent

    rows = [r for r in hf_rows("BBEH/bbeh", "train", limit=5000)
            if r["task"] == task][:fetch]
    if not rows:
        raise ValueError(f"no BBEH rows for task {task!r}")
    tasks = [Task(id=str(i), prompt=r["input"],
                  meta={"gold": str(r["target"]).strip()})
             for i, r in enumerate(rows)]
    ds = split_dataset(tasks, ratios=(0.5, 0.25, 0.25), seed=seed,
                       name=f"BBEH/{task}")

    def _label(text: str) -> str:
        found = re.findall(r"\(([A-Za-z0-9]+)\)", str(text))
        return found[-1].upper() if found else ""

    def reward(t, output: str) -> float:
        gold = _label(t.meta["gold"])
        got = _label(output)
        return 1.0 if gold and got == gold else 0.0

    agent = LLMAgent(
        completion,
        solve_template="{artifact}\n\n{prompt}\n\n"
                       "End with a line containing only the chosen option, "
                       "like (A).",
        propose_template=(
            "You are improving the instruction given to a solver. It scored "
            "{reward:.2f} on this problem.\n\n"
            "Current instruction:\n{artifact}\n\n"
            "Problem:\n{prompt}\n\n"
            "Its answer (wrong or low-scoring):\n{output}\n\n"
            "Write an improved, general instruction. 2-3 sentences. "
            "Output only the instruction."),
    )

    def test_eval(result: EvolutionResult) -> float:
        return score_tasks(agent.solve, result.state.get("instruction", ""),
                           ds.test, reward)

    tasks_, val_frac = _capped(ds.trainval, ds.val_frac, val_cap)
    return Workload(
        tasks=tasks_, reward=reward, test_eval=test_eval, agent=agent,
        strategy=SingleSlot(key="instruction"),
        evolve_kwargs={
            "initial_state": {"instruction": "Solve the problem."},
            "artifact_id": "bbeh_prompt", "blast_radius": 0.2,
            "held_out_frac": val_frac, "rounds": 10_000,
            "self_verify": self_verify, "eval_concurrency": eval_concurrency,
        })


#: A seed artifact already scoring above this on test leaves nothing to measure.
#: The second lesson, after the first: `--pool 1600 --top-k 40` passed the split
#: check with 26 test tasks and then reported 0.923 for almost every arm, because
#: `--top-k` is ACE's *difficulty* knob and lowering it to buy a bigger split made
#: the task easy. Split size and difficulty are separate properties and both have
#: to hold; only one of them can be checked without spending anything.
MAX_SEED_TEST_SCORE = 0.85

#: Below this many test tasks the quality column cannot resolve anything worth
#: reporting. Learned the expensive way: a run launched with a `--pool` that
#: happened to yield a **2-task** test split reported `test=1.000` for two
#: different arms -- 2/2 both times -- and had to be thrown away. The split is
#: cheap to check and the run is not, so it is checked first.
MIN_TEST_TASKS = 20


def split_sizes(args) -> tuple:
    """(train, val, test) for one seed, without touching a model.

    The dataset loaders are the ports' own, and they need no completion to build
    a split -- so the plan can report what the run will actually be measured on
    before a single call is paid for.
    """
    if args.dataset == "hotpotqa":
        from examples.gepa import gepa_prompt_evolution as gepa
        return gepa.load_dataset(args.fetch, seed=0).sizes()
    if args.dataset == "bbeh":
        from agentdescent.dataloader import split_dataset
        rows = [r for r in hf_rows("BBEH/bbeh", "train", limit=5000)
                if r["task"] == args.bbeh_task][:args.fetch]
        tasks = [Task(id=str(i), prompt=r["input"], meta={})
                 for i, r in enumerate(rows)]
        return split_dataset(tasks, ratios=(0.5, 0.25, 0.25), seed=0).sizes()
    if args.dataset == "bbh":
        from agentdescent.dataloader import split_dataset
        rows = hf_rows("lukaemon/bbh", "test", config=args.bbh_task,
                       limit=args.fetch)
        tasks = [Task(id=str(i), prompt=r["input"], meta={})
                 for i, r in enumerate(rows)]
        return split_dataset(tasks, ratios=(0.5, 0.25, 0.25), seed=0).sizes()
    if args.dataset == "gsm8k":
        from agentdescent.dataloader import split_dataset
        rows = hf_rows("openai/gsm8k", "train", config="main", limit=args.fetch)
        tasks = [Task(id=str(i), prompt=r["question"], meta={}) 
                 for i, r in enumerate(rows)]
        return split_dataset(tasks, ratios=(0.5, 0.25, 0.25), seed=0).sizes()
    from examples.ace import ace_context_evolution as ace
    return ace.load_dataset(args.pool, args.top_k, seed=0).sizes()


class _SeedOnly:
    """The un-evolved artifact, shaped like an `EvolutionResult` for `test_eval`.

    `test_eval` reads `rendered` (ACE) or `state` (GEPA), so the probe supplies
    both from the workload's own starting point -- no run, no rollouts, and
    nothing about the measurement differs from what an arm would be scored on.
    """

    def __init__(self, workload) -> None:
        initial = dict(workload.evolve_kwargs.get("initial_state") or {})
        strategy = workload.strategy
        if not initial and strategy is not None:
            initial = dict(strategy.initial())
        self.state = initial
        self.rendered = strategy.render(initial) if strategy is not None else ""


def _fmt(value) -> str:
    """A score, or ``n/a`` when there is not one. Never a formatted `None`."""
    return "n/a" if value is None else f"{value:.3f}"


def _workload_for(args, seed: int, completion):
    """One workload, built the way the run builds it."""
    shared = {"self_verify": args.self_verify,
              "eval_concurrency": args.eval_concurrency,
              "val_cap": args.val_cap}
    if args.dataset == "hotpotqa":
        return _hotpotqa(args.fetch, seed, completion, **shared)
    if args.dataset == "gsm8k":
        return _gsm8k(args.fetch, seed, completion, **shared)
    if args.dataset == "bbh":
        return _bbh(args.fetch, seed, completion, task=args.bbh_task, **shared)
    if args.dataset == "bbeh":
        return _bbeh(args.fetch, seed, completion, task=args.bbeh_task, **shared)
    return _finer(args.pool, args.top_k, seed, completion, **shared)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",
                   choices=["hotpotqa", "finer", "gsm8k", "bbh", "bbeh"],
                   default="hotpotqa")
    p.add_argument("--bbeh-task", default="disambiguation qa",
                   help="which BBEH subtask; the shortest is the default")
    p.add_argument("--bbh-task", default="dyck_languages",
                   help="which BIG-Bench Hard subtask; the difficulty dial")
    p.add_argument("--budget-rollouts", type=int, default=96)
    p.add_argument("--budget-calls", type=int, default=0,
                   help="0 leaves calls unbounded; see --fixed")
    p.add_argument("--fixed", choices=["rollouts", "calls"], default="rollouts",
                   help="which unit the comparison holds fixed. The other one "
                        "diverges and is printed as a confound -- they cannot "
                        "both be equalised")
    p.add_argument("--width", type=int, default=4,
                   help="N, for both merge-of-N and fork-of-N")
    p.add_argument("--seeds", default="0,1,2",
                   help="comma-separated; three is the minimum that shows a spread")
    p.add_argument("--arms", default="serial,fork,merge")
    p.add_argument("--fetch", type=int, default=48, help="hotpotqa rows")
    p.add_argument("--pool", type=int, default=800, help="finer rows to scan")
    p.add_argument("--top-k", type=int, default=120, help="finer concept cutoff")
    p.add_argument("--provider", default="openai", choices=["claude", "openai", "glm"])
    p.add_argument("--model", default="GLM-5.2")
    p.add_argument("--json", dest="json_out")
    p.add_argument("--no-self-verify", dest="self_verify", action="store_false",
                   help="skip the second rollout that measures each proposal's "
                        "own before/after delta. Halves the model cost per "
                        "proposal and removes one input from the Beta posterior "
                        "-- identical across arms, so the comparison stays valid, "
                        "but it is a different algorithm from the default and the "
                        "table has to say which one ran")
    p.add_argument("--run-concurrency", type=int, default=1,
                   help="how many of the (arm, seed) runs to execute at once. "
                        "They are independent by construction -- different seeds, "
                        "separate scratch ledgers, and no arm can observe another "
                        "-- so this changes wall-clock and nothing else. The "
                        "backend's rate limit is the real ceiling; start at 4")
    p.add_argument("--fork-concurrency", type=int, default=1,
                   help="how many forks inside one fork arm to run at once. That "
                        "arm is N runs by itself and is otherwise the slowest of "
                        "the three by a factor of N")
    p.add_argument("--eval-concurrency", type=int, default=8,
                   help="how many held-out tasks the gate scores at once. The "
                        "gate, not the rollouts, is what dominates wall-clock on "
                        "a small budget")
    p.add_argument("--headroom", action="store_true",
                   help="before anything else, score the *seed* artifact on the "
                        "test split and refuse if it is already too good. Costs "
                        "one call per test task -- a rounding error against a run, "
                        "and the only way to catch a workload that has no headroom, "
                        "which a split-size check cannot see")
    p.add_argument("--val-cap", type=int, default=0,
                   help="cap the gate's split at this many tasks, leaving train "
                        "and test alone. `--pool`/`--fetch` move all three at "
                        "once and the two that matter pull opposite ways: val is "
                        "what a run costs (swept once per candidate), test is "
                        "what it can resolve (scored twice, at the end). 0 keeps "
                        "the loader's own split")
    p.add_argument("--reflective-merge", action="store_true",
                   help="give the merge arm `reflective_merge()`: contradicting "
                        "proposals survive to the merge step and a model writes "
                        "their union, which goes straight to the acceptance gate. "
                        "Without it, a one-key artifact (GEPA's InstructionSlot) "
                        "cannot fuse at all and the merge arm is really per-round "
                        "best-of-N selection -- `fusion.contested` is 0 and says so")
    p.add_argument("--fusion-tournament", action="store_true",
                   help="rank each fused union against the individual candidates "
                        "on held-out data before the gate. Costs one extra sweep "
                        "per candidate, which is why it is off by default -- but "
                        "it is the only setting under which `fusion.contested` "
                        "and `win_rate` are populated, so a merge-vs-fork run "
                        "that wants to say whether fusion *fired* needs it")
    p.add_argument("--allow-thin-headroom", action="store_true",
                   help="run even when the seed artifact already scores above "
                        f"{MAX_SEED_TEST_SCORE}. The refusal still prints and the "
                        "number of tasks the whole comparison lives in is stated: "
                        "a result from here is bounded by that, and any table "
                        "carrying it has to say so")
    p.add_argument("--allow-small-test", action="store_true",
                   help=f"run even when the test split is under {MIN_TEST_TASKS} "
                        "tasks. It will not resolve a quality difference")
    p.add_argument("--plan", action="store_true",
                   help="print how many runs this would be, and stop")
    p.add_argument("--yes", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    budget = Budget(rollouts=args.budget_rollouts,
                    calls=args.budget_calls or None)

    # A fork arm is N runs, so the run count is not len(arms) x len(seeds) and
    # guessing it wrong is how a "small" sweep turns into an afternoon of API.
    runs = len(seeds) * sum(args.width if a == "fork" else 1 for a in arms)
    print(f"Dataset  : {args.dataset}")
    print(f"Arms     : {', '.join(arms)} (N={args.width})")
    print(f"Budget   : {budget.rollouts} rollouts"
          + (f", {budget.calls} calls" if budget.calls else ", calls unbounded")
          + f"; held fixed in {args.fixed}")
    print(f"Seeds    : {seeds}")
    print(f"Runs     : {runs} evolve() calls, "
          f"~{runs * budget.rollouts} rollouts in total")
    print("Optimizer: the engine's default aggregator -- the port's own search "
          "strategy is deliberately not used, so a difference cannot come from it")
    print(f"self_verify: {args.self_verify}"
          + ("" if args.self_verify else
             "  (each proposal's own before/after delta is not measured)"))
    try:
        ntr, nva, nte = split_sizes(args)
    except Exception as e:  # noqa: BLE001 - a cold cache needs the network
        print(f"Split    : could not be checked offline ({type(e).__name__})")
        ntr = nva = nte = None
    else:
        print(f"Split    : {ntr} train / {nva} val (the gate) / {nte} test "
              "(nothing sees this)")
        if nte < MIN_TEST_TASKS:
            print(f"\nREFUSED: {nte} test tasks cannot resolve a quality "
                  f"difference -- every number would be a count out of {nte}. "
                  f"Raise --pool/--fetch until the test split is at least "
                  f"{MIN_TEST_TASKS}, or pass --allow-small-test if a coarse "
                  "number is genuinely what you want.", file=sys.stderr)
            if not args.allow_small_test:
                return 2
    if args.plan and not args.headroom:
        return 0
    if len(seeds) < 3:
        print(f"warning: {len(seeds)} seed(s). docs/algo-ace.md records one "
              "configuration moving 4.8 points between two runs; one number per "
              "arm is not a result.", file=sys.stderr)
    if not confirm(args):
        return 0

    usage = Usage()
    completion = completion_for(args, usage=usage)

    if args.headroom:
        # The seed artifact is where every arm starts, so its score on test is
        # the ceiling the run is trying to climb from. Measured on seed 0 only:
        # this is a go/no-go on the *workload*, and paying for three of them to
        # decide whether to pay for fifteen would defeat the point.
        probe = _workload_for(args, 0, completion)
        seed_score = probe.test_eval(_SeedOnly(probe))
        print(f"Headroom : the seed artifact already scores {seed_score:.3f} "
              f"on test")
        if seed_score > MAX_SEED_TEST_SCORE:
            print(f"\nREFUSED: nothing to measure. Every arm starts at "
                  f"{seed_score:.3f} and the test split has {nte} tasks, so the "
                  f"whole comparison lives in the {(1 - seed_score) * (nte or 0):.0f} "
                  "task(s) above it. Raise the difficulty (ACE: --top-k; GEPA: a "
                  "harder --fetch sample) rather than the size.", file=sys.stderr)
            if not args.allow_thin_headroom:
                return 3
            print(f"    proceeding anyway: the comparison lives in "
                  f"{(1 - seed_score) * (nte or 0):.0f} task(s), and every number "
                  "from this run is bounded by that.", file=sys.stderr)
        if args.plan:
            return 0

    # Workloads are built once per seed and *before* any run starts, so every arm
    # at a seed sees byte-identical data -- and so a dataset download cannot
    # happen concurrently from several threads.
    workloads = {seed: _workload_for(args, seed, completion) for seed in seeds}

    def _progress(label: str):
        """Per-round output, because an arm is 20+ minutes of silence otherwise.

        `evolve(verbose=True)` prints to stdout from several threads at once and
        interleaves into an unreadable mess; `on_round` is the structured hook,
        and one line per round under the same lock as everything else keeps a
        concurrent sweep legible. It also makes a stall visible: a run that has
        stopped producing rounds looks exactly like a slow one from the outside,
        which is how an earlier sweep sat for 51 minutes on a single round.
        """
        def hook(info) -> None:
            with printing:
                print(f"    {label} r{info.round:<3} reward={info.held_out_reward:.3f} "
                      f"{info.rollouts} rollouts / {info.calls} calls "
                      f"+{info.committed}/-{info.rejected}", flush=True)
        return hook

    def _build(name: str, workload, seed: int) -> ArmResult:
        hooked = replace(workload, evolve_kwargs={
            **workload.evolve_kwargs, "on_round": _progress(f"{name}/s{seed}")})
        workload = hooked
        if name == "serial":
            return serial(workload, budget=budget, seed=seed)
        if name == "fork":
            return best_of_n_fork(workload, args.width, budget=budget, seed=seed,
                                  concurrency=args.fork_concurrency)
        if name == "merge":
            if args.fusion_tournament:
                # Only the merge arm can hold a tournament, and only a
                # tournament populates `contested`/`win_rate`. Installed here
                # rather than on the shared workload so the other two arms keep
                # the gate they would have had without this flag.
                workload = replace(workload, evolve_kwargs={
                    **workload.evolve_kwargs, "fusion_tournament": True})
            if args.reflective_merge:
                # Only the merge arm. serial has one proposal per step and fork
                # never merges, so installing it there would change nothing and
                # make the arms differ in more than one way.
                # One meter for the arm *and* its merger. Built here rather
                # than inside `merge_of_n` because the synthesis calls come from
                # a policy, and a policy spending on a throwaway `Usage` makes
                # the arm under test look cheaper than it is.
                arm_usage = Usage()
                merged = dict(workload.evolve_kwargs)
                merged["policies"] = Policies(
                    **reflective_merge(completion_for(args, usage=arm_usage)))
                w = replace(workload, evolve_kwargs=merged)
                return merge_of_n(w, args.width, budget=budget, seed=seed,
                                  usage=arm_usage)
            return merge_of_n(workload, args.width, budget=budget, seed=seed)
        raise ValueError(f"unknown arm {name!r}")

    jobs = [(name, seed) for seed in seeds for name in arms]
    printing = threading.Lock()

    def _run_job(job):
        name, seed = job
        try:
            arm = _build(name, workloads[seed], seed)
        except Exception as e:  # noqa: BLE001 - one dead arm is not nine
            # Every completed arm has already been paid for. Losing them because
            # a later one hit a dropped connection is the most expensive possible
            # response to a transient, and it is what happened before this
            # existed: three finished arms discarded by the fourth.
            with printing:
                print(f"  {name:<12} seed={seed}  FAILED "
                      f"{type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
            return None
        # `test_reward` is None when the scoring pass failed, which is the whole
        # point of it being Optional -- and formatting None with `:.3f` raises,
        # from inside the pool, taking the sweep down for the reason the
        # Optional existed to prevent. Every number here goes through `_fmt`.
        with printing:          # threads interleave; a torn line is unreadable
            print(f"  {arm.arm:<12} seed={seed}  {arm.rollouts} rollouts / "
                  f"{arm.calls} calls  dev={_fmt(arm.dev_reward)} "
                  f"test={_fmt(arm.test_reward)}"
                  + (f" oracle={_fmt(arm.test_oracle)}"
                     if arm.test_oracle is not None else "")
                  + (f"  ERROR {arm.error}" if arm.error else ""))
        return arm

    if args.run_concurrency > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=args.run_concurrency) as pool:
            collected = list(pool.map(_run_job, jobs))
    else:
        collected = [_run_job(job) for job in jobs]

    results: List[ArmResult] = [a for a in collected if a is not None]
    if not results:
        print("every arm failed; nothing to compare.", file=sys.stderr)
        return 4
    if len(results) < len(jobs):
        print(f"\n{len(jobs) - len(results)} of {len(jobs)} arms failed outright "
              "and are absent from the table below -- read the seed counts, not "
              "the arm names.", file=sys.stderr)

    comparison = compare(results, fixed=args.fixed)
    print()
    print(to_markdown(comparison))
    print()
    print(f"total model spend: {usage.summary()}")

    # `contested` before `win_rate`, always: a merge arm whose fusions never
    # fired is not evidence about merging, and the two are indistinguishable in
    # a quality table. Printed for every merge arm, so a run that answered
    # nothing says so on its own.
    for arm in results:
        if arm.fusion is not None:
            print(f"fusion  {arm.arm} seed={arm.seed}: {arm.fusion.summary()}")
        elif arm.arm.startswith("merge"):
            print(f"fusion  {arm.arm} seed={arm.seed}: no tournament held "
                  "(pass --fusion-tournament to populate contested/win_rate)")

    if "merge" in arms and "fork" in arms:
        merge_name, fork_name = f"merge-of-{args.width}", f"fork-of-{args.width}"
        if comparison.underpowered(merge_name, fork_name):
            # "We looked and found no difference" and "we could not have found
            # one" read identically in a table and mean opposite things about
            # what to do next.
            print(f"\n{len(seeds)} seed(s): too few to compare {merge_name} with "
                  f"{fork_name} at all. Not a null result -- an experiment that "
                  "could not have produced one.")
        elif comparison.separates(merge_name, fork_name):
            print(f"\n{merge_name} is above {fork_name} on every seed.")
        else:
            print(f"\n{merge_name} and {fork_name} overlap across seeds: this "
                  "budget on this dataset did not distinguish merging from "
                  "selecting. That is the result, and it belongs in "
                  "docs/results.md as one.")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({"dataset": args.dataset, "width": args.width,
                       "budget": {"rollouts": budget.rollouts,
                                  "calls": budget.calls},
                       "fixed": args.fixed, "seeds": seeds,
                       "model": args.model, "provider": args.provider,
                       "arms": [r.__dict__ for r in results]},
                      fh, indent=2, default=lambda o: getattr(o, "__dict__", str(o)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
