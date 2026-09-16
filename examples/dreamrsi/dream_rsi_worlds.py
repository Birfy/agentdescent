"""Dream-RSI: explore online, dream in the recorded worlds, redeploy.

    "My exploration policy costs a whole discovery run to evaluate. I already
    ran one. Can the run I paid for score the policies I have not tried?"

Dream-RSI (Zheng et al., 2026) says yes, and this example is its loop on a cheap
synthetic domain, so the machinery can be exercised end to end with no model and
no sandbox.

* **artifact**: the source of an exploration policy -- ``select(ctx, n)`` over
  the revealed part of a discovery tree, returning the batch of nodes to
  continue. The seed is the paper's manually designed *parallel refining*
  policy (:data:`~agentdescent.dream.PARALLEL_REFINE_SEED`).
* **task**: one recorded discovery tree, replayed. Revealing a node costs
  nothing -- the score is already in the record -- so a task that used to be a
  whole inner search is now a few hundred dictionary lookups.
* **reward**: Equation 1 -- best score revealed, minus ``b1`` per continuation,
  plus ``b2`` times the mean batch size -- normalised by what the world itself
  could give.
* **propose**: :func:`~agentdescent.dream.replay_reflector` shows the model the
  round-by-round trajectory, the decomposition of Equation 1, and what earlier
  revisions scored; it rewrites the policy.
* **redeploy**: the selected policy runs the next online rollout, whose tree
  joins the pool. That is the recursion.

**What the offline run demonstrates, and what it does not.** With
``--offline`` the reflector is a fixed script rather than a model, so the run
shows the *mechanism*: that a policy which stops once branches saturate is
scored above the seed on worlds the seed itself recorded, is selected, and then
halves the cost of the next online rollout. It is not a measurement of a model's
ability to find such a policy, and no number here corresponds to anything in the
paper.

**The limitation to keep in view.** A replay world contains only what the
recording policy explored. Dreaming can discover a cheaper, better-ordered, or
better-batched walk over that record; it cannot discover a direction nobody
took. That is the paper's own framing -- the simulator is *"a grounded model of
the portion of the discovery space that has already been observed"* -- and it is
why the pool keeps every world instead of only the newest.

Run::

    python -m examples.dreamrsi.dream_rsi_worlds --dry-run
    python -m examples.dreamrsi.dream_rsi_worlds --offline --yes
    python -m examples.dreamrsi.dream_rsi_worlds --provider openai \\
        --model deepseek-v4-flash --thinking disabled --rounds 4 --yes
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Sequence

from agentdescent.agents import Completion, Usage, with_retries
from agentdescent.dream import (PARALLEL_REFINE_SEED, DreamResult, DreamRound,
                                ReplayObjective, dream_rsi, exploration_policy,
                                explore, replay_reflector)
from agentdescent.meta import cached_completion

from examples._common import (add_standard_args, budget_kwargs, completion_for,
                              confirm, worker_count)
from examples._measure import usage_dict
from examples.dreamrsi._world import (FAMILIES, SOURCE, TARGET, branch_report,
                                      discovery_agent)

__all__ = ["OFFLINE_PROPOSAL", "build_parser", "main", "offline_model",
           "run_loop", "validate"]


#: What ``--offline`` proposes instead of calling a model.
#:
#: Not a placeholder: it is the policy this domain's structure rewards, and
#: writing it out is how the example states what the run is supposed to find.
#: Branches approach a hidden ceiling geometrically, so two attempts are enough
#: to rank them and everything spent after that on a branch that cannot win is
#: pure cost. The seed policy refines all of them to the end anyway.
#:
#: Two details in it were each paid for by a run on the example world, and both
#: are things Appendix B.2 tells the policy-development agent in prose:
#:
#: * **It prunes rather than stopping.** A policy that simply stops early --
#:   three waves and out -- scores *below* the seed on the worlds the seed
#:   recorded, 0.755 against 0.758 mean replay value over eight of them. It wins
#:   five of the eight and loses the mean, because a branch's ceiling is not yet
#:   visible at depth three and the worlds where it guesses wrong cost more than
#:   the continuations it saved were worth.
#: * **It ranks a branch by its best score, not by its latest.** Ranking on the
#:   frontier node's own score drops any branch whose last attempt happened to
#:   fail -- 15% of them here. Anchoring on the branch's best instead takes the
#:   same policy from 0.734 to 0.779 on those eight worlds, and from 4 wins to
#:   7. `tests/test_dream_rsi.py` keeps the losing variant around under
#:   `RANK_ON_LATEST`, because the dreaming gate *commits* it and only the
#:   pool-wide argmax refuses it.
OFFLINE_PROPOSAL = '''class Policy:
    # Rank a branch by the best score anywhere on it, not by its latest attempt:
    # a failed refinement is repairable and must not erase the branch's anchor.
    # Open every branch, measure each one twice, then follow the leaders only.
    def select(self, ctx, n):
        legal = [c for c in ctx.candidates if c.state.get("legal") == "1"]
        if not legal:
            return []
        roots = [c for c in legal if c.parent is None]
        frontier = [c for c in legal if c.parent is not None]
        if not frontier:
            return ([roots[0]] * n) if roots else []
        anchor = {}
        for c in ctx.candidates:
            branch = c.per_task.get("branch")
            if branch is None or c.score is None:
                continue
            if branch not in anchor or c.score > anchor[branch]:
                anchor[branch] = c.score
        frontier.sort(key=lambda c: -anchor.get(c.per_task.get("branch"), 0.0))
        shallow = min(c.per_task.get("depth", 0.0) for c in frontier)
        if shallow < 2.0 and len(frontier) >= 2:
            return frontier[:n]                 # measure every branch twice
        return frontier[:max(1, n // 2)]        # then follow the leaders only
'''


#: The share of the pool the dreaming gate holds out.
#:
#: Worlds are the scarce thing here -- each one is an online rollout -- so the
#: gate is small however this is set, and 0.4 is the smallest fraction that
#: leaves it able to separate anything. Measured on this domain: with a pool of
#: four worlds and ``0.5`` the gate is two, the pruning policy ties on them, and
#: L1 governance turns every tie into an oracle veto -- three dreaming sweeps,
#: ``{'oracle-rejected': 3}``, nothing committed. Eight worlds at ``0.4`` is a
#: gate of three, and the same proposal commits.
HELD_OUT_FRAC = 0.4


def offline_model(source: str = OFFLINE_PROPOSAL) -> Completion:
    """A scripted reflector: every call returns ``source``.

    The offline arm exists so the loop is exercised by the test suite and by a
    reader with no API key. It deliberately proposes the *same* policy every
    time, because what is being demonstrated is the selection step -- the
    proposal is a given, and whether it is adopted is the question.
    """

    def complete(prompt: str) -> str:
        return "```python\n" + source + "```"

    return complete


def run_loop(complete: Completion, *, rounds: int, workers: int,
             online_rounds: int, replay_rounds: int, repeats: int,
             dream_rounds: int, dream_workers: int, family_name: str, seed: int,
             cost_weight: float, parallel_weight: float, max_seconds: float,
             on_round=None, extra: Optional[dict] = None) -> DreamResult:
    """The outer loop: :func:`~agentdescent.dream.dream_rsi` on one family.

    ``workers`` and ``dream_workers`` are separate, and ``--serial`` moves only
    the second. ``W`` is the *discovery agent's* parallelism -- it is in the
    paper, it is the batch cap the policy plays against, and it is the ``N/k``
    term of Equation 1, so a run with ``W=1`` is a different algorithm rather
    than a serial control of this one. What this repository parallelises, and
    what the control arm has to remove, is the **offline phase**: the paper
    revises the policy ``M`` times in sequence, and ``meta_evolve`` runs those
    revisions concurrently and merges them.
    """
    family = FAMILIES[family_name]
    objective = ReplayObjective.scaled(
        max_nodes=max(1, workers * online_rounds), n_workers=workers,
        cost_weight=cost_weight, parallel_weight=parallel_weight)
    spec = exploration_policy()
    return dream_rsi(
        discovery_agent(family, seed=seed),
        spec=spec,
        propose=replay_reflector(complete, spec),
        objective=objective,
        rounds=rounds, n_workers=workers, online_rounds=online_rounds,
        replay_rounds=replay_rounds, online_repeats=repeats,
        dream_rounds=dream_rounds, dream_workers=dream_workers,
        root_score=family.root_score, held_out_frac=HELD_OUT_FRAC,
        max_seconds=max_seconds, on_round=on_round, **(extra or {}))


def validate(rendered: str, *, family_name: str, workers: int,
             online_rounds: int, seeds: Sequence[int],
             cost_weight: float, parallel_weight: float) -> Dict[str, float]:
    """Deploy the seed policy and the evolved one on **fresh** worlds.

    The number that matters is not the replay value -- that is measured on
    worlds the loop already saw. It is what each policy does *online*, on a
    family and seeds the loop never touched: what it finds, and what it spent
    finding it. A policy that wins on replay value and loses here has been fitted
    to the record rather than improved.
    """
    family = FAMILIES[family_name]
    spec = exploration_policy()
    objective = ReplayObjective.scaled(
        max_nodes=max(1, workers * online_rounds), n_workers=workers,
        cost_weight=cost_weight, parallel_weight=parallel_weight)
    report: Dict[str, float] = {}
    for label, source in (("seed", PARALLEL_REFINE_SEED), ("evolved", rendered)):
        policy = exploration_policy(source).compile(source)
        best: List[float] = []
        cost: List[int] = []
        attempts = 0
        for index in seeds:
            tree = explore(discovery_agent(family, seed=index), policy,
                           n_workers=workers, max_rounds=online_rounds,
                           root_score=family.root_score, attempt_offset=attempts,
                           name=f"v{index}")
            attempts += max(0, len(tree) - 1)
            found = tree.best()
            best.append(family.root_score if found is None else found)
            cost.append(len(tree) - 1)
        report[f"{label}_best"] = sum(best) / len(best)
        report[f"{label}_calls"] = sum(cost) / len(cost)
    report["quality_delta"] = report["evolved_best"] - report["seed_best"]
    report["call_ratio"] = (report["seed_calls"] / report["evolved_calls"]
                            if report["evolved_calls"] else float("inf"))
    # Unused here beyond the report, but it is what `spec` is for: the evolved
    # source has to still pass the gate at the end of the run, not only when it
    # was proposed.
    spec.compile(rendered)
    return report


def format_report(report: Dict[str, float]) -> str:
    return "\n".join([
        f"{'policy':<10}{'best found':>12}{'agent calls':>14}",
        f"{'seed':<10}{report['seed_best']:>12.4f}{report['seed_calls']:>14.1f}",
        f"{'evolved':<10}{report['evolved_best']:>12.4f}{report['evolved_calls']:>14.1f}",
        f"{'delta':<10}{report['quality_delta']:>+12.4f}"
        f"{report['call_ratio']:>13.2f}x",
    ])


def progress():
    def on_round(record: DreamRound) -> None:
        note = "redeployed" if record.redeployed else record.stop_reason or "kept"
        print(f"  [round {record.index}] worlds={record.pool_size} "
              f"online_calls={record.online_nodes} "
              f"best={record.online_best if record.online_best is None else round(record.online_best, 4)} "
              f"V {record.value_before:.4f} -> {record.value_after:.4f}  {note}")

    return on_round


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_standard_args(parser, model_default="deepseek-v4-flash",
                      max_seconds_default=1800.0, include_val_cap=False)
    # An OpenAI-compatible endpoint, and a lag budget of one: a dreaming
    # rollout is a whole replay, and a three-version lag is three policies old.
    parser.set_defaults(provider="openai", async_ratio=1)
    parser.add_argument("--rounds", type=int, default=3,
                        help="outer iterations t; each one deploys the policy online")
    parser.add_argument("--workers", type=int, default=4,
                        help="W -- parallel continuations, online and in replay")
    parser.add_argument("--online-rounds", type=int, default=6,
                        help="K1 -- decision rounds per online rollout")
    parser.add_argument("--replay-rounds", type=int, default=12,
                        help="K2 -- decision rounds per replay")
    parser.add_argument("--repeats", type=int, default=8,
                        help=("online rollouts per outer iteration. The paper "
                              "deploys once, which leaves H_1 a single tree; "
                              "evolve() refuses fewer than four tasks, and "
                              "dreaming on one world while the gate holds out "
                              "that same world is the fit-to-the-record failure "
                              "the whole design is trying to see. 1 follows the "
                              "paper and starts dreaming at iteration 4"))
    parser.add_argument("--dream-rounds", type=int, default=3,
                        help="meta_evolve rounds per dreaming phase")
    parser.add_argument("--family", choices=sorted(FAMILIES), default=SOURCE.name,
                        help="which world generator the loop runs on")
    parser.add_argument("--validate-family", choices=sorted(FAMILIES),
                        default=TARGET.name,
                        help=("the family the final deployment is scored on. "
                              "Different from --family by default: a policy that "
                              "only wins on the generator it was evolved against "
                              "is a fit, and this is the column that says so"))
    parser.add_argument("--validate-seeds", type=int, default=24,
                        help="fresh worlds per policy in the final report")
    parser.add_argument("--cost-weight", type=float, default=0.25,
                        help=("b1 * max_nodes -- what spending the whole "
                              "continuation budget costs, in units of score"))
    parser.add_argument("--parallel-weight", type=float, default=0.10,
                        help=("b2 * W -- what running at full parallelism earns. "
                              "Deliberately below --cost-weight: set them equal "
                              "and the two terms cancel at k = max_nodes / W, "
                              "which makes the objective blind to how many "
                              "continuations a full-length rollout spent"))
    parser.add_argument("--offline", action="store_true",
                        help=("script the reflector instead of calling a model. "
                              "Exercises the whole loop with no API key; it "
                              "demonstrates the mechanism, not a model's ability "
                              "to find the policy"))
    parser.add_argument("--thinking", choices=("disabled", "enabled", "default"),
                        default="default",
                        help=("send `thinking` in the request body. One policy "
                              "rewrite does not need a reasoning preamble, and "
                              "`disabled` saves minutes per call on Ark/Volcengine"))
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--api-timeout", type=float, default=180.0,
                        help="seconds per model call before it is retried")
    parser.add_argument("--out", default="", help="write the full result here as JSON")
    parser.add_argument("--pool-out", default="",
                        help=("write the simulator pool here as JSON. The worlds "
                              "are the expensive half of the method and they "
                              "serialise; a later run can dream in them without "
                              "paying for them again"))
    parser.add_argument("--completion-cache", default="",
                        help="memoise the reflector's prompt -> text here")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    workers = args.workers
    # `--serial` removes the *merge*, which here is the dreaming phase's, not
    # the discovery agent's parallelism -- see `run_loop`.
    dream_workers = worker_count(args, min(4, max(2, workers)))
    budget = workers * args.online_rounds
    plan = (f"dream-rsi: {args.rounds} outer iterations on {args.family}, "
            f"{args.repeats} online rollout(s) each at W={workers} x K1="
            f"{args.online_rounds} ({budget} agent calls per rollout); dream over "
            f"the pool ({args.dream_rounds} meta rounds, K2={args.replay_rounds}); "
            f"objective b1*N={args.cost_weight} b2*W={args.parallel_weight}; "
            f"validate on {args.validate_seeds} fresh {args.validate_family} worlds; "
            f"dream workers={dream_workers}; "
            + ("scripted reflector (--offline)" if args.offline else
               f"{args.provider}/{args.model} temperature={args.temperature} "
               f"thinking={args.thinking}"))
    print(plan)
    if args.dry_run:
        print("[dry-run] no model API was accessed; the discovery domain is "
              "synthetic. Seed policy:\n" + PARALLEL_REFINE_SEED)
        return 0
    if not args.offline and not confirm(args):
        return 0
    api = Usage()
    if args.offline:
        complete: Completion = offline_model()
    else:
        options: Dict[str, object] = {}
        if args.thinking != "default":
            options["thinking"] = {"type": args.thinking}
        complete = with_retries(
            completion_for(args, usage=api, max_tokens=args.max_tokens,
                           timeout=args.api_timeout, temperature=args.temperature,
                           retries=1, **options),
            attempts=4, backoff=3.0)
        if args.completion_cache:
            complete = cached_completion(
                complete, args.completion_cache,
                key_extra=f"{args.model}|{args.temperature}|{args.max_tokens}|{args.thinking}")
    started = time.monotonic()
    result = run_loop(complete, rounds=args.rounds, workers=workers,
                      online_rounds=args.online_rounds,
                      replay_rounds=args.replay_rounds, repeats=args.repeats,
                      dream_rounds=args.dream_rounds, dream_workers=dream_workers,
                      family_name=args.family, seed=args.seed,
                      cost_weight=args.cost_weight,
                      parallel_weight=args.parallel_weight,
                      max_seconds=args.max_seconds, on_round=progress(),
                      extra=budget_kwargs(args))
    print("[policy]\n" + result.rendered)
    if result.pool.worlds:
        print("[last world, by branch] (head, attempts, best) "
              + str(branch_report(result.pool.worlds[-1].tree)))
    fresh = range(1_000_000 + args.seed * 1_000,
                  1_000_000 + args.seed * 1_000 + args.validate_seeds)
    report = validate(result.rendered, family_name=args.validate_family,
                      workers=workers, online_rounds=args.online_rounds,
                      seeds=list(fresh), cost_weight=args.cost_weight,
                      parallel_weight=args.parallel_weight)
    print(f"\ndeployed on fresh {args.validate_family} worlds:")
    print(format_report(report))
    payload = {
        "plan": plan, "seed": args.seed, "offline": args.offline,
        "serial": args.serial, "dream_workers": dream_workers,
        "wall_seconds": time.monotonic() - started,
        "rendered": result.rendered, "rounds": [r.to_payload() for r in result.rounds],
        "worlds": len(result.pool), "validation": report,
        "objective": {"cost_weight": args.cost_weight,
                      "parallel_weight": args.parallel_weight,
                      "max_nodes": budget, "n_workers": workers},
        "usage": usage_dict(api),
    }
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1)
        print(f"[result saved] {args.out}")
    if args.pool_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.pool_out)) or ".",
                    exist_ok=True)
        with open(args.pool_out, "w", encoding="utf-8") as handle:
            handle.write(result.pool.to_json())
        print(f"[pool saved] {args.pool_out} ({len(result.pool)} worlds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
