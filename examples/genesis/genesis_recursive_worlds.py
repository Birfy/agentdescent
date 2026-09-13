"""EvoX Genesis -- persistent recursive worlds, on the AgentDescent engine.

Upstream reference
------------------
Paper: *Persistent Recursive Worlds Enable Autonomous Software Evolution*,
Beichen Huang, Zhenyu Liang, Bowen Zheng, Ran Cheng (arXiv:2608.10450v3).
Repository: https://github.com/EMI-Group/genesis (v0.12.6, AGPL-3.0, Elixir).

Preserved from that revision:

* ``w = (v, p)`` -- an agent is situated by an accepted version and a
  repository-relative path (:mod:`examples.genesis._world`).
* ``(v, p) ⇝ (v, q)`` -- recursive delegation moves the path and **not** the
  version; only an accepted event advances the project
  (:mod:`examples.genesis._delegation`).
* The spatial contract: an agent writes only inside its own subtree
  (:mod:`examples.genesis._spatial`).
* The parent merges what its children return with a three-way merge, so two
  children editing different parts of one file both survive
  (:mod:`examples.genesis._octopus`, upstream ``Git.merge_octopus/2``).
* The responsible parent's verdict -- accept / reject / ask again -- with
  upstream's monotone rule, *partial progress is accepted*
  (:mod:`examples.genesis._judge`, upstream ``agents/manager.ex:58``).
* ``CONTEXT.md`` is part of the accepted version, in both directions: the chain
  from the root down is what an entering agent is given, its **routing table is
  where a manager may delegate**, a manager that opens an unrouted node records
  it at its own level, and a refusal is written there even though the refused
  code is not (paper, appendix 1.4; upstream ``core/context_node.ex`` and the
  routing-table clauses of ``agents/prompt_fragments.ex``).
* Human-supplied validation stays human-supplied: ``spec/**`` is frozen to every
  proposal and restored pristine before scoring.

Intentional differences
-----------------------
* **The benchmark is not reproduced, and is not claimed.** Upstream's formation
  run is 123.4 h, US$44.38 and one sample; its continuation and MESA runs are one
  sample each. This port runs a compact formation domain that finishes offline
  (:mod:`examples.genesis._domain`) and reports only what that measures.
* **Acceptance is upstream's, and the engine's is the control arm.** The port
  runs the parent's monotone rule because ``docs/port-fidelity.md`` puts the
  acceptance rule in the *must not change* column; ``--engine-gate`` swaps in the
  shipped Beta posterior so the two can be compared. On this domain they are not
  interchangeable, and the page says why: a formation run's first steps are
  structure, structure moves no case, and a gate that requires a measured
  improvement never lets the run start.
* **"Request more work" is reconstructed, not native.** ``AcceptDecision`` is a
  boolean whose ``category`` is a closed vocabulary, so a parent's *not yet*
  becomes an accepted partial step plus a request the next round's manager
  re-delegates from.
* **The world is L2, not L1.** The audit gate runs *before* acceptance and, above
  ``FAST_MAX``, vetoes every candidate that does not strictly improve -- which
  contradicts "partial progress is accepted". The reason the artifact can honestly
  sit at L2 is that the agents cannot reach the evaluator at all: the suite, its
  expectations and the harness that runs them live outside the artifact, and
  ``spec/**`` is refused to every proposal and restored pristine before scoring.
* **Tensor parallelism is refused rather than used.** TP's ownership map is fixed
  before round 0 and a key outside it is a violation, so every newly created file
  would be discarded -- and creating files is what formation *is*. The contract is
  enforced in the strategy instead, where creation is free.
* **Depth is 2-3 here, 4-8 upstream**, because the domain is three nodes deep.
  The number reported is the depth *observed*, never the depth configured.
* **Only the routing table is maintained automatically.** Upstream every
  ``:read_write`` agent keeps its node's ``CONTEXT.md`` current -- intent, API
  surface, known issues -- and the read-only roles exist to do nothing else. Here
  the two writes an agent makes on its own are the routing entry for a node it
  opened and the refusal note for a child it rejected; an LLM executor may write
  more, and nothing requires it to. Skills (``.agents/skills/``) are not loaded
  at all.
* **Multi-repository work, the desktop shell and the dashboard are out of scope.**
  Upstream's task-level accept is a human action on that dashboard
  (``EvoGit.Review.merge_branch/2,3``); here the task-level gate is the
  aggregator, and the parent's judging is the part that runs inside an episode.
"""

from __future__ import annotations

import argparse

from agentdescent import Policies, evolve
from agentdescent.evolution import EvolvingArtifact
from agentdescent.agents import Usage
from agentdescent.governance import SKILL_BLAST_RADIUS, classify
from examples._common import (add_standard_args, budget_kwargs, completion_for,
                              confirm, report_engine, worker_count)

from ._delegation import RecursiveDelegation
from . import _domain as minilang
from . import _jqx as jqx
from . import _md as md
from . import _stackvm as stackvm
from ._judge import ParentJudge
from ._octopus import OctopusConflict, git_available
from ._spatial import SpatialContract
from ._world import WorldLog

#: The formation domains. Three, and each answers something the one before could
#: not: minilang shows the mechanism, stackvm gives the recursion somewhere to go,
#: jqx comes out as a program you can run -- its command-line entry point is
#: frozen beside the specification, so a finished run is software rather than a
#: package nobody can invoke -- and md cannot be passed by code that merely
#: parses, because four of its ten stages are invariants rather than values. All
#: four are stand-ins for upstream's 123.4-hour compiler run and say so.
DOMAINS = {"minilang": minilang, "stackvm": stackvm, "jqx": jqx, "md": md}

DOMAIN_BLURB = {
    "minilang": "an integer expression language (2 nodes deep, 4 files)",
    "stackvm": "a stack machine and its assembler (4 nodes deep, 10 files)",
    "jqx": "a JSON query tool with a frozen CLI (4 nodes, 9 files, 4 stages)",
    "md": ("Lennard-Jones molecular dynamics (6 nodes, 15 files, 10 stages, "
           "4 of them invariants)"),
}


def build_tasks(domain: str = "minilang"):
    """The selected domain's loader -- the boundary ``--dry-run`` must not cross."""
    return DOMAINS[domain].build_tasks()


PORT_NAME = "EvoX Genesis (persistent recursive worlds)"
PORT_AUTHOR = "chendanyang"
UPSTREAM_RELEASED_CODE = "https://github.com/EMI-Group/genesis @ v0.12.6"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_standard_args(
        parser, model_default=None, max_seconds_default=120.0,
        model_help=("optional: let a model be the manager and executor agents "
                    "(else rule-based offline actors -- see the selected "
                    "--domain module)"))
    # Upstream's unit is the agent episode ("1,019 archived agent episodes"), not
    # a round or a generation, so that is the flag -- and because a root episode
    # is one rollout it is already the rollout budget, which is why `--workers`
    # divides it rather than multiplying it.
    parser.add_argument("--domain", default="minilang", choices=sorted(DOMAINS),
                        help=("which software world to grow: minilang is two "
                              "nodes deep, stackvm four"))
    parser.add_argument("--episodes", type=int, default=24,
                        help="total ROOT episodes; one root episode is one rollout")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--depth", type=int, default=3,
                        help="maximum recursive delegation depth (v,p) -> (v,q)")
    parser.add_argument("--engine-gate", action="store_true",
                        help=("the control arm: accept on the engine's Beta "
                              "posterior instead of the parent's monotone rule. "
                              "Changes the acceptance rule; see the module docstring"))
    parser.add_argument("--keyed-union", action="store_true",
                        help=("the control arm: drop the three-way merge, so two "
                              "agents editing one file contradict and one is "
                              "dropped on held-out score (the engine's default)"))
    parser.add_argument("--no-accountability", action="store_true",
                        help=("make a manager a pure router: skip upstream's "
                              "review-and-accountability turn at its own node "
                              "after its children return"))
    parser.add_argument("--no-parent-tests", action="store_true",
                        help=("stop the parent running the suite on a child's "
                              "work before accepting it; leaves only the scope "
                              "check, which is the cheap half of the rule"))
    parser.add_argument("--write-repo", default="", metavar="DIR",
                        help=("write the grown repository here. The ledger is "
                              "scratch and is reaped on exit, so without this the "
                              "thing the run produced is not kept"))
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    # One root episode is one rollout, so the shared budget flag maps onto the
    # port's own iteration flag rather than adding a second budget beside it.
    if getattr(args, "budget_rollouts", None):
        args.episodes = args.budget_rollouts
    args.workers = worker_count(args, args.workers)
    rounds = max(1, args.episodes // max(1, args.workers))

    merge = "keyed union (control)" if args.keyed_union else "git three-way (octopus)"
    gate = ("engine Beta gate (control)" if args.engine_gate
            else "parent (partial progress accepted)")
    print(f"Algorithm: {PORT_NAME} (port author: {PORT_AUTHOR})")
    print(f"Upstream : {UPSTREAM_RELEASED_CODE}")
    print(f"Dataset  : compact formation domain -- {args.domain}: "
          f"{DOMAIN_BLURB[args.domain]}, staged suite; "
          "NOT upstream's 123.4h C-compiler run")
    print(f"Plan     : model={args.model or 'offline rule-based actors'}, "
          f"episodes={args.episodes} root ({rounds} rounds x {args.workers} workers), "
          f"max depth={args.depth}")
    print(f"Merge    : {merge}" + ("" if git_available() else
                                   "  [git missing: every contested file falls back]"))
    print(f"Gate     : {gate}")
    print("Parent   : scope check"
          + ("" if args.no_parent_tests else " + integration test on each child's work")
          + ("; manager routes only" if args.no_accountability
             else "; manager reviews and writes its own node last"))

    if args.dry_run:
        print("Data     : deferred (dry-run performs no network access)")
        print("\n[dry-run] plan only; no dataset or model API was accessed.")
        return

    spec = DOMAINS[args.domain]
    tasks = build_tasks(args.domain)
    artifact = EvolvingArtifact("world", blast_radius=SKILL_BLAST_RADIUS)
    print(f"Governance: world blast_radius={artifact.blast_radius} -> "
          f"{classify(artifact).name} (the agents cannot reach the evaluator: the "
          f"suite lives outside the artifact); frozen to every proposal: "
          f"{', '.join(spec.FROZEN)}")
    print(f"Loaded   : {len(tasks)} validation cases over "
          f"{len(set(t.meta['kind'] for t in tasks))} stages; "
          f"{len(spec.initial_files())} files in the repository, "
          "none of them implementation")

    usage = Usage()
    complete = None
    if args.model:
        if not confirm(args):
            return
        complete = completion_for(args, usage=usage)

    log = WorldLog()
    strategy = SpatialContract(initial_files=spec.initial_files(), frozen=spec.FROZEN,
                               log=log, max_files_per_diff=6)
    delegation = RecursiveDelegation(
        manager=spec.llm_manager(complete) if complete else spec.offline_manager,
        executor=(spec.llm_executor(complete) if complete else spec.offline_executor),
        log=log, max_depth=args.depth, max_edits=4, contracts=spec.FROZEN,
        # The parent's own test run, inside the episode, on one child's work --
        # the half of the upstream rule the acceptance gate cannot see.
        review=None if args.no_parent_tests else spec.suite_review(tasks),
        accountability=not args.no_accountability)
    judge = ParentJudge(log=log, enabled=not args.engine_gate)
    octopus = None if args.keyed_union else OctopusConflict()

    run = spec.make_runner()

    def _superseded(rendered, task, output, reward_):
        # `evolve()` requires a `propose` before it installs the bundle's
        # proposal policy over it. Raising rather than returning None so that a
        # bundle that failed to install is a loud failure, not a silent round of
        # nothing.
        raise AssertionError("Policies(proposal=RecursiveDelegation) was not installed")

    print(f"\nGrowing the world ({args.workers} workers, "
          f"{'barrier-free' if args.asynchronous else 'synchronous DP'})...\n")
    result = evolve(
        tasks, spec.reward, run=run, propose=_superseded, strategy=strategy,
        artifact_id="world", blast_radius=SKILL_BLAST_RADIUS,
        rounds=rounds, n_workers=args.workers,
        max_concurrency=1 if args.asynchronous else args.workers,
        asynchronous=args.asynchronous, async_ratio=args.async_ratio,
        max_seconds=args.max_seconds if args.asynchronous else None,
        # Upstream runs no local before/after re-check: a child returns its work
        # and the parent judges it. Re-running every proposal here would also
        # double the number of child processes the domain spawns.
        self_verify=False,
        held_out_frac=0.4, eval_concurrency=args.eval_concurrency or 8,
        seed=args.seed, usage=usage,
        policies=Policies(proposal=delegation, acceptance=judge,
                          **({} if octopus is None else {"conflict": octopus})),
        **budget_kwargs(args),
    )

    print(f"held-out reward : {result.final_reward:.3f}")
    print(f"outcomes        : {result.outcomes()}")
    print(f"stop reason     : {result.stop_reason}")
    print(f"error           : {result.error or 'none'}")
    print(f"world           : root episodes={result.rollouts}  {log.summary()}  "
          f"truncated_edits={delegation.truncated}  "
          f"routes_opened={delegation.routes_opened}  "
          f"mistaken_nodes={delegation.mistaken_nodes}")
    print(f"parent          : sibling_merges={delegation.sibling_merges}  "
          f"sibling_conflicts={delegation.sibling_conflicts}  "
          f"requests raised/handled/unmet={delegation.requests_raised}/"
          f"{delegation.adopted_requests}/{delegation.unmet_requests}  "
          f"node_relative_paths={delegation.resolved_relative}  "
          f"accountability={delegation.accountability_edits}/"
          f"{delegation.accountability_declined}")
    if octopus is not None:
        print(f"merge           : merged={octopus.merged} conflicted={octopus.conflicted}")
    if not args.engine_gate:
        print(f"gate            : accepted={judge.accepted} rejected={judge.rejected} "
              f"partial={judge.partial}")
    if log.pending_rework:
        print(f"open rework     : {sorted(log.pending_rework)}")
    print(f"model usage     : {usage.summary()}")
    report_engine(result)

    if args.write_repo:
        plan = result.write_to(args.write_repo)
        print(f"\nwrote {len(plan.get('written', []))} files to {args.write_repo}")


if __name__ == "__main__":
    main()
