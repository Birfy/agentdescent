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
import posixpath

from agentdescent import Policies, evolve
from agentdescent.evolution import EvolvingArtifact
from agentdescent.agents import Usage
from agentdescent.filetree import load_tree, match_any
from agentdescent.governance import SKILL_BLAST_RADIUS, classify
from agentdescent.sampling import DifficultyWeighted
from agentdescent.staleness import get_policy
from examples._common import (ConcurrencyGauge, add_standard_args, budget_kwargs,
                              completion_for, confirm, report_engine, worker_count)

from ._delegation import RecursiveDelegation
from . import _domain as minilang
from . import _fly as fly
from . import _jqx as jqx
from . import _md as md
from . import _stackvm as stackvm
from ._architect import ArchitectPhase, harness_record, missing_sections
from ._claude_code import ClaudeCodeExecutor, claude_code_available
from ._extract import ExtractPhase
from ._judge import ParentJudge
from ._review import CompletionJudge, ParentCodeReview, chain_reviews
from ._suite import TEST_FAILURE
from ._suite import cold_start, preflight
from ._octopus import OctopusConflict, git_available
from ._spatial import SpatialContract
from ._worktree import Rollout, WorktreeLedger, git_worktrees_available
from ._world import CONTEXT_FILE, WorldLog, parse_routing

#: The formation domains. Four, and each answers something the one before could
#: not: minilang shows the mechanism, stackvm gives the recursion somewhere to go,
#: jqx comes out as a program you can run -- its command-line entry point is
#: frozen beside the specification, so a finished run is software rather than a
#: package nobody can invoke -- and md takes the oracle out of the scoring path
#: altogether, because a frozen test suite is what a human actually writes and
#: what upstream validates against. All four are stand-ins for upstream's
#: 123.4-hour compiler run and say so. `fly` is the fifth and the first that grows a
#: *product* rather than a library -- a simulated animal, an HTTP backend and the page a
#: person watches it learn on -- and the first shipped with no reference implementation
#: at all, which is a cost `_fly` states rather than hides.
DOMAINS = {"minilang": minilang, "stackvm": stackvm, "jqx": jqx, "md": md,
           "fly": fly}

#: "No wall-clock budget", spelled as a number the runtime can add to `time.time()`.
#: `float("inf")` cannot be: the shutdown deadline is `t0 + max_seconds` and joining a
#: thread on it raises `OverflowError: timestamp out of range for platform time_t`.
#: A day is past any run this port has taken and is still a timestamp.
UNBOUNDED_SECONDS = 86_400.0

#: One catalogue line per domain, for the header. What the *agents* are briefed with
#: is `spec.OBJECTIVE` -- `objective_for` below -- and the two are deliberately not the
#: same string: see `_md.OBJECTIVE` for the run that proves why.
DOMAIN_BLURB = {
    "minilang": ("an integer expression language (2 nodes deep, 4 files), "
                 "staged suite"),
    "stackvm": ("a stack machine and its assembler (4 nodes deep, 10 files), "
                "staged suite"),
    "jqx": ("a JSON query tool with a frozen CLI (4 nodes, 9 files, 4 stages), "
            "staged suite"),
    "md": ("Lennard-Jones molecular dynamics with a frozen driver (6 nodes, "
           "14 files), test suite -- invariants, not values"),
    "fly": ("a Drosophila brain, three assays and a web page to watch it learn on, "
            "test suite -- a product, not a library"),
}


def objective_for(domain: str) -> str:
    """The brief handed to phase 1, the Context Extractor and the completion judge.

    A domain that has not written one falls back to its catalogue line, which is what
    every domain used to get and is enough to run -- just not enough to design against.
    """
    return getattr(DOMAINS[domain], "OBJECTIVE", DOMAIN_BLURB[domain])


def package_root(domain: str) -> str:
    """The directory the library itself lives in -- `src`, in every domain here.

    Read off the domain's own entry point rather than declared twice: `ENTRY` is
    `src/__init__.py`, and what the suite imports is the package around it.
    """
    entry = getattr(DOMAINS[domain], "ENTRY", "src/__init__.py")
    return posixpath.dirname(entry)


def build_tasks(domain: str = "minilang"):
    """The selected domain's loader -- the boundary ``--dry-run`` must not cross."""
    return DOMAINS[domain].build_tasks()


PORT_NAME = "EvoX Genesis (persistent recursive worlds)"
PORT_AUTHOR = "chendanyang"
UPSTREAM_RELEASED_CODE = "https://github.com/EMI-Group/genesis @ v0.12.6"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_standard_args(
        # No wall-clock budget unless one is asked for. It was 120 s back when
        # `--async` was an opt-in experiment and `max_seconds` was passed only in
        # that arm; barrier-free by default, that dormant number became the thing
        # that ended every run -- a 4 000-episode budget stopping after 12 rollouts.
        # `--episodes` is the budget; this is a stopwatch for when you want one.
        parser, model_default=None, max_seconds_default=0.0,
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
    parser.add_argument("--max-turns", type=int, default=24,
                        help="turns inside one --executor claude-code episode. "
                             "Upstream's runs allow 128 for a non-root episode")
    parser.add_argument("--cold-start", action="store_true",
                        help="start from the goal, the contract and the suite only: "
                             "no node CONTEXT.md records, no routing tables, no "
                             "skills. The run has to write its own decomposition, "
                             "which is the paper's first phase and the part a "
                             "pre-seeded tree hands it for free")
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
    parser.add_argument("--worktrees", action="store_true",
                        help="give every episode its own git worktree, commit its "
                             "work there, and remove the worktree after -- upstream's "
                             "scheduling model. The run then leaves a real "
                             "phylogenetic graph: one commit per episode, a parent's "
                             "merge carrying every child commit as a parent")
    parser.add_argument("--sync", action="store_true",
                        help="put the round barrier back. The default is barrier-free "
                             "with nothing discarded for lag, which is upstream's "
                             "arrangement; this restores the engine's synchronous DP, "
                             "which is what every published number on the page was "
                             "measured under")
    parser.add_argument("--signal-weighted", action="store_true",
                        help="sample tasks by how much signal each carries instead of "
                             "in order, down-weighting a test that always passes. "
                             "Off by default because it is UNMEASURED here: the "
                             "observation it answers is real (one run spent 2 003 "
                             "rollouts to buy 28 episodes) but the offline arm reaches "
                             "1.000 too fast to show the waste, so nothing has been "
                             "shown to improve")
    parser.add_argument("--staleness", choices=("guarded", "full", "reflective"),
                        default="reflective",
                        help="what happens to a proposal built on a version that has "
                             "since moved. `reflective` (default) replays it on the "
                             "current head and keeps it if it still improves -- never "
                             "discarded for being late, only for no longer helping, "
                             "which is the parent's own rule applied to a lagging "
                             "contribution. `guarded` (the engine's default) drops it "
                             "past a lag budget, which upstream never does; `full` "
                             "takes it as-is, which is not a merge and can revert "
                             "newer work. Only reachable under the barrier-free mode")
    parser.add_argument("--executor", choices=("completion", "claude-code"),
                        default="completion",
                        help="what one episode is. `completion` (default): one model "
                             "call returning whole files, which is this port's "
                             "largest distance from upstream -- an episode there is a "
                             "tool-using session of up to 2 048 turns. `claude-code`: "
                             "one headless Claude Code session per episode, in a "
                             "throwaway worktree, with the frozen paths denied and "
                             "the network tools off. It signs in with the LOCAL claude "
                             "CLI's credentials, not --model's endpoint, and it is "
                             "much slower and dearer per episode")
    parser.add_argument("--mode", choices=("given", "a", "b"), default="given",
                        help="how the Context Tree comes to exist, which is what "
                             "`Genesis.run` branches on. `given` (default): the "
                             "domain's hand-written records -- honest about being "
                             "upstream's phase 1 done by a person. `b`: an architect "
                             "designs the tree, then implementation grows it (a new "
                             "codebase). `a`: a context extractor reads an existing "
                             "repository and writes the tree over it, which is what "
                             "upstream does to a codebase that has code and no "
                             "records -- needs --continue-from")
    parser.add_argument("--continue-from", default="", metavar="DIR",
                        help="start from the repository in DIR rather than from the "
                             "domain's empty one. With --mode a this is the Context "
                             "Tree extraction upstream runs on an existing codebase; "
                             "the frozen contract is always the domain's own")
    parser.add_argument("--architect", action="store_true",
                        help="upstream's Phase 1: an agent designs the CONTEXT.md tree "
                             "-- intent, API surface, constraints, routing tables -- "
                             "before any code is written, and the implementation phase "
                             "then grows what it designed. Implies --cold-start, since "
                             "there is nothing to design if the tree is given. Needs "
                             "--model")
    parser.add_argument("--complete-task", action="store_true",
                        help="let the root agent end the run when it judges the "
                             "objective delivered, instead of exhausting the round "
                             "budget. Asked only once every test it can see passes")
    parser.add_argument("--no-parent-review", action="store_true",
                        help="skip the parent reading its child's diff (one model "
                             "call per returned child). The tests alone cannot see "
                             "a function that computes nothing")
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
    # Barrier-free by default, and nothing discarded for being late. Upstream has no
    # round barrier -- an agent commits, releases its worktree and is re-queued, and
    # the archive shows 22 episodes overlapping -- and it never drops a contribution
    # for lag: the parent merges what its child returns, whatever the child branched
    # from. `--sync` restores the barrier, which is what the published numbers used.
    # `--serial` is the one-worker arm of the published comparison, so it means the
    # barrier too: the shared contract refuses `--serial --async` outright, and it
    # is right to -- a one-worker asynchronous run is neither arm.
    args.asynchronous = not (args.sync or getattr(args, 'serial', False))
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
          f"{DOMAIN_BLURB[args.domain]}; "
          "NOT upstream's 123.4h C-compiler run")
    print(f"Plan     : model={args.model or 'offline rule-based actors'}, "
          f"episodes={args.episodes} root ({rounds} rounds x {args.workers} workers), "
          f"max depth={args.depth}")
    print("Budget   : " + f"{args.episodes} root episodes"
          + (f", or {args.max_seconds:.0f}s, whichever comes first"
             if args.max_seconds and args.asynchronous else "")
          + ("; the root agent may end it sooner" if args.complete_task and args.model
             else ""))
    print(f"Staleness: {args.staleness}"
          + (" (no effect: nothing is stale at a barrier)" if not args.asynchronous else
             "  -- a lagging proposal is replayed on the current head and kept if it "
             "still improves" if args.staleness == "reflective" else
             "  -- a lagging proposal is taken as-is, which is not a merge and can "
             "revert newer work" if args.staleness == "full" else
             "  -- a proposal that lagged past the budget is discarded, which upstream "
             "never does"))
    print(f"Merge    : {merge}" + ("" if git_available() else
                                   "  [git missing: every contested file falls back]"))
    print(f"Gate     : {gate}")
    print("Episode  : " + ("one headless Claude Code session in a throwaway worktree, "
                           f"up to {args.max_turns} turns, frozen paths denied, network "
                           "tools off -- billed to the local CLI's credentials"
                           if args.executor == "claude-code" else
                           "one model call returning whole files"))
    print("Workspace: " + ("a git worktree per episode, a commit per episode, and "
                           "the worktree removed after" if args.worktrees else
                           "in-memory states (--worktrees for upstream's model)"))
    print("Ends when: " + ("the root agent says the objective is delivered, or the "
                           "rounds run out" if args.complete_task and args.model else
                           "the rounds run out (--complete-task asks the root agent)"))
    print("Parent   : scope check"
          + ("" if args.no_parent_tests else " + integration test on each child's work")
          + ("" if args.no_parent_review or not args.model else
             " + reads the diff (upstream's code-quality rejection)")
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
    audited = [t for t in tasks if t.meta.get("audit")]
    print(f"Scoring  : {spec.SCORING}")
    if args.architect:
        args.mode = "b"                    # the older spelling of the same thing
    initial = (cold_start(spec.initial_files())
               if args.cold_start or args.mode == "b" else spec.initial_files())
    if args.continue_from:
        # An existing repository, with the domain's contract restored over it: the
        # suite and the specification are the human's in every mode.
        existing = {k: v for k, v in load_tree(args.continue_from).items()
                    if not match_any(k, spec.FROZEN)}
        initial = dict(existing, **{k: v for k, v in spec.initial_files().items()
                                    if match_any(k, spec.FROZEN)})
        if args.mode == "a":
            initial = {k: v for k, v in initial.items()
                       if not k.endswith(CONTEXT_FILE) or k.startswith("spec/")}
    if args.mode == "a" and not args.continue_from:
        print("--mode a is upstream's existing-codebase path: it reads a repository "
              "and writes the Context Tree over it, so it needs --continue-from DIR. "
              "A formation domain starts empty and there is nothing to read.")
        return
    print(f"Loaded   : {len(tasks)} {spec.CASE_NOUN} over "
          f"{len(set(t.meta['kind'] for t in tasks))} {spec.GROUP_NOUN}; "
          f"{len(initial)} files in the repository, "
          "none of them implementation")
    print("Start    : " + (f"{len(initial)} files from {args.continue_from}, its own "
                           "records stripped -- phase A writes the tree over it"
                           if args.mode == "a" else
                           f"{len(initial)} files from {args.continue_from}, contract "
                           "restored" if args.continue_from else "")
          + ("one generated record mapping the repository root to the package it "
             "holds; everything below it designed in phase 1 by an architect agent, "
             "from the goal, the contract and the suite" if args.mode == "b" else
                           "cold -- the goal, the contract and the suite; no node "
                           "records, no routing tables, no skills. The run writes "
                           "its own decomposition."
                           if args.cold_start else
                           f"{sum(1 for p in initial if p.endswith(CONTEXT_FILE))} "
                           "CONTEXT.md records with routing tables, human-written: "
                           "the decomposition is given, not grown (--cold-start "
                           "takes it away)"))
    # Two different things live in the held-out tail depending on the domain, and
    # conflating them is how the jqx run stalled: sampled inputs the search has not
    # seen (a generalisation estimate), or requirements the search is not allowed to
    # see (a bug). md holds out neither -- its tail is an audit set that is not in
    # the repository at all, so every requirement drives the search.
    print(f"Held out : {len(tasks) - int(round(len(tasks) * (1 - spec.HELD_OUT_FRAC)))}"
          f" of {len(tasks)}"
          + (f" -- an audit set the agents cannot read: {len(audited)} tests "
             f"injected only at scoring time" if audited else
             " -- the tail of the case list"))

    usage = Usage()
    #: How many model calls were in flight at once. The engine's barrier-free
    #: concurrency *is* `n_workers` and its merge side runs at `max_concurrency=1`,
    #: so this should come back equal to `--workers`; it is reported because
    #: `usage.seconds / wallclock` looks like it answers the same question and does
    #: not -- `seconds` spans phases that run before `evolve()` does.
    args._concurrency = ConcurrencyGauge()
    complete = None
    if args.model:
        if not confirm(args):
            return
        complete = completion_for(args, usage=usage)
        # One call before the run: a wrong endpoint is otherwise 400 episodes of
        # silent nothing, which reads exactly like a broken mechanism.
        preflight(complete)

    # Upstream's Phase 1, and it is a separate root agent there for a reason: the tree
    # is designed before anything is written against it, and Phase 2 is handed
    # "the architecture, directory structure, CONTEXT.md routing tables ... already in
    # place (created by an Architect agent)".
    if getattr(spec, "REQUIRES_MODEL", False) and complete is None:
        print(f"--domain {args.domain} needs --model: it ships no reference "
              "implementation, so there is no offline rule-based actor to fall back "
              "on -- and one written for it would be the design the run is meant to "
              "invent")
        return

    if args.mode == "a":
        if complete is None:
            print("--mode a needs --model: extracting a Context Tree is reading code, "
                  "and a rule-based reader would be describing what it was told")
            return
        extractor = ExtractPhase(complete, contracts=spec.CONTRACTS,
                                 max_depth=args.depth + 1,
                                 skip=tuple(p.split("/")[0] for p in spec.FROZEN
                                            if "/" in p))
        initial = extractor.extract(initial, objective_for(args.domain))
        print(f"\nPhase A  : context extractor {extractor.summary()}")
        for path in extractor.nodes:
            record = initial[f"{path}/{CONTEXT_FILE}" if path else CONTEXT_FILE]
            print(f"           {path or './':<24} {len(parse_routing(record))} routes")

    architect = None
    if args.mode == "b":
        if complete is None:
            print("--mode b needs --model: there is no offline architect, and a "
                  "rule-based one would be the decomposition it is meant to invent")
            return
        # Phase 1 designs the *codebase*, and the repository around it is the
        # harness: see `_architect.HARNESS_RECORD` for the three-out-of-three run
        # that established the difference is not cosmetic.
        root = package_root(args.domain)
        initial[CONTEXT_FILE] = harness_record(root, spec.FROZEN,
                                               objective_for(args.domain))
        architect = ArchitectPhase(complete, contracts=spec.CONTRACTS,
                                   max_depth=args.depth, root_path=root)
        initial = architect.design(initial, objective_for(args.domain))
        print(f"\nPhase 1  : architect {architect.summary()}")
        for path in architect.nodes:
            record = initial[f"{path}/{CONTEXT_FILE}" if path else CONTEXT_FILE]
            gaps = missing_sections(record)
            print(f"           {path or './':<24} {len(parse_routing(record))} routes"
                  + (f", missing {', '.join(gaps)}" if gaps else ""))

    log = WorldLog()
    strategy = SpatialContract(initial_files=initial, frozen=spec.FROZEN,
                               log=log, max_files_per_diff=6)
    # `manager.ex` states the parent's validation as three things: review the child's
    # results, run the tests, and reject anti-patterns it can see in the code. The
    # middle one is a number and was all this port had; the zero-field run is what
    # that cost. Both now, tests first because they are free.
    code_review = (None if args.no_parent_review or complete is None else
                   ParentCodeReview(complete, contracts=spec.CONTRACTS))
    ledger = WorktreeLedger() if args.worktrees else None
    if ledger is not None and not git_worktrees_available():
        print("Worktrees: git worktree is unavailable here -- running without it")
        ledger = None
    sessions = None
    if args.executor == "claude-code":
        if not claude_code_available():
            print("--executor claude-code needs the `claude` CLI on PATH")
            return
        # The frozen globs are denied inside the session as well as enforced outside
        # it, and the failure block is the domain's own wording.
        # The failure template is the domain's, not a constant: a blind domain must
        # not hand the session the assertion's source, which is the whole point of it.
        # And the session runs the model the run asked for -- left unset it takes the
        # CLI's default, which on a `--provider claude-cli` run means the architect is
        # on one model and every executor episode on another.
        sessions = ClaudeCodeExecutor(frozen=spec.FROZEN,
                                      failure=getattr(spec, "FAILURE", TEST_FAILURE),
                                      model=(args.model if args.provider == "claude-cli"
                                             else None),
                                      max_turns=args.max_turns)
    delegation = RecursiveDelegation(
        rollout_factory=(None if ledger is None else lambda: Rollout(ledger)),
        manager=spec.llm_manager(complete) if complete else spec.offline_manager,
        executor=(sessions if sessions is not None else
                  spec.llm_executor(complete) if complete else spec.offline_executor),
        log=log, max_depth=args.depth, max_edits=4, contracts=spec.CONTRACTS,
        readonly=spec.FROZEN,
        # Three links, and the middle one is the only one that can ask "did you test
        # what you just wrote". A domain whose tests are the agents' own supplies it.
        review=chain_reviews(None if args.no_parent_tests else spec.suite_review(tasks),
                             (getattr(spec, "own_review", lambda: None)()
                              if not args.no_parent_tests else None),
                             code_review),
        accountability=not args.no_accountability)
    judge = ParentJudge(log=log, enabled=not args.engine_gate)
    octopus = None if args.keyed_union else OctopusConflict()

    run = spec.make_runner()

    # Upstream nothing watches a number: an agent decides its objective is met and
    # calls `complete_task`. Asked only when every test the agents can see passes,
    # which is upstream's own precondition, and never shown the held-out reward --
    # that is this port's measurement, computed from tests no agent may read.
    completion = None
    if args.complete_task and complete is not None:
        completion = CompletionJudge(
            complete, tasks=tasks, run=run, reward=spec.reward,
            # The strategy's window, not the proposal policy's: the engine renders the
            # accepted artifact every round, while a proposal policy sees a state only
            # when it is asked for a proposal -- and it stops being asked exactly when
            # the run is finished. Scoring `delegation.last_state` meant scoring a
            # stale tree forever, which is why the first run with --complete-task
            # asked nobody anything.
            state_of=lambda: (strategy.last_rendered if strategy.last_rendered
                              is not None else delegation.last_state),
            contracts=spec.CONTRACTS,
            suite_failures=getattr(spec, "suite_failures", None),
            objective=objective_for(args.domain))

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
        # Say the budget outright rather than letting `rounds` be reinterpreted:
        # `--episodes` is a count of ROOT episodes in both arms, and under the
        # barrier-free default that is exactly what a worker rollout is.
        **({"max_rollouts": args.episodes} if args.asynchronous
           else {"rounds": rounds}),
        n_workers=args.workers,
        max_concurrency=1 if args.asynchronous else args.workers,
        asynchronous=args.asynchronous, async_ratio=args.async_ratio,
        # The barrier-free runtime treats the wall clock as a required bound and
        # defaults it to twenty seconds, so `None` is not "unbounded" there -- it is
        # twenty seconds, and a 4 000-episode run ended after four rollouts twice
        # before that was clear. `--episodes` is the budget; this is infinite unless
        # a stopwatch was asked for.
        max_seconds=((args.max_seconds or UNBOUNDED_SECONDS) if args.asynchronous
                     else None),
        # Upstream runs no local before/after re-check: a child returns its work
        # and the parent judges it. Re-running every proposal here would also
        # double the number of child processes the domain spawns.
        self_verify=False,
        stop_when=completion,
        held_out_frac=spec.HELD_OUT_FRAC,
        eval_concurrency=args.eval_concurrency or 8,
        seed=args.seed, usage=usage,
        policies=Policies(proposal=delegation, acceptance=judge,
                          staleness=get_policy(args.staleness),
                          # Round-robin "spends rollouts uniformly, including on tasks
                          # the agent already solves" -- its own words -- and that is
                          # where a late run's budget goes: one model run spent 2 003
                          # rollouts to buy 28 episodes. Upstream's manager works on
                          # failures and never on a test that passes, so the weighted
                          # sampler is the more faithful one -- and it is off by
                          # default because no measurement here shows it helping. The
                          # offline arm reaches 1.000 before the waste can appear.
                          **({"task_sampler": DifficultyWeighted()}
                             if args.signal_weighted else {}),
                          **({} if octopus is None else {"conflict": octopus})),
        **budget_kwargs(args),
    )

    # The whole suite in one process, which is what `mix test` is. A per-task score
    # cannot see one test poisoning the next: this run's own output is the example,
    # 65/65 per test and five failures in one interpreter, from a function that
    # rebound its own name on first call.
    whole = getattr(spec, "suite_failures", None)
    if whole is not None and delegation.last_state is not None:
        state = result.state if isinstance(result.state, dict) else dict(result.state)
        failures = whole(state, audit=True)
        if getattr(getattr(spec, "FLY", None), "blind", False) or getattr(
                spec, "REQUIRES_MODEL", False) and not hasattr(spec, "reference_tree"):
            # A blind domain's repository suite is the agents' own, so there is no
            # denominator from `tasks` to report it against -- count what is there.
            written = sum(1 for p in state if p.endswith(".py")
                          and p.rsplit("/", 1)[-1].startswith("test_"))
            print(f"their own suite : {written} test file(s), "
                  + ("all passing" if not failures else f"{len(failures)} failing:\n    "
                     + "\n    ".join(failures[:6])))
        else:
            print(f"suite, 1 process: {len(tasks) - len(failures)}/{len(tasks)}"
                  + ("" if not failures else
                     "   <- these pass one-per-process and fail together:\n    "
                     + "\n    ".join(failures[:6])))
    label = "audit reward" if audited else "held-out reward"
    print(f"{label:<16}: {result.final_reward:.3f}"
          + ("   (tests no agent ever saw)" if audited else ""))
    print(f"outcomes        : {result.outcomes()}")
    print(f"stop reason     : {result.stop_reason}"
          + ("" if completion is None else
             f"  (root agent asked {completion.asked}x, last said "
             f"{completion.verdict or 'nothing'}"
             + (f": {completion.reason}" if completion.reason else "") + ")"))
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
          f"{delegation.accountability_declined}  "
          f"context_updates={delegation.record_updates}")
    if ledger is not None:
        print(f"workspace       : {ledger.summary()}")
    if sessions is not None:
        print(f"claude code     : {sessions.summary()}")
    if code_review is not None:
        print(f"parent review   : read={code_review.reviewed} "
              f"rejected={code_review.rejected} unparsed={code_review.unparsed}")
    if octopus is not None:
        print(f"merge           : merged={octopus.merged} conflicted={octopus.conflicted}")
    if not args.engine_gate:
        print(f"gate            : accepted={judge.accepted} rejected={judge.rejected} "
              f"partial={judge.partial}")
    if log.pending_rework:
        print(f"open rework     : {sorted(log.pending_rework)}")
    print(f"model usage     : {usage.summary()}"
          + (f", peak {args._concurrency.peak} concurrent"
             if args._concurrency.peak else ""))
    report_engine(result)

    if args.write_repo:
        plan = result.write_to(args.write_repo)
        print(f"\nwrote {len(plan.get('written', []))} files to {args.write_repo}")


if __name__ == "__main__":
    main()
