"""ERA tree search on SWE-bench Science -- one tree per task, expanded by an agent.

    "SWE-bench Science evaluates coding agents on software-engineering tasks
    drawn from scientific-computing repositories. The release contains 119
    tasks across 20 scientific domains, with isolated environments and separate
    programmatic verifiers."
    -- OpenMOSS-Team/SWE-bench-Science

What the search optimises
-------------------------
A candidate is a **patch to a repository** -- the release's own
``model.patch``, ``git diff --cached --binary <baseline>``, the artifact its
`pre_artifacts.sh` hands to the verifier. The root of every tree is the empty
patch, so a tree starts at exactly what the unmodified baseline scores and
every gain is measured against the repository as published.

Correctness is not a matter of degree at the finish line: the benchmark's reward
is 1 only when the public reproduction exits 0 **and** every private test
passes. That is what this port reports. It is not what the tree ranks on -- a
binary reward is flat almost everywhere and PUCT's exploitation term would rank
a patch that fixes four checks of five level with one that deleted the module --
so the tree ranks on the **pass rate over the checks it is allowed to see**,
with a fraction of the private suite held back and never shown to it. Same
choice, same reason, as ranking LLM-SRBench on ``-log10(NMSE)`` instead of
Acc(0.1).

The mutation is an agent, not a reply
-------------------------------------
Every other ERA task in this repository rewrites one file per expansion, and one
model call can do that. These tasks are repository-scale, so an expansion here
is a **coding-agent session**: Claude Code (or Codex, or any command-line agent)
run inside a git checkout of the task with the parent node's patch already
applied, able to read the whole source tree and to *run* it -- ``run-in-env``
executes inside the task's own offline container, where its dependencies are
installed. Whatever the agent leaves in the checkout is diffed and becomes the
node. ``--agent completion`` is the control arm: one model call, no tools, asked
for a unified diff.

One tree per task
-----------------
Each benchmark task gets its own flat-PUCT tree, its own root, its own held-back
tests and its own result. They are separate searches over separate program
spaces -- a fix found for a plasma-stability solver is not a node in a genomics
task's tree and could not be selected there.

Everything about the search itself is `era_empirical_software.py`: the flat-PUCT
tree, the visit reservation, the staleness handling, the aggregator, the
governance layer. This module supplies a
:class:`~examples.era._era_domain.Domain` per task and the command line the
other ERA ports share.

The protocol is ERA's, not the benchmark's
------------------------------------------
On the leaderboard an agent gets one attempt per task and no verifier feedback.
Here a tree of agent sessions is scored between attempts against part of the
private suite, and the best node is kept. Same tasks, same pinned images, same
grader, **different experiment** -- so a number from here is a number about this
search, not a leaderboard row. It is said here, in the result file, in
`_era_swe_science.py`, and in `docs/algo-era.md`.

Requirements
------------
A Docker daemon (the benchmark is distributed as 238 pinned ``linux/amd64``
images) and a command-line coding agent on PATH. Roughly 1.6 GB of images per
task.

Run
---
    python -m examples.era.era_swe_science --dry-run
    python -m examples.era.era_swe_science --tasks 001 --iterations 4 --workers 2 --yes
    python -m examples.era.era_swe_science --tasks unrestricted --iterations 6 --yes
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from agentdescent.agents import Usage, cli_agent
from agentdescent.evolution import EvolvingArtifact
from agentdescent.governance import classify

from examples._common import (
    add_standard_args,
    completion_for,
    confirm,
    worker_count,
)
from examples.era._era_domain import Domain
from examples.era._era_support import UPSTREAM_COMMIT
from examples.era._era_swe_science import (
    DEFAULT_TASKS,
    FEEDBACK_LEVELS,
    RELEASE_AGENT_TIMEOUT,
    RELEASE_GITHUB,
    RELEASE_REPO,
    RELEASE_REVISION,
    RELEASE_TASKS,
    RELEASE_VERIFIER_TIMEOUT,
    Suite,
    agent_version,
    docker_backend,
    envelope,
    evaluate_patch,
    extract_patch,
    framework_score,
    grade,
    load_manifest,
    make_agent_mutation,
    make_completion_mutation,
    parse_selection,
    prepare_suite,
)
from examples.era.era_empirical_software import (
    _usage_dict,
    _utc_now,
    _write_json,
    run_agentdescent_era,
)


ARTIFACT_ID = "era_program"
DEFAULT_OUTPUT = Path("era-swe-science-result.json")

#: The agents this port knows how to launch. `command` takes whatever
#: `--agent-command` names, so an agent that is not listed here is one flag
#: away rather than a fork.
AGENTS = ("claude-code", "codex", "command", "completion")

#: Claude Code in print mode, with the tools an edit-and-run loop needs and
#: nothing that could reach outside the checkout. The prompt goes over stdin
#: because `--allowedTools` is variadic and would swallow a positional one.
CLAUDE_CODE = ("claude", "-p", "--permission-mode", "acceptEdits",
               "--allowedTools", "Bash Edit Write Read Glob Grep TodoWrite")
CODEX = ("codex", "exec", "--full-auto")

EMPTY_WARNING = (
    "For this task that means an agent session that left the checkout "
    "unchanged -- it timed out, refused, or could not find the defect. The "
    "node is still appended, scoring what its parent's patch scores.")


def agent_command(args: argparse.Namespace) -> Tuple[str, ...]:
    """The argv prefix for one agent session; the prompt arrives on stdin."""
    if args.agent == "command":
        if not args.agent_command:
            raise SystemExit("--agent command needs --agent-command '<argv>'")
        return tuple(shlex.split(args.agent_command))
    base = CLAUDE_CODE if args.agent == "claude-code" else CODEX
    return base + (("--model", args.model) if args.model else ())


def agent_environment(args: argparse.Namespace) -> Dict[str, str]:
    """Environment overrides for one agent session, beyond the workspace's PATH.

    ``--thinking disabled`` sets ``MAX_THINKING_TOKENS=0``, which is the CLI's
    own budget knob rather than anything this port invents. It matters twice
    over: extended thinking is most of a session's wall-clock, so turning it off
    is what makes a *deep* tree affordable -- and the variable is inherited from
    whatever shell launched the run, so a result file that did not record it
    could not be repeated. `agent_sessions.thinking_tokens` carries the value
    that was in force.
    """
    if args.thinking == "disabled":
        return {"MAX_THINKING_TOKENS": "0"}
    if args.thinking == "enabled" and not os.getenv("MAX_THINKING_TOKENS"):
        return {"MAX_THINKING_TOKENS": "31999"}
    return {}


def build_launch(args: argparse.Namespace, usage: Usage):
    """`launch(workspace, env)` -> the callable that runs one agent session."""
    command = agent_command(args)
    overrides = agent_environment(args)

    def launch(workspace: str, env: Dict[str, str]):
        return cli_agent(command, workspace=workspace, env={**env, **overrides},
                         via_stdin=True, timeout=args.agent_timeout, usage=usage)

    return launch


def swe_science_domain(suite: Suite, *, verifier_timeout: float,
                       max_patch_bytes: int, agent_timeout: float,
                       ask_promise: bool, public: bool, feedback: str,
                       cache: Optional[Dict[str, Any]] = None) -> Domain:
    """One SWE-bench Science task, in the four terms the ERA search needs."""
    return Domain(
        name=(f"SWE-bench Science task {suite.task_id} ({suite.domain}) -- "
              f"{suite.title}"),
        entrypoint="model.patch",
        metric_key="pass_rate",
        metric_better="higher",
        initial_program="",
        initial_summary="the unmodified baseline repository (empty patch)",
        evaluate=lambda patch, shard_ids: evaluate_patch(
            patch, suite=suite, shards=shard_ids, timeout=verifier_timeout,
            public=public, max_patch_bytes=max_patch_bytes, cache=cache),
        reward=framework_score,
        prompt=lambda program: envelope(program, suite=suite,
                                        agent_timeout=agent_timeout,
                                        ask_promise=ask_promise,
                                        feedback=feedback),
        task_prompt=lambda index: (
            f"Score the patch for SWE-bench Science task {suite.task_id} on "
            f"held-out check set {index}."),
        test_shards=suite.test_range(),
        data_summary=suite.data_summary(),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    # `model_default=None`: the mutation operator is a coding-agent CLI, which
    # has a model of its own. `--model` is forwarded to it when given, and the
    # completion control arm requires one.
    add_standard_args(parser, model_default=None, max_seconds_default=1800.0,
                      eval_concurrency_default=None,
                      model_help=("model id passed to the agent CLI (its own "
                                  "default when unset); required for "
                                  "--agent completion"))
    parser.set_defaults(async_ratio=1)
    parser.add_argument("--tasks", default="default",
                        help=(f"comma-separated task ids and inclusive ranges "
                              f"(`002,005-007`, the release's own syntax), "
                              f"`default` for the {len(DEFAULT_TASKS)} spanning "
                              f"domains, `unrestricted` for the release's own "
                              f"96-task default selection, or `all` for every "
                              f"one of the {RELEASE_TASKS}"))
    parser.add_argument("--list-tasks", action="store_true",
                        help="print the release's task table and exit")
    parser.add_argument("--allow-restricted-licenses", action="store_true",
                        help=("run tasks the release gates on GPL-family, "
                              "non-commercial or restricted-material licences. "
                              "The flag selects task bundles; it does not "
                              "replace the upstream licence obligations"))
    parser.add_argument("--release", type=Path, default=None,
                        help=("a local download of the release "
                              f"({RELEASE_REPO}) to read the manifest and "
                              "instructions from, instead of fetching them"))
    parser.add_argument("--agent", default="claude-code", choices=AGENTS,
                        help=("the mutation operator. `claude-code` and `codex` "
                              "run that CLI in the checkout; `command` runs "
                              "--agent-command; `completion` is the control arm "
                              "-- one model call, no tools, asked for a diff"))
    parser.add_argument("--agent-command", default="",
                        help="argv for --agent command; the prompt goes to stdin")
    parser.add_argument("--agent-timeout", type=float, default=1800.0,
                        help=(f"wall clock for one agent session. The release "
                              f"allows {int(RELEASE_AGENT_TIMEOUT)}s per task "
                              f"for a single attempt; a tree spends several, so "
                              f"the default here is lower and a run that raises "
                              f"it is not comparable to one that did not"))
    parser.add_argument("--staleness", default="guarded",
                        choices=["guarded", "reflective", "full"],
                        help=("what to do with an expansion proposed against a "
                              "head the merger has since moved. The tree is "
                              "append-only, so `full` is the honest default for "
                              "a comparison and `guarded` the conservative one"))
    parser.add_argument("--iterations", type=int, default=4,
                        help=("FUTS expansions per task (upstream's "
                              "num_iterations). Each task gets its own tree, and "
                              "each expansion is one agent session"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--shards", type=int, default=4,
                        help=("check sets the search may score against, cut from "
                              "the visible half of the private suite. The engine "
                              "needs at least 4 to split train from held-out"))
    parser.add_argument("--test-shards", type=int, default=2,
                        help="further check sets, cut from the held-back tests")
    parser.add_argument("--held-back-frac", type=float, default=0.25,
                        help=("fraction of each task's private tests the search "
                              "never sees, in any shard or prompt. 0 turns the "
                              "split off, and a run with it off cannot report a "
                              "held-back figure"))
    parser.add_argument("--feedback", default="public", choices=FEEDBACK_LEVELS,
                        help=("how much of an evaluation a child's prompt may "
                              "quote. `public` is the public reproduction's own "
                              "output and nothing else -- the information the "
                              "benchmark itself gives an agent, and the default. "
                              "`counts` adds how many visible private checks "
                              "passed, with no names. `tests` adds pytest's "
                              "failure sections, which embed the **body of the "
                              "failing test**: it hands over the hidden suite's "
                              "assertions, and a resolve rate measured under it "
                              "is not a measurement of the benchmark's task"))
    parser.add_argument("--no-public", action="store_true",
                        help=("leave the public reproduction out of the score. "
                              "It is the benchmark's own first condition, so "
                              "this is a deviation, not a speed knob"))
    parser.add_argument("--held-out-frac", type=float, default=0.5)
    parser.add_argument("--c-puct", type=float, default=1.0,
                        help="upstream's exploration constant (futs.search default)")
    parser.add_argument("--prior-exponent", type=float, default=0.0,
                        help=("weight on the agent's own rating of a direction "
                              "in the PUCT prior -- AlphaZero's P(s,a), which "
                              "upstream ERA leaves uniform at 1/N. 0 is "
                              "upstream and the default. Above 0 the prompt "
                              "asks for the rating, at no extra session"))
    parser.add_argument("--verifier-timeout", type=float,
                        default=RELEASE_VERIFIER_TIMEOUT,
                        help=("wall clock for one verifier container "
                              f"(the release's own [verifier] timeout_sec is "
                              f"{int(RELEASE_VERIFIER_TIMEOUT)})"))
    parser.add_argument("--max-patch-bytes", type=int, default=400_000,
                        help=("a candidate larger than this is an invalid node "
                              "rather than a scored one -- a runaway session "
                              "that vendored a wheel into the checkout would "
                              "otherwise be carried in the ledger and the "
                              "result file"))
    parser.add_argument("--workspace-root", type=Path, default=None,
                        help="where agent checkouts are made (default: $TMPDIR)")
    parser.add_argument("--keep-workspaces", action="store_true",
                        help="do not delete an expansion's checkout afterwards")
    parser.add_argument("--no-pull", action="store_true",
                        help="fail rather than fetch a pinned image that is missing")
    parser.add_argument("--scorecard", type=Path, nargs="+", default=None,
                        metavar="RESULT.json",
                        help=("aggregate regraded result files into the "
                              "leaderboard's own columns (OVERALL, PUBLIC, "
                              "PRIVATE, FAIL2PASS, PASS2PASS). Arithmetic over "
                              "grades already in hand -- no container, no agent"))
    parser.add_argument("--grade-node", type=int, default=None, metavar="INDEX",
                        help=("with --regrade, score this tree node instead of "
                              "the winner. `--grade-node 1` is the benchmark's "
                              "own protocol: the first expansion from the "
                              "baseline, with no verifier in the loop choosing "
                              "it. 0 is the untouched repository"))
    parser.add_argument("--regrade", type=Path, default=None,
                        help=("re-run the release's grader over the patches a "
                              "finished result file already holds, and write "
                              "the new numbers back beside the old ones. No "
                              "agent, no search -- grading is a pure function "
                              "of the patch"))
    # The control arm's knobs. They move nothing unless --agent completion.
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--thinking", choices=("disabled", "enabled", "default"),
                        default="default",
                        help=("extended thinking. For the agent arms this sets "
                              "the CLI's own MAX_THINKING_TOKENS (0 when "
                              "disabled); `default` inherits whatever the "
                              "launching shell had, which is why the effective "
                              "value is recorded in the result file. Thinking is "
                              "most of a session's wall-clock, so turning it off "
                              "is what makes a deep tree affordable"))
    parser.add_argument("--api-timeout", type=float, default=600.0)
    parser.add_argument("--shutdown-grace", type=float, default=120.0)
    parser.add_argument("--quality-target", type=float)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _make_completion(args: argparse.Namespace, usage: Usage):
    """The control arm's model call, dispatched through the shared helper."""
    options: Dict[str, Any] = {}
    if args.thinking != "default":
        options["thinking"] = {"type": args.thinking}
    return completion_for(args, usage=usage, max_tokens=args.max_tokens,
                          timeout=args.api_timeout, temperature=args.temperature,
                          **options)


def _task_payload(suite: Suite, domain: Domain, run: Any,
                  baseline_grade: Dict[str, Any], best_grade: Dict[str, Any],
                  quality_target: Optional[float]) -> Dict[str, Any]:
    baseline = run.baseline_test_metrics.get("pass_rate")
    best = run.best_test_metrics.get("pass_rate")
    return {
        "task_id": suite.task_id,
        "title": suite.title,
        "domain": suite.domain,
        "status": "completed" if run.result.error is None else "partial",
        # The benchmark's own number, from its own grader over its whole
        # private suite -- read once for the root patch and once for the best
        # node, and never by the search.
        "benchmark_reward_baseline": int(baseline_grade.get("reward", 0) or 0),
        "benchmark_reward_best": int(best_grade.get("reward", 0) or 0),
        "benchmark_grade_baseline": baseline_grade,
        "benchmark_grade_best": best_grade,
        # The held-back tests: did the gain reach past what the search could see?
        "held_back_tests": len(suite.held_back),
        "held_back_pass_rate_baseline": baseline,
        "held_back_pass_rate_best": best,
        "held_back_gain": domain.gain(baseline, best),
        "visible_pass_rate_baseline": run.tree.root().program.metrics.get("pass_rate"),
        "visible_pass_rate_best": run.tree.best().program.metrics.get("pass_rate"),
        "nodes": len(run.tree.nodes),
        "wall_seconds": run.wall_seconds,
        "best_patch_chars": len(run.tree.best().program.code),
        # The artifact itself, so the result file is self-contained: `--regrade`
        # reads it back, and a reader can see what was actually submitted rather
        # than a character count.
        "best_patch": run.tree.best().program.code,
        # Every node's artifact, not only the winner's. The tree's `best()` is
        # an **oracle selector** -- it reads the private suite -- so a result
        # file that kept only the winner could not answer the one question the
        # benchmark actually asks: what does a *single* attempt resolve? With
        # these, `--regrade` can score any node, and pass@1 is the first
        # expansion's patch rather than a rerun.
        "node_patches": {str(node.index): node.program.code
                         for node in run.tree.nodes},
        "observation": run.summary(quality_target),
    }


def task_scorecard(baseline: Dict[str, Any], after: Dict[str, Any]
                   ) -> Dict[str, Any]:
    """One task's row in the leaderboard's own columns.

    The published columns are not one metric. `OVERALL` is Pass@1 -- every
    applicable private test passing, all or nothing, per task. `PUBLIC` is
    per task too. `PRIVATE`, `FAIL2PASS` and `PASS2PASS` are per **test**, and
    the last two are defined against the *baseline's* outcome for that same
    test: fail-to-pass is the repair, pass-to-pass is the regression check.
    Which is why both grades are needed here, with their per-test maps.
    """
    before = baseline.get("tests") or {}
    now = after.get("tests") or {}
    shared = [name for name in now if name in before]
    f2p = [n for n in shared if before[n] == "failed"]
    p2p = [n for n in shared if before[n] == "passed"]
    private = after.get("private") or {}
    return {
        "resolved": int(after.get("reward", 0) or 0),
        "public": int((after.get("public") or {}).get("passed", 0) or 0),
        "private_passed": int(private.get("passed", 0) or 0),
        "private_total": int(private.get("collected", 0) or 0),
        "f2p_passed": sum(1 for n in f2p if now[n] == "passed"),
        "f2p_total": len(f2p),
        "p2p_passed": sum(1 for n in p2p if now[n] == "passed"),
        "p2p_total": len(p2p),
        "per_test_available": bool(before and now),
    }


def scorecard(paths: Sequence[Path]) -> int:
    """Aggregate regraded result files into the leaderboard's columns.

    Pure arithmetic over grades already in hand -- no container, no agent. The
    leaderboard's denominator is **119 tasks**; this prints the denominator it
    actually had, because a rate over a subset is not the published number and
    a table that hid that is how it would be quoted as one.
    """
    rows: List[Dict[str, Any]] = []
    attempts, graded_nodes, feedback = set(), set(), set()
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        attempts.add(payload.get("config", {}).get("iterations"))
        feedback.add(payload.get("config", {}).get("feedback"))
        for entry in payload.get("tasks", []):
            graded_nodes.add(entry.get("graded_node"))
            card = entry.get("scorecard")
            if card is None:
                raise SystemExit(
                    f"{path} task {entry['task_id']} has no scorecard: regrade it "
                    "first (--regrade PATH [--grade-node 1]) so the per-test "
                    "outcomes are recorded.")
            rows.append(dict(card, task_id=entry["task_id"]))
    if not rows:
        raise SystemExit("no tasks in the given result files")

    def rate(num, den):
        return (100.0 * num / den) if den else None

    n = len(rows)
    tot = lambda key: sum(r[key] for r in rows)              # noqa: E731
    single = graded_nodes == {1} or attempts == {1}
    print(f"Scorecard: SWE-bench Science, {n} task(s)")
    print(f"  protocol : {'Pass@1 -- one attempt per task' if single else 'NOT Pass@1'}"
          f" (iterations={sorted(a for a in attempts if a is not None)}, "
          f"graded node={sorted(g for g in graded_nodes if g is not None) or 'best'})")
    print(f"  feedback : {sorted(f for f in feedback if f)}")
    print()
    print(f"  {'column':<12} {'value':>8}   {'counts':>14}")
    print(f"  {'-'*12} {'-'*8}   {'-'*14}")
    for label, num, den in (
        ("OVERALL", tot("resolved"), n),
        ("PUBLIC", tot("public"), n),
        ("PRIVATE", tot("private_passed"), tot("private_total")),
        ("FAIL2PASS", tot("f2p_passed"), tot("f2p_total")),
        ("PASS2PASS", tot("p2p_passed"), tot("p2p_total")),
    ):
        value = rate(num, den)
        print(f"  {label:<12} {('%.2f%%' % value) if value is not None else 'n/a':>8}"
              f"   {num:>6} / {den:<6}")
    print()
    print("  per task:")
    for r in sorted(rows, key=lambda r: r["task_id"]):
        print(f"    {r['task_id']}  resolved={r['resolved']}  public={r['public']}  "
              f"private={r['private_passed']}/{r['private_total']}  "
              f"f2p={r['f2p_passed']}/{r['f2p_total']}  "
              f"p2p={r['p2p_passed']}/{r['p2p_total']}")
    print()
    print(f"  The published leaderboard's denominator is 119 tasks and its "
          f"ISSUE/EXPERT/ENGINEERING columns split them 52/49/18 by task "
          f"paradigm.\n  This is {n} task(s) and the release ships no per-task "
          f"paradigm label, so those three columns cannot be reproduced here.")
    return 0


def _resolve_summary(entries: List[Dict[str, Any]], baseline: int, resolved: int,
                     attempts: Optional[int], graded_node: Optional[int]
                     ) -> Dict[str, Any]:
    """Name the rate after what it measures, and never after what it is not.

    The benchmark's `resolve rate` is **one attempt per task**. What this port
    produces by default is the reward of the tree's best node -- chosen by a
    selector that reads the private suite, out of `--iterations` attempts. Those
    are different numbers, and a field called `resolve_rate` carrying the second
    one is how the second gets quoted as the first.

    So the benchmark's name is emitted only when the run earns it: a single
    expansion per task, or a regrade of node 1, which is that first expansion
    with nothing in the loop choosing it. Otherwise the field says
    `best_of_n_resolve_rate` and carries the `n` beside it.
    """
    n = len(entries)
    rate = (resolved / n) if n else None
    single = (attempts == 1) or (graded_node == 1)
    summary: Dict[str, Any] = {
        "tasks_run": n,
        "resolved_baseline": baseline,
        "resolved": resolved,
        "attempts_per_task": attempts,
    }
    if single:
        summary["resolve_rate"] = rate
        summary["selection"] = "none -- one attempt per task, as the benchmark scores it"
    else:
        summary["best_of_n_resolve_rate"] = rate
        summary["selection"] = (
            "ORACLE -- the tree's best node, ranked on the private suite. This is "
            "not the benchmark's resolve rate; run --iterations 1, or regrade "
            "with --grade-node 1, for that.")
    return summary


def regrade(args: argparse.Namespace) -> int:
    """Re-run the release's grader over a finished run's saved patches.

    Grading is a pure function of `(patch, task)`, so a run whose *grading* step
    was wrong does not have to be paid for twice. It was wrong once, and the
    fix is what this exists for: `grade()` used to run the grader baked into the
    verifier image, and some of those images carry a stale one that collects
    nothing and scores every submission 0.

    The previous numbers are kept beside the new ones under `regraded`, because
    a result file that quietly changed its own answer is worse than one that
    was wrong.
    """
    path = Path(args.regrade)
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload.get("config", {})
    manifest = load_manifest(release=args.release)
    entries = payload.get("tasks", [])
    print(f"Regrade  : {len(entries)} task(s) in {path}, with the release's own "
          f"bundle grader")
    for index, entry in enumerate(entries, start=1):
        task_id = entry["task_id"]
        suite = prepare_suite(
            task_id, manifest=manifest,
            shards=int(config.get("shards", args.shards)),
            test_shards=int(config.get("test_shards", args.test_shards)),
            held_back_frac=float(config.get("held_back_frac", args.held_back_frac)),
            seed=int(config.get("seed", args.seed)), release=args.release,
            pull=not args.no_pull)
        if args.grade_node is not None:
            nodes = entry.get("node_patches") or {}
            if str(args.grade_node) not in nodes:
                raise SystemExit(
                    f"task {task_id} has no node {args.grade_node} recorded. "
                    "Only runs made after `node_patches` was added carry every "
                    "node's artifact; older ones keep the winner alone.")
            patch = nodes[str(args.grade_node)]
        else:
            patch = entry.get("best_patch")
            if patch is None:
                sidecar = path.with_name(f"{path.stem}-task{task_id}-best.patch")
                patch = sidecar.read_text(encoding="utf-8") if sidecar.is_file() else ""
                entry["best_patch"] = patch
        baseline_grade = grade("", suite=suite, timeout=args.verifier_timeout,
                               release=args.release)
        best_grade = (baseline_grade if not patch.strip() else
                      grade(patch, suite=suite, timeout=args.verifier_timeout,
                            release=args.release))
        entry["graded_node"] = args.grade_node
        entry["scorecard"] = task_scorecard(baseline_grade, best_grade)
        entry["regraded"] = {
            "previous_reward_baseline": entry.get("benchmark_reward_baseline"),
            "previous_reward_best": entry.get("benchmark_reward_best"),
            "previous_grade_baseline": entry.get("benchmark_grade_baseline"),
            "previous_grade_best": entry.get("benchmark_grade_best"),
        }
        entry["benchmark_grade_baseline"] = baseline_grade
        entry["benchmark_grade_best"] = best_grade
        entry["benchmark_reward_baseline"] = int(baseline_grade.get("reward", 0) or 0)
        entry["benchmark_reward_best"] = int(best_grade.get("reward", 0) or 0)
        print(f"[{index}/{len(entries)}] {task_id}: reward "
              f"{entry['regraded']['previous_reward_best']} -> "
              f"{entry['benchmark_reward_best']} "
              f"(private {best_grade.get('private', {}).get('passed')}"
              f"/{best_grade.get('private', {}).get('collected')})", flush=True)
    resolved = sum(entry["benchmark_reward_best"] for entry in entries)
    baseline = sum(entry["benchmark_reward_baseline"] for entry in entries)
    payload.setdefault("summary", {}).update(
        _resolve_summary(entries, baseline, resolved,
                         int(config.get("iterations", 0)) or None,
                         args.grade_node))
    payload["regraded_at"] = _utc_now()
    payload["regraded_reason"] = (
        "the run's own grading step ran the grader baked into each verifier "
        "image; some of those are stale (task 034's collects nothing and scores "
        "every submission 0). Re-run with the release's own task-bundle grader, "
        "which is what its tests/docker-compose.yaml mounts. The search itself "
        "is untouched -- grading is a pure function of the patch.")
    _write_json(path, payload)
    print(f"\nregraded : resolved {baseline} -> {resolved} of {len(entries)}"
          + (f" (node {args.grade_node}, one attempt per task)"
             if args.grade_node is not None else " (best of "
             f"{config.get('iterations')}, oracle-selected)")
          + f", output={path}")
    return 0


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.workers = worker_count(args, args.workers)
    if args.budget_rollouts:
        args.iterations = args.budget_rollouts
    if getattr(args, "reflective_merge", False):
        raise SystemExit(
            "--reflective-merge is not supported by the ERA port: a candidate is "
            "a whole patch rather than a delta, and fusing a round's expansions "
            "into one would delete tree nodes that FUTS selects from. "
            "Use --staleness {guarded,reflective,full}.")
    if args.list_tasks:
        manifest = load_manifest(release=args.release)
        for task_id in sorted(manifest):
            row = manifest[task_id]
            gate = row.get("license_gate", "none")
            print(f"{task_id}  {row.get('language', ''):<14} "
                  f"{row.get('domain', ''):<38} "
                  f"{'' if gate == 'none' else '[' + gate + '] '}"
                  f"{row.get('title', '')}")
        return 0

    if args.scorecard:
        return scorecard(args.scorecard)
    if args.regrade:
        # Ahead of the run plan, because none of it applies: a regrade spends no
        # agent session and reads the patches a finished run already produced.
        print("Algorithm: ERA Flat UCB tree search (FUTS) on AgentDescent")
        print(f"Benchmark: SWE-bench Science, {RELEASE_REPO}@{RELEASE_REVISION[:12]}")
        print(f"Evaluator: the release's own task-bundle grader in its own pinned "
              f"verifier image, {docker_backend() or 'NO DOCKER DAEMON'}")
        if args.dry_run:
            print("[dry-run] no release file, image, container, or agent was "
                  "accessed.")
            return 0
        if docker_backend() is None:
            raise SystemExit("regrading runs the release's grader in its own "
                             "image, and this host has no Docker daemon.")
        return regrade(args)

    mode = "async" if args.asynchronous else ("serial" if args.serial else "sync")
    backend = docker_backend()
    print("Algorithm: ERA Flat UCB tree search (FUTS) on AgentDescent")
    print(f"Benchmark: SWE-bench Science -- {RELEASE_TASKS} tasks, one tree each, "
          f"pass rate over the checks the search may see")
    print(f"Release  : {RELEASE_REPO}@{RELEASE_REVISION[:12]} ({RELEASE_GITHUB})")
    print(f"Evaluator: the release's own grader in its own pinned verifier "
          f"image, {backend or 'NO DOCKER DAEMON -- this run will fail'}")
    print(f"Mutation : one {args.agent} session per expansion, in a checkout "
          f"bind-mounted into the task's offline environment image")
    print(
        f"Plan     : mode={mode}, tasks={args.tasks}, iterations={args.iterations}"
        f"/task, workers={args.workers}, c_puct={args.c_puct}, "
        f"agent_timeout={args.agent_timeout:g}s"
    )
    print(f"Feedback : {args.feedback} -- what a child's prompt may quote of "
          f"its parent's evaluation"
          + ("  [the visible tests' own tracebacks, which embed their source]"
             if args.feedback == "tests" else ""))
    print(f"Split    : {args.held_back_frac:g} of each private suite held back "
          f"from the search; the reported reward is the release's own grader "
          f"over the whole suite")
    if args.prior_exponent > 0.0:
        print(f"Prior    : the agent's own rating in P(s,a), exponent "
              f"{args.prior_exponent} (upstream ERA: uniform 1/N)")
    artifact = EvolvingArtifact(ARTIFACT_ID, blast_radius=0.6)
    print(
        f"Governance: generated patch blast_radius={artifact.blast_radius} "
        f"-> {classify(artifact).name}"
    )
    if args.dry_run:
        print("[dry-run] no release file, image, container, or agent was accessed.")
        return 0

    if backend is None:
        raise SystemExit(
            "SWE-bench Science is distributed as pinned Docker images and this "
            "host has no Docker daemon. There is no offline substitute: the "
            "task's dependencies, its public fixtures and its held-out tests "
            "all live in those images.")
    if args.shards < 4:
        raise SystemExit(
            f"--shards {args.shards} is too few: a shard is one rollout task, "
            "and the engine needs at least 4 to split train from held-out.")
    if not confirm(args):
        return 0

    manifest = load_manifest(release=args.release)
    tasks = parse_selection(args.tasks, manifest)
    gated = [task_id for task_id in tasks
             if manifest[task_id].get("license_gate", "none") != "none"]
    if gated and not args.allow_restricted_licenses:
        raise SystemExit(
            f"{len(gated)} selected task(s) carry a licence gate "
            f"({', '.join(gated[:8])}{'...' if len(gated) > 8 else ''}). The "
            "release excludes them from its default selection because they hold "
            "GPL/LGPL/AGPL-family code, academic non-commercial sources, or "
            "restricted third-party materials. Pass "
            "--allow-restricted-licenses only after confirming your use is "
            "permitted.")

    agent_usage = Usage()
    model_usage = Usage()
    sessions: Dict[str, int] = {}
    launch = None
    complete_control = None
    if args.agent == "completion":
        if not args.model:
            raise SystemExit("--agent completion needs --model")
        complete_control = _make_completion(args, model_usage)
    else:
        launch = build_launch(args, agent_usage)

    started = time.monotonic()
    entries: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    for index, task_id in enumerate(tasks, start=1):
        actor_usage = Usage()
        try:
            suite = prepare_suite(
                task_id, manifest=manifest, shards=args.shards,
                test_shards=args.test_shards, held_back_frac=args.held_back_frac,
                seed=args.seed, release=args.release, pull=not args.no_pull)
        except Exception as exc:
            print(f"[{index}/{len(tasks)}] {task_id}: skipped -- "
                  f"{type(exc).__name__}: {exc}", flush=True)
            failures.append({"task_id": task_id, "stage": "prepare",
                             "error": f"{type(exc).__name__}: {exc}"})
            continue
        print(f"\n[{index}/{len(tasks)}] task {task_id} ({suite.domain}, "
              f"{suite.language}): {suite.title}\n"
              f"    {len(suite.tests)} private test(s) -- "
              f"{len(suite.visible)} visible in {args.shards} shard(s), "
              f"{len(suite.held_back)} held back"
              f"{' (none: too few tests to split)' if not suite.held_back else ''}",
              flush=True)

        # Deterministic evaluations, memoised for this task only. A shard that
        # repeats a test because the suite was too small to fill it, and a
        # rollout that re-scores a head the merger has not moved, then cost no
        # container at all. Deliberately not persisted: a cache that outlived
        # the run would change what a rerun measures.
        cache: Dict[Any, Any] = {}
        domain = swe_science_domain(
            suite, verifier_timeout=args.verifier_timeout,
            max_patch_bytes=args.max_patch_bytes, agent_timeout=args.agent_timeout,
            ask_promise=args.prior_exponent > 0.0, public=not args.no_public,
            feedback=args.feedback, cache=cache)
        if launch is not None:
            mutate = make_agent_mutation(
                suite, launch=launch, run_root=args.workspace_root,
                keep=args.keep_workspaces, counter=sessions,
                on_event=lambda note: print(f"    {note}", flush=True))
        else:
            mutate = make_completion_mutation(
                suite, complete_control, run_root=args.workspace_root,
                counter=sessions)
        try:
            run = run_agentdescent_era(
                mutate,
                mode=mode,
                iterations=args.iterations,
                workers=args.workers,
                shards=args.shards,
                test_shards=args.test_shards,
                held_out_frac=args.held_out_frac,
                c_puct=args.c_puct,
                prior_exponent=args.prior_exponent,
                candidate_timeout=args.verifier_timeout,
                max_code_length=args.max_patch_bytes,
                async_ratio=args.async_ratio,
                staleness=args.staleness,
                max_seconds=args.max_seconds,
                shutdown_grace=args.shutdown_grace,
                seed=args.seed,
                usage=actor_usage,
                eval_concurrency=args.eval_concurrency,
                domain=domain,
                extract=extract_patch,
                empty_warning=EMPTY_WARNING,
                # Never "solved". A shard here is a handful of tests and a
                # rollout that clears all of them says nothing about the
                # held-back ones the benchmark's verdict rests on -- while
                # `evolve`'s default would take it as a reason to skip that
                # worker's expansion, and the run would quietly spend less of
                # the budget than its own result file reports. Upstream FUTS
                # expands the selected node whatever it scored.
                solved_threshold=float("inf"),
                verbose=True,
            )
        except Exception as exc:
            print(f"[{index}/{len(tasks)}] {task_id}: failed -- "
                  f"{type(exc).__name__}: {exc}", flush=True)
            failures.append({"task_id": task_id, "stage": "search",
                             "error": f"{type(exc).__name__}: {exc}"})
            continue

        baseline_grade = grade("", suite=suite, timeout=args.verifier_timeout)
        best_patch = run.tree.best().program.code
        best_grade = (baseline_grade if not best_patch.strip()
                      else grade(best_patch, suite=suite,
                                 timeout=args.verifier_timeout))
        entry = _task_payload(suite, domain, run, baseline_grade, best_grade,
                              args.quality_target)
        entry["actor_usage"] = _usage_dict(actor_usage)
        entries.append(entry)
        patch_path = args.output.with_name(
            f"{args.output.stem}-task{suite.task_id}-best.patch")
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        patch_path.write_text((best_patch.rstrip() + "\n") if best_patch.strip()
                              else "", encoding="utf-8")
        print(f"[{index}/{len(tasks)}] {task_id}: benchmark reward "
              f"{entry['benchmark_reward_baseline']} -> "
              f"{entry['benchmark_reward_best']}, held-back pass rate "
              f"{entry['held_back_pass_rate_baseline']} -> "
              f"{entry['held_back_pass_rate_best']}, nodes={entry['nodes']}, "
              f"wall={entry['wall_seconds']:.1f}s, patch -> {patch_path}",
              flush=True)

    wall = time.monotonic() - started
    resolved = sum(entry["benchmark_reward_best"] for entry in entries)
    resolved_baseline = sum(entry["benchmark_reward_baseline"] for entry in entries)
    improved = sum(1 for entry in entries if (entry["held_back_gain"] or 0.0) > 0.0)
    payload: Dict[str, Any] = {
        "experiment": "ERA on AgentDescent -- SWE-bench Science, one tree per task",
        "status": "completed" if entries and not failures else "partial",
        "completed_at": _utc_now(),
        "upstream_commit": UPSTREAM_COMMIT,
        "benchmark": f"SWE-bench Science ({RELEASE_REPO}@{RELEASE_REVISION})",
        "protocol": (
            "ERA's, not the benchmark's: the leaderboard gives an agent one "
            "attempt per task with no verifier feedback, while this scores a "
            "tree of agent sessions against the visible part of each private "
            "suite and keeps the best node. `benchmark_reward_*` is the "
            "release's own grader over the whole private suite; the search "
            "never reads it."),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items() if key not in ("output", "yes")
        },
        "summary": {
            "tasks_run": len(entries),
            "tasks_failed": len(failures),
            "tasks_improved_on_held_back": improved,
            **_resolve_summary(entries, resolved_baseline, resolved,
                               args.iterations, None),
            "wall_seconds": wall,
        },
        "tasks": entries,
        "failures": failures,
        "agent_sessions": {
            "agent": args.agent,
            "command": " ".join(agent_command(args)) if args.agent != "completion"
                       else "",
            # The CLI carries its own model, so a result that named only the
            # command could not be repeated.
            "version": (agent_version(agent_command(args))
                        if args.agent != "completion" else ""),
            "model": args.model or "",
            # Inherited from the launching shell unless --thinking says
            # otherwise, and most of a session's wall-clock either way.
            "thinking": args.thinking,
            "thinking_tokens": (agent_environment(args).get("MAX_THINKING_TOKENS")
                                or os.getenv("MAX_THINKING_TOKENS", "")),
            "timeout_s": args.agent_timeout,
            **{key: value for key, value in sorted(sessions.items())},
        },
        "agent_usage": _usage_dict(agent_usage),
        "model_usage": _usage_dict(model_usage),
    }
    _write_json(args.output, payload)
    print(
        f"\ncompleted: {len(entries)} task(s), resolved {resolved_baseline} -> "
        f"{resolved} by the release's own grader, {improved} improved on their "
        f"held-back tests, wall={wall:.1f}s, "
        f"agent_sessions={sessions.get('sessions', 0)}, output={args.output}"
    )
    return 0 if entries and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
