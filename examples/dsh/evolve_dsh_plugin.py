"""Evolve a **dsh plugin's own source**, gated on its test suite.

The skill runner next door evolves text the agent reads. This evolves code the
harness *executes*, which is why it sits at L1: every merge goes through the
oracle, and the governance surface is frozen no matter what you pass.

    # what would happen, and what is frozen -- calls nothing
    python -m examples.dsh.evolve_dsh_plugin --plugin dsh-plugin-mine --dry-run

    # list what this profile has, and which of them can be evolved
    python -m examples.dsh.evolve_dsh_plugin --list

    # a real run: your tasks, your entrypoint, the plugin's own tests as the gate
    python -m examples.dsh.evolve_dsh_plugin --plugin dsh-plugin-mine \\
        --tasks q.jsonl --entrypoint "dsh --profile web -p" --model deepseek-v4-flash

``--entrypoint`` is the command a rollout runs to answer one task with the
candidate mounted; the task's prompt is appended as the final argument.

**Only a linked checkout can be evolved.** A plugin installed from the registry
is built output with no source behind it, and the next install would erase any
edit -- silently, at a time unrelated to the run. The runner refuses those and
says how to re-add them.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from typing import List, Optional

from agentdescent.backends import dsh
from agentdescent.dsh import list_plugins
from agentdescent.dsh.plugin import (
    FROZEN_PLUGIN_PATHS,
    NotEvolvable,
    evolve_dsh_plugin,
    resolve_evolvable_plugin,
)

from .evolve_dsh_skill import SCORERS, build_objective, load_tasks


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Evolve a dsh plugin's source, gated on its own tests.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plugin", help="package name, as the profile depends on it")
    ap.add_argument("--profile", default="web")
    ap.add_argument("--tasks", help="JSONL of {prompt, gold?}")
    ap.add_argument("--entrypoint", default=None,
                    help="command a rollout runs; the prompt is appended")
    ap.add_argument("--list", action="store_true",
                    help="show this profile's plugins and whether each is evolvable")
    ap.add_argument("--score", default="contains", choices=sorted(SCORERS))
    ap.add_argument("--check", default=None, help="rung B: a command that judges the output")
    ap.add_argument("--check-timeout", type=float, default=30.0)
    ap.add_argument("--test-cmd", default="npm test", help="the gate (default: npm test)")
    ap.add_argument("--model", default="deepseek-v4-flash", help="reflector model")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if args.list:
        found = list_plugins(args.profile)
        if not found:
            print(f"profile {args.profile!r} depends on no out-of-tree plugins.")
            return 1
        for plugin in found:
            why = "linked" if plugin.linked else "registry — not evolvable"
            kind = "bundle" if plugin.is_bundle else "library"
            print(f"{plugin.name:<32} {kind:<8} {why:<24} {plugin.path}")
        return 0

    if not args.plugin or not args.tasks:
        ap.error("--plugin and --tasks are required (or use --list)")

    try:
        plugin = resolve_evolvable_plugin(args.plugin, args.profile)
    except NotEvolvable as refusal:
        print(str(refusal), file=sys.stderr)
        return 1

    tasks = load_tasks(args.tasks)
    reward, objective = build_objective(args, tasks)
    entrypoint = shlex.split(args.entrypoint) if args.entrypoint else None

    print(f"plugin    : {plugin.name} ({'bundle' if plugin.is_bundle else 'library'})")
    print(f"path      : {plugin.path}")
    print(f"tasks     : {len(tasks)} from {args.tasks}")
    print(f"objective : {objective}")
    print(f"gate      : {args.test_cmd}")
    print(f"layer     : L1 — every merge goes through the oracle")
    print("frozen    : " + ", ".join(FROZEN_PLUGIN_PATHS))
    print(f"entrypoint: {' '.join(entrypoint) if entrypoint else '(required for a real run)'}")

    if args.dry_run:
        print("\ndry run — nothing was called, nothing was written.")
        return 0
    if entrypoint is None:
        print("\n--entrypoint is required for a real run.", file=sys.stderr)
        return 1

    result = evolve_dsh_plugin(
        plugin, tasks, entrypoint=entrypoint, reflect_with=dsh(model=args.model),
        score=reward, test_cmd=tuple(shlex.split(args.test_cmd)),
        rounds=args.rounds, n_workers=args.workers, max_concurrency=args.workers,
        verbose=True)

    print(f"\nfinal reward : {result.final_reward:.3f}")
    print(f"stop reason  : {result.stop_reason}")
    print(f"outcomes     : {result.outcomes()}")
    print(f"error        : {result.error}")
    print("\nnot installed. An L1 change is reviewed before it lands: inspect "
          "result.state, then write it back deliberately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
