#!/usr/bin/env python3
"""Meta-curriculum: the meta loop generates the problems it is judged on.

A challenger generates frontier *problems* for the decision rule being evolved:
the sampler mixes them with the seed problems, the rule climbs toward them, a
problem it solves is counted learned and the challenger pushes the generation
spec outward. The validation set is fixed and never touched by the challenger.

Offline, deterministic, no key:

    python -m examples.meta.curriculum_meta_demo
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile

from agentdescent import (
    ConsolidationStore,
    MetaChallengerConfig,
    MetaOutcome,
    MetaProblemChallenger,
    ParamSlot,
    curriculum_meta_run,
)
from agentdescent.sampling import DifficultyWeighted


def _c_of(rendered: str) -> float:
    m = re.search(r"^c: ([0-9.]+)", rendered, re.M)
    return float(m.group(1)) if m else 0.2


def slot() -> ParamSlot:
    return ParamSlot(factory=lambda c: DifficultyWeighted(c=c),
                     params={"c": 0.2}, bounds={"c": (0.05, 1.0)})


def offline_propose(rendered: str, task, output: str, reward: float) -> str:
    c = _c_of(rendered)
    if reward < 0.7:
        return f"c: {min(1.0, c + 0.1):.2f}"
    return f"c: {c:.2f}"


def helpful(value, seed):
    # a synthetic landscape, monotone in c: every nudge strictly improves the
    # held-out gate, so the rule is free to climb toward the challenger's edge
    c = getattr(value, "c", 0.2)
    r = min(1.0, 0.3 + 0.5 * c)
    return MetaOutcome(curve=[r, min(1.0, r + 0.05)], final=r)


def helpful_val(value, seed):
    return helpful(value, seed)


def frontier_problem(d: float):
    def problem(value, seed):
        if getattr(value, "c", 0.2) >= d:
            return MetaOutcome(curve=[1.0, 1.0], final=1.0)    # solved
        return MetaOutcome(curve=[0.1, 0.2], final=0.2)         # on the edge
    return problem


def make_challenger(start_d: float) -> MetaProblemChallenger:
    state = {"d": start_d}

    def generate(descriptor, seed):
        m = re.search(r"d:([0-9.]+)", descriptor)
        return {"d": float(m.group(1)) if m else state["d"]}

    def validate(config):
        return frontier_problem(config["d"])

    def propose(descriptor, report):
        m = re.search(r"d:([0-9.]+)", descriptor)
        d = float(m.group(1)) if m else state["d"]
        if report.get("learned", 0) > 0:
            state["d"] = min(1.0, d + 0.1)
        return f"d:{state['d']:.2f}"

    return MetaProblemChallenger(
        generate=generate, validate=validate, propose=propose,
        spec=f"d:{start_d:.2f}",
        config=MetaChallengerConfig(generated_per_round=2, frontier_share=0.4))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", type=int, default=6)
    args = parser.parse_args()

    store = ConsolidationStore(tempfile.mkdtemp(prefix="agentdescent-meta-cur-"))
    report, result = curriculum_meta_run(
        store=store, slot="task_sampler", spec=slot(),
        challenger=make_challenger(start_d=0.5), propose=offline_propose,
        problems={"ta": helpful, "tb": helpful}, val_problems={"va": helpful_val},
        config={"dataset": "meta-cur", "scorer": "exact"},
        seeds=(0, 1, 2, 3), val_seeds=(10, 11),
        rounds=args.rounds, n_workers=1,
        repo_dir=tempfile.mkdtemp(prefix="agentdescent-meta-cur-repos-"))

    f = report["frontier"]
    print(f"evolved slot value:\n{report['value']}")
    print(f"verdict: {report['verdict']} -- {report['reason']}")
    print(f"mean gain: {report['mean_gain']:+.3f}  transfer: "
          f"{report['transfer'] if report['transfer'] is None else round(report['transfer'], 3)}")
    print(f"frontier: generated={f['generated_total']} invalid={f['invalid_total']} "
          f"learned={f['learned']} stuck={f['stuck']} "
          f"r0_mean={f['r0_mean']:.3f} frontier_n={f['frontier_n']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
