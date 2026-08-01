"""Worker: rollout + propose (design doc, section 3.1).

Each worker holds a *snapshot* of the ledger (its ``base_version``), runs tasks
against it, and turns observed failures into a diff plus an evidence card.  In a
real deployment the "propose" step is an LLM reflecting on a trajectory; in the
reference domain it is a deterministic corrector with tunable noise, so we can
exercise the aggregator's conflict resolution (a noisy worker proposes a wrong
label) and fusion (different workers fix different keywords).

Workers never mutate the ledger directly -- they only emit evidence cards.  All
mutation goes through the :class:`~agentdescent.aggregator.Aggregator`, which is the
sole "optimizer" (design doc, section 4).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from .evolvable import Diff, EvidenceCard, VersionVector, stable_hash
from .domains.router import RouterSkill, RouterTask


@dataclass
class Worker:
    worker_id: str
    gold: Dict[str, str]        # the proposal oracle (an LLM stands in here)
    noise: float = 0.0          # prob. of proposing a wrong label (contradiction)
    max_ops: int = 4            # keep diffs inside the trust region
    seed: int = 0
    # optional per-rollout latency (seconds) -- models the real cost of a
    # rollout (tool calls, HPC queues, LLM latency). Used by the efficiency
    # experiments to make parallelism/asynchrony observable in wall-clock.
    rollout_latency: Optional[Callable[[], float]] = None

    def __post_init__(self) -> None:
        self._rng = random.Random(stable_hash((self.worker_id, self.seed)) & 0xFFFF)

    def run(
        self,
        skill: RouterSkill,
        base_version: VersionVector,
        tasks: Sequence[RouterTask],
    ) -> Optional[EvidenceCard]:
        """One rollout: classify tasks, propose a corrective diff for failures."""
        if self.rollout_latency is not None:
            time.sleep(self.rollout_latency())  # releases the GIL -> real overlap
        failures = [t for t in tasks if skill.classify(t.text) != t.label]
        if not failures:
            return None

        # choose up to max_ops distinct failing keywords to fix this round.
        seen: Dict[str, RouterTask] = {}
        for t in failures:
            if t.keyword not in seen:
                seen[t.keyword] = t
            if len(seen) >= self.max_ops:
                break

        ops: Dict[str, str] = {}
        for kw, task in seen.items():
            if self._rng.random() < self.noise:
                # propose a plausible-but-wrong label (contradiction source).
                wrong = self._rng.choice([l for l in set(self.gold.values()) if l != task.label]
                                         or [task.label])
                ops[kw] = wrong
            else:
                ops[kw] = self.gold[kw]

        diff = Diff(
            diff_id=f"{self.worker_id}:{'-'.join(sorted(ops))}:{base_version.get(skill.id, 0)}",
            target=skill.id,
            ops=ops,
            author=self.worker_id,
        )

        # local before/after measurement on the failing tasks (the "gradient").
        before = skill.accuracy(list(seen.values()))
        after = skill.apply(diff).accuracy(list(seen.values()))

        return EvidenceCard(
            diff=diff,
            base_version={skill.id: base_version.get(skill.id, 0)},
            touched=[skill.id],
            before_after_delta=after - before,
            trajectory_refs=list(seen.values()),
            cost_tokens=len(tasks) * 10,
            cost_wallclock=float(len(tasks)),
        )
