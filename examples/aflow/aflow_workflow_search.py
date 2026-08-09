"""Mechanism microport: **AFlow** (FoundationAgents/AFlow@3f457218, ICLR 2025).

Preserved: the *actual* selection rule at the pinned revision -- **soft mixed
probability**, ``λ·uniform + (1−λ)·softmax(α·(s−s_max))`` over the top-k scored
workflows (`data_utils.select_round`; it is not UCT and keeps no visit counts) --
and expansion prompts carrying the selected parent's **experience**: its score,
and every modification already tried from it with whether it helped, which is
upstream's ``experience.json`` and its ``format_experience``.

**The scores are scaled by 100 before the softmax**, exactly as
``select_round`` does, and that single line is the difference between AFlow and
a coin flip. Upstream's α is 0.2 *against percentages*, so the effective
temperature is 20 against accuracies in ``[0, 1]``. Without the scaling, on a
pool scoring 0.50 / 0.40 / 0.25 / 0.10 the pick distribution is
``[0.265, 0.257, 0.245, 0.233]`` -- uniform to three digits -- where upstream's
is ``[0.688, 0.158, 0.079, 0.075]``. A port missing it still runs, still reports
"soft mixed probability", and has no exploitation at all.

``α = 0.2`` and ``λ = 0.3`` are the pinned code's own defaults
(``DataUtils.DEFAULT_ALPHA`` / ``DEFAULT_LAMBDA``), not the paper's 0.4 / 0.2.
Where the released code and the paper disagree about a constant, the code is what
produced the published numbers.

The workflow is a :class:`~examples._method_policy.FieldSlots` artifact (solve
node, review node, modification note), so field edits union-merge and contested
fields are model-merged without ranking evaluations.

Boundaries: two fixed model nodes instead of code-level graph rewrites; search
depth is the candidate budget instead of 20 rounds; upstream regenerates
*without bound* until the proposed modification is one it has not tried from
that parent, and this retries once.
"""

from __future__ import annotations

import json
import math
import random
import threading
from typing import Dict, List, Optional, Sequence, Tuple

from agentdescent.evolution import Task
from agentdescent.policies import Policies
from agentdescent.selection import SelectionContext, SingleHead

from examples._measure import parse_json_object
from examples._method_policy import FieldSlots, MethodPolicy, read_fields
from examples._method_runner import standard_main
from examples._money_domain import (STARTING_INSTRUCTION, feedback,
                                    money_reward, money_splits, solve_money)


FIDELITY = "mechanism_microport"

REVIEW_INSTRUCTION = "Check the arithmetic and return only the corrected answer."

#: `select_round` computes `scores = [item["score"] * 100 ...]` before the
#: softmax. Upstream's benchmark scores are accuracies in [0, 1] exactly like
#: this port's, so the scaling is not a unit conversion -- it *is* the
#: temperature, and dropping it flattens the distribution to uniform.
SCORE_SCALE = 100.0

#: `DataUtils.DEFAULT_ALPHA` / `DEFAULT_LAMBDA` at the pinned revision.
ALPHA = 0.2
LAMBDA = 0.3

#: `run.py --sample`, default 4.
TOP_K = 4


class SoftMixed(SingleHead):
    """AFlow's node selection: λ-uniform mixed with a score softmax over top-k.

    The pool is the top-k *by score*. The seed workflow competes for a place
    like any other and is not force-included: upstream's ``get_top_rounds``
    moves round 1 to the front of the list **only when it already made the
    cut**, and ``select_round`` re-sorts by score immediately afterwards, so
    that reordering changes nothing and membership is all that is left.
    """

    def __init__(self, alpha: float = ALPHA, lam: float = LAMBDA,
                 top_k: int = TOP_K, seed: int = 0,
                 scale: float = SCORE_SCALE) -> None:
        self.alpha = alpha
        self.lam = lam
        self.top_k = top_k
        self.scale = scale
        self.rng = random.Random(seed)

    def probabilities(self, scores: Sequence[float]) -> List[float]:
        """`_compute_probabilities`, on scores already multiplied by `scale`."""
        scaled = [s * self.scale for s in scores]
        s_max = max(scaled)
        weights = [math.exp(self.alpha * (s - s_max)) for s in scaled]
        total = sum(weights)
        return [self.lam / len(scaled) + (1 - self.lam) * w / total
                for w in weights]

    def select(self, ctx: SelectionContext, n: int) -> Sequence:
        candidates = list(ctx.candidates)
        if len(candidates) <= 1:
            return super().select(ctx, n)
        pool = sorted(candidates, key=lambda c: c.score or 0.0,
                      reverse=True)[:self.top_k]
        probs = self.probabilities([c.score or 0.0 for c in pool])
        return [self.rng.choices(pool, weights=probs, k=1)[0] for _ in range(n)]


class Experience:
    """Upstream's ``experience.json``, in memory and keyed by parent workflow.

    Each entry is one modification tried *from* that parent, with the parent's
    score before it and whether it helped -- ``create_experience_data`` plus
    ``update_experience``. ``format_experience`` renders both the successes and
    the failures as things to prohibit, because what it is preventing is
    *repetition*, not failure.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rows: Dict[str, List[Tuple[str, float, Optional[bool]]]] = {}

    def record(self, parent: str, modification: str, before: float) -> None:
        with self._lock:
            self._rows.setdefault(parent, []).append((modification, before, None))

    def resolve(self, parent: str, modification: str, after: float) -> None:
        with self._lock:
            rows = self._rows.get(parent, [])
            for index, (mod, before, done) in enumerate(rows):
                if mod == modification and done is None:
                    rows[index] = (mod, before, bool(after > before))
                    break

    def tried(self, parent: str) -> List[str]:
        with self._lock:
            return [mod for mod, _, _ in self._rows.get(parent, ())]

    def render(self, parent: str, score: float) -> str:
        with self._lock:
            rows = list(self._rows.get(parent, ()))
        if not rows:
            return f"No experience data found for this workflow (score {score:.3f})."
        lines = [f"Original Score: {score:.3f}",
                 "These are some conclusions drawn from experience:", ""]
        for mod, before, succeeded in rows:
            if succeeded is False:
                lines.append(f"-Absolutely prohibit {mod} (Score: {before:.3f})")
            else:
                lines.append(f"-Absolutely prohibit {mod}")
        lines.append(
            "\nNote: Take into account past failures and avoid repeating the "
            "same mistakes, as these failures indicate that these approaches are "
            "ineffective. You must fundamentally change your way of thinking, "
            "rather than simply rewording the instruction.")
        return "\n".join(lines)


def build(seed: int) -> MethodPolicy:
    experience = Experience()

    def solve(llm, rendered: str, task: Task) -> str:
        graph = read_fields(rendered)
        draft = solve_money(
            llm, graph.get("solve_instruction", STARTING_INSTRUCTION), task,
            subphase="solve")
        return llm(
            (
                f"Problem:\n{task.prompt}\n\nDraft answer:\n{draft}\n\n"
                f"ReviewAndRevise node:\n{graph.get('review_instruction', REVIEW_INSTRUCTION)}\n\n"
                "Return only the final answer."
            ),
            subphase="review",
            unit=task.id,
        )

    def propose(llm, rendered: str, task: Task, output: str,
                reward: float) -> Optional[str]:
        prompt = (
            "You are AFlow's graph optimizer. Expand the selected workflow "
            "from its execution feedback while preserving exactly two nodes: "
            "Solve, then ReviewAndRevise.\n\n"
            f"Current workflow:\n{rendered}\n\n"
            f"{experience.render(rendered, reward)}\n"
            f"Reward: {reward}\n{feedback(task, output)}\n\n"
            "Return JSON only with modification, solve_instruction, and "
            "review_instruction strings."
        )
        raw = llm(prompt, unit=task.id)
        # `check_modification`: a modification already tried from this parent is
        # regenerated rather than accepted. Upstream loops without bound; one
        # retry is the declared boundary here, and the second attempt is what
        # `proposal_calls_per_candidate = 2` reserves.
        note = _modification(raw)
        if note and note in experience.tried(rendered):
            raw = llm(
                prompt + f"\n\nYou already proposed {note!r} from this workflow "
                         "and it was rejected for that reason. Propose a "
                         "different modification.",
                unit=f"{task.id}:regenerate")
            note = _modification(raw)
        experience.record(rendered, note or "(unparseable modification)", reward)
        return raw

    def _modification(raw: str) -> str:
        try:
            return str(parse_json_object(raw).get("modification", ""))[:200]
        except (KeyError, TypeError, ValueError):
            return ""

    train, held_out, test = money_splits(seed)
    return MethodPolicy(
        name="aflow",
        fidelity=FIDELITY,
        notes=(
            "The pinned revision's soft mixed selection (not UCT) runs through the population aggregator over the archive of committed workflows, with upstream's own alpha=0.2, lambda=0.3 and its score-by-100 scaling -- without which the distribution is uniform.",
            "Expansion prompts carry the selected parent's score and every modification already tried from it, as upstream's experience.json and format_experience do.",
            "A modification already tried from this parent is regenerated once, as check_modification does without bound.",
            "Every candidate keeps the same two model nodes, fixing workflow topology across modes.",
        ),
        strategy=FieldSlots(
            fields={
                "solve_instruction": STARTING_INSTRUCTION,
                "review_instruction": REVIEW_INSTRUCTION,
                "modification": "root workflow",
            },
            parse=parse_json_object,
        ),
        train_tasks=tuple(train),
        held_out_tasks=tuple(held_out),
        test_tasks=tuple(test),
        solve=solve,
        propose=propose,
        reward=money_reward,
        proposal_calls_per_candidate=2,
        engine=Policies(selection=SoftMixed(seed=seed)),
        reflective=True,
    )


def main(argv=None) -> int:
    return standard_main(build, argv)


if __name__ == "__main__":
    raise SystemExit(main())
