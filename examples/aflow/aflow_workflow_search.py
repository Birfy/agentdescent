"""Mechanism microport: **AFlow** (FoundationAgents/AFlow@3f457218, ICLR 2025).

Preserved: the *actual* selection rule at the pinned revision -- **soft mixed
probability**, ``λ·uniform + (1−λ)·softmax(α·(s−s_max))`` over the top-k scored
workflows **plus the seed workflow** (`data_utils._compute_probabilities`; it is
not UCT and keeps no visit counts) -- and expansion prompts that carry the
selected parent's **experience**: its prior modifications and whether each
helped, upstream's ``experience.json``.

The workflow is a :class:`~examples._method_policy.FieldSlots` artifact (solve
node, review node, modification note), so field edits union-merge and contested
fields are model-merged without ranking evaluations.

Boundaries: two fixed model nodes instead of code-level graph rewrites; search
depth is the candidate budget instead of 20 rounds; paper hyper-parameters
α=0.4, λ=0.2 (the pinned code itself uses 0.2/0.3).
"""

from __future__ import annotations

import math
import random
import threading
from typing import Dict, List, Optional, Sequence

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


class SoftMixed(SingleHead):
    """AFlow's node selection: λ-uniform mixed with a score softmax over top-k.

    The seed workflow (the first candidate) is always kept in the pool --
    upstream's "including the initial workflow ensures persistent exploration".
    """

    def __init__(self, alpha: float = 0.4, lam: float = 0.2, top_k: int = 4,
                 seed: int = 0) -> None:
        self.alpha = alpha
        self.lam = lam
        self.top_k = top_k
        self.rng = random.Random(seed)

    def select(self, ctx: SelectionContext, n: int) -> Sequence:
        candidates = list(ctx.candidates)
        if len(candidates) <= 1:
            return super().select(ctx, n)
        pool = sorted(candidates, key=lambda c: c.score or 0.0,
                      reverse=True)[:self.top_k]
        if candidates[0] not in pool:
            pool.append(candidates[0])
        s_max = max(c.score or 0.0 for c in pool)
        weights = [math.exp(self.alpha * ((c.score or 0.0) - s_max)) for c in pool]
        total = sum(weights)
        probs = [self.lam / len(pool) + (1 - self.lam) * w / total for w in weights]
        return [self.rng.choices(pool, weights=probs, k=1)[0] for _ in range(n)]


def build(seed: int) -> MethodPolicy:
    experience: Dict[str, List[str]] = {}
    lock = threading.Lock()

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
        with lock:
            past = list(experience.get(rendered, ()))[-3:]
        history = (
            "Experience on this workflow (do not repeat these modifications):\n"
            + "\n".join(f"- {entry}" for entry in past)
            if past else "Experience on this workflow: none yet."
        )
        raw = llm(
            (
                "You are AFlow's graph optimizer. Expand the selected workflow "
                "from its execution feedback while preserving exactly two nodes: "
                "Solve, then ReviewAndRevise.\n\n"
                f"Current workflow:\n{rendered}\n\n{history}\n"
                f"Reward: {reward}\n{feedback(task, output)}\n\n"
                "Return JSON only with modification, solve_instruction, and "
                "review_instruction strings."
            ),
            unit=task.id,
        )
        try:
            note = str(parse_json_object(raw).get("modification", ""))[:200]
        except ValueError:
            note = "(unparseable modification)"
        with lock:
            experience.setdefault(rendered, []).append(
                f"{note} (trigger reward {reward})")
        return raw

    train, held_out, test = money_splits(seed)
    return MethodPolicy(
        name="aflow",
        fidelity=FIDELITY,
        notes=(
            "The pinned revision's soft mixed selection (not UCT) is declared at the selection seam; the engine's single-head ledger keeps it degenerate until multi-head support lands.",
            "Expansion prompts carry the selected parent's modification experience, as upstream's experience.json does.",
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
        proposal_calls_per_candidate=1,
        engine=Policies(selection=SoftMixed(seed=seed)),
        reflective=True,
    )


def main(argv=None) -> int:
    return standard_main(build, argv)


if __name__ == "__main__":
    raise SystemExit(main())
