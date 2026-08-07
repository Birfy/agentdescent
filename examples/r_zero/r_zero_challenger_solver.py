"""Inference analogue: **R-Zero** (Chengsong-Huang/R-Zero@5699329d).

Preserved: separate Challenger and Solver roles with **separate updates** (two
plain-text fields, so role edits union-merge and contested roles are
model-merged); the Challenger's uncertainty signal from **repeated solver
samples** -- two solver attempts per generated task give an agreement rate, and
``min(p̂, 1−p̂)`` (upstream's reward, maximal at 50%) is surfaced in the
Challenger update; **GRPO's group-relative shape** at the engine's acceptance
seam via :class:`~agentdescent.advantage.AdvantageAcceptance`; and frontier
targeting via :class:`~agentdescent.sampling.DifficultyWeighted`, whose
``4p(1−p)`` weight shares its peak and zeros with ``min(p̂,1−p̂)`` exactly.

Boundaries: verbal role memories replace two GRPO-trained checkpoints; no
BLEU-cluster repetition penalty; evaluation carts are frozen from the seed.
"""

from __future__ import annotations

from typing import Optional

from agentdescent.advantage import AdvantageAcceptance
from agentdescent.defaults import DefaultAcceptance
from agentdescent.evolution import Task
from agentdescent.policies import Policies
from agentdescent.sampling import DifficultyWeighted

from examples._measure import canonical_json, parse_json_object
from examples._method_policy import (FieldSlots, MethodPolicy, clip_text,
                                     read_fields)
from examples._method_runner import standard_main
from examples._money_domain import STARTING_INSTRUCTION, parse_integer_answer
from examples._selfplay_domain import (CartTask, mixed_reward, proposer_prompt,
                                       selfplay_splits, solver_prompt,
                                       trajectory, validate_generated)


FIDELITY = "inference_analogue"

_FALLBACK = CartTask((125, 240), (1, 2))
CHALLENGER_SEED = "Increase curriculum difficulty gradually."


def build(seed: int) -> MethodPolicy:
    def solve(llm, rendered: str, task: Task) -> str:
        memory = read_fields(rendered)
        solver_memory = memory.get("solver_memory", STARTING_INSTRUCTION)
        if task.meta.get("kind") == "frozen":
            return llm(solver_prompt(task.prompt, solver_memory),
                       subphase="solver", unit=task.id)
        slot = int(task.meta["slot"])
        raw = llm(
            proposer_prompt("R-Zero Challenger", slot,
                            memory.get("challenger_memory", CHALLENGER_SEED)),
            subphase="challenger", unit=task.id)
        valid = True
        try:
            cart = validate_generated(parse_json_object(raw))
        except ValueError:
            cart, valid = _FALLBACK, False
        # Two solver samples give the Challenger its uncertainty signal; the
        # first is the trajectory's answer.
        finals = [
            llm(solver_prompt(cart.question, solver_memory),
                subphase="solver", unit=task.id)
            for _ in range(2)
        ]
        parsed = [parse_integer_answer(f.strip()) for f in finals]
        agreement = float(parsed[0] is not None and parsed[0] == parsed[1])
        record = parse_json_object(trajectory(cart, finals[0], valid=valid))
        record["agreement"] = agreement
        return canonical_json(record)

    def propose(llm, rendered: str, task: Task, output: str,
                score: float) -> Optional[str]:
        try:
            agreement = float(parse_json_object(output).get("agreement", 1.0))
        except ValueError:
            agreement = 1.0
        p_hat = (score + agreement * score) / 2 if score else agreement / 2
        uncertainty = min(p_hat, 1.0 - p_hat)
        challenger = llm(
            (
                "Update only R-Zero's Challenger policy toward valid tasks at "
                "the Solver frontier. Uncertainty reward min(p,1-p) is "
                f"currently {uncertainty:.2f} (maximal when the Solver agrees "
                "with itself half the time). Return one general lesson.\n\n"
                f"Trajectory: {output[:300]}\nSolver reward: {score}"
            ),
            subphase="challenger_update",
            unit=task.id,
        )
        solver = llm(
            (
                "Update only R-Zero's Solver policy from verifier feedback. "
                "State the answer representation and omit item-specific "
                "numbers.\n\n"
                f"Trajectory: {output[:300]}\nReward: {score}"
            ),
            subphase="solver_update",
            unit=task.id,
        )
        return canonical_json(
            {
                "challenger_memory": clip_text(challenger),
                "solver_memory": clip_text(solver),
            }
        )

    train, held_out, test = selfplay_splits(seed, "r_zero")
    return MethodPolicy(
        name="r_zero",
        fidelity=FIDELITY,
        notes=(
            "Separate Challenger and Solver role memories receive separate updates and union-merge across fields.",
            "Two solver samples per generated task give the Challenger upstream's min(p,1-p) uncertainty signal.",
            "AdvantageAcceptance carries GRPO's group-relative shape at the acceptance seam; DifficultyWeighted's 4p(1-p) matches the frontier target.",
            "Verbal role memories replace two GRPO checkpoints; the BLEU repetition penalty is omitted; evaluation carts are frozen.",
        ),
        strategy=FieldSlots(
            fields={"challenger_memory": CHALLENGER_SEED,
                    "solver_memory": STARTING_INSTRUCTION},
            parse=parse_json_object,
        ),
        train_tasks=tuple(train),
        held_out_tasks=tuple(held_out),
        test_tasks=tuple(test),
        solve=solve,
        propose=propose,
        reward=mixed_reward,
        proposal_calls_per_candidate=2,
        engine=Policies(
            task_sampler=DifficultyWeighted(),
            acceptance=AdvantageAcceptance(DefaultAcceptance(0.5, 64, 4000)),
        ),
        reflective=True,
    )


def main(argv=None) -> int:
    return standard_main(build, argv)


if __name__ == "__main__":
    raise SystemExit(main())
