"""Mechanism microport: **PromptBreeder** (arXiv:2309.16797, no released code).

Algorithm 1, as written. A population of ``N`` units, each holding a task-prompt
and a mutation-prompt; a binary tournament that samples two units uniformly,
scores both on a **random batch of training data**, mutates the winner with an
operator **sampled uniformly from all nine**, and **replaces the loser** with the
result; the highest-fitness unit is returned.

Three pieces of that were missing and are now here, because each changes the
search rather than the bookkeeping:

* **Replacement.** The population was an archive that only grew, so no unit ever
  died and the selection pressure the replacement step creates was absent.
  :class:`~examples.promptbreeder._promptbreeder_population.PromptBreederPopulation`
  keeps ``|P| = N`` and overwrites the last tournament's loser.
* **Fitness on a random training batch.** The tournament ranked on the held-out
  split -- the acceptance gate's own signal, which makes the tournament a second
  gate rather than a fitness measure. It now scores a random batch of the
  training split, resampled per tournament, as the paper does.
* **All nine operators, sampled uniformly.** Three ran in fixed rotation.
  Rotation guarantees each operator's share and sampling does not, so a run's
  operator mix was an assumption rather than an observation; the mix is now
  reported.

Initialisation is the paper's too: ``N`` units, each the problem description
mutated by a mutation-prompt and a thinking-style drawn from hand-designed sets.

The population lives in the engine's ``aggregator_factory`` seam, and the genome
is a :class:`~examples._method_policy.FieldSlots` artifact, so the two prompts
are separate plain-text ledger keys -- edits to different fields union-merge, and
contested fields are model-merged, which is the paper's *prompt crossover*
arriving through the merge layer.

Boundaries: the population is budget-sized rather than the paper's 50 units and
the fitness batch is 4 items rather than its 100; the unit carries no few-shot
context, so context shuffling is expressed as prompt crossover over the
population rather than over exemplars.
"""

from __future__ import annotations

import json
import random
from typing import Optional, Sequence

from agentdescent.evolution import Task
from agentdescent.policies import Policies
from agentdescent.selection import SelectionContext, SingleHead

from examples._measure import parse_json_object
from examples._method_policy import FieldSlots, MethodPolicy, read_fields
from examples._method_runner import standard_main
from examples._money_domain import (STARTING_INSTRUCTION, money_reward,
                                    money_splits, solve_money)
from examples.promptbreeder._promptbreeder_operators import (
    MUTATION_PROMPTS, PROBLEM_DESCRIPTION, THINKING_STYLES, TWO_STEP,
    OperatorSampler, apply_step_prompt, build_prompt)
from examples.promptbreeder._promptbreeder_population import (
    PopulationView, promptbreeder_population)


FIDELITY = "mechanism_microport"

MUTATION_PROMPT = MUTATION_PROMPTS[0]

#: Units in the population. The paper uses 50; this is budget-sized, and the
#: only constraint that matters is that a tournament has two units to sample.
POPULATION_SIZE = 8

#: Training items per fitness evaluation. The paper evaluates a unit "on a random
#: batch of training data" (its batch is 100 of the full set) rather than on the
#: whole split, and the sampling is part of the algorithm: it is what keeps a
#: tournament a noisy comparison rather than a verdict, and it is the difference
#: between 2 and 2 x |train| model calls per replication event.
FITNESS_BATCH = 4


class BinaryTournament(SingleHead):
    """Retained for the engine's declared-selection check.

    Algorithm 1's tournament needs to *evaluate* both sampled units and *replace*
    the loser, neither of which a `SelectionPolicy` can do -- it is handed
    candidates with cached scores and returns one. The real tournament therefore
    lives in
    :class:`~examples.promptbreeder._promptbreeder_population.PromptBreederPopulation`;
    this class stays because declaring a selection policy is what tells the
    runner to install a population layer at all, and it stays *equivalent* so
    that the two cannot disagree about who wins.
    """

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def select(self, ctx: SelectionContext, n: int) -> Sequence:
        candidates = list(ctx.candidates)
        if len(candidates) <= 1:
            return super().select(ctx, n)
        winners = []
        for _ in range(n):
            pair = self.rng.sample(candidates, 2)
            winners.append(max(
                pair, key=lambda c: (c.score is None, c.score or 0.0)))
        return winners


def seed_population(llm, size: int, seed: int) -> list:
    """The paper's initialisation: ``N`` units from description x style x prompt.

    A run that starts from one instruction has no population to sample for its
    first tournaments, and the EDA and lineage operators -- three of the nine --
    have no input until enough children have committed. That is not a slow start,
    it is a different algorithm for the first third of the budget.
    """
    rng = random.Random(seed)
    units = []
    for _ in range(max(0, size - 1)):
        mutation_prompt = rng.choice(MUTATION_PROMPTS)
        style = rng.choice(THINKING_STYLES)
        reply = llm(
            f"{mutation_prompt}\n\n"
            f"INSTRUCTION: {PROBLEM_DESCRIPTION}\n"
            f"A thinking style to fold in: {style}\n\n"
            "Write a reusable instruction for solving problems in this domain. "
            "Do not invent a specific numeric answer.\n"
            'Return JSON only, with a "task_prompt" string and nothing else.',
            unit="seed-population")
        try:
            task_prompt = str(parse_json_object(reply).get("task_prompt", "")).strip()
        except (KeyError, TypeError, ValueError):
            task_prompt = ""
        if task_prompt:
            units.append({"task_prompt": task_prompt[:900],
                          "mutation_prompt": mutation_prompt})
    return units


def build(seed: int) -> MethodPolicy:
    view = PopulationView()
    sampler = OperatorSampler(seed)

    def solve(llm, rendered: str, task: Task) -> str:
        genome = read_fields(rendered)
        return solve_money(llm, genome.get("task_prompt", STARTING_INSTRUCTION),
                           task)

    def propose(llm, rendered: str, task: Task, output: str,
                reward: float) -> Optional[str]:
        genome = read_fields(rendered)
        genome.setdefault("task_prompt", STARTING_INSTRUCTION)
        genome.setdefault("mutation_prompt", MUTATION_PROMPT)
        drawn = sampler.draw(lambda name: build_prompt(
            name, genome, task, output, reward, view, sampler.rng()))
        if drawn is None:
            return None
        name, prompt = drawn
        reply = llm(prompt, unit=f"{task.id}:{name}")
        if name not in TWO_STEP:
            return reply
        # Hypermutation is two steps in the paper: write a mutation-prompt, then
        # *apply it to the task-prompt*. Stopping after step one returns a
        # candidate whose task-prompt is unchanged, which cannot move the score
        # at all -- one seed spent over a third of its budget on exactly that and
        # finished at 0.000.
        try:
            fresh = str(parse_json_object(reply).get("mutation_prompt", "")).strip()
        except (KeyError, TypeError, ValueError):
            fresh = ""
        if not fresh:
            return reply
        applied = llm(
            apply_step_prompt(fresh, genome["task_prompt"], task, output, reward),
            unit=f"{task.id}:{name}:apply")
        try:
            task_prompt = str(parse_json_object(applied).get("task_prompt", "")).strip()
        except (KeyError, TypeError, ValueError):
            task_prompt = ""
        if not task_prompt:
            return reply
        return json.dumps({"task_prompt": task_prompt, "mutation_prompt": fresh})

    built = []

    def population(ctx):
        return promptbreeder_population(
            ctx, view=view, size=POPULATION_SIZE, batch=FITNESS_BATCH,
            seed_units=seed_population(ctx.llm, POPULATION_SIZE, seed),
            built=built)

    def report() -> str:
        # The operator is *sampled*, so which nine ran is an observation. Saying
        # in the docs that the mix is reported, while printing it nowhere, is a
        # documented property with no code behind it.
        mix = ", ".join(f"{k}={v}" for k, v in sorted(
            sampler.histogram().items(), key=lambda kv: -kv[1]))
        lines = [f"  operators: {mix or 'none drawn'}"]
        skipped = {k: v for k, v in sampler.unavailable.items() if v}
        if skipped:
            lines.append("  declined (input absent): " + ", ".join(
                f"{k}={v}" for k, v in sorted(skipped.items(), key=lambda kv: -kv[1])))
        if built:
            units = built[0].snapshot()
            lines.append(f"  population ({len(units)}/{POPULATION_SIZE}), "
                         f"fitness on a {FITNESS_BATCH}-item train batch:")
            for prompt, score in sorted(units, key=lambda u: -u[1]):
                lines.append(f"    {score:.3f}  {prompt!r}")
        return "\n".join(lines)

    train, held_out, test = money_splits(seed)
    return MethodPolicy(
        name="promptbreeder",
        fidelity=FIDELITY,
        notes=(
            "Algorithm 1 as written: fixed-size population, binary tournament on a random training batch, and the loser replaced by the mutated winner.",
            "All nine mutation operators, sampled uniformly per replication event; the realised mix is reported rather than assumed.",
            "Initialisation is the paper's -- N units from the problem description crossed with hand-designed mutation-prompts and thinking-styles.",
            "Population is budget-sized rather than the paper's 50 units, and the unit carries no few-shot context, so context shuffling is expressed as prompt crossover over the population.",
        ),
        strategy=FieldSlots(
            fields={"task_prompt": STARTING_INSTRUCTION,
                    "mutation_prompt": MUTATION_PROMPT},
            parse=parse_json_object,
        ),
        train_tasks=tuple(train),
        held_out_tasks=tuple(held_out),
        test_tasks=tuple(test),
        solve=solve,
        propose=propose,
        reward=money_reward,
        # Two, not one: the paper's hypermutation operators are two model calls,
        # and declaring one would make the budget check fail on a run that spent
        # exactly what the algorithm asks for.
        proposal_calls_per_candidate=2,
        engine=Policies(selection=BinaryTournament(seed)),
        reflective=True,
        population=population,
        report=report,
    )


def main(argv=None) -> int:
    return standard_main(build, argv)


if __name__ == "__main__":
    raise SystemExit(main())
