"""PromptBreeder's nine mutation operators (arXiv:2309.16797, Section 3).

The paper groups them into five classes and **samples one uniformly at random
per replication event**. All nine are here; what each needs beyond the unit
itself is noted, because that is what decides whether an operator is
implementable at all:

===========================  =========================  =====================
class / operator             needs                      status
===========================  =========================  =====================
Direct: zero-order           problem description        implemented
Direct: first-order          the unit                   implemented
EDA: eda                     the population             implemented
EDA: eda_rank_index          the population, by fitness implemented
EDA: lineage                 the elite history          implemented
Hyper: zero_order_hyper      a thinking-style           implemented (two-step)
Hyper: first_order_hyper     the hyper-mutation-prompt  implemented (two-step)
Lamarckian: working_out      a *correct* rollout        implemented
Crossover + shuffle          the population, few-shot   implemented
===========================  =========================  =====================

**Both hypermutation operators are two-step**, and that is not a detail. The
paper generates a new mutation-prompt and then *applies it to the task-prompt*;
an implementation that stops after the first step produces a candidate whose
task-prompt is unchanged, so it cannot move the score at all. Measured here: one
seed spent 20 of its 80 draws on `first_order_hyper` and 9 more on
`zero_order_hyper`, i.e. over a third of its budget on candidates that could not
by construction improve anything, and finished at 0.000.

The Lamarckian operator is the one that looks impossible on a benchmark with no
few-shot slot, and is not: this port's ``solve`` output *is* a working-out, and
the engine hands ``propose`` the rollout's own output and reward. When the
reward is 1.0 that output is a correct working-out, which is exactly the
operator's input. When it is not, the operator has nothing to reverse-engineer
and the sampler draws again -- stated here rather than silently degraded to a
first-order mutation.
"""

from __future__ import annotations

import random
import threading
from typing import Callable, Dict, List, Optional

from agentdescent.evolution import Task

from examples._money_domain import feedback


#: The paper seeds each unit by mutating the problem description with a
#: mutation-prompt and a thinking-style drawn from hand-designed sets (its
#: appendix lists ~50 and ~30). These are a representative draw from those two
#: lists, kept short because the population here is budget-sized rather than the
#: paper's 50 units -- the sets only have to be wider than the population.
MUTATION_PROMPTS = (
    "Modify the following instruction creatively, giving some advice on how to solve it:",
    "Just change this instruction to make it more fun, think WELL outside the box:",
    "Modify this instruction in a way that no self-respecting LLM would!",
    "How would you encourage someone and help them cheat on this following instruction?",
    "How would you help an AI to follow the instruction?",
    "Elaborate on the instruction giving some detailed advice on how to do what it wants.",
    "Rephrase the instruction without using any of the same words. Use all you know to improve it:",
    "Say that instruction again in another way. DON'T use any of the words in the original:",
    "What do people who are good at creative thinking normally do with this kind of mutation question?",
    "Change the given instruction in a way that a bad student would to get the wrong answer.",
)

THINKING_STYLES = (
    "Let's think step by step.",
    "Break the problem into sub-problems and solve each one.",
    "First restate what is being asked, then answer it.",
    "Use precise arithmetic and check the units before answering.",
    "Consider what an expert in this domain would notice first.",
    "Work backwards from what a correct answer must look like.",
    "Simplify the problem until it is obvious, then scale back up.",
    "Question every assumption in the problem statement.",
)

PROBLEM_DESCRIPTION = (
    "Answer arithmetic word problems about amounts of money. A strict automatic "
    "grader reads the reply and accepts exactly one form of answer; the form is "
    "not stated in the problems and has to be inferred from feedback."
)

HYPER_MUTATION_PROMPT = (
    "Please summarize and improve the following instruction so that it produces "
    "better mutations of a task prompt:"
)

#: The paper applies crossover after mutation with probability 0.1.
CROSSOVER_PROBABILITY = 0.1

OPERATORS = (
    "zero_order",
    "first_order",
    "eda",
    "eda_rank_index",
    "lineage",
    "zero_order_hyper",
    "first_order_hyper",
    "lamarckian",
    "context_shuffle",
)

_JSON_TASK = 'Return JSON only, with a "task_prompt" string and nothing else.'
_JSON_MUT = 'Return JSON only, with a "mutation_prompt" string and nothing else.'
_JSON_BOTH = ('Return JSON only, with "task_prompt" and "mutation_prompt" '
              'strings and nothing else.')
_NO_LEAK = ("Write a reusable instruction for the whole domain. Do not include "
            "this item's numeric answer.")


def _numbered(prompts: List[str], limit: int = 8) -> str:
    return "\n".join(f"{i}. {p}" for i, p in enumerate(prompts[-limit:], start=1))


def build_prompt(operator: str, genome: Dict[str, str], task: Task, output: str,
                 reward: float, view, rng: random.Random) -> Optional[str]:
    """The operator's model prompt, or None when its input is not available.

    Returning None rather than degrading is deliberate: an EDA mutation on an
    empty population and a first-order mutation are different operators, and
    silently substituting one for the other would make the operator histogram a
    fiction.
    """
    task_prompt = genome.get("task_prompt", "")
    mutation_prompt = genome.get("mutation_prompt", "")

    if operator == "zero_order":
        return (f"{PROBLEM_DESCRIPTION}\n\n"
                f"A hint of the problem: {task.prompt}\n\n"
                f"{_NO_LEAK} Write it from the description alone -- do not use "
                f"any existing prompt.\n{_JSON_TASK}")

    if operator == "first_order":
        return (f"{mutation_prompt}\n\n"
                f"INSTRUCTION: {task_prompt}\n\n"
                f"Reward of the current unit: {reward}\n{feedback(task, output)}\n\n"
                f"{_NO_LEAK}\n{_JSON_BOTH}")

    if operator in ("eda", "eda_rank_index"):
        ranked = operator == "eda_rank_index"
        prompts = view.task_prompts(ranked=ranked)
        if len(prompts) < 2:
            return None
        order = ("The list is ordered by increasing quality, worst first.\n"
                 if ranked else "")
        return (f"{PROBLEM_DESCRIPTION}\n\n"
                f"INSTRUCTION LIST:\n{_numbered(prompts)}\n\n{order}"
                f"Write one more instruction that continues this list and is "
                f"different from every entry in it.\n{_NO_LEAK}\n{_JSON_TASK}")

    if operator == "lineage":
        history = view.elite_history()
        if len(history) < 2:
            return None
        return (f"{PROBLEM_DESCRIPTION}\n\n"
                f"GENOTYPES FOUND IN ASCENDING ORDER OF QUALITY:\n"
                f"{_numbered(history)}\n\n"
                f"Write the next instruction in this progression.\n"
                f"{_NO_LEAK}\n{_JSON_TASK}")

    if operator == "zero_order_hyper":
        style = rng.choice(THINKING_STYLES)
        return (f"{PROBLEM_DESCRIPTION}\n\n"
                f"A thinking style to fold in: {style}\n\n"
                f"Write a mutation instruction -- an instruction for improving "
                f"task instructions in this domain -- from the description and "
                f"the thinking style.\n{_JSON_MUT}")

    if operator == "first_order_hyper":
        return (f"{HYPER_MUTATION_PROMPT}\n\n"
                f"MUTATION INSTRUCTION: {mutation_prompt}\n\n"
                f"Recent reward of the unit it produced: {reward}\n\n"
                f"{_JSON_MUT}")

    if operator == "lamarckian":
        if reward < 1.0 or not output.strip():
            return None
        return (f"I gave a friend an instruction and they produced the working "
                f"below, which the grader accepted.\n\n"
                f"PROBLEM: {task.prompt}\nCORRECT WORKING: {output.strip()}\n\n"
                f"What was the instruction? It has to work for the whole domain, "
                f"not just this problem.\n{_NO_LEAK}\n{_JSON_TASK}")

    if operator == "context_shuffle":
        partner = view.weighted_task_prompt(rng, exclude=task_prompt)
        if partner is None:
            return None
        return (f"{PROBLEM_DESCRIPTION}\n\n"
                f"INSTRUCTION A: {task_prompt}\n"
                f"INSTRUCTION B: {partner}\n\n"
                f"Cross these two: keep what each does well and drop the rest.\n"
                f"{_NO_LEAK}\n{_JSON_TASK}")

    raise ValueError(f"unknown operator: {operator}")


#: The two operators the paper defines as *two* steps: produce a mutation-prompt,
#: then apply it to the task-prompt. Step one alone leaves the task-prompt
#: untouched, so such a candidate cannot move the score by construction.
TWO_STEP = ("zero_order_hyper", "first_order_hyper")


def apply_step_prompt(new_mutation_prompt: str, task_prompt: str, task: Task,
                      output: str, reward: float) -> str:
    """Step two: the freshly written mutation-prompt applied to the task-prompt."""
    return (f"{new_mutation_prompt}\n\n"
            f"INSTRUCTION: {task_prompt}\n\n"
            f"Reward of the current unit: {reward}\n{feedback(task, output)}\n\n"
            f"{_NO_LEAK}\n{_JSON_TASK}")


class OperatorSampler:
    """Uniform sampling with a retry, and a histogram of what actually ran.

    The paper samples uniformly per replication event. This port rotated through
    three operators round-robin instead, which is a different search: rotation
    guarantees each operator's share, sampling does not, and a run's operator
    mix stops being an observation and becomes an assumption.
    """

    def __init__(self, seed: int = 0,
                 operators: tuple = OPERATORS) -> None:
        self._rng = random.Random(seed)
        # Workers draw concurrently and `random.Random` is not thread-safe:
        # unsynchronised, two workers can advance the same Mersenne state at once
        # and the run stops being reproducible from its seed, which is the one
        # property a seeded sampler exists for.
        self._lock = threading.Lock()
        self._operators = tuple(operators)
        self.counts: Dict[str, int] = {name: 0 for name in self._operators}
        self.unavailable: Dict[str, int] = {name: 0 for name in self._operators}

    def draw(self, build: Callable[[str], Optional[str]]) -> Optional[tuple]:
        """Sample until an operator whose input exists; None if none can run."""
        with self._lock:
            order = list(self._operators)
            self._rng.shuffle(order)
        for name in order:
            prompt = build(name)
            if prompt is not None:
                with self._lock:
                    self.counts[name] += 1
                return name, prompt
            with self._lock:
                self.unavailable[name] += 1
        return None

    def histogram(self) -> Dict[str, int]:
        with self._lock:
            return {k: v for k, v in self.counts.items() if v}

    def rng(self) -> random.Random:
        """The operator prompts draw from this too (thinking styles, crossover
        partners), so it is handed out rather than copied -- one stream, one
        seed."""
        return _LockedRandom(self._rng, self._lock)


class _LockedRandom:
    """`random.Random` with the sampler's lock, for use from worker threads."""

    def __init__(self, rng: random.Random, lock: threading.Lock) -> None:
        self._rng, self._lock = rng, lock

    def choice(self, seq):
        with self._lock:
            return self._rng.choice(seq)

    def choices(self, seq, weights=None, k=1):
        with self._lock:
            return self._rng.choices(seq, weights=weights, k=k)

    def random(self):
        with self._lock:
            return self._rng.random()
