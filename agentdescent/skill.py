"""The shortest path from a dataset to an evolved skill.

Evolving a skill needs three things that are genuinely yours -- your data, how to
score an answer, and which model -- plus a page of boilerplate that is the same
every time: wrapping rows in :class:`~agentdescent.evolution.Task`, writing the
``run`` lambda that puts the skill in front of the question, writing the same
last-number regex, and picking a dozen knobs. :func:`evolve_skill` supplies the
boilerplate and leaves the three real decisions to you.

    from agentdescent import evolve_skill
    from agentdescent.agents import openai_compatible
    from agentdescent.dataloader import hf_rows

    rows = hf_rows("openai/gsm8k", config="main", split="train", limit=64)
    result = evolve_skill(rows, model=openai_compatible(model="deepseek-v4-flash"),
                          prompt="question", gold="answer", score="last_number")
    print(result.rendered)          # the skill it learned

It is a **thin wrapper**, not a different system: it builds ordinary arguments and
calls :func:`~agentdescent.evolution.evolve`, returns the same
:class:`~agentdescent.evolution.EvolutionResult`, and forwards any extra keyword
argument straight through. Reach for `evolve()` directly the moment you want
something it does not express -- nothing here is load-bearing.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Union

from .agents import Completion
from .evolution import (
    Completion as _Completion,  # noqa: F401  (re-exported type for readers)
    EvolutionResult,
    SingleSlot,
    Task,
    evolve,
    reflector,
    tasks_from,
)
from . import rewards as _rewards

__all__ = ["evolve_skill", "SCORERS"]

#: Named scorers accepted by ``score=``. Pass a callable for anything else.
SCORERS = {
    "last_number": _rewards.last_number,
    "exact": _rewards.exact_match,
    "contains": _rewards.contains,
    "numeric_close": _rewards.numeric_close,
}

_DEFAULT_TEMPLATE = "{skill}\n\n{prompt}"


def evolve_skill(
    data: Sequence[Any],
    model: Completion,
    *,
    prompt: str = "prompt",
    gold: str = "gold",
    score: Union[str, Callable] = "last_number",
    instruction: str = "You are a helpful assistant.",
    template: str = _DEFAULT_TEMPLATE,
    reflect_with: Optional[Completion] = None,
    **evolve_kwargs: Any,
) -> EvolutionResult:
    """Evolve one instruction (a "skill") against a dataset, in one call.

    Parameters
    ----------
    data:
        Rows (dicts) from any source, or ready-made :class:`Task` objects. Rows go
        through :func:`~agentdescent.evolution.tasks_from`.
    model:
        The completion your agent uses -- see :mod:`agentdescent.agents`.
    prompt, gold:
        Which columns hold the question and the expected answer. Ignored when
        ``data`` is already Tasks.
    score:
        A name from :data:`SCORERS` or your own ``(task, output) -> float``.
    instruction:
        The starting skill. Everything the run learns replaces this.
    template:
        How the skill meets the question. Must contain ``{skill}`` and
        ``{prompt}`` -- change it to put the skill somewhere else (a suffix, a
        section header, inside a larger scaffold).
    reflect_with:
        The model that proposes improvements. Defaults to ``model``; a cheap
        reflector for an expensive agent is a good trade.
    **evolve_kwargs:
        Passed to :func:`~agentdescent.evolution.evolve` and override the defaults
        chosen here (``asynchronous=True``, a different ``strategy=``, an
        ``aggregator_factory=``, ...). ``shuffle=True`` is worth knowing about:
        rows arrive in dataset order and the train/held-out split is positional,
        so grouped data otherwise holds out one end of the file.

    The defaults it picks -- and why they are defaults, not decisions
    ----------------------------------------------------------------
    Worker count and round count scale with the dataset, and early stopping is on,
    so a small dataset does not buy 15 rounds of nothing. Every one is overridable:

    * ``n_workers = max_concurrency = min(8, len(train tasks))``
    * ``rounds = 8``, ``patience = 3``, ``target_reward = 0.98``
    * ``held_out_frac = 0.3``
    """
    if not data:
        raise ValueError("evolve_skill() got no data")
    if "{skill}" not in template or "{prompt}" not in template:
        raise ValueError(
            f"template must contain {{skill}} and {{prompt}}, got {template!r}")

    tasks = (list(data) if isinstance(data[0], Task)
             else tasks_from(data, prompt=prompt, gold=gold))

    if callable(score):
        reward = score
    elif score in SCORERS:
        reward = SCORERS[score]()
    else:
        raise ValueError(
            f"unknown score={score!r}; use one of {sorted(SCORERS)} or pass a "
            f"callable (task, output) -> float")

    held_out_frac = evolve_kwargs.pop("held_out_frac", 0.3)
    n_train = max(1, int(len(tasks) * (1 - held_out_frac)))
    workers = max(1, min(8, n_train))

    defaults = {
        "rounds": 8,
        "n_workers": workers,
        "max_concurrency": workers,
        "target_reward": 0.98,
        "patience": 3,
    }
    # asynchronous runs shard across n_workers themselves and warn about
    # max_concurrency, so do not hand them a knob they would only reject.
    if evolve_kwargs.get("asynchronous"):
        defaults.pop("max_concurrency")
    for key, value in defaults.items():
        evolve_kwargs.setdefault(key, value)

    def run(rendered: str, task: Task) -> str:
        return model(template.format(skill=rendered, prompt=task.prompt))

    # setdefault, not keyword arguments: an explicit run=/propose=/strategy= from
    # the caller must win, and passing both would raise "multiple values for".
    evolve_kwargs.setdefault("run", run)
    evolve_kwargs.setdefault("propose", reflector(reflect_with or model))
    evolve_kwargs.setdefault("strategy", SingleSlot(initial_value=instruction))
    return evolve(tasks, reward, held_out_frac=held_out_frac, **evolve_kwargs)
