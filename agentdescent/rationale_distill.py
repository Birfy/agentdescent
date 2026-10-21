"""Rationale distillation: the artifact learns to carry its own reasoning.

STaR (Self-Taught Reasoner) boots a model by distilling its own successful
reasoning back into it. Every evolution run here produces rollouts -- successes
and near-misses -- and then the artifact dies with the run; nothing distills
*why* it succeeded into the artifact itself. This module does: ``extract_rationale``
turns one rollout into a rule (a success says how it answered, a near-miss says
what it needed); ``distill`` dedupes the rationales; and ``rationale_evolve``
runs the loop -- evolve, roll out, distill the rationales, and **fold them back
into the rendered artifact** so a later session seeds from a value that knows
what it learned -- with a report showing how many rationales the artifact
learned to carry and what it now says about itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .evolution import EvolutionResult, Task, evolve
from .meta import MetaOutcome

__all__ = ["extract_rationale", "distill", "RationaleReport", "rationale_evolve"]


def extract_rationale(task: Any, output: Any, success: bool) -> str:
    """One rollout's rationale as a rule the skill can carry.

    A success says what it answered, a near-miss what it failed with --
    **domain-neutral**, so the text never assumes a cap/difficulty model and a
    near-miss at a solved difficulty is not off by one. Pass a custom
    ``rationale=`` to :func:`distill` / :func:`rationale_evolve` for a
    domain-specific extractor."""
    if success:
        return f"answered {output!r} on {task.prompt}"
    return f"failed with {output!r} on {task.prompt}"


def distill(
    records: Sequence[Tuple[Task, Any, bool]],
    *,
    rationale: Callable[[Task, Any, bool], str] = extract_rationale,
) -> List[str]:
    """The rationales the artifact will carry: deduped rules from the rollouts,
    near-miss rationales included -- they are the artifact's own notes about
    what it still needs. Dedup is on the **exact rationale text**, so the
    output value stays semantic content (``-1`` and ``1``, ``A`` and ``a`` are
    different rules, not formatting to normalise away)."""
    out: List[str] = []
    seen: set = set()
    for task, output, success in records:
        rule = rationale(task, output, success)
        if rule in seen:
            continue
        seen.add(rule)
        out.append(rule)
    return out


@dataclass(frozen=True)
class RationaleReport:
    """What :func:`rationale_evolve` distilled: how many rationales the skill
    now carries, and the skill itself."""

    skill: str
    n_rationales: int
    successes: int


def rationale_evolve(
    problems: Mapping[str, Callable],
    *,
    run: Callable[[str, Task], Any],
    reward: Callable[[Task, Any], float],
    propose: Callable[[str, Task, Any, float], Optional[str]],
    seed_skill: str = "cap:0",
    solved: float = 0.999,
    rounds: int = 3,
    n_workers: int = 1,
    rationale: Callable[[Task, Any, bool], str] = extract_rationale,
    **evolve_kwargs: Any,
) -> Tuple[RationaleReport, EvolutionResult]:
    """Evolve the skill, roll out, distill its own successes back into it.

    Runs ``evolve()`` with the skill as the artifact; then every task is scored
    once with the final value, each rollout is classified by ``reward`` against
    ``solved``, its rationale is extracted (via ``rationale``, domain-neutral
    by default), and the distilled notes are **folded into the rendered
    artifact** (and its state) -- an existing notes block is replaced, not
    duplicated, so re-seeding from a previous result stays clean. The report
    shows the distilled skill, the rationale count and the success count.
    """
    from .strategies import SingleSlot

    if len(problems) < 4:
        raise ValueError("rationale_evolve() needs at least 4 problems to "
                         "split train/held-out")
    tasks = list(problems.values())
    strategy = SingleSlot(initial_value=seed_skill)

    def run_task(rendered: str, task: Task) -> Any:
        return run(rendered, task)

    def reward_fn(task: Task, output: Any) -> float:
        return float(reward(task, output))

    result = evolve(
        tasks, reward_fn, run=run_task, propose=propose, strategy=strategy,
        rounds=rounds, n_workers=n_workers, **evolve_kwargs)
    records: List[Tuple[Task, Any, bool]] = []
    for task in tasks:
        output = run_task(result.rendered or seed_skill, task)
        records.append((task, output, float(reward(task, output)) >= solved))
    rationales = distill(records, rationale=rationale)
    notes = "# What I learned from my own runs\n" + "".join(
        f"- {r}\n" for r in rationales) if rationales else \
        "# What I learned from my own runs\n(no rationales yet)"
    # fold the distilled notes into the rendered artifact, replacing any
    # earlier notes block so a re-seed does not accumulate duplicates
    carried_source = (result.rendered or seed_skill).strip()
    if "# What I learned" in carried_source:
        carried_source = carried_source.split("# What I learned", 1)[0].rstrip()
    carried = f"{carried_source}\n\n{notes}" if carried_source else notes
    result.rendered = carried
    result.state = {"value": carried}
    report = RationaleReport(
        skill=notes, n_rationales=len(rationales),
        successes=sum(1 for _, _, s in records if s))
    try:
        result.meta = {"n_rationales": len(rationales),
                       "successes": report.successes}
    except AttributeError:
        pass
    return report, result
