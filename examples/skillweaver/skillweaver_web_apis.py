"""Environment analogue: **SkillWeaver** (OSU-NLP-Group/SkillWeaver@f2a63d65).

Preserved, with the paper's own **three-stage** naming: Skill Proposal is the
task pool plus difficulty sampling; Skill Synthesis's *Practice* is the engine
rollout and its *reward model* is the engine's self-verify rollout graded by
the deterministic site; Skill Honing is the proposal call, driven by the
**execution feedback** of running the attempt against the site (first failed
call), never by a required trace. APIs accumulate in a per-page library and
union-merge across parallel workers.

Boundaries: a deterministic settings service replaces Dockerized WebArena;
key-match retrieval replaces the paper's API-doc retrieval.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple

from agentdescent.evolution import Task
from agentdescent.policies import Policies
from agentdescent.sampling import DifficultyWeighted

from examples._measure import canonical_json, parse_json_object
from examples._method_policy import MethodPolicy, SkillLibrary
from examples._method_runner import standard_main


FIDELITY = "environment_analogue"

PAGES = (
    ("/settings/profile", "timezone", "UTC"),
    ("/settings/profile", "language", "English"),
    ("/settings/display", "theme", "dark"),
    ("/settings/display", "density", "compact"),
    ("/settings/account", "region", "EU"),
    ("/settings/account", "currency", "EUR"),
    ("/settings/privacy", "tracking", "off"),
    ("/settings/privacy", "visibility", "private"),
    ("/settings/alerts", "email", "enabled"),
    ("/settings/alerts", "sms", "disabled"),
    ("/settings/accessibility", "contrast", "high"),
    ("/settings/accessibility", "motion", "reduced"),
)

BROWSER_CALLS = "open, wait, fill, click, assert (as 'call:argument')"


def _required(task: Task) -> List[str]:
    return [item.lower() for item in (
        f"open:{task.meta['page']}",
        "wait:hydration-complete",
        f"fill:{task.meta['field']}={task.meta['value']}",
        "click:save",
        "assert:saved-toast",
    )]


def _call_list(text: str, key: str) -> List[str]:
    payload = parse_json_object(text)
    value = payload.get(key)
    if not isinstance(value, list) or not value or len(value) > 12:
        raise ValueError(f"{key} must be a non-empty bounded list")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{key} must contain strings")
    return [item.strip().lower() for item in value]


def simulate(calls: Sequence[str], task: Task) -> Tuple[bool, str]:
    """Run browser calls against the deterministic site; report the failure."""
    required = _required(task)
    cursor = 0
    for call in calls:
        if cursor < len(required) and call == required[cursor]:
            cursor += 1
    if cursor == len(required):
        return True, "saved-toast asserted: setting persisted"
    missing_verb = required[cursor].split(":", 1)[0]
    return False, (
        f"the site did not persist the change: a '{missing_verb}' step the "
        f"page requires never succeeded in order (progress "
        f"{cursor}/{len(required)}). The page hydrates before accepting input "
        f"and confirms with a toast. Available calls: {BROWSER_CALLS}."
    )


def _api_value(text: str) -> str:
    calls = _call_list(text, "calls")
    return canonical_json({"calls": calls})


def _tasks() -> List[Task]:
    return [
        Task(
            id=f"web:{index}",
            prompt=f"set {field} to {value} on {page}",
            meta={"page": page, "field": field, "value": value},
        )
        for index, (page, field, value) in enumerate(PAGES)
    ]


def _split(seed: int) -> Tuple[List[Task], List[Task], List[Task]]:
    rows = _tasks()
    random.Random(seed).shuffle(rows)
    return rows[:4], rows[4:8], rows[8:12]


def _retrieve(rendered: str, page: str) -> str:
    sections = {}
    current = None
    for line in rendered.splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    exact = sections.get(f"api {page}")
    generic = sections.get("api generic")
    chosen = exact if exact is not None else generic
    return "\n".join(chosen) if chosen else "{}"


def build(seed: int) -> MethodPolicy:
    def solve(llm, rendered: str, task: Task) -> str:
        page = str(task.meta["page"])
        return llm(
            (
                "Operate a settings website using the reusable API below. "
                "Instantiate its placeholders for the task. Available calls: "
                f"{BROWSER_CALLS}.\n\n"
                f"Task: {task.prompt}\nReusable API: {_retrieve(rendered, page)}\n\n"
                'Return JSON only as {"calls": ["browser-call", ...]}.'
            ),
            unit=task.id,
        )

    def reward(task: Task, output: str) -> float:
        try:
            calls = _call_list(output, "calls")
        except ValueError:
            return 0.0
        ok, _ = simulate(calls, task)
        return float(ok)

    def propose(llm, rendered: str, task: Task, output: str,
                score: float) -> Optional[str]:
        try:
            calls = _call_list(output, "calls")
            _, site_feedback = simulate(calls, task)
        except ValueError:
            site_feedback = (
                "the site rejected the attempt: it was not a JSON call list. "
                f"Available calls: {BROWSER_CALLS}."
            )
        raw = llm(
            (
                "SkillWeaver HONE: repair the reusable API from the site's "
                "execution feedback. Keep {page}, {field}, and {value} "
                "placeholders generic.\n\n"
                f"Practice task: {task.prompt}\nAttempt: {output[:500]}\n"
                f"Reward: {score}\nSite feedback: {site_feedback}\n\n"
                'Return JSON only as {"calls": ["...", ...]}.'
            ),
            unit=task.id,
        )
        # The reusable placeholder API evolves under the generic key --
        # held-out tasks live on disjoint pages, so only a placeholder
        # API can pass the gate. Per-page keys remain for specialization.
        return f"api generic: {raw}"

    train, held_out, test = _split(seed)
    categories = ["api generic"] + sorted({f"api {p}" for p, _, _ in PAGES})
    return MethodPolicy(
        name="skillweaver",
        fidelity=FIDELITY,
        notes=(
            "The paper's three stages map to: Proposal = task pool + difficulty sampling; Synthesis = rollout + self-verify reward model; Honing = the repair call.",
            "Honing sees the site's execution feedback, never a required trace.",
            "APIs accumulate per page and union-merge across parallel workers.",
            "A deterministic settings service replaces Dockerized WebArena.",
        ),
        strategy=SkillLibrary(
            categories=categories,
            value_validator=_api_value,
            initial_entries={
                "api generic": canonical_json(
                    {"calls": ["open:{page}", "fill:{field}={value}",
                               "click:save"]}),
            },
        ),
        train_tasks=tuple(train),
        held_out_tasks=tuple(held_out),
        test_tasks=tuple(test),
        solve=solve,
        propose=propose,
        reward=reward,
        proposal_calls_per_candidate=1,
        engine=Policies(task_sampler=DifficultyWeighted()),
        reflective=False,
        self_verify=True,
    )


def main(argv=None) -> int:
    return standard_main(build, argv)


if __name__ == "__main__":
    raise SystemExit(main())
