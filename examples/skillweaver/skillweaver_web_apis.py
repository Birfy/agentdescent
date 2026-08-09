"""Environment analogue: **SkillWeaver** (OSU-NLP-Group/SkillWeaver@f2a63d65).

Preserved, with the paper's own **three-stage** naming: Skill Proposal is the
task pool plus difficulty sampling; Skill Synthesis's *Practice* is the engine
rollout and its *reward model* is the engine's self-verify rollout; Skill Honing
is the proposal call, driven by the **execution feedback** of running the attempt
against the site (first failed call), never by a required trace. The knowledge
base is updated **only when the success check passes** -- ``explore.py``:
``if success_check["success"]: await knowledge_base.update(...)`` -- which is the
engine's self-verify plus acceptance gate here.

An API key holds the **latest** accepted call sequence, not an accumulation: the
per-page entry is rewritten when a better one is accepted, as `SkillLibrary`
does. Upstream's knowledge base likewise carries one current definition per
function name and bumps ``metadata["global_version"]``.

Two departures worth naming rather than folding into "a deterministic service
replaces WebArena":

* **The success check is a model upstream and the environment here.**
  ``check_success_simple`` asks a separate LM (``success_check_lm``, gpt-4o) to
  judge the trajectory and a screenshot. A model critic can be wrong in both
  directions; the deterministic site cannot. This port therefore has a *cleaner*
  reward than the paper, not merely a cheaper one, and a skill that would have
  slipped past a model judge does not slip past this.
* **Upstream separates exploring from testing on a schedule.**
  ``_should_perform_test`` alternates explore and test iterations, and
  ``update`` shows the synthesis model only functions with
  ``test_count > 0`` (``is_tested``). Verification is a scheduled phase over the
  library there, and a per-proposal re-roll here.

Boundary: key-match retrieval replaces the paper's API-doc retrieval.
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

#: 48, not 12: `run_port` refuses a run whose train split is smaller than
#: the worker count, so twelve tasks split 4/4/4 capped this port at four
#: workers with a four-task gate.
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
    ("/settings/profile", "pronouns", "they"),
    ("/settings/profile", "handle", "birdy"),
    ("/settings/display", "font", "serif"),
    ("/settings/display", "zoom", "125"),
    ("/settings/account", "plan", "team"),
    ("/settings/account", "seats", "12"),
    ("/settings/privacy", "indexing", "off"),
    ("/settings/privacy", "retention", "30d"),
    ("/settings/alerts", "digest", "weekly"),
    ("/settings/alerts", "quiet-hours", "22-07"),
    ("/settings/accessibility", "captions", "on"),
    ("/settings/accessibility", "cursor", "large"),
    ("/settings/security", "mfa", "totp"),
    ("/settings/security", "session", "8h"),
    ("/settings/security", "recovery", "email"),
    ("/settings/security", "devices", "trusted"),
    ("/settings/billing", "invoice", "pdf"),
    ("/settings/billing", "cycle", "annual"),
    ("/settings/billing", "tax-id", "absent"),
    ("/settings/billing", "receipts", "on"),
    ("/settings/integrations", "webhook", "disabled"),
    ("/settings/integrations", "api-key", "rotated"),
    ("/settings/integrations", "scope", "read"),
    ("/settings/integrations", "callback", "https"),
    ("/settings/editor", "wrap", "on"),
    ("/settings/editor", "tabs", "spaces"),
    ("/settings/editor", "theme", "solarized"),
    ("/settings/editor", "autosave", "5s"),
    ("/settings/search", "history", "off"),
    ("/settings/search", "suggestions", "on"),
    ("/settings/search", "safe", "strict"),
    ("/settings/search", "region", "global"),
    ("/settings/team", "role", "member"),
    ("/settings/team", "invites", "admin-only"),
    ("/settings/team", "visibility", "internal"),
    ("/settings/team", "domain", "verified"),
)

BROWSER_CALLS = "open, wait, fill, click, assert (as 'call:argument')"


#: What each required browser call needs, beyond its verb. The site's message
#: says *"the page hydrates before accepting input and confirms with a toast"* --
#: it hints the concepts and, under string equality, still demanded the exact
#: tokens `wait:hydration-complete` and `assert:saved-toast`. Measured:
#: `wait:hydration`, `wait:page-hydrated`, `assert:toast` and
#: `assert:success-toast` all did the right thing and were refused, as was
#: `fill:timezone = UTC` for the spaces. A verb whose argument the site only
#: gestures at is matched on the verb; a verb whose argument comes from the task
#: must carry it.
_STEPS = (
    ("open", ("page",)),
    ("wait", ()),                 # that it hydrates is hinted; the token is not
    ("fill", ("field", "value")),
    ("click", ()),
    ("assert", ()),               # that it toasts is hinted; the token is not
)


def _required(task: Task) -> List[str]:
    """The canonical rendering of the required sequence, for display and tests."""
    return [item.lower() for item in (
        f"open:{task.meta['page']}",
        "wait:hydration-complete",
        f"fill:{task.meta['field']}={task.meta['value']}",
        "click:save",
        "assert:saved-toast",
    )]


def _matches(call: str, index: int, task: Task) -> bool:
    """Does this call perform the required step at `index`?

    Verb plus *content*. What still has to be discovered is the **sequence** --
    that the page must be waited on before it accepts input, and confirmed after
    it is saved -- which is the API the paper is about.
    """
    verb, _, argument = call.partition(":")
    want_verb, fields = _STEPS[index]
    if verb.strip() != want_verb:
        return False
    return all(str(task.meta[f]).lower() in argument for f in fields)


def _call_list(text: str, key: str) -> List[str]:
    payload = parse_json_object(text)
    value = payload.get(key)
    if not isinstance(value, list) or not value or len(value) > 12:
        raise ValueError(f"{key} must be a non-empty bounded list")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{key} must contain strings")
    return [item.strip().lower() for item in value]


def _diagnose(calls: Sequence[str], cursor: int, task: Task, verb: str) -> str:
    """Which of the three failures this is, without naming what is expected."""
    if any(_matches(call, cursor, task) for call in calls):
        return (f"a '{verb}' step is present and does the right thing, but the "
                f"page reached it after steps it has to follow")
    if any(call.partition(":")[0].strip() == verb for call in calls):
        return (f"a '{verb}' step is present but its argument is not what the "
                f"page acted on")
    return f"a '{verb}' step the page requires never succeeded"


def simulate(calls: Sequence[str], task: Task) -> Tuple[bool, str]:
    """Run browser calls against the deterministic site; report the failure."""
    required = _required(task)
    cursor = 0
    for call in calls:
        if cursor < len(required) and _matches(call, cursor, task):
            cursor += 1
    if cursor == len(required):
        return True, "saved-toast asserted: setting persisted"
    verb = _STEPS[cursor][0]
    # Three outcomes, not two, because they call for three different repairs:
    # the step is missing, or it is written but in the wrong place, or it is
    # written in the right shape with the wrong content. Collapsing the last two
    # sends an agent reordering a call whose *value* is wrong. Telling an agent
    # that wrote a `fill` that no `fill` succeeded makes it write another one --
    # the failure that kept Voyager at 0.000 for three seeds.
    detail = _diagnose(calls, cursor, task, verb)
    return False, (
        f"the site did not persist the change: {detail} (progress "
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
    third = len(rows) // 3
    return rows[:third], rows[third:2 * third], rows[2 * third:3 * third]


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
            "The library is updated only when the success check passes, as knowledge_base.update is called only under success_check['success'].",
            "A page key holds the latest accepted API rather than an accumulation, as upstream keeps one current definition per function name.",
            "The success check is the deterministic site, where upstream asks a separate LM to judge a trajectory and a screenshot -- a cleaner reward than the paper's, not just a cheaper one.",
            "Upstream alternates explore and test iterations on a schedule and synthesises only from tested functions; verification here is a per-proposal re-roll.",
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
