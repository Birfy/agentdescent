"""The parent's review of what a child returned -- upstream's other half.

``agents/manager.ex`` states the parent's validation as three things, in this order:

    **Validation.** Review subagent results. Run tests to validate changes. Check
    for code quality: duplicated code (copy-paste instead of reusing existing
    helpers), defensive code that silently swallows errors (empty catch blocks
    returning defaults -- these create impossible-to-debug silent failures), and
    missing test coverage. Reject work that introduces these anti-patterns.

and the architect's third phase is the same shape: "Run `cargo build` and tests;
review the implementation. Delegate fixes/refinements to `subagent_manager` if
needed."

This port had only the middle one. :func:`examples.genesis._suite.Suite.review`
runs the tests and rejects a regression, which is a number; *reading the change* was
missing, and the cost of missing it is on the record. A run returned a registry whose
minimum-image expression bound ``// 1`` after the multiplication, so every pair in a
periodic box fell past the cutoff and the force field was identically zero. The tests
could not see it -- zero satisfies every invariant they asserted -- and the run
reported 1.000. A reviewer *reading* that function has a fair chance: it is four
lines, and the question "does this compute anything" is one a reader asks and a
pass-count cannot.

So: one model call per returned child, asking upstream's questions about the diff.
The verdict is the parent's, as everywhere else in this port -- a rejection sends the
work back with a reason and the next round re-delegates from it.
"""

from __future__ import annotations

import json
import re
from typing import Callable, Mapping, Optional, Sequence, Tuple

from ._delegation import Edit
from ._world import normalise

__all__ = ["CODE_REVIEW_PROMPT", "COMPLETE_TASK_PROMPT", "CompletionJudge",
           "ParentCodeReview", "chain_reviews"]

#: Upstream's questions, and nothing else. "Missing test coverage" is in
#: ``manager.ex``'s list and is left out here on purpose: in this port the suite is
#: frozen and the agents cannot write tests, so asking would invite a rejection for
#: something no child is allowed to fix.
CODE_REVIEW_PROMPT = """You are the manager agent at `{path}`, reviewing work a child \
agent at `{child}` just returned. You did not write this and you are accountable for it.

{context}

The child was asked to: {objective}

It returned these files. This is the whole of each one:

{diff}

Review it the way a responsible owner does, and answer ONLY these questions:

1. **Does it actually compute something?** A function that returns a constant, an
   empty list, zeros, or its input unchanged where real work was required is the
   failure that matters most -- a suite of invariants cannot see it, and a reader can.
2. **Is it a stub or a placeholder?** `pass`, `NotImplementedError`, `TODO`, a
   docstring with no body, a hard-coded answer for one case.
3. **Does it silently swallow errors?** An empty `except` returning a default, or
   `if x is None: return 0`. A silent failure is worse than a crash.
4. **Does it duplicate logic that already exists** in the files listed above it,
   instead of calling it?
5. **Does it contradict the contract** you were given -- a signature, a parameter, a
   documented formula?

Reply with ONE JSON object and nothing else:
{{"verdict": "accept" | "reject", "reason": "<one sentence, empty when accepting>"}}

Reject only for something you can point at in the text above. A change that is
incomplete is **not** grounds for rejection -- partial progress is accepted, and the
next round continues from it. Wrong, fake, or silently broken is."""


class ParentCodeReview:
    """``review(parent, returned)`` -- the parent reads the diff before accepting.

    Installs where the integration check does, so a domain can have either or both;
    :func:`chain_reviews` runs the cheap deterministic one first.
    """

    def __init__(self, complete, *, contracts: Sequence[str] = (),
                 max_chars: int = 12_000):
        self._complete = complete
        self._contracts = tuple(contracts)
        self._max_chars = max_chars
        #: How many children were read, and how many of them were sent back.
        self.reviewed = 0
        self.rejected = 0
        self.unparsed = 0

    def __call__(self, parent, returned: Sequence[Edit]) -> Optional[Tuple[str, str]]:
        work = [e for e in returned if e.kind == "work" and e.content is not None]
        if not work:
            return None                      # bookkeeping only: nothing to read
        prompt = CODE_REVIEW_PROMPT.format(
            path=normalise(parent.world.path) or "./",
            child=_child_of(work),
            context=parent.world.situate(parent.state, contracts=self._contracts),
            objective=parent.objective,
            diff=_render(work, self._max_chars))
        self.reviewed += 1
        try:
            reply = self._complete(prompt) or ""
        except Exception:  # noqa: BLE001 - a dead call must not reject the work
            return None
        verdict, reason = _parse(reply)
        if verdict != "reject":
            # An unparseable reply is not a rejection. The child did the work; a
            # reviewer that cannot speak is not evidence against it.
            self.unparsed += int(verdict is None)
            return None
        self.rejected += 1
        return ("rejected", f"parent review: {reason or 'unspecified'}")


def chain_reviews(*reviews) -> Optional[Callable]:
    """Run each review in order and return the first refusal.

    Tests before the reviewer, because the tests are free and deterministic and a
    regression needs no second opinion. ``None`` entries are skipped, so a caller can
    pass a disabled review without branching.
    """
    active = [r for r in reviews if r is not None]
    if not active:
        return None
    if len(active) == 1:
        return active[0]

    def review(parent, returned):
        for one in active:
            verdict = one(parent, returned)
            if verdict is not None:
                return verdict
        return None

    return review


def _child_of(work: Sequence[Edit]) -> str:
    owners = {normalise(e.owner) for e in work}
    return ", ".join(sorted(owners)) or "./"


def _render(work: Sequence[Edit], budget: int) -> str:
    """Whole files, not a patch: a reviewer judging four lines of arithmetic needs
    the arithmetic, and the edits are whole files everywhere else in this port."""
    out, spent = [], 0
    for edit in work:
        body = edit.content or ""
        room = max(0, budget - spent)
        if not room:
            out.append(f"# {edit.path}\n(not shown: the review budget is spent)")
            continue
        shown = body if len(body) <= room else body[:room] + "\n... [truncated] ..."
        spent += len(shown)
        out.append(f"# {edit.path}\n```\n{shown}\n```")
    return "\n\n".join(out)


def _parse(reply: str) -> Tuple[Optional[str], str]:
    """``(verdict, reason)``; ``(None, "")`` when the reply is not usable."""
    match = re.search(r"\{.*\}", reply, re.S)
    if not match:
        return None, ""
    try:
        data = json.loads(match.group(0))
    except Exception:  # noqa: BLE001 - malformed model output, not a bug
        return None, ""
    if not isinstance(data, Mapping):
        return None, ""
    verdict = str(data.get("verdict", "")).strip().lower()
    reason = " ".join(str(data.get("reason", "")).split())[:240]
    if verdict not in ("accept", "reject"):
        return None, reason
    return verdict, reason


#: What the root agent is asked, once the tests it can see are all passing. Upstream's
#: own bar, from ``runtime/genesis.ex``: "Call complete_task only when the codebase is
#: complete, functional, and polished -- when you can confidently say the original
#: objective has been 100% delivered."
COMPLETE_TASK_PROMPT = """You are the root agent of this task, deciding whether it is \
finished.

{context}

The original objective:
{objective}

The given test suite is the definition of done for your work, and **all {total} of \
its tests now pass**. That is necessary and it is not sufficient: a suite passes on \
stubs if the stubs are shaped right.

Look at what is in the tree above and answer for the codebase, not for the suite:

1. Is every part the objective asked for actually implemented, with real working code
   rather than a stub, a placeholder, or a hard-coded answer?
2. Is anything there that should not be -- a dead module shadowed by another, a file
   that cannot be imported, a leftover scaffold?
3. Would you hand this to someone as finished?

Reply with ONE JSON object and nothing else:
{{"complete": true | false, "reason": "<one sentence>"}}

`false` is the safe answer and costs only more rounds. Say `true` only when you can \
say the objective has been 100% delivered."""


class CompletionJudge:
    """``stop_when(info)`` -- the run ends when the root agent says it is done.

    Upstream nothing watches a number. An agent decides its objective is met and calls
    ``complete_task``; the root agent's completion ends the phase, and a human merges
    or rejects afterwards on the dashboard. This port ends when ``evolve()``'s round
    budget ends, which is why a finished domain still reports ``stop reason: rounds``
    -- nobody decided it was done.

    So: ask. Two gates, in upstream's order, and the cheap one first.

    * **The tests it can see must all pass.** "When tests are given to guide
      development, aim for a 100% pass rate on the given test suites if possible --
      treat them as the definition of done for your work" (``runtime/genesis.ex``).
      Below that, the question is not asked and no call is spent.
    * **Then the root agent judges the codebase**, not the suite. A suite passes on
      stubs if the stubs are shaped right, and this port has measured exactly that
      twice -- a shadowed lexer and an unimportable module on stackvm, a zero force
      field on md.

    The held-out reward is deliberately **not** shown to it: that number is this
    port's measurement, it is computed from tests no agent may see, and handing it
    over would make the decision a threshold again.
    """

    def __init__(self, complete, *, tasks, run, reward, state_of,
                 suite_failures=None, contracts: Sequence[str] = (),
                 root_path: str = "", objective: str = ""):
        self._complete = complete
        #: ``state -> [failure]``, the whole suite in one process. Preferred when the
        #: domain has one, because that is what `mix test` is and what "run ALL tests"
        #: means: a per-test score cannot see one test poisoning the next.
        self._suite_failures = suite_failures
        self._driven = [t for t in tasks if not t.meta.get("audit")]
        self._run = run
        self._reward = reward
        self._state_of = state_of
        self._contracts = tuple(contracts)
        self._root = root_path
        self._objective = objective
        #: The last version this judge scored, and what it concluded about it. A round
        #: that accepted nothing is the same codebase, and re-running fifty tests on it
        #: would cost more than the rounds it is watching.
        self._scored = None
        self._answer = False
        #: Rounds where the suite was green and the question was therefore asked.
        self.asked = 0
        #: What it said, the last time it said anything.
        self.verdict = ""
        self.reason = ""

    def __call__(self, info) -> bool:
        state = self._state_of()
        if state is None:
            return False
        from agentdescent.filetree import canonical
        from ._world import LocalWorld

        rendered = canonical(state)
        if rendered == self._scored:
            return self._answer              # nothing was accepted: the same codebase
        self._scored, self._answer = rendered, False
        if self._suite_failures is not None:
            if self._suite_failures(state):
                return False                 # not the question yet -- and no call
        else:
            for task in self._driven:
                if self._reward(task, self._run(rendered, task)) < 1.0:
                    return False
        self.asked += 1
        world = LocalWorld(version=int(getattr(info, "round", 0)), path=self._root)
        try:
            reply = self._complete(COMPLETE_TASK_PROMPT.format(
                context=world.situate(state, contracts=self._contracts),
                objective=self._objective, total=len(self._driven))) or ""
        except Exception:  # noqa: BLE001 - a dead call is not a completion
            return False
        verdict, reason = _parse_flag(reply, "complete")
        self.verdict, self.reason = ("complete" if verdict else
                                     ("incomplete" if verdict is not None else
                                      "unparsed")), reason
        self._answer = bool(verdict)
        return self._answer


def _parse_flag(reply: str, key: str) -> Tuple[Optional[bool], str]:
    match = re.search(r"\{.*\}", reply, re.S)
    if not match:
        return None, ""
    try:
        data = json.loads(match.group(0))
    except Exception:  # noqa: BLE001 - malformed model output, not a bug
        return None, ""
    if not isinstance(data, Mapping) or not isinstance(data.get(key), bool):
        return None, " ".join(str(data.get("reason", "")).split())[:240]
    return bool(data[key]), " ".join(str(data.get("reason", "")).split())[:240]
