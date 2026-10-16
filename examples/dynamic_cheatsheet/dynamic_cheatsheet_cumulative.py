"""Mechanism microport: **Dynamic Cheatsheet, cumulative mode (DC-Cu)**.

Upstream traced: ``suzgunmirac/dynamic-cheatsheet@5cfe3c3`` (arXiv:2504.07952),
``language_model.py`` ``approach_name == "DynamicCheatsheet_Cumulative"`` and the
two prompt files ``prompts/generator_prompt.txt`` /
``prompts/curator_prompt_for_dc_cumulative.txt``.

The mechanism, per test query, in order:

1. **Gen** -- answer the query with the current cheatsheet ``M_i`` in context.
2. **Cur** -- a second completion reads ``(M_i, x_i, y_i)`` and writes the whole
   next cheatsheet ``M_{i+1}``. It sees the question and *the model's own
   answer*; it never sees a gold label and never sees the grader's verdict.
   Usefulness is the model's own judgement, which is the point of test-time
   learning without labels.

Preserved: one persistent cheatsheet string, rewritten wholesale by the curator
(no append-only deltas -- that is exactly what ACE adds); the cheatsheet starts
as ``(empty)``; the curator prompt carries no gold and no reward; the generator
prompt always contains the current cheatsheet; the Game-of-24 domain upstream
reports its largest gains on.

Deliberate substitutions, stated as such:

* **No code execution.** Upstream lets the generator emit ``EXECUTE CODE!`` and
  runs it. This microport is offline, so code snippets live in the cheatsheet
  as text and the generator cannot run them.
* **Unparseable curator output is refused, not substituted.** Upstream's
  ``extract_cheatsheet`` returns the *old* cheatsheet when the
  ``<cheatsheet>`` block is missing. The porting policy forbids fallback
  substitution, so such a proposal costs its candidate, is counted invalid and
  produces no diff.
* **Online learning vs. the held-out gate.** DC-Cu is not hill-climbing on a
  validation split: every curator rewrite is kept. The default therefore
  installs :class:`CommitEveryUpdate`; ``--gated`` swaps in the framework's
  strict held-out gate for an ACE-like offline mode, where a rewrite that
  drops useful content is rejected.
* **Skill layer, not L1.** The cheatsheet declares ``blast_radius=0.2`` (ACE's
  value). The shared runner's default 0.6 puts an artifact in the L1 layer,
  where the runtime audits every commit and vetoes any that did not raise
  held-out reward *in front of* the acceptance policy -- which would make
  "keep every rewrite" gated anyway.
* **Parallel curators.** Upstream is one sequential stream. Under
  ``evolve()`` several workers rewrite the same cheatsheet from the same base;
  those contradictions are merged by the shared reflective merge (one model
  call). That is the only place the runner's merge machinery departs from the
  strictly sequential upstream.

Out of scope: DC-RS / DC-CumulativeRetrieval (top-k retrieval needs an embedding
seam), and the AIME/GPQA datasets. ACE (``examples/ace``) is the structured
successor to this mechanism.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from agentdescent.evolution import Task
from agentdescent.evolvable import Diff
from agentdescent.policies import AcceptDecision, Policies

from examples._method_policy import MethodPolicy, ValidatedSlot
from examples._method_runner import standard_main
from examples.dynamic_cheatsheet.game24 import (TARGET, game24_reward,
                                                game24_splits)


FIDELITY = "mechanism_microport"

#: Upstream's initial cheatsheet (``cheatsheet = "(empty)"`` in run_benchmark.py).
INITIAL_CHEATSHEET = "(empty)"

#: Hard guard against a runaway completion; upstream sets no cap (its prompt asks
#: for circa 2000-2500 words, roughly 15k characters).
MAX_CHEATSHEET_CHARS = 20_000

#: Upstream ``prompts/generator_prompt.txt``, condensed: same sections, same
#: ``FINAL ANSWER: <answer>`` contract, minus the code-execution protocol this
#: offline port has no sandbox for. ``[[CHEATSHEET]]`` / ``[[QUESTION]]`` are
#: upstream's own placeholders.
GENERATOR_TEMPLATE = """# GENERATOR (PROBLEM SOLVER)

Instruction: You are an expert problem-solving assistant tasked with analyzing and solving various questions using a combination of your expertise and provided reference materials. Each task will include:
1. A specific question or problem to solve
2. A cheatsheet containing relevant strategies, patterns, and examples from similar problems

## 1. ANALYSIS & STRATEGY
- Carefully analyze both the question and cheatsheet before starting
- Search for and identify any applicable patterns, strategies, or examples within the cheatsheet
- Create a structured approach to solving the problem at hand

## 2. SOLUTION DEVELOPMENT
- Present your solution using clear, logical steps that others can follow and review
- Check and verify all assumptions and intermediate calculations

## 3. FINAL ANSWER FORMAT
ALWAYS present your final answer in the following format:

FINAL ANSWER:
<answer>
(final answer)
</answer>

N.B. Make sure that the final answer is properly wrapped inside the <answer> block.

-----

CHEATSHEET:
'''
[[CHEATSHEET]]
'''

-----
-----

Now it is time to solve the following question.

CURRENT INPUT:
'''
[[QUESTION]]
'''"""

#: The domain's task statement, upstream's ``PREDEFINED_PROMPTS["GameOf24"]``.
GAME24_PROMPT = (
    "Let's play a game called 24. You'll be given four integers, and your "
    "objective is to use each number only once, combined with any of the four "
    "arithmetic operations (addition, subtraction, multiplication, and "
    "division) and parentheses, to achieve a total of 24. For example, if the "
    "input is 4, 7, 8, and 8, the output could be (7 - (8 / 8)) * 4 = 24. "
    "Please present a single expression that evaluates to 24."
)

#: Upstream ``prompts/curator_prompt_for_dc_cumulative.txt``, condensed to its
#: purpose, responsibilities, memory-item format and the "copy forward what you
#: want to keep" warning. **There is deliberately no placeholder for a gold
#: answer or a reward** -- upstream has none, and
#: ``test_the_curator_template_has_no_gold_or_reward_slot`` pins that.
CURATOR_TEMPLATE = """# CHEATSHEET REFERENCE CURATOR

#### 1. Purpose and Goals
As the Cheatsheet Curator, you are tasked with creating a continuously evolving reference designed to help solve a wide variety of tasks. The cheatsheet's purpose is to consolidate verified solutions, reusable strategies, and critical insights into a single, well-structured resource. After seeing each input, you should improve the content of the cheatsheet, synthesizing lessons, insights, tricks, and errors learned from past problems and adapting to new challenges.

#### 2. Core Responsibilities
- Curate and preserve knowledge: select and document only the most relevant, most useful, and most actionable solutions and strategies, while preserving old content of the cheatsheet.
- Maintain accuracy: ensure that all entries are accurate, clear, and well-contextualized.
- Refine and update content: incorporate new insights, remove repetitions or trivial information.
Before updating the cheatsheet, first assess the correctness of the provided solution and strategically incorporate code blocks, insights, and solutions into the new cheatsheet. You are not told whether the solution is right: judge it yourself.

#### 3. Memory item format
<memory_item>
<description>
[Briefly describe the problem context, purpose, and key aspects of the solution.] (Reference: Q1, Q2, etc.)
</description>
<example>
[A well-documented snippet, worked-out solution, or efficient strategy.]
</example>
</memory_item>
** Count:  [Number of times this strategy has been used to solve a problem.]

N.B. Keep in mind that once the cheatsheet is updated, any previous content not directly included will be lost and cannot be retrieved. Therefore, make sure to explicitly copy any (or all) relevant information from the previous cheatsheet to the new cheatsheet!!!

#### 4. Cheatsheet Template
NEW CHEATSHEET:
```
<cheatsheet>

Version: [Version Number]

SOLUTIONS, IMPLEMENTATION PATTERNS, AND CODE SNIPPETS
<memory_item>
[...]
</memory_item>

GENERAL META-REASONING STRATEGIES
<memory_item>
[...]
</memory_item>

</cheatsheet>
```

N.B. Make sure that all information related to the cheatsheet is wrapped inside the <cheatsheet> block.

-----
-----

## PREVIOUS CHEATSHEET

[[PREVIOUS_CHEATSHEET]]

-----
-----

## CURRENT INPUT

[[QUESTION]]

-----
-----

## MODEL ANSWER TO THE CURRENT INPUT

[[MODEL_ANSWER]]"""

_PLACEHOLDER = re.compile(r"\[\[([A-Z_]+)\]\]")
_CHEATSHEET_BLOCK = re.compile(r"<cheatsheet>(.*?)</cheatsheet>",
                               re.DOTALL | re.IGNORECASE)
_TAGS = re.compile(r"</?cheatsheet>", re.IGNORECASE)


def fill(template: str, **values: str) -> str:
    """Substitute ``[[NAME]]`` placeholders in ONE pass.

    Upstream chains ``str.replace``, so a model answer that happens to contain
    ``[[PREVIOUS_CHEATSHEET]]`` is itself substituted. One regex pass substitutes
    only the template's own slots and never re-scans inserted text.
    """
    return _PLACEHOLDER.sub(lambda m: values.get(m.group(1), m.group(0)),
                            template)


def generator_prompt(cheatsheet: str, task: Task) -> str:
    question = f"{GAME24_PROMPT}\n\nInput: {task.prompt}"
    return fill(GENERATOR_TEMPLATE, CHEATSHEET=cheatsheet, QUESTION=question)


def curator_prompt(cheatsheet: str, task: Task, output: str) -> str:
    """The curator's whole view: previous memory, the question, the model's own
    answer. No gold, no reward, no grader verdict."""
    question = f"{GAME24_PROMPT}\n\nInput: {task.prompt}"
    return fill(CURATOR_TEMPLATE, PREVIOUS_CHEATSHEET=cheatsheet,
                QUESTION=question, MODEL_ANSWER=output)


def extract_cheatsheet(response: str) -> str:
    """The text of the last ``<cheatsheet>`` block, strictly.

    Upstream falls back to the old cheatsheet when the block is absent; here a
    missing block is a ``ValueError`` so the proposal is counted invalid.
    """
    blocks = _CHEATSHEET_BLOCK.findall(response or "")
    if not blocks:
        raise ValueError("curator reply has no <cheatsheet> block")
    return blocks[-1].strip()


def check_cheatsheet(text: str) -> str:
    """Shape check shared by curator proposals and reflective-merge output.

    Strips stray ``<cheatsheet>`` tags (a merged text may carry them), refuses an
    empty cheatsheet and a runaway one. Whether the content is *useful* is not
    decided here: that is the held-out gate's job under ``--gated``, and nobody's
    job in online mode, exactly as upstream.
    """
    cleaned = _TAGS.sub("", text or "").strip()
    if not cleaned:
        raise ValueError("empty cheatsheet")
    if len(cleaned) > MAX_CHEATSHEET_CHARS:
        raise ValueError("cheatsheet exceeds the runaway guard")
    return cleaned


@dataclass
class CheatsheetSlot(ValidatedSlot):
    """The cheatsheet: one plain-text value, replaced wholesale by the curator.

    ``to_diff`` is the single validation point. It requires upstream's
    ``<cheatsheet>`` block (strictly -- see :func:`extract_cheatsheet`), then
    hands the inner text to :class:`ValidatedSlot`, which applies
    :func:`check_cheatsheet`. The merge validator is the tag-free
    ``check_cheatsheet`` itself, so a synthesised merge clears the same shape
    check as a proposal that arrived on its own.
    """

    def to_diff(self, state, proposal, author, base_version,
                target) -> Optional[Diff]:
        try:
            inner = extract_cheatsheet(proposal or "")
        except ValueError:
            self._rejected()
            return None
        return super().to_diff(state, inner, author, base_version, target)


class CommitEveryUpdate:
    """DC-Cu's actual rule: the curator's rewrite *is* the new memory.

    Upstream has no acceptance test of any kind -- ``cheatsheet = new_cheatsheet``
    after every query. Shape was already checked in ``to_diff``, so by the time a
    candidate reaches this policy it is exactly upstream's situation, and it is
    accepted unconditionally.
    """

    def accept(self, ctx) -> AcceptDecision:
        return AcceptDecision(accept=True,
                              detail="online: every curator rewrite is kept")


def build(seed: int, *, gated: bool = False) -> MethodPolicy:
    """``gated=False`` (default) is online DC-Cu; ``gated=True`` is the offline,
    ACE-like variant where the framework's held-out gate vets each rewrite."""

    def solve(llm, rendered: str, task: Task) -> str:
        return llm(generator_prompt(rendered, task), unit=task.id)

    def propose(llm, rendered: str, task: Task, output: str,
                reward: float) -> Optional[str]:
        # `reward` is deliberately unused: it is the grader's verdict, and DC's
        # curator judges the answer itself. Upstream curates after every query,
        # right or wrong, so there is no early return either.
        return llm(curator_prompt(rendered, task, output), unit=task.id)

    train, held_out, test = game24_splits(seed)
    return MethodPolicy(
        name="dynamic_cheatsheet",
        fidelity=FIDELITY,
        notes=(
            "DC-Cu (cumulative, no retrieval) traced at suzgunmirac/dynamic-cheatsheet@5cfe3c3: Gen answers with the current cheatsheet in context, Cur rewrites the whole cheatsheet from (memory, question, the model's own answer).",
            "The curator never sees a gold answer or the grader's verdict; its prompt has no slot for either, and the reward argument is not read.",
            "Default is online: every curator rewrite is committed (upstream has no acceptance test). --gated installs the framework's held-out gate, an ACE-like offline mode that can reject a rewrite which drops useful content.",
            "A curator reply without a <cheatsheet> block is refused and counted invalid; upstream's extract_cheatsheet returns the old cheatsheet instead, which is a fallback substitution.",
            "No code execution: upstream's generator can run its own Python via EXECUTE CODE!; snippets here are text in the cheatsheet only.",
            "The cheatsheet is skill-layer context (blast_radius=0.2, as ACE): at the runner's L1 default of 0.6 the runtime's commit audit vetoes any non-improving rewrite regardless of the acceptance policy, so online mode would not be online.",
            "Concurrent curator rewrites of one base are merged by the shared reflective merge; upstream is one sequential stream.",
            "Vendored 36-puzzle Game-of-24 slice replaces upstream's full Game-of-24 set; DC-RS / CumulativeRetrieval (embeddings) and AIME/GPQA are out of scope.",
        ),
        strategy=CheatsheetSlot(initial_value=INITIAL_CHEATSHEET,
                                key="cheatsheet",
                                validator=check_cheatsheet),
        train_tasks=tuple(train),
        held_out_tasks=tuple(held_out),
        test_tasks=tuple(test),
        solve=solve,
        propose=propose,
        reward=game24_reward,
        proposal_calls_per_candidate=1,
        engine=Policies() if gated else Policies(acceptance=CommitEveryUpdate()),
        reflective=True,
        # Skill-layer context, like ACE's playbook. Above 0.3 the runtime audits
        # every commit and vetoes any that did not raise held-out reward, which
        # would make the accept-everything online mode gated by the back door.
        blast_radius=0.2,
    )


def _gated_flag(parser) -> None:
    parser.add_argument(
        "--gated",
        action="store_true",
        help=("offline, ACE-like mode: vet every curator rewrite with the "
              "framework's held-out acceptance gate. The default is online "
              "DC-Cu, which keeps every rewrite as upstream does"))


def main(argv=None) -> int:
    return standard_main(build, argv, extra_args=_gated_flag,
                         build_kwargs=lambda args: {"gated": args.gated})


if __name__ == "__main__":
    raise SystemExit(main())
