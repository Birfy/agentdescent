"""Mechanism microport: **TextGrad** (zou-group/textgrad@75e912e2).

Ported, and only this: the **prompt ``Variable`` + one textual-gradient step**
of ``evaluation/prompt_optimization.py``. A minibatch's loss is backpropagated
as natural language (``loss.backward()``), then ``TextualGradientDescent.step()``
asks the optimizer model to rewrite the variable from that critique, and the
rewrite is kept only if validation did not get worse
(``run_validation_revert``). The three pieces map onto the engine's seams:

* ``loss.backward()`` -> :func:`backward_prompt`, the base-case prompt of
  ``LLMCall._construct_llm_base_backward_prompt`` (``CONVERSATION_TEMPLATE`` +
  ``CONVERSATION_START_INSTRUCTION_BASE`` + ``OBJECTIVE_INSTRUCTION_BASE`` +
  ``EVALUATE_VARIABLE_INSTRUCTION``), sent under upstream's
  ``BACKWARD_SYSTEM_PROMPT``. This is the *textual gradient*: feedback on the
  variable, explicitly **not** a new version of it.
* ``optimizer.step()`` -> :func:`tgd_prompt`, upstream's ``TGD_PROMPT_PREFIX`` +
  ``CONSTRAINT_PROMPT_ADDITION`` + ``TGD_PROMPT_SUFFIX`` under
  ``OPTIMIZER_SYSTEM_PROMPT``. The new value is whatever sits between
  ``<IMPROVED_VARIABLE>`` tags, as ``new_text.split(tag0)[1].split(tag1)[0]``.
* ``run_validation_revert`` -> the engine's held-out acceptance gate, with the
  stricter rule declared below. No ``aggregator_factory`` is needed.

So a proposal is **two model calls** (``proposal_calls_per_candidate=2``): the
gradient, then the step. The loss TextGrad backpropagates is often free-form
critique; the *score* that gates the port stays a real metric (GSM8K's
last-number match), and the loss text handed to the backward pass is the
evaluator's reading of the attempt, as upstream's ``StringBasedFunction`` output
is for its QA tasks.

Declared departures, each stated rather than implied:

* **The gate is stricter than upstream's revert.** ``run_validation_revert``
  reverts only when ``val_performance < previous``, so a rewrite that *ties* the
  previous validation score is **kept**. The engine's default gate accepts a
  candidate only when the Beta posterior ``P(candidate_rate > base_rate)``
  clears an annealed threshold, and a tie is resolved by rejecting: ties and
  marginal gains are refused, increasingly so as the version grows. Matching
  upstream exactly would take a thin ``aggregator_factory``; this port documents
  the difference instead.
* **Minibatch of one per worker.** Upstream sums ``batch_size=3`` losses before
  one ``backward()``. Here a proposal is one rollout, so each worker's gradient
  is a batch of one and concurrent workers' rewrites meet in the shared
  reflective merge -- the analogue of summing gradients, not the same operator.
* **A correct attempt proposes nothing.** Upstream's backward prompt tells the
  gradient engine to give *no feedback* when the variable "is already working
  well"; a free skip (``None``) is the same rule without a wasted step call.
* **One prompt, no system role.** The completion seam takes one string, so the
  upstream system prompts are prepended to the user turn.
* **The reusability constraint is upstream's own knob.** ``constraints=`` on
  ``TextualGradientDescent`` renders ``CONSTRAINT_PROMPT_ADDITION``; the one used
  here keeps a rewrite from baking the training question's numbers into a prompt
  that is read against other questions.
* **A reply without the tags is an invalid proposal, not a fallback.** Upstream
  raises ``IndexError``; here the candidate is spent and counted, and the
  strategy never substitutes the raw reply.
* Out of scope: multi-module compound systems, ``tg.sum`` over losses,
  gradient memory / momentum, and the molecule / code / solution-optimization
  demos. The package is never imported; this file mirrors the update rule.

The domain is **GSM8K**, which upstream's prompt-optimization task list also
carries. The gold answer reaches the backward prompt only as the evaluator's
"it wanted" line, exactly what the other reflective ports show their proposer;
the worked example in the prompts is invented, never a graded item.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from typing import Optional

from agentdescent.evolution import Task
from agentdescent.evolvable import Diff
from agentdescent.strategies import rule_id

from examples._method_policy import MethodPolicy, ValidatedSlot, clip_text
from examples._method_runner import standard_main
from examples._gsm8k_domain import (SAMPLE_ENV, STARTING_INSTRUCTION, feedback,
                                    gsm8k_reward, gsm8k_splits, solve_gsm8k)


FIDELITY = "mechanism_microport"

UPSTREAM = "zou-group/textgrad@75e912e210864b61999781778cdf756d4468120f"

#: ``optimizer.new_variable_tags`` default.
TAGS = ("<IMPROVED_VARIABLE>", "</IMPROVED_VARIABLE>")

#: The ``role_description`` upstream gives the system-prompt Variable in
#: ``prompt_optimization.py``.
ROLE = ("structured system prompt to a somewhat capable language model that "
        "specifies the behavior and strategies for the QA task")

#: ``textgrad/autograd/llm_backward_prompts.py:BACKWARD_SYSTEM_PROMPT``.
BACKWARD_SYSTEM_PROMPT = (
    "You are part of an optimization system that improves a given text (i.e. the "
    "variable). You are the gradient (feedback) engine. Your only responsibility "
    "is to give intelligent and creative feedback and constructive criticism to "
    "variables, given an objective specified in <OBJECTIVE_FUNCTION> "
    "</OBJECTIVE_FUNCTION> tags. The variables may be solutions to problems, "
    "prompts to language models, code, or any other text-based variable. Pay "
    "attention to the role description of the variable, and the context in which "
    "it is used. You should assume that the variable will be used in a similar "
    "context in the future. Only provide strategies, explanations, and methods to "
    "change in the variable. DO NOT propose a new version of the variable, that "
    "will be the job of the optimizer. Your only job is to send feedback and "
    "criticism (compute 'gradients'). For instance, feedback can be in the form "
    "of 'Since language models have the X failure mode...', 'Adding X can fix "
    "this error because...', 'Removing X can improve the objective function "
    "because...', 'Changing X to Y would fix the mistake ...', that gets at the "
    "downstream objective.\n"
    "If a variable is already working well (e.g. the objective function is "
    "perfect, an evaluation shows the response is accurate), you should not give "
    "feedback."
)

#: ``textgrad/optimizer/optimizer_prompts.py:OPTIMIZER_SYSTEM_PROMPT`` with the
#: tags substituted, as ``TextualGradientDescent.__init__`` does.
OPTIMIZER_SYSTEM_PROMPT = (
    "You are part of an optimization system that improves text (i.e., variable). "
    "You will be asked to creatively and critically improve prompts, solutions to "
    "problems, code, or any other text-based variable. You will receive some "
    "feedback, and use the feedback to improve the variable. The feedback may be "
    "noisy, identify what is important and what is correct. Pay attention to the "
    "role description of the variable, and the context in which it is used. This "
    f"is very important: You MUST give your response by sending the improved "
    f"variable between {TAGS[0]} {{improved variable}} {TAGS[1]} tags. The text "
    "you send between the tags will directly replace the variable."
)

#: ``TextualGradientDescent(constraints=[...])``.
CONSTRAINTS = (
    "The variable is read while solving other problems it has not seen, so it "
    "must not mention this conversation's specific numbers, names or answer.",
    "The variable must remain a general system prompt for solving grade-school "
    "math word problems whose reply ends with the final number.",
)


def backward_prompt(system_prompt: str, task: Task, output: str) -> str:
    """The textual gradient request: ``_construct_llm_base_backward_prompt``."""
    conversation = (
        f"<LM_SYSTEM_PROMPT> {system_prompt} </LM_SYSTEM_PROMPT>\n\n"
        f"<LM_INPUT> {task.prompt} </LM_INPUT>\n\n"
        f"<LM_OUTPUT> {(output or '').strip()} </LM_OUTPUT>\n\n"
        f"<EVALUATION>\n{feedback(task, output)}\n</EVALUATION>\n\n"
    )
    return (
        f"{BACKWARD_SYSTEM_PROMPT}\n\n"
        f"You will give feedback to a variable with the following role: "
        f"<ROLE> {ROLE} </ROLE>. Here is an evaluation of the variable using a "
        f"language model:\n\n{conversation}"
        "<OBJECTIVE_FUNCTION>Your goal is to give feedback and criticism to the "
        "variable given the above evaluation output. Our only goal is to improve "
        "the above metric, and nothing else. </OBJECTIVE_FUNCTION>\n\n"
        f"We are interested in giving feedback to the {ROLE} for this "
        "conversation. Specifically, give feedback to the following span of "
        f"text:\n\n<VARIABLE> {system_prompt} </VARIABLE>\n\n"
        f"Given the above history, describe how the {ROLE} could be improved to "
        "improve the <OBJECTIVE_FUNCTION>. Be very creative, critical, and "
        "intelligent.\n"
    )


def tgd_prompt(system_prompt: str, task: Task, output: str,
               gradient: str) -> str:
    """The update request: ``construct_tgd_prompt`` with constraints on."""
    context = (
        "Here is a conversation:\n\n<CONVERSATION>"
        f"<LM_SYSTEM_PROMPT> {system_prompt} </LM_SYSTEM_PROMPT>\n\n"
        f"<LM_INPUT> {task.prompt} </LM_INPUT>\n\n"
        f"<LM_OUTPUT> {(output or '').strip()} </LM_OUTPUT>\n\n"
        "</CONVERSATION>\n\n"
        "This conversation is potentially part of a larger system. The output is "
        "used as response from a language model\n\n"
        f"Here is the feedback we got for {ROLE} in the conversation:\n\n"
        f"<FEEDBACK>{gradient.strip()}</FEEDBACK>\n\n"
    )
    constraints = "\n".join(f"Constraint {i + 1}: {c}"
                            for i, c in enumerate(CONSTRAINTS))
    return (
        f"{OPTIMIZER_SYSTEM_PROMPT}\n\n"
        f"Here is the role of the variable you will improve: <ROLE>{ROLE}</ROLE>."
        f"\n\nThe variable is the text within the following span: <VARIABLE> "
        f"{system_prompt} </VARIABLE>\n\n"
        f"Here is the context and feedback we got for the variable:\n\n"
        f"<CONTEXT>{context}</CONTEXT>\n\n"
        f"Improve the variable ({ROLE}) using the feedback provided in <FEEDBACK> "
        "tags.\n"
        f"You must follow the following constraints:\n\n"
        f"<CONSTRAINTS>{constraints}</CONSTRAINTS>\n\n"
        f"Send the improved variable in the following format:\n\n"
        f"{TAGS[0]}{{the improved variable}}{TAGS[1]}\n\n"
        f"Send ONLY the improved variable between the {TAGS[0]} tags, and "
        "nothing else."
    )


#: Handed to `to_diff` when there is no gradient to step on. It carries no
#: improved-variable tags, so it is refused and counted like any malformed reply.
NO_GRADIENT = "[textgrad: the gradient engine returned no feedback]"


def extract_improved(reply: str) -> str:
    """``new_text.split(tags[0])[1].split(tags[1])[0].strip()`` -- or ``""``.

    Upstream raises ``IndexError`` when either tag is missing. The empty string
    is this port's version of that: ``TextVariable.to_diff`` refuses it as an
    invalid proposal, which is counted and costs its candidate; the raw reply is
    never used in its place.
    """
    parts = (reply or "").split(TAGS[0])
    if len(parts) < 2:
        return ""
    body, closer, _ = parts[1].partition(TAGS[1])
    return body.strip() if closer else ""


def _instruction(text: str) -> str:
    value = clip_text(text)
    if not value:
        raise ValueError("empty instruction")
    return value


@dataclass
class TextVariable(ValidatedSlot):
    """The prompt ``Variable``: one text slot, rewritten whole by each step.

    Differs from :class:`ValidatedSlot` in two ways that are TextGrad's own.
    A step whose rewrite equals the current value changes nothing, so it is a
    **no-op** -- no diff, counted in ``noop_steps`` -- and the diff id is
    **content-addressed** -- the id hashes the text, not the order it arrived
    in, so one author's repeated critiques to the same text give one id for a
    base version. The id still carries the author, so two *workers* proposing the
    same text are distinct diffs, as for every other slot in this repository.
    """

    noop_steps: int = 0

    def to_diff(self, state, proposal, author, base_version,
                target) -> Optional[Diff]:
        # The proposal is the optimizer's raw reply; the improved span is read
        # here, once, so a reply without the tags is counted as invalid rather
        # than dropped upstream of this method.
        try:
            value = self.validator(extract_improved(proposal or ""))
        except (KeyError, TypeError, ValueError):
            self._rejected()
            return None
        if not value:
            self._rejected()
            return None
        if state.get(self.key) == value:
            with self._lock:
                self.noop_steps += 1
            return None
        return Diff(diff_id=f"{author}:{rule_id(value)}:{base_version}",
                    target=target, ops={self.key: value}, author=author)


@contextlib.contextmanager
def _committed_sample(active: bool):
    """Point the GSM8K domain at its committed 96-row slice for one ``build``.

    Only ``--dry-run`` asks for this, so that inspecting the configuration needs
    no network and no ``pyarrow``. It is scoped to the call and restored, and a
    live run never reaches it: the sample must never stand in for the benchmark
    silently, which is why it is not a fallback on a failed fetch.
    """
    if not active:
        yield
        return
    previous = os.environ.get(SAMPLE_ENV)
    os.environ[SAMPLE_ENV] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(SAMPLE_ENV, None)
        else:
            os.environ[SAMPLE_ENV] = previous


def build(seed: int, *, dry_run: bool = False) -> MethodPolicy:
    """``dry_run=True`` reads the committed GSM8K sample instead of fetching."""

    def solve(llm, rendered: str, task: Task) -> str:
        return solve_gsm8k(llm, rendered, task)

    def propose(llm, rendered: str, task: Task, output: str,
                reward: float) -> Optional[str]:
        # The gradient engine is told to give no feedback to a variable that is
        # already working; skipping is that rule, and costs no budget.
        if reward >= 1.0:
            return None
        gradient = llm(backward_prompt(rendered, task, output),
                       subphase="backward", unit=task.id)
        if not (gradient or "").strip():
            # Nothing to step on. Not "": the proposal adapter drops falsy
            # returns before `to_diff`, which would leave this spent candidate
            # out of `invalid_proposals`. A non-empty reply with no tags reaches
            # `to_diff`, which counts it and produces no diff.
            return NO_GRADIENT
        reply = llm(tgd_prompt(rendered, task, output, gradient),
                    subphase="step", unit=task.id)
        # The raw reply, tags and all: `TextVariable.to_diff` reads the span, so
        # a malformed one is a non-empty proposal that gets counted there.
        return reply or NO_GRADIENT

    with _committed_sample(dry_run):
        train, held_out, test = gsm8k_splits(seed)
    dry_run_note = (
        "dry-run: the splits are the committed GSM8K sample "
        f"({len(train)}/{len(held_out)}/{len(test)}); nothing was fetched. A "
        "live run loads GSM8K and uses 64/64/64.",) if dry_run else ()
    return MethodPolicy(
        name="textgrad",
        fidelity=FIDELITY,
        notes=(
            f"Traced against {UPSTREAM}: evaluation/prompt_optimization.py, textgrad/optimizer/optimizer.py (TextualGradientDescent), optimizer_prompts.py, autograd/llm_backward_prompts.py.",
            "loss.backward() maps to a gradient call (the base-case backward prompt under BACKWARD_SYSTEM_PROMPT); optimizer.step() maps to a TGD call whose rewrite is read from <IMPROVED_VARIABLE> tags; the two are the proposal's two reserved calls.",
            "run_validation_revert maps to the engine's held-out gate, which is stricter: upstream keeps a rewrite that ties the previous validation score (it reverts only on val < previous), while the Beta-posterior gate needs P(candidate > base) above an annealed threshold, so ties and marginal gains are rejected.",
            "The score gating the port is GSM8K's last-number match; the free-form critique is the loss text, not the gate.",
            "A minibatch is one rollout per worker; concurrent workers' rewrites meet in the reflective merge rather than being summed as tg.sum does.",
            "A correct attempt proposes nothing, as the backward system prompt asks the gradient engine to give no feedback to a working variable.",
            "A reply missing the improved-variable tags is an invalid proposal, counted and never substituted; upstream raises IndexError.",
            "Multi-module systems, gradient memory, momentum and the molecule / code / solution demos are out of scope; the textgrad package is never imported.",
        ) + dry_run_note,
        strategy=TextVariable(initial_value=STARTING_INSTRUCTION,
                              validator=_instruction),
        train_tasks=tuple(train),
        held_out_tasks=tuple(held_out),
        test_tasks=tuple(test),
        solve=solve,
        propose=propose,
        reward=gsm8k_reward,
        proposal_calls_per_candidate=2,
        reflective=True,
    )


def main(argv=None) -> int:
    return standard_main(build, argv,
                         build_kwargs=lambda a: {"dry_run": a.dry_run})


if __name__ == "__main__":
    raise SystemExit(main())
