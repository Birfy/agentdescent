"""Harness self-evolution, faithful port: **ADAS -- Meta Agent Search**.

Paper : "Automated Design of Agentic Systems", Shengran Hu, Cong Lu, Jeff Clune,
        2024 (arXiv:2408.08435; ICLR 2025).
Repo  : https://github.com/ShengranHu/ADAS
Dataset: **MGSM** (Multilingual Grade-School Math), the light math benchmark
        ADAS searches on (`dataset/mgsm/*.tsv`; Shi et al., 2022).

Where ACE/GEPA evolve a *skill* (an L2 prompt/context), ADAS evolves the
**agentic system itself** -- the control flow that orchestrates the model. That
is a **harness** change: high blast radius -> AgentDescent's **L1** governance
layer (`classify()` below prints the layer). It runs **through `evolve()`** like
ACE/GEPA: an `AgentDesignStrategy` turns a proposed agent (JSON) into a `Diff`,
and a custom `aggregator_factory` (`MetaSearchAggregator`) is the keep-all
archive with bootstrap-CI fitness. Meta Agent Search is the loop:

    1. seed an ARCHIVE with hand-designed building blocks (CoT, Self-Consistency,
       Reflexion, Debate, Step-back, Quality-Diversity, Role-Assignment);
    2. a META-AGENT, conditioned on the *entire archive* (designs + fitness),
       proposes the next agent as a structured program, then does two Reflexion
       refinement rounds;
    3. EVALUATE it on the MGSM validation set; fitness = bootstrap-CI mean;
    4. **keep-all** append to the archive; repeat. Return the best test fitness.

Faithful-but-safe substitution (documented, not hidden): ADAS represents each
agent as a model-written Python `forward()` that it `exec`s. Executing arbitrary
model code is unsafe, so here an agent is a **composable control-flow program**
in a small, validated DSL (`AGENT_BLOCKS`) run by a safe interpreter. The Meta
Agent Search *loop*, the seed archive, MGSM scoring, and keep-all archive are
faithful; only the agent *substrate* is a safe DSL instead of raw `exec`.

Parent conditioning is pluggable: ADAS conditions the meta-agent on the whole
archive; `--select dgm` instead samples which prior agents to surface using the
**Darwin Godel Machine** rule (Zhang et al., 2025, arXiv:2505.22954):
`p_i proportional to sigmoid(10*(score-0.5)) * 1/(1+children)` -- DGM's own
setting (a self-modifying coding agent on SWE-bench) needs Docker and is out of
scope, but its open-ended selection rule ports directly and is unit-tested.

    python -m examples.adas.adas_meta_agent_search --dry-run       # plan only, zero network
    python -m examples.adas.adas_meta_agent_search --model claude-haiku-4-5 --generations 6
    python -m examples.adas.adas_meta_agent_search --select dgm --langs en,es
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from agentdescent.agents import Usage
from agentdescent.aggregator import AggregatorProtocol, MergeOutcome, MergeReport
from agentdescent.dataloader import Dataset, fetch_text, split_dataset
from agentdescent.evolvable import Diff, EvidenceCard
from agentdescent.evolution import EvolvingArtifact, Task, evolve, rule_id
from agentdescent.governance import classify
from agentdescent.ledger import CASConflict, Ledger
from examples._common import (add_standard_args, completion_for, confirm,
                              is_openai_compatible, worker_count,
                              budget_kwargs)

MGSM_URL = "https://raw.githubusercontent.com/ShengranHu/ADAS/main/dataset/mgsm/mgsm_{lang}.tsv"
# ADAS's MGSM language set (utils.ALL_LANGUAGES).
ALL_LANGUAGES = ["bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]


# ===========================================================================
# The agent substrate: a safe, composable control-flow DSL (replaces exec'd code)
# ===========================================================================

Completion = Callable[[str], str]
AGENT_BLOCKS = {"cot", "cot_sc", "reflexion", "debate", "step_back",
                "role_assignment", "ensemble"}


def _extract_int(text: str) -> Optional[str]:
    """Pull the final integer answer out of free-form model output."""
    m = re.search(r"answer\s*(?:is)?\s*[:=]?\s*(-?[\d,]+)", text, re.IGNORECASE)
    if not m:
        nums = re.findall(r"-?[\d,]+", text)
        if not nums:
            return None
        m_val = nums[-1]
    else:
        m_val = m.group(1)
    return m_val.replace(",", "")


def _majority(answers: List[Optional[str]]) -> Optional[str]:
    votes = Counter(a for a in answers if a is not None)
    return votes.most_common(1)[0][0] if votes else None


def canonical(program: dict) -> str:
    """A design's identity: its JSON with **sorted keys**.

    Every dedup in the search is string equality on this -- `to_diff` drops a
    proposal identical to the head, `step()` commits only when the best design
    differs from it, and the evaluation cache keys on it (an artifact's cache key
    is what it renders to, and a design IS what this artifact renders). Plain
    `json.dumps` preserves *insertion* order, which for a proposal is whatever
    key order the model happened to emit -- so `{"block":"cot_sc","k":3}` and
    `{"k":3,"block":"cot_sc"}` were two different designs. The duplicate passed
    the dedup, was scored on every val item (~330 model calls at this size),
    landed the same fitness as the design it duplicated, failed the strict
    `>` test, and showed up as one more `+0/-1`. Sorted, the duplicate is
    recognised for free -- and if one slips through anyway it now hits the
    evaluation cache instead of being re-run."""
    return json.dumps(program, sort_keys=True)


def validate_program(program: dict, depth: int = 0) -> bool:
    """Reject any program that is not built purely from known DSL blocks."""
    if depth > 4 or not isinstance(program, dict):
        return False
    block = program.get("block")
    if block not in AGENT_BLOCKS:
        return False
    if block == "ensemble":
        kids = program.get("children", [])
        return bool(kids) and all(validate_program(c, depth + 1) for c in kids)
    return True


#: Ceiling on model calls per question for a *proposed* program. The DSL nests,
#: so cost is multiplicative: an `ensemble` of five 2-round/4-role `debate`s is 45
#: calls per question, and evaluation is candidates x items x this number. One
#: such proposal costs more than the entire rest of the search, and the meta-agent
#: has no reason not to make it -- "compose more blocks" is exactly what it is
#: asked to do. The most expensive *seed* is 5, so this leaves real headroom.
MAX_PROGRAM_CALLS = 10


def program_cost(program: dict, depth: int = 0) -> int:
    """Model calls this program makes per question -- the unit of the run's budget.

    Used for two things: the pre-run budget estimate, and the cap that keeps one
    proposal from eating the whole search (:data:`MAX_PROGRAM_CALLS`)."""
    if depth > 4 or not isinstance(program, dict):
        return 0
    block = program.get("block")
    if block == "cot":
        return 1
    if block == "cot_sc":
        return max(1, min(int(program.get("k", 3) or 3), Interpreter.max_samples))
    if block == "reflexion":
        return 1 + max(0, int(program.get("n", 1) or 0))
    if block == "step_back":
        return 2
    if block == "debate":
        roles = program.get("roles") or ["a", "b", "c"]
        return len(roles) * max(1, int(program.get("rounds", 1) or 1)) + 1
    if block == "role_assignment":
        return 2
    if block == "ensemble":
        return sum(program_cost(c, depth + 1) for c in program.get("children", []))
    return 0


@dataclass
class Interpreter:
    """Runs a DSL agent program over one MGSM question via LLM calls."""

    complete: Completion
    max_samples: int = 5

    def run(self, program: dict, question: str) -> Optional[str]:
        block = program["block"]
        if block == "cot":
            return self._cot(question)
        if block == "cot_sc":
            k = min(int(program.get("k", 3)), self.max_samples)
            return _majority([self._cot(question) for _ in range(k)])
        if block == "reflexion":
            return self._reflexion(question, int(program.get("n", 1)))
        if block == "step_back":
            return self._step_back(question)
        if block == "debate":
            return self._debate(question, program.get("roles") or
                                ["Math Professor", "Grade School Teacher", "Math Enthusiast"],
                                int(program.get("rounds", 1)))
        if block == "role_assignment":
            return self._role_assignment(question, program.get("roles") or
                                         ["Math Professor", "Grade School Teacher"])
        if block == "ensemble":
            return _majority([self.run(c, question) for c in program["children"]])
        return None

    def _ask(self, prompt: str) -> str:
        return self.complete(prompt)

    def _cot(self, q: str) -> Optional[str]:
        return _extract_int(self._ask(
            f"Solve the math problem. Think step by step, then end with "
            f"`Answer: <integer>`.\n\nProblem: {q}"))

    def _step_back(self, q: str) -> Optional[str]:
        principles = self._ask(f"What general principles/steps solve this kind of "
                               f"problem? Be brief.\n\nProblem: {q}")
        return _extract_int(self._ask(
            f"Using these principles:\n{principles}\n\nNow solve step by step and "
            f"end with `Answer: <integer>`.\n\nProblem: {q}"))

    def _reflexion(self, q: str, n: int) -> Optional[str]:
        answer = self._ask(f"Solve step by step, end with `Answer: <integer>`.\n\n"
                           f"Problem: {q}")
        for _ in range(max(0, n)):
            answer = self._ask(
                f"Problem: {q}\n\nYour previous attempt:\n{answer}\n\nCritique it "
                f"for errors, then give a corrected solution ending with "
                f"`Answer: <integer>`.")
        return _extract_int(answer)

    def _debate(self, q: str, roles: List[str], rounds: int) -> Optional[str]:
        transcript = ""
        for _ in range(max(1, rounds)):
            for role in roles:
                reply = self._ask(
                    f"You are a {role}. Problem: {q}\n\nDebate so far:\n{transcript}\n\n"
                    f"Give your reasoning and a tentative `Answer: <integer>`.")
                transcript += f"\n[{role}] {reply}\n"
        return _extract_int(self._ask(
            f"Given this debate, state the final consensus ending with "
            f"`Answer: <integer>`.\n\n{transcript}"))

    def _role_assignment(self, q: str, roles: List[str]) -> Optional[str]:
        choice = self._ask(f"Which single expert is best for this problem? "
                           f"Choose one of {roles}. Reply with just the name.\n\n{q}")
        role = next((r for r in roles if r.lower() in choice.lower()), roles[0])
        return _extract_int(self._ask(
            f"You are a {role}. Solve step by step, end with `Answer: <integer>`."
            f"\n\nProblem: {q}"))


# ===========================================================================
# The seed archive: ADAS's hand-designed MGSM building blocks (mgsm_prompt.py)
# ===========================================================================


def seed_archive() -> List[dict]:
    """The seven seeds ADAS starts MGSM search from, as safe DSL programs."""
    return [
        {"name": "Chain-of-Thought", "thought": "Step-by-step reasoning.",
         "program": {"block": "cot"}},
        {"name": "Self-Consistency with Chain-of-Thought",
         "thought": "Sample multiple CoT paths and take the majority answer.",
         "program": {"block": "cot_sc", "k": 3}},
        {"name": "Self-Refine (Reflexion)",
         "thought": "Iteratively critique and improve the answer.",
         "program": {"block": "reflexion", "n": 1}},
        {"name": "LLM Debate",
         "thought": "Different roles debate to reach a better answer.",
         "program": {"block": "debate", "rounds": 1}},
        {"name": "Step-back Abstraction",
         "thought": "First derive principles, then solve.",
         "program": {"block": "step_back"}},
        {"name": "Quality-Diversity",
         "thought": "Ensemble diverse strategies and vote.",
         "program": {"block": "ensemble", "children": [
             {"block": "cot"}, {"block": "step_back"}, {"block": "reflexion", "n": 1}]}},
        {"name": "Dynamic Assignment of Roles",
         "thought": "Route the problem to the best expert.",
         "program": {"block": "role_assignment"}},
    ]


# ===========================================================================
# Fitness (ADAS bootstrap_confidence_interval) + selection rules
# ===========================================================================


def bootstrap_ci(correct: List[float], n_resamples: int = 2000, seed: int = 0
                 ) -> Tuple[float, float, float]:
    """ADAS's fitness: mean accuracy with a 95% bootstrap CI (`_mgsm/utils.py`).

    Upstream uses ``num_bootstrap_samples=100000``; 2000 is a deliberate speed
    trade for an example that runs in seconds. The CI half-width it produces
    differs by well under a percentage point at these sample sizes, and the archive
    ranks on the mean either way -- but it *is* a deviation, so it is named here
    rather than left for a reader to diff against the repo."""
    if not correct:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(correct)
    means = []
    for _ in range(n_resamples):
        means.append(sum(correct[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    mean = sum(correct) / n
    lo = means[int(0.025 * n_resamples)]
    hi = means[int(0.975 * n_resamples)]
    return mean, lo, hi


def dgm_parent_weights(scores: List[float], children: List[int]) -> List[float]:
    """Darwin Godel Machine parent-selection weights (DGM_outer.py `score_child_prop`).

    ``p_i proportional to sigmoid(10*(score-0.5)) * 1/(1+children_i)``: favour
    high performers, discount already-explored parents (open-ended novelty)."""
    raw = []
    for s, c in zip(scores, children):
        sig = 1.0 / (1.0 + math.exp(-10.0 * (s - 0.5)))
        nov = 1.0 / (1.0 + c)
        raw.append(sig * nov)
    total = sum(raw) or 1.0
    return [r / total for r in raw]


# ===========================================================================
# Meta Agent Search: the archive-conditioned proposal loop
# ===========================================================================

_META_SYSTEM = (
    "You are an expert machine-learning researcher designing agentic systems. "
    "You compose control-flow building blocks to solve the Multilingual "
    "Grade-School Math benchmark (MGSM). Return a WELL-FORMED JSON object.")

_META_TMPL = """Here is the archive of agents discovered so far (with fitness):

{archive}

You may compose ONLY these building blocks into a `program` tree:
- {{"block": "cot"}}
- {{"block": "cot_sc", "k": <2-5>}}
- {{"block": "reflexion", "n": <1-3>}}
- {{"block": "step_back"}}
- {{"block": "debate", "roles": [<role strings>], "rounds": <1-2>}}
- {{"block": "role_assignment", "roles": [<role strings>]}}
- {{"block": "ensemble", "children": [<two or more programs>]}}

Design the NEXT agent: it should be interesting, novel versus the archive, and
likely to improve fitness. Output ONLY a JSON object with keys "thought",
"name", and "program". The "program" must use only the blocks above.

Two constraints on the design:
- It may make at most {max_calls} model calls per question (`cot`=1, `cot_sc`=k,
  `reflexion`=1+n, `step_back`=2, `role_assignment`=2, `debate`=roles*rounds+1,
  `ensemble`=the sum of its children). A costlier proposal is discarded unrun.
- The fitness intervals overlap heavily at this validation-set size. Prefer a
  design that is structurally different from the archive over one that adds a
  refinement step to whichever entry happens to lead.
{reflexion}"""

_REFLEXION_1 = ("\nReflect: is this genuinely novel versus the archive and "
                "well-formed? Revise and output the improved JSON.")
_REFLEXION_2 = ("\nReflect again: is the program valid (only allowed blocks) and "
                "is it the most promising next step? Output the final JSON.")


def _render_archive(archive: List[dict]) -> str:
    """The meta-agent's whole view of the search so far.

    Shows the 95% bootstrap CI beside the mean, as ADAS's own archive does. The
    mean alone reads as an exact ranking, so a meta-agent conditioned on it chases
    gaps that are pure sampling noise on a validation set this size -- and then
    proposes a *more elaborate* version of whatever happened to win. The interval
    is already computed by :func:`bootstrap_ci`; it was just being discarded."""
    lines = []
    for i, a in enumerate(archive):
        fit = a.get("fitness")
        if isinstance(fit, (int, float)):
            ci = a.get("ci")
            fit_s = f"{fit:.3f}"
            if ci:
                fit_s += f", 95% CI [{ci[0]:.3f}, {ci[1]:.3f}]"
        else:
            fit_s = "unevaluated"
        lines.append(f"[{i}] {a['name']} (fitness={fit_s}; "
                     f"{program_cost(a['program'])} model calls per question)\n"
                     f"    thought: {a.get('thought','')}\n"
                     f"    program: {json.dumps(a['program'])}")
    return "\n".join(lines)


def _json_objects(text: str):
    """Every balanced ``{...}`` span in ``text``, outermost first.

    The single greedy ``\\{.*\\}`` this replaces spans from the first brace to the
    last one in the whole reply, so any prose containing a brace after the JSON --
    a closing note, a second worked example, a ```json fence followed by
    commentary -- made the match unparseable and threw the proposal away. A
    balanced scan gives each candidate span its own chance."""
    starts = [i for i, c in enumerate(text) if c == "{"]
    for start in starts:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    yield text[start:i + 1]
                    break


def _parse_agent(text: str, max_calls: int = MAX_PROGRAM_CALLS) -> Optional[dict]:
    """The first balanced JSON span that is a well-formed, affordable agent."""
    for span in _json_objects(text or ""):
        try:
            obj = json.loads(span)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or "program" not in obj:
            continue
        if not validate_program(obj["program"]):
            continue
        if max_calls and program_cost(obj["program"]) > max_calls:
            continue                       # affordable is part of well-formed here
        obj.setdefault("name", "Unnamed Agent")
        obj.setdefault("thought", "")
        return obj
    return None


def propose_agent(complete: Completion, archive: List[dict], debug_max: int = 3
                  ) -> Optional[dict]:
    """Meta-agent proposes a new agent, with two Reflexion refinement rounds.

    The refinement rounds are *revisions*, so the last one is normally the answer
    -- but only if it parsed. This used to return ``_parse_agent(text)`` on
    whatever the final round happened to emit, which throws away a perfectly good
    agent from round 0 or 1 whenever round 2 exhausts its retries on malformed
    output. A generation produces one proposal, so that is a whole generation
    spent for nothing. Keep the newest valid draft instead, and carry it forward
    as the draft the next round refines."""
    reflexions = ["", _REFLEXION_1, _REFLEXION_2]
    best: Optional[dict] = None
    draft = ""
    for i, refl in enumerate(reflexions):
        prompt = f"{_META_SYSTEM}\n\n" + _META_TMPL.format(
            archive=_render_archive(archive), max_calls=MAX_PROGRAM_CALLS,
            reflexion=(refl if i else ""))
        if i and draft:
            prompt += f"\n\nYour current draft:\n{draft}"
        for _ in range(debug_max):
            text = complete(prompt)
            agent = _parse_agent(text)
            if agent is not None:
                best, draft = agent, json.dumps(agent)
                break
    return best


# ===========================================================================
# Dataset: MGSM (loaded dependency-free from ADAS's TSVs)
# ===========================================================================


def download_mgsm(lang: str) -> List[Tuple[str, str]]:
    text = fetch_text(MGSM_URL.format(lang=lang), cache_subdir="mgsm",
                      filename=f"mgsm_{lang}.tsv")
    out = []
    for line in text.splitlines():
        if "\t" not in line:
            continue
        q, a = line.split("\t", 1)
        out.append((q, a))
    return out


def build_examples(langs: List[str], per_lang: int, seed: int = 0
                   ) -> List[Tuple[str, str]]:
    rng = random.Random(seed)
    pool: List[Tuple[str, str]] = []
    for lang in langs:
        rows = download_mgsm(lang)
        rng.shuffle(rows)
        pool.extend(rows[:per_lang])
    rng.shuffle(pool)
    return pool


def language_of(langs: List[str], per_lang: int, seed: int = 0) -> dict:
    """``question -> language`` for the same pool :func:`build_examples` returns.

    MGSM's languages are not equally hard (measured with ``deepseek-v4-flash``:
    ``en`` 0.964, ``ja`` 0.880), so a hard subset drawn from several of them is a
    mixture -- and an unstratified split can hand validation one mix and test
    another. That difference then shows up as a lift, or hides one."""
    out = {}
    for lang in langs:
        for q, _ in download_mgsm(lang):
            out[q] = lang
    return out


def score_mgsm(target: str, prediction: Optional[str]) -> bool:
    """ADAS's exact MGSM scoring (utils.score_mgsm): strip commas/trailing zeros."""
    if prediction is None:
        return False
    if "." in prediction:
        prediction = prediction.rstrip("0").rstrip(".")
    return target.replace(",", "") == prediction.replace(",", "")


def evaluate_agent(interp: Interpreter, program: dict, examples: List[Tuple[str, str]],
                   concurrency: int = 8) -> List[float]:
    """Score a program on each example, concurrently.

    This was a sequential list comprehension, and it is the whole run: every
    candidate is scored on every example, and each score is a *multi-step* program
    (debate and self-consistency make several model calls per question). Serially
    that made ADAS's effective concurrency ~1 no matter how many workers the
    engine was given -- measured, a 6-example generation had not finished after 25
    minutes. `Interpreter` holds only a completion, so the runs are independent.
    """
    if not examples:
        return []
    from concurrent.futures import ThreadPoolExecutor

    def one(ex):
        q, a = ex
        return 1.0 if score_mgsm(a, interp.run(program, q)) else 0.0

    with ThreadPoolExecutor(max(1, min(concurrency, len(examples)))) as pool:
        return list(pool.map(one, examples))


def load_dataset(langs: List[str], per_lang: int, seed: int = 0,
                 ratios=(0.5, 0.25, 0.25)) -> Dataset:
    """MGSM (q, a) examples split into train / val (search) / test."""
    return split_dataset(build_examples(langs, per_lang, seed=seed),
                         ratios=ratios, seed=seed, name="MGSM")


EVAL_CONCURRENCY = 16          # set from --eval-concurrency; I/O bound, so high

#: Token budget per model call. Far above what any single reply here needs -- the
#: meta-agent's JSON is ~250 tokens and an answer is a handful -- because on a
#: *reasoning* model the budget is spent on hidden reasoning first and the visible
#: content is what is left. Measured on ``deepseek-v4-flash`` at the library
#: default of 4096: **0 of 4** meta-agent calls returned anything at all, each
#: burning its entire 4096 on reasoning; at 16384 all 4 returned a well-formed
#: 740-1068 character design. You are billed for tokens generated, not for the
#: cap, so the headroom is free.
MAX_TOKENS = 16384


class EmptyCompletionGuard:
    """Wraps a completion and counts the replies that come back blank.

    An empty completion is the most dangerous failure mode this example has,
    because it is *silent*: `_extract_int("")` is `None`, `score_mgsm` scores that
    as a wrong answer, and a starved run therefore reports a low accuracy that
    looks exactly like a model that cannot do the problems. Nothing raises, so
    every gate downstream -- fitness, the archive, the lift -- is computed over
    answers that were never generated. Counting them turns a corrupted
    measurement into a visible one."""

    def __init__(self, complete: Completion) -> None:
        self._complete = complete
        self.blank = 0
        self.total = 0
        self._lock = threading.Lock()

    def __call__(self, prompt: str) -> str:
        out = self._complete(prompt)
        if not (out or "").strip():
            with self._lock:
                self.blank += 1
        with self._lock:
            self.total += 1
        return out

    def report(self) -> Optional[str]:
        if not self.blank:
            return None
        return (f"{self.blank}/{self.total} model calls returned EMPTY content "
                f"({self.blank / self.total:.1%}). On a reasoning model that means "
                f"the token budget was spent on hidden reasoning -- raise "
                f"--max-tokens. Every blank reply was scored as a wrong answer, so "
                f"the numbers below understate the agents by an unknown amount.")


def fmt_score(x: Optional[float]) -> str:
    """A score, or ``n/a`` when it could not be measured.

    A missing number must print as missing. Formatting `None` with `:.3f` raises,
    which would turn "we could not score the test split" into a traceback that
    also discards the validation numbers the run *did* produce."""
    return "     n/a" if x is None else f"{x:>8.3f}"


def estimate_calls(generations: int, n_train: int, n_val: int, n_test: int,
                   n_workers: int = 2) -> Tuple[int, int]:
    """``(typical, ceiling)`` model calls for a run of this shape.

    The old estimate was ``(7 + generations) * (n_val + n_test) * 3`` computed
    *before* ``--hard`` narrowed the split, so it reported the cost of a run over
    the full pool for a run that would actually happen on 8% of it. The recorded
    figure was ~9009 calls for a run that made 791 -- an order of magnitude, on
    the number a caller uses to decide whether they can afford this at all.

    A single number cannot be honest here either, because a candidate's cost is
    whatever the meta-agent proposes: anywhere from 1 call per question (`cot`) to
    :data:`MAX_PROGRAM_CALLS`. So both ends are given -- the typical figure prices
    a candidate at the mean seed cost, the ceiling at the cap. Not every
    generation yields a candidate either (a proposal is only requested when the
    trigger rollout fails), so even the low end is an over-estimate."""
    seeds = seed_archive()
    costs = [program_cost(s["program"]) for s in seeds]
    seed_calls = sum(costs) * n_val
    typical_cost = sum(costs) / len(costs)
    propose = generations * n_workers * 3          # three Reflexion rounds each
    out = []
    for per_item in (typical_cost, float(MAX_PROGRAM_CALLS)):
        trigger = generations * n_workers * per_item
        candidates = generations * n_workers * n_val * per_item
        test = 2 * n_test * per_item               # best seed + best searched
        out.append(int(seed_calls + propose + trigger + candidates + test))
    return out[0], out[1]


class _HardCache:
    """Per-question verdicts from the ``--hard`` baseline pass, kept on disk.

    The pass is one cheap call per item, but it is *every* item: at
    ``--langs`` all ``--per-lang 250`` that is 2750 calls thrown away and paid
    again on the next run, which is more than a three-generation search costs.
    The verdict depends only on (model, question), so it caches cleanly -- and
    caching it is what makes the hard subset reproducible across runs rather than
    re-drawn from a fresh sample of the model each time."""

    def __init__(self, model: str, enabled: bool = True) -> None:
        import hashlib
        import os
        self.enabled = enabled
        key = hashlib.sha256(model.encode()).hexdigest()[:12]
        root = os.path.join(os.path.expanduser("~/.cache/agentdescent"), "mgsm", "hard")
        self.path = os.path.join(root, f"direct_{key}.json")
        self._d = {}
        self._dirty = False
        self._lock = threading.Lock()
        if enabled:
            os.makedirs(root, exist_ok=True)
            try:
                with open(self.path, encoding="utf-8") as fh:
                    self._d = json.load(fh)
            except (OSError, json.JSONDecodeError):
                self._d = {}

    def count(self, questions) -> int:
        return sum(1 for q in questions if q in self._d)

    def get(self, question: str) -> Optional[float]:
        return self._d.get(question) if self.enabled else None

    def put(self, question: str, score: float) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._d[question] = score
            self._dirty = True

    def flush(self) -> None:
        if not (self.enabled and self._dirty):
            return
        try:
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump(self._d, fh, ensure_ascii=False)
        except OSError:
            pass                          # a cache miss is not a run failure


def evaluate(complete: Completion, program: dict,
             examples: List[Tuple[str, str]]) -> float:
    """Mean MGSM accuracy of an agent program on a held-out split."""
    if not examples:
        return 0.0
    return sum(evaluate_agent(Interpreter(complete), program, examples,
                              concurrency=EVAL_CONCURRENCY)) / len(examples)


# ===========================================================================
# Running Meta Agent Search THROUGH evolve() -- design strategy + archive optimizer
# ===========================================================================
#
# Like ACE/GEPA, ADAS plugs into `evolve()`: a `Strategy` turns a proposed agent
# (JSON) into a `Diff` on the one-slot "agentic system", and a custom
# `aggregator_factory` is the keep-all archive that scores each candidate
# (bootstrap-CI fitness) and keeps the best design as the dev head. The meta-agent
# (the propose step) conditions on the *whole archive* (shared via `AdasContext`),
# so it does not depend on the specific per-task input evolve() hands it.


def _weighted_sample_without_replacement(weights: List[float], k: int,
                                         rng: random.Random) -> List[int]:
    idxs, pool, w = [], list(range(len(weights))), list(weights)
    for _ in range(min(k, len(pool))):
        pick = rng.choices(range(len(pool)), weights=w, k=1)[0]
        idxs.append(pool.pop(pick))
        w.pop(pick)
    return idxs


#: Two outcomes the shared :class:`~agentdescent.aggregator.MergeOutcome`
#: vocabulary has no name for, because they are specific to a keep-all archive.
#: Both print as `+0/-1` on the driver's line and need opposite responses:
#: ALREADY_BEST means candidates were scored and none beat the incumbent (look at
#: the meta-prompt); NO_CANDIDATES means none was produced at all, because every
#: trigger rollout passed and `evolve()` never asked (look at the trigger split).
#: Plain strings, like `MergeOutcome`'s own values -- `outcomes()` keys on them.
ALREADY_BEST = "already-best"
NO_CANDIDATES = "no-candidates"


@dataclass
class AdasContext:
    select: str = "adas"
    rng_seed: int = 0
    archive: List[dict] = field(default_factory=list)
    children: List[int] = field(default_factory=list)
    seed_fitness: float = 0.0
    best_fitness: float = 0.0
    best_agent: dict = field(default_factory=dict)
    #: The best *hand-designed* seed, kept so the run can report the lift that
    #: matters -- searched-vs-seed on the test split. Reporting only the searched
    #: agent's test accuracy says nothing about whether the search helped.
    best_seed: dict = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)   # workers run concurrently


class AgentDesignStrategy:
    """The artifact is one agentic-system design; a proposal is a new agent JSON."""

    def initial(self):
        seed = seed_archive()[0]                            # start from Chain-of-Thought
        return {"design": canonical(seed["program"]),
                "name": seed["name"], "thought": seed["thought"]}

    def render(self, state):
        return state.get("design", "{}")                   # the program, for the interpreter

    def to_diff(self, state, proposal, author, base_version, target):
        agent = _parse_agent(proposal)
        if agent is None:
            return None
        design = canonical(agent["program"])
        if design == state.get("design"):
            return None
        return Diff(diff_id=f"{author}:{rule_id(design)}:{base_version}", target=target,
                    ops={"design": design, "name": agent["name"],
                         "thought": agent.get("thought", "")}, author=author)


def make_propose(ctx: AdasContext, complete: Completion):
    """The meta-agent: conditioned on the whole archive, propose the next agent."""
    rng = random.Random(ctx.rng_seed)

    def propose(rendered, task, output, score):
        conditioning = ctx.archive or seed_archive()
        if ctx.select == "dgm" and ctx.archive:            # DGM: sample who to surface
            with ctx.lock:                                 # concurrent workers share rng + archive
                weights = dgm_parent_weights([a["fitness"] for a in ctx.archive], ctx.children)
                idxs = _weighted_sample_without_replacement(weights, min(5, len(ctx.archive)), rng)
                conditioning = [ctx.archive[i] for i in idxs]
                for i in idxs:                             # "explored" -> novelty discount
                    ctx.children[i] += 1
        agent = propose_agent(complete, conditioning)
        return json.dumps(agent) if agent else None
    return propose


class MetaSearchAggregator(AggregatorProtocol):
    """ADAS's optimizer: a keep-all archive with bootstrap-CI fitness."""

    def __init__(self, ledger: Ledger, verifier, ctx: AdasContext,
                 artifact_id: str = "agentic_system", boot_seed: int = 0,
                 selection=None):
        from agentdescent.selection import Beam
        self.ledger = ledger
        self.verifier = verifier
        self.ctx = ctx
        # ADAS keeps the archive's best design as head. That rule is exactly
        # the shipped Beam(1): stable sort, first maximal wins -- the same
        # tie-break as the incremental strictly-greater tracking it replaces.
        # The first legacy port whose rule needed no local subclass.
        self.selection = selection or Beam(1)
        self.aid = artifact_id
        self.boot_seed = boot_seed
        self.cards: List[EvidenceCard] = []
        self._lock = threading.Lock()   # ingest: workers; step: one thread
        self._seeded = False

    def _fitness(self, head, design: str) -> Tuple[float, Tuple[float, float]]:
        art = head.apply(Diff(diff_id="eval", target=self.aid,
                              ops={"design": design}, author="adas"))
        # Per-instance 0/1, because the bootstrap CI needs the individual outcomes --
        # but scored concurrently. One task per `score()` call sidesteps the
        # engine's own fan-out (it short-circuits at a single task), so this loop
        # was serial even though everything under it is not.
        from concurrent.futures import ThreadPoolExecutor
        held = list(self.verifier.held_out)
        if held:
            with ThreadPoolExecutor(max(1, min(EVAL_CONCURRENCY, len(held)))) as pool:
                correct = list(pool.map(lambda t: art.score([t]), held))
        else:
            correct = []
        mean, lo, hi = bootstrap_ci(correct, seed=self.boot_seed)
        return mean, (lo, hi)

    def seed(self) -> None:
        """Score the seven hand-designed seeds -- once, and *before* round 0.

        This ran inside the first ``step()``, which is after the first round's
        meta-agents have already proposed. Their whole conditioning signal is the
        archive, so the first generation was designed against seven entries all
        reading ``unevaluated``: it could not know that Debate beats CoT here,
        because nothing had been measured yet. With ``--generations 3`` that is a
        third of the search spent blind. The aggregator is built before round 0,
        so this belongs in construction."""
        if self._seeded:
            return
        self._seeded = True
        head = self.ledger.snapshot(Ledger.DEV).get(self.aid)
        for a in seed_archive():
            mean, ci = self._fitness(head, canonical(a["program"]))
            self._record(a["name"], a["thought"], a["program"], mean, ci, seed=True)
        self.ctx.seed_fitness = self.ctx.best_fitness
        self.ctx.best_seed = dict(self.ctx.best_agent)

    def ingest(self, card: EvidenceCard) -> None:
        # ingest runs on worker threads, step on one: see AggregatorProtocol.
        with self._lock:
            self.cards.append(card)

    def _record(self, name, thought, program, fitness, ci=None, seed=False):
        agent = {"name": name, "thought": thought, "program": program,
                 "fitness": fitness, "ci": ci, "seed": seed}
        self.ctx.archive.append(agent)
        self.ctx.children.append(0)
        if fitness > self.ctx.best_fitness or not self.ctx.best_agent:
            self.ctx.best_fitness = fitness
            self.ctx.best_agent = agent
        return agent

    def step(self) -> List[MergeReport]:
        self.seed()                                        # no-op after construction
        snap = self.ledger.snapshot(Ledger.DEV)
        head = snap.get(self.aid)
        base_vv = {self.aid: snap.version.get(self.aid, 0)}

        with self._lock:
            cards, self.cards = self.cards, []
        for card in cards:
            design = card.diff.ops["design"]
            mean, ci = self._fitness(head, design)
            self._record(card.diff.ops.get("name", "agent"),
                         card.diff.ops.get("thought", ""),
                         json.loads(design), mean, ci)

        # Best-of-archive at the standard seam (version = archive index).
        from agentdescent.selection import Candidate, SelectionContext
        rows = [Candidate(artifact_id=self.aid, version=i,
                          score=a["fitness"])
                for i, a in enumerate(self.ctx.archive)]
        best_agent = self.ctx.best_agent
        if rows:
            sel = SelectionContext(head=rows[0], candidates=tuple(rows),
                                   n_workers=1)
            best_agent = self.ctx.archive[
                self.selection.select(sel, 1)[0].version]
        best_design = canonical(best_agent["program"])
        diff = Diff(diff_id="best", target=self.aid,
                    ops={"design": best_design,
                         "name": best_agent["name"],
                         "thought": best_agent.get("thought", "")},
                    author="adas")
        committed, category = None, ALREADY_BEST
        if not cards:
            category = NO_CANDIDATES
        if best_design != head.state.get("design"):        # keep the best design as head
            try:
                _, committed = self.ledger.commit(
                    head.apply(diff), base_vv, branch=Ledger.DEV,
                    message="adas: keep best design")
                category = MergeOutcome.COMMITTED
            except CASConflict:
                category = MergeOutcome.CAS_CONFLICT
        # `accepted` and `category` were left at None/"" while the archive was in
        # fact committing, so the driver's own tally -- which reads
        # `committed_version` -- printed `+0/-1` for a generation whose best design
        # was simply unchanged, and `outcomes()` bucketed every generation as
        # "unknown". The three outcomes need opposite responses: "already-best"
        # means the search found nothing better, "no-candidates" means nothing
        # reached the archive at all (the trigger rollouts all passed, so evolve()
        # never asked for a proposal), and "committed" means it did improve.
        return [MergeReport(self.aid, diff if committed is not None else None, False,
                            len(cards), len(cards), 0, 0,
                            self.ctx.best_fitness, committed,
                            f"archive={len(self.ctx.archive)} "
                            f"best={self.ctx.best_fitness:.3f} ({category})",
                            category)]


@dataclass
class SearchResult:
    archive: List[dict]
    best: dict
    seed_fitness: float
    best_fitness: float
    #: The best hand-designed seed, so the caller can score *it* on test too. On a
    #: `--hard` subset the direct baseline is 0.000 by construction, so the only
    #: comparison that says anything about the search is searched-vs-seed.
    best_seed: dict = field(default_factory=dict)
    #: ``EvolutionResult.outcomes()`` -- why each generation went as it did.
    #: The driver's own ``+0/-1`` line cannot distinguish "candidates reached the
    #: archive and none beat the incumbent" (`already-best`) from "no candidate
    #: was ever produced" (`no-candidates`, i.e. every trigger rollout passed, so
    #: `evolve()` never asked for a proposal). Those need opposite fixes -- a
    #: better meta-prompt versus a harder trigger split -- and the recorded run
    #: that concluded "the archive rejected every candidate" had no way to tell
    #: which of the two it had actually observed.
    outcomes: dict = field(default_factory=dict)
    stop_reason: str = "rounds"
    error: Optional[str] = None


def run_meta_agent_search(complete: Completion, val: List[Tuple[str, str]],
                          generations: int, select: str = "adas", seed: int = 0,
                          asynchronous: bool = False, async_ratio: int = 3,
                          max_seconds: float = 45.0, held_out_frac: float = 0.5,
                          n_workers: int = 2, max_rollouts: Optional[int] = None,
                          verbose: bool = False) -> SearchResult:
    """Drive Meta Agent Search through `evolve()` (val split into trigger/held-out)."""
    tasks = [Task(id=f"mgsm{i}", prompt=q, meta={"answer": a})
             for i, (q, a) in enumerate(val)]
    ctx = AdasContext(select=select, rng_seed=seed)
    interp = Interpreter(complete)

    def run(rendered, task):
        try:
            program = json.loads(rendered)
        except json.JSONDecodeError:
            return ""
        return interp.run(program, task.prompt) or ""

    def reward(task, output):
        return 1.0 if score_mgsm(task.meta["answer"], output) else 0.0

    def factory(ledger, verifier, audit, config, policy):
        agg = MetaSearchAggregator(ledger, verifier, ctx,
                                   artifact_id="agentic_system", boot_seed=seed)
        agg.seed()             # score the seed archive BEFORE round 0 proposes
        return agg

    out = evolve(tasks, reward, run=run, propose=make_propose(ctx, complete),
           strategy=AgentDesignStrategy(), blast_radius=0.6, artifact_id="agentic_system",
           rounds=generations, n_workers=n_workers,
           max_concurrency=1 if asynchronous else n_workers,
           asynchronous=asynchronous, async_ratio=async_ratio,
           max_seconds=max_seconds if asynchronous else None,
           # The archive judges a candidate by its own bootstrap fitness over the
           # whole held-out set and never reads `before_after_delta`, so the
           # self-verify rollout each proposal triggers is a multi-step program run
           # for a number nothing consumes -- pure cost on the most expensive
           # example in the repo.
           self_verify=False,
           eval_concurrency=EVAL_CONCURRENCY,
           held_out_frac=held_out_frac, aggregator_factory=factory, verbose=verbose,
           max_rollouts=max_rollouts)
    return SearchResult(ctx.archive, ctx.best_agent or {}, ctx.seed_fitness,
                        ctx.best_fitness, ctx.best_seed or {},
                        out.outcomes(), out.stop_reason, out.error)


# ===========================================================================
# main
# ===========================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    add_standard_args(p, max_seconds_default=60.0, eval_concurrency_default=16)
    p.add_argument("--generations", type=int, default=6)
    p.add_argument("--langs", default="en,es,fr",
                   help=f"comma-separated MGSM languages (of {','.join(ALL_LANGUAGES)})")
    p.add_argument("--per-lang", type=int, default=8, help="validation examples per language")
    p.add_argument("--select", default="adas", choices=["adas", "dgm"],
                   help="archive conditioning: adas (whole archive) or dgm (sampled)")
    p.add_argument("--hard-keep", type=int, default=None,
                   help="cap the hard subset -- evaluation cost is candidates x "
                        "items x (multi-step calls), so this is the strongest lever "
                        "on how long a run takes")
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                   help="token budget per model call. On a reasoning model the "
                        "budget is spent on hidden reasoning first, so too small a "
                        "cap returns EMPTY content -- which scores as a wrong "
                        "answer and silently understates every agent")
    p.add_argument("--workers", type=int, default=2,
                   help="meta-agents proposing per generation. A proposal is only "
                        "requested when the generation's trigger rollout FAILS, so "
                        "on a hard subset roughly half of them fire -- raising this "
                        "buys candidates per generation at the same cost each")
    p.add_argument("--hard", action="store_true",
                   help="keep only items a plain single call gets wrong -- MGSM is "
                        "already ~0.92 for a strong model, so the search has no "
                        "gradient. Those are exactly the items where agentic "
                        "structure (CoT, debate, self-refine) can help")
    p.add_argument("--no-hard-cache", dest="hard_cache", action="store_false",
                   help="re-run the --hard baseline pass instead of reusing the "
                        "cached per-question verdicts")
    p.add_argument("--train-frac", type=float, default=0.15,
                   help="share of the pool used only to TRIGGER proposals. The "
                        "meta-agent conditions on the archive, not on the task it "
                        "is handed, so a round consumes 2 of these and nothing "
                        "else -- everything left over measures fitness instead")
    p.add_argument("--test-frac", type=float, default=0.35,
                   help="share of the pool held out for the final number")
    return p


def main(argv=None) -> None:
    p = build_parser()
    args = p.parse_args(argv)
    # --serial collapses this to the upstream algorithm's own semantics:
    # one worker, nothing to merge. Applied to args so the printed plan,
    # the cost estimate and the run cannot disagree about what ran.
    args.workers = worker_count(args, args.workers)

    langs = [l.strip() for l in args.langs.split(",") if l.strip() in ALL_LANGUAGES]
    # `not` binds tighter than `and`, so the guard needs its own parentheses --
    # without them `--train-frac 0.9 --test-frac 0.5` passed and produced a
    # negative validation ratio.
    if not (0.0 < args.train_frac and 0.0 < args.test_frac
            and args.train_frac + args.test_frac < 1.0):
        p.error("--train-frac and --test-frac must be positive and sum to < 1")
    ratios = (args.train_frac, 1.0 - args.train_frac - args.test_frac, args.test_frac)
    print("Algorithm: ADAS Meta Agent Search -- harness (agentic-system) self-evolution")
    print("Dataset  : MGSM (Multilingual Grade-School Math)")
    print(f"\nPlan     : model={args.model}, generations={args.generations}, "
          f"select={args.select}")
    if args.asynchronous:
        print(f"Async    : {args.workers} meta-agents, barrier-free "
              f"(async_ratio={args.async_ratio}, max {args.max_seconds:.0f}s); "
              f"the archive rebases/discards stale designs")
    else:
        print(f"Parallel : {args.workers} meta-agents propose concurrently each "
              "generation (synchronous DP; the archive merge is the barrier)")
    if args.dry_run:
        print(f"Data     : langs={langs}, per-lang={args.per_lang}; deferred "
              "(dry-run performs no network access)")
        print("\n[dry-run] plan only; no dataset or model API was accessed.")
        return

    langmap = language_of(langs, args.per_lang, seed=args.seed)
    pool = build_examples(langs, args.per_lang, seed=args.seed)
    ds = split_dataset(pool, ratios=ratios, seed=args.seed, name="MGSM",
                       stratify_key=lambda ex: langmap.get(ex[0], "?"))
    harness = EvolvingArtifact("agentic_system", blast_radius=0.6)
    print(f"Governance: harness artifact blast_radius={harness.blast_radius} "
          f"-> {classify(harness).name} (harness changes are high-blast-radius)")
    print(f"Loaded   : langs={langs}; {len(pool)} items")
    print(f"Seeds    : {', '.join(a['name'] for a in seed_archive())}")
    print("\nExample problem:")
    print("  Q:", ds.train[0][0][:150])
    print("  A:", ds.train[0][1])
    if not confirm(args):
        return

    usage = Usage()                       # what the run actually costs
    # --timeout is an openai_compatible-only knob; claude() takes no such
    # parameter, so it stays a caller-side branch rather than a shared default.
    extra = {"timeout": args.timeout} if is_openai_compatible(args) else {}
    raw = completion_for(args, usage=usage, max_tokens=args.max_tokens, **extra)
    guard = EmptyCompletionGuard(raw)
    completion = guard
    globals()["EVAL_CONCURRENCY"] = args.eval_concurrency
    try:
        probe = completion("Solve: what is 17 * 23? Think step by step, then end "
                           "with `Answer: <integer>`.")
    except Exception as e:  # noqa: BLE001
        print(f"\nCould not reach the model ({type(e).__name__}: {e}).")
        print("For --provider openai set OPENAI_BASE_URL + OPENAI_API_KEY; "
              "for claude set ANTHROPIC_API_KEY (or `ant auth login`).")
        return
    # A *reasoning* prompt, not "reply with ok": the cheap probe passes at any
    # budget, and the whole run is reasoning prompts. An empty reply here means
    # every rollout will be blank and score as wrong, with nothing raising.
    if not (probe or "").strip():
        print(f"\nThe model returned EMPTY content for a reasoning prompt at "
              f"--max-tokens {args.max_tokens}. On a reasoning model the budget is "
              f"spent on hidden reasoning first, so every rollout would come back "
              f"blank and be scored as a wrong answer. Raise --max-tokens.")
        return

    if args.hard:
        # A plain, structure-free call is the right baseline here: what ADAS
        # searches over IS structure, so the items worth keeping are the ones a
        # single call cannot already do.
        from agentdescent.dataloader import select_hard

        cache = _HardCache(args.model, enabled=args.hard_cache)
        verdicts = {}                     # this run's view, cache on or off

        def _direct(ex):
            q, a = ex
            hit = cache.get(q)
            if hit is not None:
                verdicts[q] = hit
                return hit
            out = completion(f"{q}\n\nAnswer with the final number only.")
            # The agents are scored with `_extract_int`; scoring the baseline with
            # a different, cruder extractor (concatenating every digit character in
            # the reply) means "hard" partly meant "the two extractors disagree" --
            # and the run then measures the agents against items they were never
            # actually wrong about. One extractor for both sides, so the subset
            # means what it says.
            score = 1.0 if score_mgsm(a, _extract_int(out or "")) else 0.0
            cache.put(q, score)
            verdicts[q] = score
            return score

        n_cached = cache.count(q for q, _ in pool)
        if n_cached:
            print(f"\nBaseline : {n_cached}/{len(pool)} verdicts reused from "
                  f"{cache.path}")
        hard = select_hard(pool, _direct, keep=args.hard_keep,
                           concurrency=args.eval_concurrency)
        cache.flush()
        ds = split_dataset(hard, ratios=ratios, seed=args.seed, name="MGSM",
                           stratify_key=lambda ex: langmap.get(ex[0], "?"))
        # From the verdicts, not from len(hard): `select_hard` tops a thin result
        # up with items the baseline SOLVED, and `--hard-keep` truncates. Deriving
        # the baseline from the subset size therefore reports the top-up as model
        # error -- on a 30-item English pool it printed 0.600 for a model that had
        # in fact answered 29 of 30.
        n_wrong = sum(1 for q, _ in pool if verdicts.get(q) == 0.0)
        kept_hard = sum(1 for q, _ in hard if verdicts.get(q) == 0.0)
        print(f"Hard mode: {n_wrong}/{len(pool)} items a single call answers "
              f"incorrectly (direct accuracy {1 - n_wrong/len(pool):.3f})")
        if kept_hard < len(hard):
            print(f"           subset is {len(hard)} items, of which {kept_hard} are "
                  f"hard -- select_hard topped up a pool too thin to split")

    ntr, nva, nte = ds.sizes()
    print(f"Split    : {ntr} trigger / {nva} val (fitness) / {nte} test")
    lo, hi = estimate_calls(args.generations, ntr, nva, nte, args.workers)
    print(f"Budget   : ~{lo}-{hi} model calls (each candidate is a multi-step "
          f"program, run on every val item)")
    if nva < 20 or nte < 20:
        print("WARNING  : fewer than 20 items on a split -- at this size a single "
              "rollout moves the number by 5%+ and no lift is readable. "
              "Raise --per-lang / --langs, or drop --hard-keep.")

    print("\nSearching agentic systems (Meta Agent Search, L1 harness)...\n")
    result = run_meta_agent_search(completion, ds.trainval, args.generations,
                                   select=args.select, seed=args.seed,
                                   asynchronous=args.asynchronous, async_ratio=args.async_ratio,
                                   max_seconds=args.max_seconds, n_workers=args.workers,
                                   # the engine's held-out split IS ds.val
                                   held_out_frac=ds.val_frac, verbose=True,
                                   **budget_kwargs(args))

    # Both numbers on the SAME held-out split. Reporting only the searched agent's
    # test accuracy cannot answer the question the run exists to answer -- on a
    # `--hard` subset the structure-free baseline is 0.000 by construction, so
    # "0.4 on test" is not evidence that searching did anything. The comparison is
    # searched-vs-best-seed, and it costs one extra evaluation of one program.
    seed_prog = result.best_seed.get("program", {})
    best_prog = result.best.get("program", {})
    if not validate_program(best_prog):        # the run died before the archive filled
        print("\nNo agentic system was evaluated -- see the error above.")
        print(f"model usage: {usage.summary()}")
        return
    # The search is already done and paid for at this point: the archive holds
    # every design's validation fitness. Letting a backend failure *here* escape
    # throws all of it away and prints nothing -- which is how a run that had
    # completed its seeding and its generations reported a bare traceback. It
    # happened: an account ran out of credit during this very call, after 79
    # minutes of work. Test accuracy is the one thing that becomes unavailable;
    # everything measured on val survives, so report that and say which number is
    # missing rather than losing both.
    def _on_test(program) -> Optional[float]:
        try:
            return evaluate(completion, program, ds.test)
        except Exception as e:  # noqa: BLE001 - a backend failure, not a caller bug
            print(f"\nWARNING: could not score the test split "
                  f"({type(e).__name__}: {str(e)[:160]}).")
            print("         The validation numbers below are complete; the test "
                  "column is not available for this run.")
            return None

    seed_test = _on_test(seed_prog)
    test_acc = (seed_test if seed_prog == best_prog
                else _on_test(best_prog) if seed_test is not None else None)


    print("\n=== best discovered agentic system ===")
    print(f"name   : {result.best.get('name', '?')}")
    print(f"program: {json.dumps(best_prog, indent=2)}")
    print(f"         ({program_cost(best_prog)} model calls per question)")
    print(f"\narchive: {len(result.archive)} designs "
          f"({sum(1 for a in result.archive if a.get('seed'))} seeds + "
          f"{sum(1 for a in result.archive if not a.get('seed'))} searched)")
    # Why the generations went as they did. `already-best` and `no-candidates`
    # both print as `+0/-1` on the driver's line and need opposite fixes.
    if result.outcomes:
        print("generations: " + ", ".join(f"{k}={v}" for k, v in
                                          sorted(result.outcomes.items()))
              + f"  (stopped: {result.stop_reason})")
    if result.error:
        print(f"WARNING: the run did not finish cleanly -- {result.error}")
    lift_test = (None if seed_test is None or test_acc is None
                 else test_acc - seed_test)
    print("\n                                  val (search)   test (held out)")
    print(f"  best hand-designed seed         {result.seed_fitness:>9.3f}   "
          f"{fmt_score(seed_test):>13}   ({result.best_seed.get('name', '?')})")
    print(f"  best searched design            {result.best_fitness:>9.3f}   "
          f"{fmt_score(test_acc):>13}   ({result.best.get('name', '?')})")
    print(f"  lift                            {result.best_fitness - result.seed_fitness:>+9.3f}   "
          f"{('     n/a' if lift_test is None else f'{lift_test:>+8.3f}'):>13}")
    if args.hard:
        print("\n  (a structure-free single call scores 0.000 on this subset by "
              "construction -- the seed row is the baseline that matters)")
    blank = guard.report()
    if blank:
        print(f"\nWARNING: {blank}")
    print(f"\nmodel usage: {usage.summary()}")


if __name__ == "__main__":
    main()
