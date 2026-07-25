"""Harness self-evolution, faithful port: **ADAS -- Meta Agent Search**.

Paper : "Automated Design of Agentic Systems", Shengran Hu, Cong Lu, Jeff Clune,
        2024 (arXiv:2408.08435; ICLR 2025).
Repo  : https://github.com/ShengranHu/ADAS
Dataset: **MGSM** (Multilingual Grade-School Math), the light math benchmark
        ADAS searches on (`dataset/mgsm/*.tsv`; Shi et al., 2022).

Where ACE/GEPA evolve a *skill* (an L2 prompt/context), ADAS evolves the
**agentic system itself** -- the control flow that orchestrates the model. That
is a **harness** change: high blast radius -> Concordia's **L1** governance
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

    python -m examples.adas_meta_agent_search --dry-run       # dataset + seeds, no API
    python -m examples.adas_meta_agent_search --model claude-haiku-4-5 --generations 6
    python -m examples.adas_meta_agent_search --select dgm --langs en,es
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from concordia.agents import claude, openai_compatible
from concordia.aggregator import AggregatorProtocol, MergeReport
from concordia.dataloader import Dataset, fetch_text, split_dataset
from concordia.evolvable import Diff, EvidenceCard
from concordia.evolution import EvolvingArtifact, Task, evolve, rule_id
from concordia.governance import classify
from concordia.ledger import CASConflict, Ledger

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
        {"name": "Self-Consistency with CoT",
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
    """ADAS's fitness: mean accuracy with a 95% bootstrap CI (utils.py)."""
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
{reflexion}"""

_REFLEXION_1 = ("\nReflect: is this genuinely novel versus the archive and "
                "well-formed? Revise and output the improved JSON.")
_REFLEXION_2 = ("\nReflect again: is the program valid (only allowed blocks) and "
                "is it the most promising next step? Output the final JSON.")


def _render_archive(archive: List[dict]) -> str:
    lines = []
    for i, a in enumerate(archive):
        fit = a.get("fitness")
        fit_s = f"{fit:.3f}" if isinstance(fit, (int, float)) else "unevaluated"
        lines.append(f"[{i}] {a['name']} (fitness={fit_s})\n"
                     f"    thought: {a.get('thought','')}\n"
                     f"    program: {json.dumps(a['program'])}")
    return "\n".join(lines)


def _parse_agent(text: str) -> Optional[dict]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "program" not in obj:
        return None
    if not validate_program(obj["program"]):
        return None
    obj.setdefault("name", "Unnamed Agent")
    obj.setdefault("thought", "")
    return obj


def propose_agent(complete: Completion, archive: List[dict], debug_max: int = 3
                  ) -> Optional[dict]:
    """Meta-agent proposes a new agent, with two Reflexion refinement rounds."""
    reflexions = ["", _REFLEXION_1, _REFLEXION_2]
    text = ""
    for i, refl in enumerate(reflexions):
        prompt = f"{_META_SYSTEM}\n\n" + _META_TMPL.format(
            archive=_render_archive(archive), reflexion=(refl if i else ""))
        if i and text:
            prompt += f"\n\nYour current draft:\n{text}"
        for _ in range(debug_max):
            text = complete(prompt)
            if _parse_agent(text) is not None:
                break
    return _parse_agent(text)


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


def score_mgsm(target: str, prediction: Optional[str]) -> bool:
    """ADAS's exact MGSM scoring (utils.score_mgsm): strip commas/trailing zeros."""
    if prediction is None:
        return False
    if "." in prediction:
        prediction = prediction.rstrip("0").rstrip(".")
    return target.replace(",", "") == prediction.replace(",", "")


def evaluate_agent(interp: Interpreter, program: dict,
                   examples: List[Tuple[str, str]]) -> List[float]:
    return [1.0 if score_mgsm(a, interp.run(program, q)) else 0.0 for q, a in examples]


def load_dataset(langs: List[str], per_lang: int, seed: int = 0,
                 ratios=(0.5, 0.25, 0.25)) -> Dataset:
    """MGSM (q, a) examples split into train / val (search) / test."""
    return split_dataset(build_examples(langs, per_lang, seed=seed),
                         ratios=ratios, seed=seed, name="MGSM")


def evaluate(complete: Completion, program: dict,
             examples: List[Tuple[str, str]]) -> float:
    """Mean MGSM accuracy of an agent program on a held-out split."""
    if not examples:
        return 0.0
    return sum(evaluate_agent(Interpreter(complete), program, examples)) / len(examples)


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


@dataclass
class AdasContext:
    select: str = "adas"
    rng_seed: int = 0
    archive: List[dict] = field(default_factory=list)
    children: List[int] = field(default_factory=list)
    seed_fitness: float = 0.0
    best_fitness: float = 0.0
    best_agent: dict = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)   # workers run concurrently


class AgentDesignStrategy:
    """The artifact is one agentic-system design; a proposal is a new agent JSON."""

    def initial(self):
        seed = seed_archive()[0]                            # start from Chain-of-Thought
        return {"design": json.dumps(seed["program"]),
                "name": seed["name"], "thought": seed["thought"]}

    def render(self, state):
        return state.get("design", "{}")                   # the program, for the interpreter

    def to_diff(self, state, proposal, author, base_version, target):
        agent = _parse_agent(proposal)
        if agent is None:
            return None
        design = json.dumps(agent["program"])
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
                 artifact_id: str = "agentic_system", boot_seed: int = 0):
        self.ledger = ledger
        self.verifier = verifier
        self.ctx = ctx
        self.aid = artifact_id
        self.boot_seed = boot_seed
        self.cards: List[EvidenceCard] = []
        self._seeded = False

    def _fitness(self, head, design: str) -> float:
        art = head.apply(Diff(diff_id="eval", target=self.aid,
                              ops={"design": design}, author="adas"))
        correct = [art.score([t]) for t in self.verifier.held_out]   # 0/1 per instance
        return bootstrap_ci(correct, seed=self.boot_seed)[0]

    def ingest(self, card: EvidenceCard) -> None:
        self.cards.append(card)

    def _record(self, name, thought, program, fitness):
        agent = {"name": name, "thought": thought, "program": program, "fitness": fitness}
        self.ctx.archive.append(agent)
        self.ctx.children.append(0)
        if fitness > self.ctx.best_fitness or not self.ctx.best_agent:
            self.ctx.best_fitness = fitness
            self.ctx.best_agent = agent
        return agent

    def step(self) -> List[MergeReport]:
        snap = self.ledger.snapshot(Ledger.DEV)
        head = snap.get(self.aid)
        base_vv = {self.aid: snap.version.get(self.aid, 0)}
        if not self._seeded:                               # ADAS: seed archive first
            for a in seed_archive():
                self._record(a["name"], a["thought"], a["program"],
                             self._fitness(head, json.dumps(a["program"])))
            self.ctx.seed_fitness = self.ctx.best_fitness
            self._seeded = True

        cards, self.cards = self.cards, []
        for card in cards:
            design = card.diff.ops["design"]
            self._record(card.diff.ops.get("name", "agent"),
                         card.diff.ops.get("thought", ""),
                         json.loads(design), self._fitness(head, design))

        best_design = json.dumps(self.ctx.best_agent["program"])
        committed = None
        if best_design != head.state.get("design"):        # keep the best design as head
            try:
                _, committed = self.ledger.commit(
                    head.apply(Diff(diff_id="best", target=self.aid,
                                    ops={"design": best_design,
                                         "name": self.ctx.best_agent["name"],
                                         "thought": self.ctx.best_agent.get("thought", "")},
                                    author="adas")),
                    base_vv, branch=Ledger.DEV, message="adas: keep best design")
            except CASConflict:
                committed = None
        return [MergeReport(self.aid, None, False, len(cards), len(cards), 0, 0,
                            self.ctx.best_fitness, committed,
                            f"archive={len(self.ctx.archive)} best={self.ctx.best_fitness:.3f}")]


@dataclass
class SearchResult:
    archive: List[dict]
    best: dict
    seed_fitness: float
    best_fitness: float


def run_meta_agent_search(complete: Completion, val: List[Tuple[str, str]],
                          generations: int, select: str = "adas",
                          seed: int = 0, verbose: bool = False) -> SearchResult:
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
        return MetaSearchAggregator(ledger, verifier, ctx,
                                    artifact_id="agentic_system", boot_seed=seed)

    evolve(tasks, reward, run=run, propose=make_propose(ctx, complete),
           strategy=AgentDesignStrategy(), blast_radius=0.6, artifact_id="agentic_system",
           rounds=generations, n_workers=2, max_concurrency=2, held_out_frac=0.5,
           aggregator_factory=factory, verbose=verbose)
    return SearchResult(ctx.archive, ctx.best_agent or {}, ctx.seed_fitness, ctx.best_fitness)


# ===========================================================================
# main
# ===========================================================================


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--provider", default="claude", choices=["claude", "glm"])
    p.add_argument("--model", default="claude-haiku-4-5")
    p.add_argument("--generations", type=int, default=6)
    p.add_argument("--langs", default="en,es,fr",
                   help=f"comma-separated MGSM languages (of {','.join(ALL_LANGUAGES)})")
    p.add_argument("--per-lang", type=int, default=8, help="validation examples per language")
    p.add_argument("--select", default="adas", choices=["adas", "dgm"],
                   help="archive conditioning: adas (whole archive) or dgm (sampled)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--yes", action="store_true")
    args = p.parse_args()

    langs = [l.strip() for l in args.langs.split(",") if l.strip() in ALL_LANGUAGES]
    print("Algorithm: ADAS Meta Agent Search -- harness (agentic-system) self-evolution")
    print("Dataset  : MGSM (Multilingual Grade-School Math)")
    ds = load_dataset(langs, args.per_lang, seed=args.seed)
    harness = EvolvingArtifact("agentic_system", blast_radius=0.6)
    ntr, nva, nte = ds.sizes()
    print(f"Governance: harness artifact blast_radius={harness.blast_radius} "
          f"-> {classify(harness).name} (harness changes are high-blast-radius)")
    print(f"Loaded   : langs={langs}; {ntr} train / {nva} val (search) / {nte} test")
    print(f"Seeds    : {', '.join(a['name'] for a in seed_archive())}")
    print("\nExample problem:")
    print("  Q:", ds.train[0][0][:150])
    print("  A:", ds.train[0][1])

    calls = args.generations * 3 + (len(seed_archive()) + args.generations) * (nva + nte) * 3
    print(f"\nPlan     : model={args.model}, generations={args.generations}, select={args.select}")
    print("Parallel : 2 meta-agents propose concurrently each generation "
          "(synchronous DP; the archive merge is the barrier)")
    print(f"Budget   : up to ~{calls} model calls (multi-step agents call the model many times)")

    if args.dry_run:
        print("\n[dry-run] not calling the API. Drop --dry-run to run Meta Agent Search.")
        return

    if not args.yes and sys.stdin.isatty():
        if input("\nProceed with real API calls? [y/N] ").strip().lower() not in ("y", "yes"):
            print("aborted.")
            return

    completion = (openai_compatible(model=args.model) if args.provider == "glm"
                  else claude(model=args.model))
    try:
        completion("Reply with the single word: ok")
    except Exception as e:  # noqa: BLE001
        print(f"\nCould not reach the model ({type(e).__name__}: {e}).")
        print("For --provider glm set OPENAI_BASE_URL + OPENAI_API_KEY; "
              "for claude set ANTHROPIC_API_KEY (or `ant auth login`).")
        return

    print("\nSearching agentic systems (Meta Agent Search, L1 harness)...\n")
    result = run_meta_agent_search(completion, ds.trainval, args.generations,
                                   select=args.select, seed=args.seed, verbose=True)

    test_acc = evaluate(completion, result.best.get("program", {}), ds.test)
    print("\n=== best discovered agentic system ===")
    print(f"name   : {result.best.get('name', '?')}")
    print(f"program: {json.dumps(result.best.get('program', {}), indent=2)}")
    print(f"\nseed fitness (best hand-designed): {result.seed_fitness:.3f}")
    print(f"searched fitness (best on val)   : {result.best_fitness:.3f}")
    print(f"test accuracy (held out)         : {test_acc:.3f}")


if __name__ == "__main__":
    main()
