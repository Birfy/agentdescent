"""Skill self-evolution, faithful port: **GEPA — Reflective Prompt Evolution**.

Paper : "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement
        Learning", Lakshya A. Agrawal et al., 2025 (arXiv:2507.19457).
Repo  : https://github.com/gepa-ai/gepa  (also `dspy.GEPA`)
Dataset: **HotpotQA** (multi-hop QA, distractor setting), exact-match --
        one of GEPA's four adaptation benchmarks.

GEPA evolves the *instruction prompt* of a compound AI system with a genetic,
reflective loop. Its two distinctive mechanisms are both preserved here:

1. **Reflective mutation** (Algorithm 1 `UpdatePrompt`). On a failure the LLM
   reflects on the execution trace **and the natural-language evaluation
   feedback** (`mu_f`: the predicted vs. gold answer), then writes a *new*
   instruction. This is the `gepa_agent` propose step below.

2. **Pareto-based candidate selection** (Algorithm 2) -- the reason GEPA beats
   greedy hill-climbing. Instead of always mutating the single best-*average*
   candidate (which gets stuck in local optima), GEPA keeps a **pool** of
   candidates scored on every `D_pareto` instance and samples the next parent
   from the **per-instance Pareto frontier**, weighted by how many instances a
   candidate uniquely wins. This is `pareto_select` + `ParetoAggregator`, wired
   into `evolve()` through the `aggregator_factory=` hook -- the sanctioned way
   to *swap the whole optimizer*. The aggregator sets the dev head to the
   sampled parent, so `evolve()`'s next round mutates *it*, not the greedy best.

Only the parent-selection optimizer changes; the ledger, governance, staleness,
and statistical-acceptance machinery are the framework's. Prompt = L2 skill ->
`blast_radius=0.2`.

    python -m examples.gepa_prompt_evolution --dry-run           # dataset + estimate, no API
    python -m examples.gepa_prompt_evolution --model claude-haiku-4-5
    python -m examples.gepa_prompt_evolution --rounds 10 --workers 3

Faithful-but-tractable simplifications (documented): GEPA optimises a
multi-module compound system with a rollout budget; here the system is a single
instruction module and the minibatch is the per-round worker sample (raise
`--workers` for a bigger minibatch). The Pareto set is the held-out split.
"""

from __future__ import annotations

import argparse
import random
import re
import string
import sys
from typing import Dict, List, Optional, Sequence, Set, Tuple

from concordia.agents import claude, openai_compatible
from concordia.aggregator import AggregatorProtocol, MergeReport
from concordia.dataloader import Dataset, hf_rows, split_dataset
from concordia.evolvable import Diff, EvidenceCard
from concordia.evolution import LLMAgent, Task, evolve, rule_id
from concordia.ledger import CASConflict, Ledger

HOTPOTQA = ("hotpotqa/hotpot_qa", "validation", "distractor")   # (dataset, split, config)


# ===========================================================================
# The GEPA representation: a single evolvable instruction module
# ===========================================================================

_SEED_INSTRUCTION = (
    "Answer the question using the provided context. Think step by step across "
    "the paragraphs, then give the final answer."
)


class InstructionSlot:
    """The artifact is one instruction string; each proposal replaces it.

    (GEPA's `pi_j`: the system prompt of the module being optimised.)"""

    def __init__(self, seed: str = _SEED_INSTRUCTION):
        self.seed = seed

    def initial(self) -> Dict[str, str]:
        return {"instruction": self.seed}

    def render(self, state: Dict[str, str]) -> str:
        return state.get("instruction", self.seed)

    def to_diff(self, state, proposal, author, base_version, target) -> Optional[Diff]:
        proposal = proposal.strip()
        if not proposal or proposal == state.get("instruction"):
            return None
        rid = rule_id(proposal)
        return Diff(diff_id=f"{author}:{rid}:{base_version}", target=target,
                    ops={"instruction": proposal}, author=author)


_GENERATOR_TMPL = (
    "{artifact}\n\n{prompt}\n\n"
    "End your response with a line `Answer: <final answer>`."
)
_MUTATE_TMPL = (
    "You are optimising the instruction of a multi-hop QA system. The current "
    "instruction scored {reward:.2f} on a task.\n\n"
    "Current instruction:\n\"\"\"\n{artifact}\n\"\"\"\n\n"
    "Task the system saw:\n{prompt}\n\n"
    "System output (feedback: this was scored against the gold answer):\n{output}\n\n"
    "Reflect on why the instruction led to a wrong or low-scoring answer, then "
    "write an improved, general instruction for this QA system. Keep it concise "
    "(2-4 sentences). Output ONLY the new instruction text."
)


def gepa_agent(complete) -> LLMAgent:
    """Generator + reflective-mutation actor over a completion."""
    return LLMAgent(complete, solve_template=_GENERATOR_TMPL,
                    propose_template=_MUTATE_TMPL)


# ===========================================================================
# Algorithm 2: Pareto-based candidate selection (the crux of GEPA)
# ===========================================================================


def _dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    """Row a Pareto-dominates row b: >= on every instance, > on at least one."""
    ge = all(x >= y for x, y in zip(a, b))
    gt = any(x > y for x, y in zip(a, b))
    return ge and gt


def pareto_frontier(scores: List[List[float]]) -> Tuple[Set[int], Dict[int, int]]:
    """Return (kept candidates, win-frequency) per GEPA Algorithm 2, steps 1-4.

    ``scores[c][i]`` = candidate c's score on Pareto instance i. Steps:
    1. per-instance best; 2. union of instance-winners; 3. drop strictly
    dominated candidates; 4. frequency = #instances each survivor still wins."""
    if not scores or not scores[0]:
        return set(), {}
    n_cand, n_inst = len(scores), len(scores[0])
    # 1 + 2: candidates that tie the best score on at least one instance.
    best = [max(scores[c][i] for c in range(n_cand)) for i in range(n_inst)]
    pstar = [set(c for c in range(n_cand) if scores[c][i] == best[i])
             for i in range(n_inst)]
    C: Set[int] = set().union(*pstar) if pstar else set()
    # 3: iteratively remove any candidate strictly dominated by another in C.
    kept = set(C)
    changed = True
    while changed:
        changed = False
        for b in list(kept):
            if any(a != b and _dominates(scores[a], scores[b]) for a in kept):
                kept.discard(b)
                changed = True
                break
    # 4: frequency over the pruned per-instance frontier.
    freq: Dict[int, int] = {c: 0 for c in kept}
    for s in pstar:
        for c in (s & kept):
            freq[c] += 1
    return kept, freq


def pareto_select(scores: List[List[float]], rng: random.Random) -> int:
    """Sample a parent index ~ its Pareto win-frequency (Algorithm 2, step 5)."""
    kept, freq = pareto_frontier(scores)
    cands = [c for c in sorted(kept) if freq[c] > 0]
    if not cands:  # degenerate: fall back to best average
        return max(range(len(scores)), key=lambda c: sum(scores[c]))
    weights = [freq[c] for c in cands]
    return rng.choices(cands, weights=weights, k=1)[0]


# ===========================================================================
# The optimizer: swap evolve()'s greedy aggregator for Pareto illumination
# ===========================================================================


class ParetoAggregator(AggregatorProtocol):
    """A drop-in optimizer that realises GEPA's genetic/Pareto loop.

    Keeps a **pool** of candidate instructions (states), each with its per-
    instance score row over ``D_pareto`` (= the verifier's held-out set). Each
    round it admits the round's mutated candidates **only if they improved on
    their minibatch** (GEPA Alg. 1: "if sigma' improves"), then samples the next
    parent from the Pareto frontier and makes it the dev head, so `evolve()`
    mutates the illumination-selected parent next round."""

    def __init__(self, ledger: Ledger, verifier, audit, config, policy,
                 artifact_id: str = "gepa_prompt", seed: int = 0):
        self.ledger = ledger
        self.verifier = verifier
        self.artifact_id = artifact_id
        self.rng = random.Random(seed)
        self._cards: List[EvidenceCard] = []
        # pool: list of (state_dict, per_instance_scores, avg). Parallel lists so
        # score rows stay aligned for pareto_frontier.
        self.states: List[Dict[str, str]] = []
        self.scores: List[List[float]] = []
        self._seen: Set[str] = set()

    # -- per-instance evaluation over D_pareto -------------------------------

    def _score_row(self, artifact) -> List[float]:
        # eval_fn is a.score(ts); binary reward -> {0,1} per instance, cached.
        return [self.verifier.eval_fn(artifact, [t]) for t in self.verifier.held_out]

    def _admit(self, artifact) -> None:
        key = artifact.render()
        if key in self._seen:
            return
        self._seen.add(key)
        self.states.append(dict(artifact.state))
        self.scores.append(self._score_row(artifact))

    def best_index(self) -> int:
        return max(range(len(self.scores)),
                   key=lambda c: sum(self.scores[c])) if self.scores else -1

    @property
    def best_avg(self) -> float:
        i = self.best_index()
        return sum(self.scores[i]) / len(self.scores[i]) if i >= 0 else 0.0

    @property
    def best_state(self) -> Dict[str, str]:
        i = self.best_index()
        return dict(self.states[i]) if i >= 0 else {}

    # -- AggregatorProtocol --------------------------------------------------

    def ingest(self, card: EvidenceCard) -> None:
        self._cards.append(card)

    def step(self) -> List[MergeReport]:
        snap = self.ledger.snapshot(Ledger.DEV)
        head = snap.get(self.artifact_id)
        base_vv = {self.artifact_id: snap.version.get(self.artifact_id, 0)}
        if not self.states:                       # seed the pool with the base system
            self._admit(head)

        cards, self._cards = self._cards, []
        admitted = 0
        for card in cards:
            # GEPA accepts a child into the pool only if its minibatch improved.
            if card.before_after_delta <= 0:
                continue
            candidate = head.apply(card.diff)
            before = len(self.states)
            self._admit(candidate)
            admitted += len(self.states) - before

        # Algorithm 2: sample the next parent from the Pareto frontier.
        k = pareto_select(self.scores, self.rng)
        target_state = self.states[k]
        report_diff = None
        committed_version = None
        if target_state != head.state:            # make it the dev head
            parent = head.apply(Diff(diff_id=f"pareto:{k}", target=self.artifact_id,
                                     ops=dict(target_state), author="gepa"))
            try:
                _, committed_version = self.ledger.commit(
                    parent, base_vv, branch=Ledger.DEV,
                    message=f"gepa: select pareto parent #{k}")
                report_diff = parent.diff(head)
            except CASConflict:
                committed_version = None
        return [MergeReport(self.artifact_id, report_diff, False, len(cards),
                            admitted, 0, 0, self.best_avg, committed_version,
                            f"pool={len(self.states)} parent=#{k} best_avg={self.best_avg:.3f}")]


def pareto_aggregator_factory(artifact_id: str = "gepa_prompt", seed: int = 0):
    """Build the `aggregator_factory=` callable evolve() expects, capturing the
    live aggregator in `.last` so the example can read GEPA's best candidate."""
    holder = {}

    def factory(ledger, verifier, audit, config, policy):
        agg = ParetoAggregator(ledger, verifier, audit, config, policy,
                               artifact_id=artifact_id, seed=seed)
        holder["agg"] = agg
        return agg

    factory.holder = holder  # type: ignore[attr-defined]
    return factory


# ===========================================================================
# Dataset: HotpotQA (distractor), loaded dependency-free
# ===========================================================================


def download_hotpotqa(limit: int) -> List[dict]:
    dataset, split, config = HOTPOTQA
    return hf_rows(dataset, split, config=config, limit=limit)


def _render_context(context: dict, max_paragraphs: int = 10) -> str:
    titles = context["title"][:max_paragraphs]
    sents = context["sentences"][:max_paragraphs]
    return "\n".join(f"[{t}] {''.join(s)}" for t, s in zip(titles, sents))


def build_tasks(rows: List[dict], limit: int, seed: int = 0) -> List[Task]:
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    tasks: List[Task] = []
    for i in idx[:limit]:
        r = rows[i]
        prompt = (f"Context:\n{_render_context(r['context'])}\n\n"
                  f"Question: {r['question']}")
        tasks.append(Task(id=f"hp{i}", prompt=prompt, meta={"target": r["answer"]}))
    return tasks


# ===========================================================================
# Scoring: HotpotQA exact match (SQuAD-style normalisation)
# ===========================================================================


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _extract_answer(output: str) -> str:
    m = re.search(r"answer\s*:\s*(.+)", output, re.IGNORECASE)
    return (m.group(1) if m else output).strip()


def make_reward():
    """Exact match on the gold answer (GEPA's HotpotQA metric)."""
    def reward(task: Task, output: str) -> float:
        pred = normalize_answer(_extract_answer(output))
        gold = normalize_answer(task.meta["target"])
        return 1.0 if pred == gold and gold else 0.0
    return reward


def estimate_calls(rounds: int, workers: int, held_out: int) -> int:
    # generator + mutation per worker, plus per-instance pareto evals (cached).
    return rounds * (workers * 3 + held_out)


def load_dataset(fetch: int, ratios=(0.5, 0.25, 0.25), seed: int = 0) -> Dataset:
    """HotpotQA tasks split into train / val (D_pareto) / test."""
    tasks = build_tasks(download_hotpotqa(fetch), limit=fetch, seed=seed)
    return split_dataset(tasks, ratios=ratios, seed=seed, name="HotpotQA")


def evaluate(agent, instruction: str, tasks: List[Task], reward) -> float:
    """Score an instruction on a held-out split (the reported test metric)."""
    if not tasks:
        return 0.0
    return sum(reward(t, agent.solve(instruction, t)) for t in tasks) / len(tasks)


# ===========================================================================
# main
# ===========================================================================


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--provider", default="claude", choices=["claude", "glm"])
    p.add_argument("--model", default="claude-haiku-4-5")
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--fetch", type=int, default=48, help="HotpotQA rows to fetch")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--yes", action="store_true")
    args = p.parse_args()

    print("Algorithm: GEPA (Reflective Prompt Evolution) -- skill/prompt self-evolution")
    print("Dataset  : HotpotQA (multi-hop QA, distractor), exact-match")
    ds = load_dataset(args.fetch, seed=args.seed)
    reward = make_reward()

    ntr, nva, nte = ds.sizes()
    print(f"Loaded   : {len(ds)} tasks -> {ntr} train / {nva} val (D_pareto) / {nte} test")
    print("\nExample problem:")
    print("  Q:", ds.train[0].prompt.split('Question: ')[-1][:160])
    print("  A:", ds.train[0].meta["target"])

    est = estimate_calls(args.rounds, args.workers, nva) + nte
    print(f"\nPlan     : model={args.model}, rounds={args.rounds}, workers={args.workers}")
    print(f"Budget   : up to ~{est} model calls (cached repeats are free)")

    if args.dry_run:
        print("\n[dry-run] not calling the API. Drop --dry-run to evolve the prompt.")
        return

    if not args.yes and sys.stdin.isatty():
        if input("\nProceed with real API calls? [y/N] ").strip().lower() not in ("y", "yes"):
            print("aborted.")
            return

    completion = (openai_compatible(model=args.model) if args.provider == "glm"
                  else claude(model=args.model))
    agent = gepa_agent(completion)
    try:
        agent.solve("", Task(id="probe", prompt="Reply with the single word: ok"))
    except Exception as e:  # noqa: BLE001
        print(f"\nCould not reach the model ({type(e).__name__}: {e}).")
        print("For --provider glm set OPENAI_BASE_URL + OPENAI_API_KEY; "
              "for claude set ANTHROPIC_API_KEY (or `ant auth login`).")
        return

    factory = pareto_aggregator_factory(artifact_id="gepa_prompt", seed=args.seed)
    print("\nEvolving instruction (reflective mutation + Pareto selection, L2)...\n")
    # fit on train, Pareto-select on val (D_pareto); test stays fully held out.
    evolve(ds.trainval, reward, agent=agent,
           strategy=InstructionSlot(), initial_state={"instruction": _SEED_INSTRUCTION},
           blast_radius=0.2, artifact_id="gepa_prompt",
           rounds=args.rounds, n_workers=args.workers, held_out_frac=ds.val_frac,
           aggregator_factory=factory, verbose=True)

    agg: ParetoAggregator = factory.holder["agg"]  # type: ignore[attr-defined]
    best = agg.best_state.get("instruction", _SEED_INSTRUCTION)
    test_em = evaluate(agent, best, ds.test, reward)
    print("\n=== GEPA-optimised instruction (best average on D_pareto) ===")
    print(best)
    print(f"\ncandidates explored: {len(agg.states)}")
    print(f"seed D_pareto EM   : {agg.scores[0] and sum(agg.scores[0]) / len(agg.scores[0]):.3f}")
    print(f"best D_pareto EM   : {agg.best_avg:.3f}")
    print(f"test EM            : {test_em:.3f}  (held out, never seen by the optimizer)")


if __name__ == "__main__":
    main()
