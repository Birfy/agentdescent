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

    python -m examples.gepa.gepa_prompt_evolution --dry-run           # plan only, zero network
    python -m examples.gepa.gepa_prompt_evolution --model claude-haiku-4-5
    python -m examples.gepa.gepa_prompt_evolution --rounds 10 --workers 3

Faithful-but-tractable simplifications (documented): GEPA optimises a
multi-module compound system with a rollout budget; here the system is a single
instruction module and the minibatch is the per-round worker sample (raise
`--workers` for a bigger minibatch). The Pareto set is the held-out split.
"""

from __future__ import annotations

import threading

import argparse
import random
import re
import string
from typing import Dict, List, Optional, Sequence, Set, Tuple

from agentdescent.agents import Usage
from agentdescent.aggregator import AggregatorProtocol, MergeReport
from agentdescent.dataloader import Dataset, hf_rows, split_dataset
from agentdescent.evolvable import Diff, EvidenceCard
from agentdescent.evolution import EvolvingArtifact, LLMAgent, Task, evolve, rule_id
from agentdescent.governance import classify
from agentdescent.ledger import CASConflict, Ledger
from agentdescent.selection import ParetoFrontier, pareto_win_frequency
from examples._common import (add_standard_args, completion_for, confirm,
                              score_tasks, worker_count,
                              budget_kwargs, capped_val, report_engine)

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


#: Algorithm 2 steps 1-4, now shipped as
#: :func:`agentdescent.selection.pareto_win_frequency`. Kept under this name
#: because it is what the paper's step numbering calls it and what this port's
#: tests reach for; the implementation moved so that a *run* can name the rule
#: it used (``ParetoFrontier(mode='win_frequency')``) instead of a reader
#: having to find this file.
pareto_frontier = pareto_win_frequency


def pareto_select(scores: List[List[float]], rng: random.Random) -> int:
    """Sample a parent index ~ its Pareto win-frequency (Algorithm 2, step 5).

    The index form, for the tests and for anyone reading the algorithm rather
    than driving the engine. The engine drives
    :class:`~agentdescent.selection.ParetoFrontier` with ``mode='win_frequency'``,
    which is this function over `Candidate` rows and draws from the same rng in
    the same order -- `tests/test_port_selection_equivalence.py` steps the two
    in lockstep rather than comparing their distributions.
    """
    kept, freq = pareto_win_frequency(scores)
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
    round it admits the round's mutated candidates that did not regress on their
    feedback sample (GEPA Alg. 1: "if sigma' improves"), then samples the next
    parent from the Pareto frontier and makes it the dev head, so `evolve()`
    mutates the illumination-selected parent next round.

    **Documented deviation: the feedback minibatch has size 1.** GEPA evaluates a
    child on a minibatch of size *b* and compares means; `evolve()` gives one
    rollout per worker per round, so the comparison is a single Bernoulli draw.
    The admission test is therefore "did not regress" rather than "improved" --
    see the note at the gate. Everything downstream (Algorithm 2's per-instance
    frontier, domination pruning, win-frequency sampling) is scored on the full
    ``D_pareto`` row and is unaffected."""

    def __init__(self, ledger: Ledger, verifier, audit, config, policy,
                 artifact_id: str = "gepa_prompt", seed: int = 0,
                 eval_concurrency: int = 8, merge_round=None,
                 selection: Optional["ParetoFrontier"] = None):
        self.ledger = ledger
        self.eval_concurrency = eval_concurrency
        #: Optional :class:`~agentdescent.policies.FusionPolicy`. When set, the
        #: round's surviving diffs are combined into **one** candidate before
        #: admission, so the pool grows by one per round instead of one per
        #: worker.
        #:
        #: A policy, not a bare callable, because the shipped one needs the
        #: verifier: `ReflectiveFusion` builds its synthesis-failure fallback in
        #: `bind()`, which only the engine calls when the policy is installed
        #: through `Policies`. Used from here that never happens, and the
        #: fallback stays `None` until a merge fails and the run dies on
        #: `'NoneType' has no attribute 'select'` -- twenty minutes in, on the
        #: rare path. Binding here is what makes the policy reusable outside the
        #: engine's own wiring.
        self.merge_round = merge_round
        if merge_round is not None and hasattr(merge_round, "bind"):
            merge_round.bind(verifier)
        self.verifier = verifier
        self.artifact_id = artifact_id
        self.rng = random.Random(seed)
        # Algorithm 2 at the shipped seam. `rng=` rather than `seed=` because
        # this aggregator owns the stream and spends it elsewhere too: a
        # policy that re-seeded per call would agree on the distribution and
        # disagree with every number this port has published.
        self.selection = selection or ParetoFrontier(
            mode="win_frequency", rng=self.rng)
        self._cards: List[EvidenceCard] = []
        self._lock = threading.Lock()   # ingest: workers; step: one thread
        # pool: list of (state_dict, per_instance_scores, avg). Parallel lists so
        # score rows stay aligned for pareto_frontier.
        self.states: List[Dict[str, str]] = []
        self.scores: List[List[float]] = []
        self._seen: Set[str] = set()

    # -- per-instance evaluation over D_pareto -------------------------------

    def _score_rows(self, artifacts) -> List[List[float]]:
        """Per-instance rows over D_pareto for several candidates at once.

        The port's largest cost, and it was serial in **two** dimensions. Within
        a row it was a comprehension over D_pareto -- on 20 instances at ~34 s a
        call, eleven minutes per candidate, in silence. Across rows, `step()`
        admitted one card at a time, so a round from eight workers queued eight
        of those sweeps back to back. `evolve(eval_concurrency=)` reaches
        neither, because that knob bounds the *engine's* gate and this aggregator
        replaces the engine's.

        Both dimensions flatten into one pool rather than nesting, so in-flight
        work is exactly ``eval_concurrency`` instead of its square -- nesting two
        pools of 16 would put 256 requests on the endpoint and measure its rate
        limiter rather than the algorithm.

        Concurrency changes no result: each (candidate, instance) score is
        independent, ``eval_fn`` is pure, rows are reassembled in order, and the
        evaluation cache already memoises per (artifact, task).
        """
        artifacts = list(artifacts)
        tasks = list(self.verifier.held_out)
        if not artifacts or not tasks:
            return [[] for _ in artifacts]
        pairs = [(a, t) for a in artifacts for t in tasks]
        if len(pairs) < 2 or self.eval_concurrency < 2:
            flat = [self.verifier.eval_fn(a, [t]) for a, t in pairs]
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(
                    max_workers=min(self.eval_concurrency, len(pairs))) as pool:
                flat = list(pool.map(
                    lambda pair: self.verifier.eval_fn(pair[0], [pair[1]]), pairs))
        width = len(tasks)
        return [flat[i * width:(i + 1) * width] for i in range(len(artifacts))]

    def _score_row(self, artifact) -> List[float]:
        """One candidate's row. For callers that admit a single artifact."""
        return self._score_rows([artifact])[0]

    def _admit_many(self, artifacts) -> int:
        """Admit several candidates, scoring all their rows in one pass.

        Dedup runs first and sequentially, so which candidates enter the pool --
        and in what order -- never depends on thread timing. Only the scoring is
        concurrent, and `states`/`scores` are appended in dedup order because
        `pareto_frontier` reads them as parallel lists.
        """
        fresh = []
        for artifact in artifacts:
            key = artifact.render()
            if key in self._seen:
                continue
            self._seen.add(key)
            fresh.append(artifact)
        if not fresh:
            return 0
        for artifact, row in zip(fresh, self._score_rows(fresh)):
            self.states.append(dict(artifact.state))
            self.scores.append(row)
        return len(fresh)

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
        # ingest runs on worker threads, step on one: see AggregatorProtocol.
        with self._lock:
            self._cards.append(card)

    def step(self) -> List[MergeReport]:
        snap = self.ledger.snapshot(Ledger.DEV)
        head = snap.get(self.artifact_id)
        base_vv = {self.artifact_id: snap.version.get(self.artifact_id, 0)}
        if not self.states:                       # seed the pool with the base system
            self._admit(head)

        with self._lock:
            cards, self._cards = self._cards, []
        survivors: List[Diff] = []
        for card in cards:
            # GEPA Alg. 1 admits a child whose score on the feedback minibatch is
            # not worse than its parent's. Here that "minibatch" has size **1**:
            # `evolve()` rolls out one task per worker per round, so
            # `before_after_delta` is a single-instance measurement, and for a
            # binary reward like HotpotQA EM it is exactly {-1, 0, +1}.
            #
            # So this gate must not be `> 0`. That demanded the one sampled
            # instance flip wrong->right -- and it is the instance the mutation was
            # generated *from*, so a prompt that helps broadly but does not fix that
            # particular question was thrown away before it was ever scored.
            # Rejected candidates never enter `self.states`, never get a
            # `_score_row`, and so can never reach the Pareto frontier -- which is
            # precisely the "complementary specialist" the frontier exists to keep
            # alive. `>= 0` filters obvious regressions and leaves the selection to
            # domination pruning, which is what illumination is supposed to do.
            if card.before_after_delta < 0:
                continue
            survivors.append(card.diff)

        # **One candidate per round instead of one per worker.** Admission is what
        # this aggregator pays for: every candidate that enters the pool is scored
        # across the whole of D_pareto, so a round from N workers costs N sweeps.
        # Combining the round's survivors into a single diff first makes that one
        # sweep, which is the only way widening N makes the run *cheaper* rather
        # than merely wider.
        #
        # `fuse_diffs` is `ops.update()`, so on this port's one-key artifact it
        # would keep the last rewrite and silently drop the rest -- N-1 proposals
        # paid for and thrown away. `merge_round` is therefore a *model* merger
        # (`ReflectiveFusion`), which writes the union of the deltas.
        #
        # It changes what the Pareto pool contains: merged candidates rather than
        # individual mutations. Algorithm 2 itself -- the per-instance frontier,
        # domination pruning, win-frequency sampling -- is untouched and still
        # runs on whatever is in the pool. A row measured this way has to say so.
        if self.merge_round is not None and len(survivors) > 1:
            merged, _, _ = self.merge_round.select(head, survivors)
            if merged is not None:
                survivors = [merged]
        admitted = self._admit_many([head.apply(d) for d in survivors])

        # Algorithm 2 at the standard seam: candidates carry their per-instance
        # rows in Candidate.per_task; version carries the pool index.
        from agentdescent.selection import Candidate, SelectionContext
        task_ids = [t.id for t in self.verifier.held_out]
        rows = [Candidate(artifact_id=self.artifact_id, version=i,
                          state=dict(state),
                          score=sum(row) / len(row) if row else 0.0,
                          per_task=dict(zip(task_ids, row)))
                for i, (state, row) in enumerate(zip(self.states, self.scores))]
        ctx = SelectionContext(head=rows[0], candidates=tuple(rows), n_workers=1)
        k = self.selection.select(ctx, 1)[0].version
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


def pareto_aggregator_factory(artifact_id: str = "gepa_prompt", seed: int = 0,
                              eval_concurrency: int = 8, merge_round=None):
    """Build the `aggregator_factory=` callable evolve() expects, capturing the
    live aggregator in `.last` so the example can read GEPA's best candidate."""
    holder = {}

    def factory(ledger, verifier, audit, config, policy):
        agg = ParetoAggregator(ledger, verifier, audit, config, policy,
                               artifact_id=artifact_id, seed=seed,
                               eval_concurrency=eval_concurrency,
                               merge_round=merge_round)
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


def evaluate(agent, instruction: str, tasks: List[Task], reward,
             concurrency: int = 8) -> float:
    """Score an instruction on a held-out split (the reported test metric).

    `concurrency` was `score_tasks`'s default, so `--eval-concurrency` reached
    the Pareto rows but not the number the run is finally judged by."""
    return score_tasks(agent.solve, instruction, tasks, reward,
                       concurrency=concurrency)


# ===========================================================================
# main
# ===========================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    add_standard_args(p, max_seconds_default=45.0)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--fetch", type=int, default=48, help="HotpotQA rows to fetch")
    p.add_argument("--seed-instruction", default=_SEED_INSTRUCTION,
                   help=("the instruction the run starts from. The default is "
                         "already tuned, which starves the loop: most rollouts "
                         "answer correctly, propose nothing, and the gate rejects "
                         "the rest. Pass '' to start from nothing when the point "
                         "is to watch the loop work"))
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    # --serial collapses this to the upstream algorithm's own semantics:
    # one worker, nothing to merge. Applied to args so the printed plan,
    # the cost estimate and the run cannot disagree about what ran.
    args.workers = worker_count(args, args.workers)

    print("Algorithm: GEPA (Reflective Prompt Evolution) -- skill/prompt self-evolution")
    print("Dataset  : HotpotQA (multi-hop QA, distractor), exact-match")
    print(f"\nPlan     : model={args.model}, rounds={args.rounds}, workers={args.workers}")
    if args.asynchronous:
        print(f"Async    : {args.workers} workers, barrier-free (async_ratio={args.async_ratio}, "
              f"max {args.max_seconds:.0f}s); staleness policy rebases/discards stale diffs")
    else:
        print(f"Parallel : {args.workers} workers run concurrently each round "
              f"(synchronous DP; the aggregator merge is the barrier)")
    if args.dry_run:
        print("Data     : deferred (dry-run performs no network access)")
        print("\n[dry-run] plan only; no dataset or model API was accessed.")
        return

    ds = load_dataset(args.fetch, seed=args.seed)
    reward = make_reward()

    ntr, nva, nte = ds.sizes()
    art = EvolvingArtifact("gepa_prompt", blast_radius=0.2)
    print(f"Loaded   : {len(ds)} tasks -> {ntr} train / {nva} val (D_pareto) / {nte} test")
    # The layer this run landed in, as every other port reports it.
    print(f"Governance: instruction blast_radius={art.blast_radius} -> {classify(art).name}")
    print("\nExample problem:")
    print("  Q:", ds.train[0].prompt.split('Question: ')[-1][:160])
    print("  A:", ds.train[0].meta["target"])

    est = estimate_calls(args.rounds, args.workers, nva) + nte
    print(f"Budget   : up to ~{est} model calls (cached repeats are free)")

    if not confirm(args):
        return

    usage = Usage()                       # what the run actually costs
    completion = completion_for(args, usage=usage)
    agent = gepa_agent(completion)
    try:
        agent.solve("", Task(id="probe", prompt="Reply with the single word: ok"))
    except Exception as e:  # noqa: BLE001
        print(f"\nCould not reach the model ({type(e).__name__}: {e}).")
        print("For --provider glm set OPENAI_BASE_URL + OPENAI_API_KEY; "
              "for claude set ANTHROPIC_API_KEY (or `ant auth login`).")
        return

    # `--reflective-merge` keeps GEPA's Pareto selection and changes only what
    # enters its pool: the round's surviving diffs are merged into one candidate,
    # so admission costs one D_pareto sweep instead of one per worker. The merger
    # has to be the model one -- `fuse_diffs` on a one-key artifact keeps the last
    # rewrite and drops the rest.
    merge_round = None
    if args.reflective_merge:
        from agentdescent.fusion import ReflectiveFusion
        merge_round = ReflectiveFusion(completion)
        print("NOTE: --reflective-merge admits ONE merged candidate per round "
              "instead of one per worker. Pareto selection is unchanged; what "
              "the pool contains is not.")
    factory = pareto_aggregator_factory(artifact_id="gepa_prompt", seed=args.seed,
                                        eval_concurrency=args.eval_concurrency,
                                        merge_round=merge_round)
    print("\nEvolving instruction (reflective mutation + Pareto selection, L2)...\n")
    # fit on train, Pareto-select on val (D_pareto); test stays fully held out.
    # Keep the result: `error` is the engine's one signal that a long, paid run
    # ended on a rate limit rather than converging, and dropping it on the floor
    # is exactly what its docstring tells callers not to do.
    # D_pareto is the gate split and every admitted candidate is scored across
    # all of it, so this ratio decides how much of the run `n_workers` can
    # parallelise. Test is untouched.
    trainval, val_frac = capped_val(ds.trainval, ds.val_frac, args.val_cap)
    result = evolve(trainval, reward, agent=agent,
                    strategy=InstructionSlot(seed=args.seed_instruction),
                    initial_state={"instruction": args.seed_instruction},
                    blast_radius=0.2, artifact_id="gepa_prompt",
                    rounds=args.rounds, n_workers=args.workers,
                    max_concurrency=1 if args.asynchronous else args.workers,
                    asynchronous=args.asynchronous, async_ratio=args.async_ratio,
                    max_seconds=args.max_seconds if args.asynchronous else None,
                    held_out_frac=val_frac,
                    aggregator_factory=factory, verbose=True,
                    **budget_kwargs(args))

    agg: ParetoAggregator = factory.holder["agg"]  # type: ignore[attr-defined]
    best = agg.best_state.get("instruction", args.seed_instruction)
    test_em = evaluate(agent, best, ds.test, reward,
                       concurrency=args.eval_concurrency)
    print("\n=== GEPA-optimised instruction (best average on D_pareto) ===")
    print(best)
    print(f"\ncandidates explored: {len(agg.states)}")
    # The seed row exists only once the aggregator has stepped at least once, and
    # `scores[0] and .../len(...)` did not guard it: on an empty row that formats
    # a *list* with `:.3f` (TypeError), and with no row at all it is an IndexError
    # -- both raised in the final print of a run the engine had deliberately kept
    # alive to report.
    seed_row = agg.scores[0] if agg.scores else []
    seed_em = f"{sum(seed_row) / len(seed_row):.3f}" if seed_row else "n/a (no round completed)"
    print(f"seed D_pareto EM   : {seed_em}")
    print(f"best D_pareto EM   : {agg.best_avg:.3f}")
    print(f"test EM            : {test_em:.3f}  (held out, never seen by the optimizer)")
    print(f"stopped            : {result.stop_reason}")
    report_engine(result)
    if result.error:
        print(f"WARNING: the run did not finish cleanly -- {result.error}")
    print(f"model usage: {usage.summary()}")


if __name__ == "__main__":
    main()
