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

import argparse
import random
import re
import string
from typing import Dict, List, Optional, Sequence, Set, Tuple

from agentdescent.agents import Usage
from agentdescent.dataloader import Dataset, hf_rows, split_dataset
from agentdescent.evolvable import Diff
from agentdescent.evolution import EvolvingArtifact, LLMAgent, Task, evolve, rule_id
from agentdescent.governance import classify
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


# The Pareto selection and aggregator live in the package registry, so a spec
# can name `"algorithm": "gepa"` and run the same implementation (#384).
from agentdescent.algorithms.gepa import (  # noqa: E402
    ParetoAggregator,
    pareto_aggregator_factory,
    pareto_frontier,
    pareto_select,
)


# ===========================================================================
# Dataset: HotpotQA (distractor), loaded dependency-free
# ===========================================================================


def download_hotpotqa(limit: int, source: str = "hf") -> List[dict]:
    """HotpotQA validation rows, from HuggingFace or a ModelScope mirror.

    ``source="hf"`` (default) uses the datasets-server; ``source="modelscope"``
    fetches the same rows from ``OpenDataLab/HotpotQA`` on ModelScope, for
    networks where the HuggingFace endpoint is unreachable but ModelScope is
    up. Rows are identical either way (``context`` is the dict shape examples
    read)."""
    if source == "modelscope":
        from agentdescent.dataloader import hf_rows_modelscope
        return hf_rows_modelscope(limit=limit)
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


def load_dataset(fetch: int, ratios=(0.5, 0.25, 0.25), seed: int = 0,
                 source: str = "hf") -> Dataset:
    """HotpotQA tasks split into train / val (D_pareto) / test.

    ``source`` is passed to :func:`download_hotpotqa` -- ``"hf"`` (default)
    via the datasets-server, or ``"modelscope"`` for the ModelScope mirror."""
    tasks = build_tasks(download_hotpotqa(fetch, source=source), limit=fetch,
                        seed=seed)
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
                    pipelined_gate=args.asynchronous and args.pipelined_gate,
                    gate_workers=args.gate_workers,
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
