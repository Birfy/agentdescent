"""Skill self-evolution, faithful port: **ACE — Agentic Context Engineering**.

Paper : "Agentic Context Engineering: Evolving Contexts for Self-Improving
        Language Models", Qizheng Zhang et al., 2025 (arXiv:2510.04618).
Repo  : https://github.com/ace-agent/ace
Dataset: **FiNER-139** (financial XBRL tagging), the light finance benchmark
        ACE evaluates on (Loukas et al., ACL 2022; HF `nlpaueb/finer-139`).

ACE evolves the *context* (a "playbook" of lessons) instead of the weights, via
three roles that map exactly onto AgentDescent's `evolve()` loop:

    ACE role     AgentDescent piece                     what it does
    ---------    --------------------------------    ------------------------------
    Generator    LLMAgent.solve (the run)            solves a task with the playbook
    Reflector    LLMAgent.propose (ACE template)     distils ONE delta bullet from a
                                                      failed trajectory
    Curator      the Aggregator (this framework)     **deterministic, non-LLM** merge:
                                                      dedup + statistical acceptance

The two ACE design principles are preserved:

* **Incremental delta updates (no monolithic rewrite).** The `ACEPlaybook`
  strategy only ever *appends* a new itemised bullet (a `Diff` with one new
  content-addressed key) or leaves the playbook untouched -- it never regenerates
  the whole context, so ACE's "context collapse" cannot happen.
* **Grow-and-refine de-duplication.** Before a bullet is admitted, `to_diff`
  drops it if it is identical or *near-duplicate* (lexical Jaccard proxy for
  ACE's embedding de-dup) of an existing bullet in the same section.

ACE's per-bullet *helpful / harmful* counters become the aggregator's per-diff
**Beta-posterior acceptance**: a bullet is committed only if it raises held-out
reward (evidence it was helpful), and rejected otherwise -- the discrete-space
analogue ACE's counters approximate. Skill layer -> `blast_radius=0.2` (L2).

    python -m examples.ace.ace_context_evolution --dry-run          # plan only, zero network
    python -m examples.ace.ace_context_evolution --model claude-haiku-4-5
    python -m examples.ace.ace_context_evolution --top-k 10 --rounds 6

Faithful-but-tractable simplifications (documented, not hidden): we use FiNER's
single-entity sentences and restrict to the `--top-k` most frequent XBRL tags so
a learned lesson transfers to held-out problems; ACE's full setup also runs
AppWorld (a heavy simulator) which this dependency-free example omits.
"""

from __future__ import annotations

import threading

import argparse
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

from agentdescent.agents import Usage
from agentdescent.dataloader import Dataset, hf_feature_names, hf_rows, split_dataset
from agentdescent.evolvable import Diff
from agentdescent.evolution import EvolvingArtifact, LLMAgent, Task, evolve, rule_id
from agentdescent.governance import classify
from agentdescent.parallel import DataParallel
from agentdescent.sampling import DifficultyWeighted, RoundRobin
from examples._common import (add_standard_args, completion_for, confirm,
                              score_tasks, worker_count,
                              budget_kwargs, capped_val, report_engine)

FINER = ("nlpaueb/finer-139", "validation", "finer-139")   # (dataset, split, config)


# ===========================================================================
# The ACE representation: an itemised, incremental-delta playbook
# ===========================================================================


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def jaccard(a: str, b: str) -> float:
    """Lexical overlap -- a dependency-free proxy for ACE's embedding de-dup."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class ACEPlaybook:
    """ACE's context object: sectioned, itemised bullets, grown by delta only.

    State is ``{bullet_id: "SECTION :: text"}``. A proposal is ``"section ::
    lesson"``. `to_diff` **appends** a new content-addressed bullet, or returns
    ``None`` when the lesson duplicates (exact or near-duplicate) an existing one
    -- ACE's *grow-and-refine*. It never rewrites existing bullets, so the
    playbook accumulates detail instead of collapsing into a lossy summary."""

    def __init__(self, title: str = "# Playbook (ACE)", dedup_threshold: float = 0.8):
        self.title = title
        self.dedup_threshold = dedup_threshold

    def initial(self) -> Dict[str, str]:
        return {}

    @staticmethod
    def _split(entry: str) -> Tuple[str, str]:
        section, _, text = entry.partition(" :: ")
        return (section or "general"), text

    def render(self, state: Dict[str, str]) -> str:
        if not state:
            return f"{self.title}\n(empty -- no lessons yet)"
        by_section: Dict[str, List[str]] = {}
        for bid in sorted(state):
            section, text = self._split(state[bid])
            by_section.setdefault(section, []).append(f"  - [{bid}] {text}")
        out = [self.title]
        for section in sorted(by_section):
            out.append(f"## {section}")
            out.extend(by_section[section])
        return "\n".join(out)

    def to_diff(self, state, proposal, author, base_version, target) -> Optional[Diff]:
        m = re.match(r"\s*([\w\- /]+?)\s*::\s*(.+)", proposal, re.DOTALL)
        if m:
            section, text = m.group(1).strip().lower(), m.group(2).strip()
        else:
            section, text = "general", proposal.strip()
        if not text:
            return None
        entry = f"{section} :: {text}"
        bid = rule_id(entry)
        if bid in state:
            return None  # exact duplicate
        # grow-and-refine: reject a near-duplicate of an existing bullet in the
        # same section (proactive de-dup, ACE Section 3.3).
        for other in state.values():
            other_section, other_text = self._split(other)
            if other_section == section and jaccard(text, other_text) >= self.dedup_threshold:
                return None
        return Diff(diff_id=f"{author}:{bid}:{base_version}", target=target,
                    ops={bid: entry}, author=author)


# ACE separates the Reflector from the Curator "to avoid conflating what to learn
# with how to store it" (paper Section 3.2). The Curator is this framework's
# deterministic aggregator, so here we only need the Reflector prompt.
_GENERATOR_TMPL = (
    "You are an expert financial-analysis agent tagging figures in SEC filings "
    "with US-GAAP XBRL concepts.\n\n{artifact}\n\n"
    "Use the playbook above. Read the sentence and output ONLY the single XBRL "
    "tag for the amount wrapped in <<>>. Choose exactly one tag from the "
    "candidate list; output the tag name and nothing else.\n\n{prompt}"
)
_REFLECTOR_TMPL = (
    "You are the ACE Reflector. The agent mis-tagged an XBRL amount (score "
    "{reward:.2f}).\n\nPlaybook so far:\n{artifact}\n\nTask:\n{prompt}\n\n"
    "The agent answered:\n{output}\n\n"
    "Diagnose the specific reasoning error and distil ONE concise, general, "
    "reusable lesson that would fix this and similar cases. Format your answer "
    "as `section :: lesson` where `section` is the XBRL concept family (e.g. "
    "`revenue`, `debt`, `shares`). Output only that one line, or NONE."
)


def ace_agent(complete) -> LLMAgent:
    """Generator + Reflector bundled as an LLMAgent over a completion."""
    return LLMAgent(complete, solve_template=_GENERATOR_TMPL,
                    propose_template=_REFLECTOR_TMPL)


# ===========================================================================
# Dataset: FiNER-139 (financial XBRL tagging), loaded dependency-free
# ===========================================================================


def _entities(tokens: List[str], tags: List[int], names: List[str]
              ) -> List[Tuple[int, int, str]]:
    """Return contiguous (start, end, entity_type) spans (BIO decoding)."""
    spans, i = [], 0
    while i < len(tags):
        name = names[tags[i]]
        if name.startswith("B-"):
            etype = name[2:]
            j = i + 1
            while j < len(tags) and names[tags[j]] == f"I-{etype}":
                j += 1
            spans.append((i, j, etype))
            i = j
        else:
            i += 1
    return spans


def download_finer(pool: int) -> Tuple[List[dict], List[str]]:
    """Fetch `pool` validation rows and the XBRL label vocabulary (via dataloader)."""
    dataset, split, config = FINER
    rows = hf_rows(dataset, split, config=config, limit=pool)
    names = hf_feature_names(dataset, split, "ner_tags", config=config)
    return rows, names


def build_tasks(rows: List[dict], names: List[str], limit: int, top_k: int,
                seed: int = 0) -> List[Task]:
    """Single-entity FiNER sentences restricted to the top-k frequent tags.

    Faithful to FiNER (real filings, real US-GAAP concepts) but framed so a
    learned lesson transfers: each task highlights one amount and asks for its
    tag, chosen from the k most frequent concepts in the pool."""
    import random

    single = []
    for r in rows:
        spans = _entities(r["tokens"], r["ner_tags"], names)
        if len(spans) == 1:
            single.append((r, spans[0]))
    freq = Counter(etype for _, (_, _, etype) in single)
    keep = {etype for etype, _ in freq.most_common(top_k)}
    candidates = sorted(keep)
    cand_block = "Candidate tags: " + ", ".join(candidates)

    pool = [(r, span) for r, span in single if span[2] in keep]
    rng = random.Random(seed)
    rng.shuffle(pool)

    tasks: List[Task] = []
    for idx, (r, (start, end, etype)) in enumerate(pool[:limit]):
        toks = list(r["tokens"])
        toks[start] = "<<" + toks[start]
        toks[end - 1] = toks[end - 1] + ">>"
        sentence = " ".join(toks)
        prompt = f"{cand_block}\n\nSentence:\n{sentence}"
        tasks.append(Task(id=f"finer{idx}", prompt=prompt,
                          meta={"target": etype, "candidates": candidates}))
    return tasks


# ===========================================================================
# Scoring
# ===========================================================================


def _canonical_tag(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def make_reward():
    """Exact match on the XBRL concept (case/punctuation-insensitive)."""
    def reward(task: Task, output: str) -> float:
        gold = _canonical_tag(task.meta["target"])
        out = _canonical_tag(output)
        if out == gold:
            return 1.0
        # accept the gold tag appearing as a standalone answer token.
        for tok in re.split(r"[\s,]+", output.strip()):
            if _canonical_tag(tok) == gold:
                return 1.0
        return 0.0
    return reward


def estimate_calls(rounds: int, workers: int, held_out: int) -> int:
    per_round = workers * 3 + (4 * min(8, held_out) + 2 * held_out)
    return rounds * per_round


def load_dataset(pool: int, top_k: int, ratios=(0.5, 0.25, 0.25), seed: int = 0) -> Dataset:
    """FiNER single-entity tasks, split train/val/test (stratified by concept)."""
    rows, names = download_finer(pool)
    tasks = build_tasks(rows, names, limit=10 ** 9, top_k=top_k, seed=seed)
    return split_dataset(tasks, ratios=ratios, seed=seed,
                         stratify_key=lambda t: t.meta["target"], name="FiNER-139")


def evaluate(agent, rendered: str, tasks: List[Task], reward,
             concurrency: int = 8) -> float:
    """Score a rendered playbook on a held-out split (the reported test metric).

    ``concurrency`` is not decorative: it defaulted to ``score_tasks``' 8 and
    nothing passed it, so a run asked for ``--eval-concurrency 1`` still scored
    its test split eight at a time. Measured on the serial matrix arm -- one
    worker, eval concurrency 1 -- that put 334s of model time inside a 209s
    wall-clock, an overlap of 1.6x on the run that is supposed to be the
    fully-serial control.
    """
    return score_tasks(agent.solve, rendered, tasks, reward,
                       concurrency=concurrency)


# ===========================================================================
# main
# ===========================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    add_standard_args(p, max_seconds_default=30.0)
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--top-k", type=int, default=120,
                   help="restrict to the k most frequent XBRL concepts -- this is "
                        "the difficulty knob, and the default is the first value "
                        "that demonstrates anything. deepseek-v4-flash scores 1.000 "
                        "at 10 (nothing to learn) and 0.850 at 40 (headroom, but no "
                        "bullet beats the baseline); at 120 it goes 0.844 -> 0.889")
    p.add_argument("--pool", type=int, default=800,
                   help="FiNER validation rows to scan for single-entity sentences")
    p.add_argument("--sampler", choices=["round-robin", "difficulty"],
                   default="round-robin",
                   help="which task a worker rolls out next (agentdescent.sampling)")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    # --serial collapses this to the upstream algorithm's own semantics:
    # one worker, nothing to merge. Applied to args so the printed plan,
    # the cost estimate and the run cannot disagree about what ran.
    args.workers = worker_count(args, args.workers)

    print("Algorithm: ACE (Agentic Context Engineering) -- skill/context self-evolution")
    print("Dataset  : FiNER-139 (financial XBRL tagging)")
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

    ds = load_dataset(args.pool, args.top_k, seed=args.seed)
    if len(ds) < 4:
        print(f"Only {len(ds)} single-entity tasks in the pool; raise --pool.")
        return
    reward = make_reward()

    ntr, nva, nte = ds.sizes()
    art = EvolvingArtifact("ace_context", blast_radius=0.2)
    print(f"Loaded   : {len(ds)} single-entity tasks, top-{args.top_k} concepts")
    # Which governance layer this run actually landed in -- the line that tells a
    # configuration mistake from a result, and which `docs/governance.md` says
    # every port prints.
    print(f"Governance: context blast_radius={art.blast_radius} -> {classify(art).name}")
    print(f"Splits   : {ntr} train / {nva} val / {nte} test")
    print(f"Concepts : {', '.join(ds.train[0].meta['candidates'])}")
    print("\nExample problem:")
    print("  Q:", ds.train[0].prompt[-200:])
    print("  A:", ds.train[0].meta["target"])

    est = estimate_calls(args.rounds, args.workers, nva) + nte
    print(f"Budget   : up to ~{est} model calls (cached repeats are free)")

    if not confirm(args):
        return

    usage = Usage()                       # what the run actually costs
    completion = completion_for(args, usage=usage)
    agent = ace_agent(completion)
    try:
        agent.solve("", Task(id="probe", prompt="Reply with the single word: ok"))
    except Exception as e:  # noqa: BLE001
        print(f"\nCould not reach the model ({type(e).__name__}: {e}).")
        print("For --provider glm set OPENAI_BASE_URL + OPENAI_API_KEY; "
              "for claude set ANTHROPIC_API_KEY (or `ant auth login`).")
        return

    mode = "async, barrier-free" if args.asynchronous else "synchronous DP"
    print(f"\nEvolving context (Generator + Reflector + deterministic Curator, L2; {mode})...\n")
    # fit on train, gate on val (evolve's held-out); test stays fully held out.
    # FiNER's stratified split hands back a *large* val -- 82 of 211 tasks at
    # `--pool 1600 --top-k 120`, because a 120-way concept key leaves most strata
    # with one member. The Curator scores every admitted candidate across all of
    # it, so val, not the rollout, is what a run costs: at 82 the gate is 97% of
    # the work and `n_workers` has almost nothing left to parallelise. `--val-cap`
    # is the only lever on that ratio and it moves neither train nor test -- the
    # cap's leftovers become train. A capped gate is a noisier gate, which is a
    # real cost and the reason this is a flag rather than a default.
    trainval, val_frac = capped_val(ds.trainval, ds.val_frac, args.val_cap)
    result = evolve(trainval, reward, agent=agent,
                    strategy=ACEPlaybook(), parallel=DataParallel(),
                    blast_radius=0.2, artifact_id="ace_playbook",
                    rounds=args.rounds, n_workers=args.workers,
                    max_concurrency=1 if args.asynchronous else args.workers,
                    task_sampler=(DifficultyWeighted() if args.sampler == "difficulty"
                                  else RoundRobin()),
                    asynchronous=args.asynchronous, async_ratio=args.async_ratio,
                    max_seconds=args.max_seconds if args.asynchronous else None,
                    held_out_frac=val_frac,
                    # Never passed, so the gate ran at `evolve()`'s default of 8
                    # whatever `--eval-concurrency` said. The flag reached only
                    # the final test measurement, and even a `--serial
                    # --eval-concurrency 1` arm scored its candidates eight at a
                    # time: measured 563s of model time inside a 476s wall-clock
                    # on the one-worker control, which is 1.18x of concurrency in
                    # the arm whose whole job is to have none.
                    eval_concurrency=args.eval_concurrency,
                    verbose=True,
                    **budget_kwargs(args))

    test_acc = evaluate(agent, result.rendered, ds.test, reward,
                        concurrency=args.eval_concurrency)
    print("\n=== evolved ACE playbook ===")
    print(result.rendered)
    # Round 0's score, not the seed playbook's: by the time `history[0]` is
    # recorded that round's merge has already happened. And guarded, because a
    # run that died before finishing a round still returns a usable result --
    # indexing an empty history turned that into an IndexError.
    if result.history:
        print(f"\nval accuracy : {result.history[0].held_out_reward:.3f} (round 0) "
              f"-> {result.final_reward:.3f} (final)")
    else:
        print(f"\nval accuracy : {result.final_reward:.3f} (no round completed)")
    print(f"test accuracy: {test_acc:.3f}  (held out, never seen by the Curator)")
    print(f"bullets curated: {len(result.state)}")
    print(f"stop reason  : {result.stop_reason}")
    report_engine(result)
    if result.error:
        print(f"WARNING: the run did not finish cleanly -- {result.error}")
    print(f"model usage  : {usage.summary()}")


if __name__ == "__main__":
    main()
