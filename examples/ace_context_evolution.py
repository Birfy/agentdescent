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

    python -m examples.ace_context_evolution --dry-run          # dataset + estimate, no API
    python -m examples.ace_context_evolution --model claude-haiku-4-5
    python -m examples.ace_context_evolution --top-k 10 --rounds 6

Faithful-but-tractable simplifications (documented, not hidden): we use FiNER's
single-entity sentences and restrict to the `--top-k` most frequent XBRL tags so
a learned lesson transfers to held-out problems; ACE's full setup also runs
AppWorld (a heavy simulator) which this dependency-free example omits.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

from agentdescent.agents import claude, openai_compatible
from agentdescent.dataloader import Dataset, hf_feature_names, hf_rows, split_dataset
from agentdescent.evolvable import Diff
from agentdescent.evolution import LLMAgent, Task, evolve, rule_id
from agentdescent.parallel import DataParallel

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


def evaluate(agent, rendered: str, tasks: List[Task], reward) -> float:
    """Score a rendered playbook on a held-out split (the reported test metric)."""
    if not tasks:
        return 0.0
    return sum(reward(t, agent.solve(rendered, t)) for t in tasks) / len(tasks)


# ===========================================================================
# main
# ===========================================================================


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--provider", default="claude", choices=["claude", "glm"])
    p.add_argument("--model", default="claude-haiku-4-5")
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--top-k", type=int, default=10,
                   help="restrict to the k most frequent XBRL concepts")
    p.add_argument("--pool", type=int, default=800,
                   help="FiNER validation rows to scan for single-entity sentences")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--async", dest="asynchronous", action="store_true",
                   help="run barrier-free (async_evolve): workers never wait for the merge")
    p.add_argument("--async-ratio", type=int, default=3, help="staleness lag budget")
    p.add_argument("--max-seconds", type=float, default=30.0, help="async wall-clock budget")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--yes", action="store_true")
    args = p.parse_args()

    print("Algorithm: ACE (Agentic Context Engineering) -- skill/context self-evolution")
    print("Dataset  : FiNER-139 (financial XBRL tagging)")
    ds = load_dataset(args.pool, args.top_k, seed=args.seed)
    if len(ds) < 4:
        print(f"Only {len(ds)} single-entity tasks in the pool; raise --pool.")
        return
    reward = make_reward()

    ntr, nva, nte = ds.sizes()
    print(f"Loaded   : {len(ds)} single-entity tasks, top-{args.top_k} concepts")
    print(f"Splits   : {ntr} train / {nva} val / {nte} test")
    print(f"Concepts : {', '.join(ds.train[0].meta['candidates'])}")
    print("\nExample problem:")
    print("  Q:", ds.train[0].prompt[-200:])
    print("  A:", ds.train[0].meta["target"])

    est = estimate_calls(args.rounds, args.workers, nva) + nte
    print(f"\nPlan     : model={args.model}, rounds={args.rounds}, workers={args.workers}")
    if args.asynchronous:
        print(f"Async    : {args.workers} workers, barrier-free (async_ratio={args.async_ratio}, "
              f"max {args.max_seconds:.0f}s); staleness policy rebases/discards stale diffs")
    else:
        print(f"Parallel : {args.workers} workers run concurrently each round "
              f"(synchronous DP; the aggregator merge is the barrier)")
    print(f"Budget   : up to ~{est} model calls (cached repeats are free)")

    if args.dry_run:
        print("\n[dry-run] not calling the API. Drop --dry-run to evolve the playbook.")
        return

    if not args.yes and sys.stdin.isatty():
        if input("\nProceed with real API calls? [y/N] ").strip().lower() not in ("y", "yes"):
            print("aborted.")
            return

    completion = (openai_compatible(model=args.model) if args.provider == "glm"
                  else claude(model=args.model))
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
    result = evolve(ds.trainval, reward, agent=agent,
                    strategy=ACEPlaybook(), parallel=DataParallel(),
                    blast_radius=0.2, artifact_id="ace_playbook",
                    rounds=args.rounds, n_workers=args.workers, max_concurrency=args.workers,
                    asynchronous=args.asynchronous, async_ratio=args.async_ratio,
                    max_seconds=args.max_seconds if args.asynchronous else None,
                    held_out_frac=ds.val_frac, verbose=True)

    test_acc = evaluate(agent, result.rendered, ds.test, reward)
    print("\n=== evolved ACE playbook ===")
    print(result.rendered)
    print(f"\nval accuracy : {result.history[0].held_out_reward:.3f} "
          f"-> {result.final_reward:.3f}")
    print(f"test accuracy: {test_acc:.3f}  (held out, never seen by the Curator)")
    print(f"bullets curated: {len(result.state)}")


if __name__ == "__main__":
    main()
