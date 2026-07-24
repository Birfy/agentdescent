"""Real skill self-evolution on a real dataset, driven by a real LLM.

Evolves a "skill playbook" (accumulated problem-solving lessons) on a
**BIG-Bench-Hard** task, using a **Claude** agent to both solve tasks and
propose new rules. BBH tasks are deliberately hard for LLMs and scored by
exact match / graded overlap, so there is genuine headroom for a learned skill
to raise the score -- which is what makes this a meaningful demo rather than a
strong model that is already perfect.

    # see the dataset + a cost estimate, no API calls:
    python -m examples.skill_evolution --dry-run

    # the real thing (needs ANTHROPIC_API_KEY or `ant auth login`):
    python -m examples.skill_evolution
    python -m examples.skill_evolution --task word_sorting --model claude-haiku-4-5
    python -m examples.skill_evolution --task logical_deduction_seven_objects --rounds 5

How it works: each round, parallel workers run held-out-*train* problems through
the current playbook via Claude, and on a failure ask Claude to propose one new
lesson. The aggregator dedupes, fuses complementary lessons, and commits a
lesson only if it improves score on a held-out split -- so unhelpful lessons are
rejected automatically.

Cost note: an LLM run makes many calls (rollouts + held-out scoring + the
aggregator's cheap-eval subsets). The defaults are small on purpose; the script
prints an estimate and asks before spending.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from difflib import SequenceMatcher
from typing import Dict, List, Optional

from concordia.evolution import Task, evolve, claude_agent

BBH_URL = "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/bbh/{task}.json"
CACHE_DIR = os.path.expanduser("~/.cache/concordia/bbh")


# -- dataset loading ---------------------------------------------------------


def download_bbh(task: str) -> List[dict]:
    """Fetch a BBH task's examples (cached locally after first download)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{task}.json")
    if not os.path.exists(path):
        url = BBH_URL.format(task=task)
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = resp.read()
        with open(path, "wb") as f:
            f.write(data)
    with open(path) as f:
        return json.load(f)["examples"]


def build_tasks(examples: List[dict], limit: int, seed: int = 0) -> List[Task]:
    """Turn raw BBH examples into Concordia Tasks (deterministic shuffle)."""
    import random
    rng = random.Random(seed)
    idx = list(range(len(examples)))
    rng.shuffle(idx)
    tasks = []
    for i in idx[:limit]:
        ex = examples[i]
        target = str(ex["target"]).strip()
        tasks.append(Task(id=f"ex{i}", prompt=ex["input"].strip(),
                          meta={"target": target, "mc": is_multiple_choice(target)}))
    return tasks


# -- scoring (the reward the skill is optimized against) ---------------------


def is_multiple_choice(target: str) -> bool:
    return bool(re.fullmatch(r"\([A-Z]\)", target.strip()))


def extract_choice(text: str) -> Optional[str]:
    """Pull the chosen option letter out of free-form model output."""
    matches = re.findall(r"\(([A-Za-z])\)", text)
    if matches:
        return matches[-1].upper()
    m = re.search(r"\b([A-Z])\b\s*$", text.strip())
    return m.group(1).upper() if m else None


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower()).rstrip(".")


def make_reward(mc: bool):
    """Exact-match for multiple-choice tasks; graded overlap for free-form."""
    def reward(task: Task, output: str) -> float:
        target = task.meta["target"]
        if mc:
            pred, gold = extract_choice(output), extract_choice(target)
            return 1.0 if pred and gold and pred == gold else 0.0
        # free-form (e.g. word_sorting): token-order similarity, so partial
        # progress is rewarded and format errors are penalized.
        return SequenceMatcher(None, _norm(output), _norm(target)).ratio()
    return reward


# -- cost estimate -----------------------------------------------------------


def estimate_calls(rounds: int, workers: int, held_out: int) -> int:
    """Rough upper bound on model calls, for the cost warning."""
    per_round = workers * 3 + (4 * min(8, held_out) + 2 * held_out)
    return rounds * per_round


# -- main --------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="word_sorting",
                   help="BIG-Bench-Hard task name (e.g. word_sorting, "
                        "logical_deduction_seven_objects, dyck_languages)")
    p.add_argument("--model", default="claude-opus-4-8",
                   help="Claude model id (use claude-haiku-4-5 for cheap runs)")
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--train", type=int, default=12)
    p.add_argument("--heldout", type=int, default=10)
    p.add_argument("--dry-run", action="store_true",
                   help="load the dataset and print an estimate, no API calls")
    p.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    args = p.parse_args()

    print(f"Dataset : BIG-Bench-Hard / {args.task}")
    examples = download_bbh(args.task)
    total = args.train + args.heldout
    tasks = build_tasks(examples, limit=total)
    mc = tasks[0].meta["mc"]
    reward = make_reward(mc)

    print(f"Loaded  : {len(examples)} examples; using {len(tasks)} "
          f"({args.train} train / {args.heldout} held-out)")
    print(f"Scoring : {'exact-match (multiple choice)' if mc else 'graded token overlap'}")
    print("\nExample problem:")
    print("  Q:", tasks[0].prompt[:200] + ("..." if len(tasks[0].prompt) > 200 else ""))
    print("  A:", tasks[0].meta["target"][:120])

    est = estimate_calls(args.rounds, args.workers, args.heldout)
    print(f"\nPlan    : model={args.model}, rounds={args.rounds}, "
          f"workers={args.workers}")
    print(f"Budget  : up to ~{est} Claude calls "
          f"(cached repeats are free; use --model claude-haiku-4-5 to cut cost)")

    if args.dry_run:
        print("\n[dry-run] not calling the API. Drop --dry-run to evolve the skill.")
        return

    if not args.yes and sys.stdin.isatty():
        if input("\nProceed with real API calls? [y/N] ").strip().lower() not in ("y", "yes"):
            print("aborted.")
            return

    try:
        agent = claude_agent(model=args.model)
        # fail fast with a clear message if credentials are missing.
        agent.complete("Reply with the single word: ok")
    except Exception as e:  # noqa: BLE001 - surface any auth/setup problem plainly
        print(f"\nCould not reach Claude ({type(e).__name__}: {e}).")
        print("Set ANTHROPIC_API_KEY or run `ant auth login`, then retry.")
        return

    print("\nEvolving skill...\n")
    result = evolve(tasks, reward, agent=agent, rounds=args.rounds,
                    n_workers=args.workers, artifact_id="skill",
                    held_out_frac=args.heldout / total, verbose=True)

    print("\n=== evolved skill playbook ===")
    print(result.rendered)
    label = "held-out accuracy" if mc else "held-out score"
    print(f"\n{label}: {result.history[0].held_out_reward:.3f} "
          f"-> {result.final_reward:.3f}")
    print(f"lessons learned: {len(result.state)}")


if __name__ == "__main__":
    main()
