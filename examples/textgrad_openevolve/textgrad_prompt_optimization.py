"""A small TextGrad prompt-optimization experiment on BBH word sorting.

The script reconstructs the official prompt-optimization path in a compact form:
forward responses, textual loss gradients, backward feedback to the system prompt,
a batch TGD update, and validation-set revert. Evaluation is deterministic exact
match so the reported lift does not depend on another LLM judge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from agentdescent.agents import Usage
from agentdescent.dataloader import fetch_text

from examples.textgrad_openevolve._common import (
    REPORT_DIR,
    add_model_args,
    confirm_paid_run,
    make_completion,
    require_api_environment,
    sum_usage,
    utc_now,
    write_json,
)


DATA_URL = (
    "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/"
    "main/bbh/word_sorting.json"
)
UPSTREAM_COMMIT = "75e912e210864b61999781778cdf756d4468120f"
STARTING_PROMPT = (
    "You will answer a reasoning question. Think step by step. The last line of your "
    "response should be of the following format: 'Answer: $VALUE' where VALUE is a "
    "numerical value."
)
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[&'._-][a-z0-9]+)*", re.IGNORECASE)


@dataclass(frozen=True)
class Example:
    index: int
    question: str
    target: str

    @property
    def word_count(self) -> int:
        return len(self.target.split())


@dataclass
class ItemResult:
    index: int
    correct: bool
    predicted: str
    target: str
    response: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_args(parser, max_tokens=2048, temperature=0.0)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--train-size", type=int, default=12)
    parser.add_argument("--val-size", type=int, default=12)
    parser.add_argument("--test-size", type=int, default=20)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--subset",
        choices=("longest", "first"),
        default="longest",
        help="deterministic selection within each official split",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORT_DIR / "textgrad-small-result.json",
    )
    return parser


def normalize_answer(text: str) -> Tuple[str, ...]:
    return tuple(token.lower() for token in TOKEN_RE.findall(text))


def extract_answer(response: str) -> str:
    matches = re.findall(r"(?im)^\s*(?:final\s+)?answer\s*:\s*(.*?)\s*$", response)
    if matches:
        return matches[-1].strip().strip("`*_ ")
    lines = [line.strip().strip("`*_ ") for line in response.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def score_response(response: str, target: str) -> Tuple[bool, str]:
    predicted = extract_answer(response)
    return normalize_answer(predicted) == normalize_answer(target), predicted


def wilson_interval(solved: int, total: int, z: float = 1.959963984540054) -> Dict[str, float]:
    """Two-sided 95% Wilson interval for a binomial accuracy."""
    proportion = solved / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * (proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
        ** 0.5
        / denominator
    )
    return {"low": max(0.0, center - margin), "high": min(1.0, center + margin)}


def load_bbh_splits(
    *,
    train_size: int,
    val_size: int,
    test_size: int,
    subset: str,
) -> Tuple[List[Example], List[Example], List[Example], str]:
    raw = fetch_text(
        DATA_URL,
        cache_subdir="bbh",
        filename="word_sorting.json",
        timeout=60.0,
    )
    payload = json.loads(raw)
    examples = [
        Example(index=i, question=row["input"], target=row["target"])
        for i, row in enumerate(payload["examples"])
    ]
    if len(examples) < 250:
        raise RuntimeError(f"expected at least 250 BBH examples, found {len(examples)}")

    # TextGrad's BBH loader uses exactly these positional splits.
    official = (examples[:50], examples[50:150], examples[150:250])

    def select(items: Sequence[Example], size: int) -> List[Example]:
        if size < 1 or size > len(items):
            raise ValueError(f"subset size must be in [1, {len(items)}], got {size}")
        if subset == "first":
            return list(items[:size])
        selected = sorted(items, key=lambda item: (-item.word_count, item.index))[:size]
        return sorted(selected, key=lambda item: item.index)

    return (
        select(official[0], train_size),
        select(official[1], val_size),
        select(official[2], test_size),
        hashlib.sha256(raw.encode("utf-8")).hexdigest(),
    )


def solve_prompt(system_prompt: str, question: str) -> str:
    return (
        "Act as the language model whose trainable system instruction is shown below.\n\n"
        f"<SYSTEM_INSTRUCTION>\n{system_prompt}\n</SYSTEM_INSTRUCTION>\n\n"
        f"<USER_QUESTION>\n{question}\n</USER_QUESTION>\n\n"
        "Answer the user question while following the system instruction."
    )


class PromptEvaluator:
    def __init__(self, completion, workers: int) -> None:
        self.completion = completion
        self.workers = max(1, workers)
        self._cache: Dict[Tuple[str, int], str] = {}
        self._lock = threading.Lock()
        self.cache_hits = 0

    def solve(self, system_prompt: str, item: Example) -> ItemResult:
        key = (system_prompt, item.index)
        with self._lock:
            response = self._cache.get(key)
            if response is not None:
                self.cache_hits += 1
        if response is None:
            response = self.completion(solve_prompt(system_prompt, item.question)).strip()
            with self._lock:
                self._cache[key] = response
        correct, predicted = score_response(response, item.target)
        return ItemResult(item.index, correct, predicted, item.target, response)

    def evaluate(self, system_prompt: str, items: Sequence[Example]) -> Dict[str, Any]:
        with ThreadPoolExecutor(max_workers=min(self.workers, len(items))) as pool:
            results = list(pool.map(lambda item: self.solve(system_prompt, item), items))
        solved = sum(result.correct for result in results)
        return {
            "accuracy": solved / len(results),
            "solved": solved,
            "total": len(results),
            "wilson_95": wilson_interval(solved, len(results)),
            "items": [asdict(result) for result in results],
        }


def _response_gradient_prompt(item: Example, result: ItemResult) -> str:
    status = "CORRECT" if result.correct else "INCORRECT"
    return (
        "You are the backward engine in textual gradient descent. Critique a model response "
        "against a deterministic exact-match objective. Give concise, actionable feedback; "
        "do not produce a replacement system prompt.\n\n"
        f"<QUESTION>{item.question}</QUESTION>\n"
        f"<GROUND_TRUTH>{item.target}</GROUND_TRUTH>\n"
        f"<MODEL_RESPONSE>{result.response}</MODEL_RESPONSE>\n"
        f"<OBJECTIVE_RESULT>{status}</OBJECTIVE_RESULT>"
    )


def _prompt_gradient_prompt(
    system_prompt: str,
    item: Example,
    result: ItemResult,
    response_gradient: str,
) -> str:
    return (
        "Backpropagate the response feedback through the language-model call to the trainable "
        "system instruction. Explain exactly what the instruction should clarify or preserve "
        "so future responses satisfy the objective. Return feedback, not a rewritten prompt.\n\n"
        f"<SYSTEM_INSTRUCTION>{system_prompt}</SYSTEM_INSTRUCTION>\n"
        f"<QUESTION>{item.question}</QUESTION>\n"
        f"<RESPONSE>{result.response}</RESPONSE>\n"
        f"<RESPONSE_GRADIENT>{response_gradient}</RESPONSE_GRADIENT>"
    )


def _update_prompt(system_prompt: str, gradients: Sequence[str]) -> str:
    joined = "\n".join(
        f"<FEEDBACK index=\"{index}\">{gradient}</FEEDBACK>"
        for index, gradient in enumerate(gradients, 1)
    )
    return (
        "You are TextualGradientDescent (TGD). Improve the trainable variable using the batch "
        "of textual gradients. The new instruction must generalize: do not copy benchmark "
        "questions, target answer lists, or case-specific answers. Keep it concise and make the "
        "required reasoning and final-answer format unambiguous.\n\n"
        f"<VARIABLE>{system_prompt}</VARIABLE>\n"
        f"<ROLE>structured system instruction for a word-sorting reasoning task</ROLE>\n"
        f"{joined}\n\n"
        "Return only the rewritten variable between these tags:\n"
        "<IMPROVED_VARIABLE>...</IMPROVED_VARIABLE>"
    )


def extract_improved_variable(response: str) -> str:
    match = re.search(
        r"<IMPROVED_VARIABLE>\s*(.*?)\s*</IMPROVED_VARIABLE>", response, re.DOTALL | re.I
    )
    if match:
        return match.group(1).strip()
    fenced = re.search(r"```(?:text)?\s*(.*?)```", response, re.DOTALL | re.I)
    return (fenced.group(1) if fenced else response).strip()


def validate_candidate(candidate: str, batch: Sequence[Example]) -> Tuple[bool, str]:
    if not 20 <= len(candidate) <= 6000:
        return False, f"candidate length {len(candidate)} is outside [20, 6000]"
    lowered = " ".join(candidate.lower().split())
    for item in batch:
        target = " ".join(item.target.lower().split())
        if len(target) >= 30 and target in lowered:
            return False, f"candidate copied training target from item {item.index}"
    return True, ""


def _batch_for_step(
    training: Sequence[Example], *, step: int, batch_size: int, seed: int
) -> List[Example]:
    if batch_size < 1:
        raise ValueError("batch-size must be positive")
    order = list(training)
    epoch = (step * batch_size) // len(order)
    random.Random(seed + epoch).shuffle(order)
    start = (step * batch_size) % len(order)
    return [order[(start + offset) % len(order)] for offset in range(batch_size)]


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if args.steps < 1:
        raise ValueError("steps must be positive")
    require_api_environment(args.provider)
    train, val, test, dataset_sha256 = load_bbh_splits(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        subset=args.subset,
    )

    usages = {
        "forward": Usage(),
        "response_gradient": Usage(),
        "prompt_gradient": Usage(),
        "tgd_update": Usage(),
    }
    evaluator = PromptEvaluator(make_completion(args, usages["forward"]), args.workers)
    response_backward = make_completion(args, usages["response_gradient"])
    prompt_backward = make_completion(args, usages["prompt_gradient"])
    update = make_completion(args, usages["tgd_update"])

    started = time.monotonic()
    started_at = utc_now()
    current_prompt = STARTING_PROMPT
    baseline_val = evaluator.evaluate(current_prompt, val)
    baseline_test = evaluator.evaluate(current_prompt, test)
    current_val = baseline_val
    trajectory: List[Dict[str, Any]] = []

    result: Dict[str, Any] = {
        "experiment": "TextGrad prompt optimization on BBH word_sorting",
        "status": "running",
        "started_at": started_at,
        "upstream": {
            "repository": "https://github.com/zou-group/textgrad",
            "commit": UPSTREAM_COMMIT,
            "data_url": DATA_URL,
            "dataset_sha256": dataset_sha256,
        },
        "config": {
            "provider": args.provider,
            "model": args.model,
            "thinking": args.thinking,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "api_timeout": args.api_timeout,
            "seed": args.seed,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "train_size": len(train),
            "val_size": len(val),
            "test_size": len(test),
            "workers": args.workers,
            "subset": args.subset,
            "split_rule": "official positional 50 train / 100 val / 100 test",
            "selected_ids": {
                "train": [item.index for item in train],
                "val": [item.index for item in val],
                "test": [item.index for item in test],
            },
        },
        "starting_prompt": STARTING_PROMPT,
        "baseline": {"validation": baseline_val, "test": baseline_test},
        "trajectory": trajectory,
    }
    write_json(args.output, result)

    for step in range(args.steps):
        batch = _batch_for_step(train, step=step, batch_size=args.batch_size, seed=args.seed)
        batch_results = [evaluator.solve(current_prompt, item) for item in batch]

        def backward(pair: Tuple[Example, ItemResult]) -> Dict[str, str]:
            item, item_result = pair
            response_gradient = response_backward(
                _response_gradient_prompt(item, item_result)
            ).strip()
            prompt_gradient = prompt_backward(
                _prompt_gradient_prompt(current_prompt, item, item_result, response_gradient)
            ).strip()
            return {
                "response_gradient": response_gradient,
                "prompt_gradient": prompt_gradient,
            }

        with ThreadPoolExecutor(max_workers=min(args.batch_size, len(batch))) as pool:
            gradients = list(pool.map(backward, zip(batch, batch_results)))

        raw_update = update(
            _update_prompt(current_prompt, [entry["prompt_gradient"] for entry in gradients])
        ).strip()
        candidate = extract_improved_variable(raw_update)
        valid, rejection_reason = validate_candidate(candidate, batch)
        candidate_val = evaluator.evaluate(candidate, val) if valid else None
        accepted = bool(valid and candidate_val["accuracy"] >= current_val["accuracy"])
        if accepted:
            current_prompt = candidate
            current_val = candidate_val
        elif valid:
            rejection_reason = (
                f"validation accuracy regressed from {current_val['accuracy']:.6f} "
                f"to {candidate_val['accuracy']:.6f}"
            )

        step_record = {
            "step": step + 1,
            "batch_ids": [item.index for item in batch],
            "batch_accuracy": sum(result.correct for result in batch_results) / len(batch_results),
            "prompt_before": (
                trajectory[-1]["prompt_after"] if trajectory else STARTING_PROMPT
            ),
            "candidate_prompt": candidate,
            "candidate_valid": valid,
            "candidate_validation": candidate_val,
            "accepted": accepted,
            "rejection_reason": rejection_reason,
            "prompt_after": current_prompt,
            "gradients": gradients,
            "usage_after_step": sum_usage(usages),
        }
        trajectory.append(step_record)
        result["usage"] = sum_usage(usages)
        result["cache_hits"] = evaluator.cache_hits
        result["wall_seconds_so_far"] = round(time.monotonic() - started, 6)
        write_json(args.output, result)
        print(
            f"step {step + 1}/{args.steps}: batch={step_record['batch_accuracy']:.3f}, "
            f"candidate_val={candidate_val['accuracy'] if candidate_val else 0.0:.3f}, "
            f"accepted={accepted}"
        )

    final_test = evaluator.evaluate(current_prompt, test)
    result.update(
        {
            "status": "completed",
            "completed_at": utc_now(),
            "final_prompt": current_prompt,
            "final": {"validation": current_val, "test": final_test},
            "improvement": {
                "validation_accuracy_points": round(
                    current_val["accuracy"] - baseline_val["accuracy"], 12
                ),
                "test_accuracy_points": round(
                    final_test["accuracy"] - baseline_test["accuracy"], 12
                ),
                "test_additional_solved": final_test["solved"] - baseline_test["solved"],
            },
            "usage": sum_usage(usages),
            "cache_hits": evaluator.cache_hits,
            "wall_seconds": round(time.monotonic() - started, 6),
        }
    )
    write_json(args.output, result)
    return result


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        calls = (args.val_size + args.test_size) + args.steps * (
            args.batch_size * 3 + 1 + args.val_size
        ) + args.test_size
        print(
            "TextGrad dry run: "
            f"model={args.model}, thinking={args.thinking}, steps={args.steps}, "
            f"batch={args.batch_size}, "
            f"train/val/test={args.train_size}/{args.val_size}/{args.test_size}, "
            f"upper-bound calls before cache reuse={calls}, output={args.output}"
        )
        return 0
    confirm_paid_run(args, "TextGrad BBH word-sorting experiment")
    result = run(args)
    print(
        "completed: "
        f"test {result['baseline']['test']['accuracy']:.3f} -> "
        f"{result['final']['test']['accuracy']:.3f}; "
        f"calls={result['usage']['total']['calls']}; output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
