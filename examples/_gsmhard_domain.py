"""GSM-Hard, for the six MethodPolicy ports that read a maths benchmark.

`reasoning-machines/gsm-hard` is GSM8K with the small numbers replaced by large
ones -- same 1319 questions, same wording, arithmetic that no longer fits in a
model's head. It is the benchmark this study needed and GSM8K stopped being.

**Why the move.** On GSM8K these six ports started from wildly different floors,
and the gap was the *prompt wrapper*, not the method:

=========================================  ======  =========
wrapper                                    GSM8K   GSM-Hard
=========================================  ======  =========
``solve_gsm8k`` (Self-Refine, AFlow,        0.578    0.391
PromptBreeder, Reflexion)
``policy_prompt`` (SICA, Godel)             0.938    0.688
=========================================  ======  =========

Measured on the same 64 items, same model, same moment. SICA and Godel began at
0.95-0.98 and had nowhere to go: their reported gains, +0.016 and +0.036, sat
inside the study's +-0.09 noise floor. GSM-Hard gives both wrappers room --
0.31 of it for the higher one, three times the noise.

**The wrapper difference stays.** It is a *fidelity* property, not an oversight:
SICA's editable surface is ``agent_prompt(question)`` returning the whole prompt,
which is upstream's shape, and forcing it through a shared system/user framing
would break the thing the port exists to reproduce. What it costs is that a
*gain* only compares within a wrapper. Final scores compare across.

**What the wrapper actually does**, ablated one part at a time on GSM8K, with
the median reply length beside each -- the grader takes the last number, so
working is free and its absence is the whole story:

===============================================  ======  =============
variant                                          score   median reply
===============================================  ======  =============
``System instruction:`` / ``User problem:``       0.578   6 chars
    labels, "Return only the final answer"
without the section labels                        0.750   9 chars
labels kept, "arithmetic problem *carefully*"     0.812   94 chars
labels kept, plus Reflexion's empty-memory        0.828   12 chars
    header
no labels, "carefully"  (``policy_prompt``)       0.938   194 chars
===============================================  ======  =============

Six characters is a bare number. The section labels make "Return only the final
answer" read as a system-level constraint the model obeys exactly; one adverb
re-opens the working and is worth +0.23 on its own.

The grader is upstream's and stays trivial: ``target`` is a number, and a reply
is scored on the last number it states. Nothing here is this repository's to
adjust -- which is the point, and why MATH-500 was rejected despite being
harder: its answers are LaTeX (``\\left( 3, \\frac{\\pi}{2} \\right)``) and would
need a symbolic comparator written here, putting the grader back in this
repository's hands.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Dict, List, Optional, Sequence, Tuple

from agentdescent.evolution import Task


DATASET = "reasoning-machines/gsm-hard"
FILENAME = "gsmhardv2.jsonl"

#: The published size. A split that does not match is refused rather than used.
EXPECTED_ROWS = 1319

_UA = {"User-Agent": "Mozilla/5.0"}


class _Redirect308(urllib.request.HTTPRedirectHandler):
    """`urllib` before 3.11 does not follow **308 Permanent Redirect**."""

    def http_error_308(self, req, fp, code, msg, headers):
        return self.http_error_301(req, fp, 301, msg, headers)

    https_error_308 = http_error_308


_OPENER = urllib.request.build_opener(_Redirect308)

STARTING_INSTRUCTION = (
    "Solve the grade-school math word problem. Return only the final answer."
)

#: Signed, comma-grouped, optionally decimal. GSM-Hard's targets are floats --
#: `-9867630.0` -- and often negative, which a pattern written for GSM8K's
#: positive integers silently mis-reads.
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

_CACHE: Dict[str, List[dict]] = {}

SAMPLE_ENV = "AGENTDESCENT_GSMHARD_SAMPLE"
SAMPLE_ROWS = 320
_SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "_gsmhard_sample.json")


def normalise(value) -> str:
    """`-9867630.0` and `-9867630` are the same answer; `,` is grouping.

    A whole-valued decimal reads as the integer, because the difference is
    formatting rather than arithmetic and the targets ship as floats.
    """
    text = str(value).replace(",", "").strip()
    if text.endswith(".0"):
        text = text[:-2]
    if "." in text:
        head, _, tail = text.partition(".")
        if set(tail) <= {"0"}:
            text = head or "0"
    return text


def parse_answer(response: str) -> Optional[str]:
    """The **last** number in the reply, normalised the way the target is."""
    numbers = _NUMBER.findall(response or "")
    return normalise(numbers[-1]) if numbers else None


def score_answer(expected: str, response: str) -> float:
    return float(parse_answer(response) == normalise(expected))


def gsmhard_reward(task: Task, output: str) -> float:
    return score_answer(str(task.meta["answer"]), output)


def feedback(task: Task, output: str) -> str:
    """What the grader read and what it wanted -- never the working.

    GSM-Hard ships a `code` field holding the Python that computes the answer,
    and it is deliberately withheld: handing a reference solution to a method
    being asked to write a better instruction turns instruction search into
    transcription.
    """
    parsed = parse_answer(output)
    shown = parsed if parsed is not None else (output or "").strip()[:120]
    return (
        f"Question: {task.prompt}\n"
        f"The evaluator read the last number in your reply: {shown!r}\n"
        f"It wanted: {normalise(task.meta['answer'])!r}\n"
        "Only that last number is graded, so the working before it is free."
    )


def solve_gsmhard(llm, instruction: str, task: Task, *, subphase: str = "") -> str:
    return llm(
        f"System instruction:\n{instruction}\n\nUser problem:\n{task.prompt}",
        subphase=subphase,
        unit=task.id,
    )


# -- loading ------------------------------------------------------------------


def _from_mirror() -> List[dict]:
    base = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
    url = f"{base}/datasets/{DATASET}/resolve/main/{FILENAME}"
    raw = _OPENER.open(urllib.request.Request(url, headers=_UA), timeout=300).read()
    return [json.loads(line) for line in raw.decode("utf-8").splitlines()
            if line.strip()]


def _cache_file() -> str:
    from agentdescent.dataloader import cache_path

    return cache_path("gsmhard", FILENAME)


def load_rows() -> List[dict]:
    """All 1319 rows, checked against the published size.

    The count is asserted rather than trusted: an interrupted fetch of GSM8K's
    raw JSONL over a slow link once returned 944 of 1319 rows and raised
    nothing, and a truncated benchmark reads exactly like a benchmark.
    """
    if os.environ.get(SAMPLE_ENV) == "1":
        with open(_SAMPLE_PATH, encoding="utf-8") as handle:
            return json.load(handle)

    if "rows" in _CACHE:
        return _CACHE["rows"]

    path = _cache_file()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if len(rows) == EXPECTED_ROWS:
            _CACHE["rows"] = rows
            return rows
        os.remove(path)          # a short cache file is a truncated download

    rows = _from_mirror()
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(
            f"GSM-Hard: got {len(rows)} rows, expected {EXPECTED_ROWS} -- "
            "a truncated download, not a short dataset")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".partial"
    with open(tmp, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    _CACHE["rows"] = rows
    return rows


#: Items per split. 64, as on GSM8K: at sixteen a single test item moves the
#: score by 0.0625 and two runs a hair apart are indistinguishable.
SPLIT_SIZE = 64


def _tasks(rows: Sequence[dict], split: str, offset: int) -> List[Task]:
    return [
        Task(
            id=f"{split}:{offset + index}",
            prompt=str(row["input"]).strip(),
            meta={"answer": normalise(row["target"]), "split": split},
        )
        for index, row in enumerate(rows)
    ]


def gsmhard_splits(seed: int, *, train: int = SPLIT_SIZE,
                   held_out: int = SPLIT_SIZE,
                   test: int = SPLIT_SIZE) -> Tuple[List[Task], List[Task], List[Task]]:
    """Disjoint train / held-out / test tasks for one seed.

    GSM-Hard ships one split, so the three are drawn from it -- **after a
    deterministic shuffle**, not as contiguous windows.

    The windows were positional at first, train and held-out from the head and
    test from the tail, and the two were not exchangeable: the seed instruction
    scored 0.73 on held-out and 0.55 on test, an 0.18 gap where re-scoring the
    same split five times moves it by 0.02 (sd 0.021). GSM-Hard is derived from
    GSM8K in order and its tail is harder, so a method was being tuned on one
    distribution and reported on another. SICA's rows are what that looks like:
    validation up +0.094 on all three seeds -- four standard deviations, a real
    improvement -- and test flat, which read as overfitting and was mostly the
    split.

    `Random(seed).shuffle` rather than a positional offset: two seeds then see
    different items *and* each seed's three splits are draws from one
    distribution, which is what a held-out set has to be for a gate decision to
    mean anything about the reported score.
    """
    import random

    rows = load_rows()
    if os.environ.get(SAMPLE_ENV) == "1":
        train = held_out = test = min(train, held_out, test, len(rows) // 10)

    order = list(range(len(rows)))
    random.Random(seed * 7919 + 13).shuffle(order)
    need = train + held_out + test
    if need > len(order):
        raise ValueError(
            f"{need} tasks requested and the dataset holds {len(order)}")
    picked = [rows[i] for i in order[:need]]
    return (
        _tasks(picked[:train], "train", 0),
        _tasks(picked[train:train + held_out], "held-out", train),
        _tasks(picked[train + held_out:], "test", train + held_out),
    )
