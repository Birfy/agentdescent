"""GSM8K, for the eleven MethodPolicy ports.

This replaces the hand-written money domain those ports shared. The difference
is not size -- it is that **nobody here wrote the questions or the grader**.

The money domain was 48 items this repository authored, graded by a rule this
repository chose, and the rule was changed mid-study when it blocked progress
(it matched the whole reply, which made the output convention and the reasoning
mutually exclusive). Both are legitimate things to do to a *fixture* and
disqualifying for a *benchmark*: a number produced against a target its author
can move is not a measurement of the method.

GSM8K brings its own questions and its own answer key. The grader is upstream's:
the gold answer is the integer after ``####``, and a reply is scored on the last
number it states. That is what every GSM8K harness does, and it is why the
"discover the output convention" gimmick disappears -- there is nothing to
discover, only arithmetic to get right.

**Expect smaller gains.** The money domain's baseline was 0.000 by construction,
because the seed instruction could not satisfy a convention no one had told it.
A capable model already answers most of GSM8K, so the headroom a method has to
work with is what remains above a real baseline rather than the whole interval.
A row that gained +0.5 there and +0.05 here has not got worse; it has stopped
being scored against a floor its own fixture installed.

Two transports, because the study runs on a host that cannot reach
``huggingface.co``:

* the **parquet files**, over ``HF_ENDPOINT`` (default ``hf-mirror.com``),
  which is the whole split in one request -- 7473 train rows and 1319 test;
* the **datasets-server** ``/rows`` API through
  :mod:`agentdescent.dataloader`, where that host is reachable.

The parquet path is tried first and the row counts are asserted, because the
failure that matters here is silent: an interrupted download of GSM8K's raw
JSONL from GitHub returned 944 of 1319 test rows with no error at all, and a
truncated benchmark reads as a benchmark.
"""

from __future__ import annotations

import io
import os
import re
import urllib.request
from typing import Dict, List, Optional, Sequence, Tuple

from agentdescent.evolution import Task


DATASET = "openai/gsm8k"
CONFIG = "main"

#: The published sizes. A split that does not match is refused rather than used.
EXPECTED_ROWS = {"train": 7473, "test": 1319}

#: `hf-mirror.com` mirrors the resolve path but returns 403 on the parquet
#: *listing* API, so the file names are pinned rather than discovered.
PARQUET = {"train": "train-00000-of-00001.parquet",
           "test": "test-00000-of-00001.parquet"}

_UA = {"User-Agent": "Mozilla/5.0"}


class _Redirect308(urllib.request.HTTPRedirectHandler):
    """`urllib` before 3.11 does not follow **308 Permanent Redirect**.

    The mirror answers the resolve path with one, so without this the loader
    raises `HTTPError: 308` and falls through to a transport that is blocked on
    the run host -- which reads as "the dataset is unavailable" rather than
    "this client is four years old".
    """

    def http_error_308(self, req, fp, code, msg, headers):
        return self.http_error_301(req, fp, 301, msg, headers)

    https_error_308 = http_error_308


_OPENER = urllib.request.build_opener(_Redirect308)

STARTING_INSTRUCTION = (
    "Solve the grade-school math word problem. Return only the final answer."
)

#: `#### <integer>` closes every gold answer in GSM8K.
_GOLD = re.compile(r"####\s*(-?[\d,]+)")

#: Any number in the reply. The last one is the answer, which is the convention
#: every GSM8K harness uses -- a model that shows its working states
#: intermediate values first.
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

_CACHE: Dict[str, List[dict]] = {}

#: A committed slice of GSM8K -- 96 rows per split -- for the **offline test
#: suite only**. This repository's suite has never needed the network and moving
#: a domain to a real dataset is no reason to change that.
#:
#: It is gated on an environment variable that only `conftest.py` sets, and never
#: used as a fallback when a fetch fails. A sample that could stand in silently
#: is the failure this module is built to refuse: a run reporting a number it
#: measured on 96 rows while its output says GSM8K would be wrong in a direction
#: nobody can see, which is exactly what the truncated 944-row download was.
SAMPLE_ENV = "AGENTDESCENT_GSM8K_SAMPLE"
SAMPLE_ROWS = 96
_SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "_gsm8k_sample.json")


def _from_sample(split: str) -> List[dict]:
    import json

    with open(_SAMPLE_PATH, encoding="utf-8") as handle:
        return json.load(handle)[split]


def gold_answer(raw: str) -> Optional[str]:
    """The integer after ``####``, normalised. None if the row is malformed."""
    match = _GOLD.search(raw or "")
    return match.group(1).replace(",", "") if match else None


def parse_answer(response: str) -> Optional[str]:
    """The **last** number in the reply, normalised the way the gold is.

    Unlike the money domain's grader this does not forbid a chain of thought:
    working through the problem and then stating the answer is what GSM8K
    expects, and the last number is the answer. A whole-valued decimal (`18.0`)
    reads as `18`, because the gold is always an integer and the difference is
    formatting rather than arithmetic.
    """
    numbers = _NUMBER.findall(response or "")
    if not numbers:
        return None
    value = numbers[-1].replace(",", "")
    if "." in value:
        head, _, tail = value.partition(".")
        if set(tail) <= {"0"}:
            value = head or "0"
    return value


def score_answer(expected: str, response: str) -> float:
    return float(parse_answer(response) == expected)


def gsm8k_reward(task: Task, output: str) -> float:
    return score_answer(str(task.meta["answer"]), output)


def feedback(task: Task, output: str) -> str:
    """What the grader read and what it wanted -- never the working.

    GSM8K ships a gold *rationale* and it is deliberately withheld: handing it
    to a method being asked to write a better instruction turns instruction
    search into transcription.
    """
    parsed = parse_answer(output)
    shown = parsed if parsed is not None else (output or "").strip()[:120]
    return (
        f"Question: {task.prompt}\n"
        f"The evaluator read the last number in your reply: {shown!r}\n"
        f"It wanted: {str(task.meta['answer'])!r}\n"
        "Only that last number is graded, so the working before it is free."
    )


def solve_gsm8k(llm, instruction: str, task: Task, *, subphase: str = "") -> str:
    return llm(
        f"System instruction:\n{instruction}\n\nUser problem:\n{task.prompt}",
        subphase=subphase,
        unit=task.id,
    )


# -- loading ------------------------------------------------------------------


def _from_parquet(split: str) -> List[dict]:
    import pyarrow.parquet as pq

    base = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
    url = f"{base}/datasets/{DATASET}/resolve/main/{CONFIG}/{PARQUET[split]}"
    request = urllib.request.Request(url, headers=_UA)
    raw = _OPENER.open(request, timeout=300).read()
    table = pq.read_table(io.BytesIO(raw))
    return [{"question": q, "answer": a} for q, a in
            zip(table.column("question").to_pylist(),
                table.column("answer").to_pylist())]


def _from_datasets_server(split: str) -> List[dict]:
    from agentdescent.dataloader import hf_rows

    return hf_rows(DATASET, split, config=CONFIG, limit=EXPECTED_ROWS[split])


def _cache_file(split: str) -> str:
    from agentdescent.dataloader import cache_path

    return cache_path("gsm8k", f"{split}.jsonl")


def load_split(split: str) -> List[dict]:
    """The whole split, checked against its published size.

    An interrupted fetch is the failure that matters: GSM8K's raw JSONL over a
    slow link returned 944 of 1319 test rows with no error raised, and a
    truncated benchmark reads exactly like a benchmark. The count is asserted
    rather than trusted.
    """
    if os.environ.get(SAMPLE_ENV) == "1":
        # Deliberately before the cache and before any transport: a suite that
        # asked for the sample must get the sample, not whatever a previous test
        # left behind.
        return _from_sample(split)

    if split in _CACHE:
        return _CACHE[split]

    import json

    path = _cache_file(split)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if len(rows) == EXPECTED_ROWS[split]:
            _CACHE[split] = rows
            return rows
        os.remove(path)          # a short cache file is a truncated download

    errors = []
    for loader in (_from_parquet, _from_datasets_server):
        try:
            rows = loader(split)
        except Exception as error:                    # noqa: BLE001 - reported
            errors.append(f"{loader.__name__}: {type(error).__name__}: {error}")
            continue
        if len(rows) != EXPECTED_ROWS[split]:
            errors.append(
                f"{loader.__name__}: got {len(rows)} rows, expected "
                f"{EXPECTED_ROWS[split]} -- a truncated split, not a short one")
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Written whole then renamed: a cache file half-written by an interrupted
        # process is exactly the truncated split this function exists to refuse.
        tmp = path + ".partial"
        with open(tmp, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
        _CACHE[split] = rows
        return rows
    raise RuntimeError(f"could not load GSM8K {split}: " + "; ".join(errors))


def _tasks(rows: Sequence[dict], split: str, offset: int) -> List[Task]:
    out = []
    for index, row in enumerate(rows):
        answer = gold_answer(str(row.get("answer", "")))
        if answer is None:
            continue
        out.append(Task(
            id=f"{split}:{offset + index}",
            prompt=str(row["question"]).strip(),
            meta={"answer": answer, "split": split},
        ))
    return out


#: Items per split. 64, not 16: at sixteen a single test item moves the score by
#: 0.0625, so "1.000" says only that sixteen questions went right and two runs a
#: hair apart are indistinguishable. Sixty-four costs about six minutes a seed
#: against the thirty-minute budget for three, which buys a resolution of 0.016.
#: It is still a *window* on GSM8K rather than the whole split -- 8792 rows at
#: this budget would be hours.
SPLIT_SIZE = 64


def gsm8k_splits(seed: int, *, train: int = SPLIT_SIZE,
                 held_out: int = SPLIT_SIZE,
                 test: int = SPLIT_SIZE) -> Tuple[List[Task], List[Task], List[Task]]:
    """Disjoint train / held-out / test tasks for one seed.

    Train and held-out come from GSM8K's **train** split and the reported test
    tasks from its **test** split, so no method tunes on what it is reported
    against. The seed offsets a window rather than shuffling, so two seeds see
    genuinely different items and a seed is reproducible without holding an
    order.
    """
    work = load_split("train")
    reported = load_split("test")
    if os.environ.get(SAMPLE_ENV) == "1":
        # The sample is 96 rows; a 64/64 request would run off the end of it and
        # hand back short splits, which is the shape of bug this domain exists to
        # catch. Tests care that the plumbing works, not how many items it saw.
        train = held_out = test = min(train, held_out, test, SAMPLE_ROWS // 3)
    work_offset = (seed * (train + held_out)) % max(1, len(work) - train - held_out)
    test_offset = (seed * test) % max(1, len(reported) - test)
    window = work[work_offset:work_offset + train + held_out]
    return (
        _tasks(window[:train], "train", work_offset),
        _tasks(window[train:], "held-out", work_offset + train),
        _tasks(reported[test_offset:test_offset + test], "test", test_offset),
    )
