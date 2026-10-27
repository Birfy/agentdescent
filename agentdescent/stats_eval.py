"""Statistical and contamination checks for an RSI adoption decision.

A point-estimate rule ("the mean gain is positive") adopts noise: with a small
held-out set a candidate can win by one lucky task. This module gives the
acceptance/report layers the two numbers a product needs:

* a **paired bootstrap confidence interval** on the per-task gain and the
  probability that the true gain is positive;
* a **contamination overlap** between training and validation text, because a
  leaked example makes every confidence interval meaningless.

No dependencies beyond the standard library; the bootstrap is seeded, so a
report is reproducible.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = ["BootstrapResult", "ContaminationReport", "bootstrap_ci",
           "contamination_overlap", "is_significant", "paired_gain"]


@dataclass(frozen=True)
class BootstrapResult:
    """A paired bootstrap of per-task gains."""

    mean: float
    lo: float
    hi: float
    prob_positive: float
    n: int
    alpha: float


def paired_gain(before: Mapping[str, float],
                after: Mapping[str, float]) -> List[float]:
    """Per-task ``after - before`` over the keys present in both sides.

    Keys are visited in a stable order so a seeded bootstrap cannot depend on
    ``PYTHONHASHSEED``. Non-finite scores are refused: a NaN would otherwise
    make the interval depend on whichever comparison it happened to hit.
    """
    diffs = []
    for key in sorted(before.keys() & after.keys(), key=str):
        try:
            before_value = float(before[key])
            after_value = float(after[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"task {key!r} has a non-numeric score") from exc
        if not math.isfinite(before_value) or not math.isfinite(after_value):
            raise ValueError(f"task {key!r} has a non-finite score")
        diffs.append(after_value - before_value)
    return diffs


def bootstrap_ci(before: Mapping[str, float], after: Mapping[str, float], *,
                 n: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> BootstrapResult:
    """Bootstrap the mean paired gain, ``after - before``.

    Resampling is over tasks. ``prob_positive`` is the share of resampled means
    above zero (a one-sided read); the interval is the central ``1-alpha`` one.
    Fewer than two paired tasks gives a zero-width interval at the mean -- the
    caller should treat that as "not enough evidence", not as significance.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if n < 1:
        raise ValueError("n must be >= 1")
    diffs = paired_gain(before, after)
    if not diffs:
        return BootstrapResult(0.0, 0.0, 0.0, 0.0, 0, alpha)
    mean = sum(diffs) / len(diffs)
    if len(diffs) == 1:
        # One pair cannot estimate a distribution. 0.5 says "no evidence"
        # rather than a 100% positive share.
        return BootstrapResult(mean, mean, mean, 0.5, 1, alpha)
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        sample = [diffs[rng.randrange(len(diffs))] for _ in range(len(diffs))]
        means.append(sum(sample) / len(diffs))
    means.sort()
    # Both ends floor, so the interval is very slightly asymmetric. That is
    # intentional: the bootstrap distribution is discrete.
    lo = means[int((alpha / 2) * (n - 1))]
    hi = means[int((1 - alpha / 2) * (n - 1))]
    positive = sum(1 for value in means if value > 0.0) / n
    return BootstrapResult(mean, lo, hi, positive, len(diffs), alpha)


def is_significant(result: BootstrapResult, *,
                   min_effect: float = 0.0,
                   alpha: Optional[float] = None) -> bool:
    """Whether the lower bound clears ``min_effect`` and the interval is usable.

    ``alpha`` defaults to the interval's own alpha. Passing a different value is
    refused rather than silently returning ``False``: a caller that computed the
    interval at 0.10 and tests at 0.05 has a bug, not a non-significant result.
    """
    if (isinstance(min_effect, bool)
            or not isinstance(min_effect, (int, float))
            or not math.isfinite(float(min_effect))):
        raise ValueError("min_effect must be a finite number")
    if alpha is not None and alpha != result.alpha:
        raise ValueError(
            f"alpha {alpha} does not match the interval's alpha {result.alpha}")
    return result.n >= 2 and result.lo > min_effect


@dataclass(frozen=True)
class ContaminationReport:
    """How much of the validation text appears in the training text.

    ``overlap`` is ``None`` when there were no n-grams to check (empty/short
    validation text). That is deliberately not ``0.0``: "we could not screen
    this" is not "this is clean". ``exact_duplicates`` counts normalized
    validation examples that also occur verbatim in training, whatever their
    length, so a two-word or CJK duplicate is still visible.
    """

    overlap: Optional[float]
    shared: Tuple[str, ...] = ()
    held_out_grams: int = 0
    exact_duplicates: int = 0
    examples: Tuple[str, ...] = ()


def _normalize(text: Any) -> str:
    return " ".join(str(text).casefold().split())


#: Split CJK characters so a short Chinese/Japanese sentence has tokens even
#: though it has no spaces.
_CJK = r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]"


def _word_grams(normalized: str, n: int) -> List[str]:
    split_cjk = re.sub(f"({_CJK})", r" \1 ", normalized)
    words = re.findall(r"\w+", split_cjk, re.UNICODE)
    return [" ".join(words[i:i + n])
            for i in range(max(0, len(words) - n + 1))]


def _char_grams(normalized: str, n: int) -> List[str]:
    chars = re.sub(r"\s+", "", normalized)
    return [chars[i:i + n] for i in range(max(0, len(chars) - n + 1))]


def _val_grams(text: str, n: int) -> Tuple[str, List[str]]:
    """Word grams when the validation text has enough words, else char grams.

    Training indexes both kinds; a short validation item is compared with the
    same kind it emitted. Comparing char grams only against word grams made a
    short item embedded in a longer prompt read as clean.
    """
    normalized = _normalize(text)
    words = _word_grams(normalized, n)
    if words:
        return "word", words
    return "char", _char_grams(normalized, n)


def contamination_overlap(train_texts: Sequence[str], val_texts: Sequence[str],
                          *, n: int = 5) -> ContaminationReport:
    """Fraction of validation n-grams that also occur in the training text.

    ``n=5`` is a word n-gram; exact duplicate examples produce an overlap of
    1.0. ``examples`` lists a few shared grams so a report can show the leak
    without printing the whole corpus. A low overlap is not proof of no
    contamination -- paraphrase and embeddings are not covered -- but a high one
    is a hard stop.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    train_texts = [text for text in train_texts if _normalize(text)]
    val_texts = [text for text in val_texts if _normalize(text)]
    train_examples = {_normalize(text) for text in train_texts}
    exact_duplicates = sum(
        1 for text in val_texts if _normalize(text) in train_examples)
    duplicate_fraction = (exact_duplicates / len(val_texts)
                          if val_texts else 0.0)
    train = {"word": Counter(), "char": Counter()}
    for text in train_texts:
        normalized = _normalize(text)
        train["word"].update(_word_grams(normalized, n))
        train["char"].update(_char_grams(normalized, n))
    val = {"word": Counter(), "char": Counter()}
    for text in val_texts:
        mode, grams = _val_grams(text, n)
        val[mode].update(grams)
    total = sum(sum(counter.values()) for counter in val.values())
    if total == 0:
        return ContaminationReport(
            overlap=(duplicate_fraction if duplicate_fraction else None),
            held_out_grams=0,
            exact_duplicates=exact_duplicates,
            examples=tuple(sorted(train_examples & {
                _normalize(text) for text in val_texts})[:5]))
    shared = []
    leaked = 0
    for mode in ("word", "char"):
        for gram, count in val[mode].items():
            if train[mode].get(gram):
                shared.append(gram)
                leaked += count
    overlap = max(leaked / total, duplicate_fraction)
    return ContaminationReport(
        overlap=overlap,
        shared=tuple(sorted(shared)),
        held_out_grams=total,
        exact_duplicates=exact_duplicates,
        examples=tuple(sorted(shared)[:5]))
