"""Collapse near-duplicate artifacts before the archive becomes noise.

Recursive self-improvement produces many *nearly* identical candidates: a
reflector changes one sentence, a strategy adds a synonym, a merge concatenates
two copies of the same rule. A scalar archive hides the bloat; a population or
QD archive pays for it: the same behaviour occupies several cells, selection
spends rollouts comparing copies, and the reflector sees "novelty" that is only
whitespace.

This module is a deterministic, dependency-free near-duplicate detector:

* **SimHash** -- 64-bit locality-sensitive hash; Hamming distance is a cheap
  estimate of cosine similarity over token weights. Good for long prompts,
  rules and code comments.
* **MinHash** -- a fixed-size signature of the token set; the fraction of equal
  positions estimates Jaccard similarity. Good for short rules and skill files.

Tokens are Unicode-aware (``\w+``), and CJK runs additionally contribute
character trigrams so two different Chinese rules are not one token. A text with
no word tokens (punctuation only) falls back to its characters; the empty string
is the only text that shares the all-zero fingerprint.

``dedupe_texts`` greedily clusters in input order, keeping the first item as the
representative. It is intentionally single-pass and deterministic: a second run
on the same input produces the same clusters, and the representative never
changes because a later near-duplicate arrived. The report exposes the members
so a caller can keep them as lineage rather than silently deleting them.

    report = dedupe_texts(rendered_artifacts, threshold=0.85)
    for cluster in report.clusters:
        print(cluster.representative, len(cluster.members))
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

__all__ = [
    "SemanticCluster",
    "DedupReport",
    "canonical_text",
    "simhash",
    "hamming",
    "simhash_similarity",
    "minhash_signature",
    "minhash_similarity",
    "dedupe_texts",
]

_TOKEN = re.compile(r"\w+", re.UNICODE)
_CJK = re.compile(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]")
_WHITESPACE = re.compile(r"\s+")


def canonical_text(text: str) -> str:
    """Lowercase, strip, and collapse whitespace without dropping word order."""
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    return _WHITESPACE.sub(" ", text.strip().lower())


def _tokens(text: str) -> List[str]:
    canon = canonical_text(text)
    words = _TOKEN.findall(canon)
    out: List[str] = []
    for word in words:
        out.append(word)
        if _CJK.search(word):
            if len(word) <= 3:
                out.append(word)
            else:
                out.extend(word[i:i + 3] for i in range(len(word) - 2))
    if not out and canon:
        out.extend(canon)
    return out


def simhash(text: str, *, bits: int = 64) -> int:
    """A SimHash fingerprint of a text, deterministic across processes."""
    if bits < 8 or bits > 64:
        raise ValueError("bits must be in [8, 64]")
    tokens = _tokens(text)
    if not tokens:
        return 0
    counts: Dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    vector = [0] * bits
    for token, weight in counts.items():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=32).digest()
        value = int.from_bytes(digest[:8], "big")
        for bit in range(bits):
            vector[bit] += weight if (value >> (bit % 64)) & 1 else -weight
    out = 0
    for bit, weight in enumerate(vector):
        if weight >= 0:
            out |= 1 << bit
    return out


def hamming(left: int, right: int) -> int:
    """Number of differing bits between two fingerprints."""
    return bin(int(left) ^ int(right)).count("1")


def simhash_similarity(left: int, right: int, *, bits: int = 64) -> float:
    """``1 - hamming/bits`` in ``[0, 1]``."""
    if bits < 1:
        raise ValueError("bits must be >= 1")
    return 1.0 - hamming(left, right) / bits


def _hash_token(token: str, seed: int) -> int:
    digest = hashlib.blake2b(f"{seed}:{token}".encode("utf-8"), digest_size=16).digest()
    return int.from_bytes(digest, "big")


def minhash_signature(text: str, *, num_perm: int = 64, seed: int = 0) -> Tuple[int, ...]:
    """A MinHash signature of the token set (not the token multiset)."""
    if num_perm < 1:
        raise ValueError("num_perm must be >= 1")
    tokens = set(_tokens(text))
    if not tokens:
        return tuple([0] * num_perm)
    signature = []
    for index in range(num_perm):
        signature.append(min(_hash_token(token, seed + index) for token in tokens))
    return tuple(signature)


def minhash_similarity(left: Sequence[int], right: Sequence[int]) -> float:
    """Fraction of equal positions in two signatures (Jaccard estimate)."""
    if len(left) != len(right):
        raise ValueError("signatures must have equal length")
    if not left:
        raise ValueError("signatures must not be empty")
    return sum(1 for a, b in zip(left, right) if a == b) / len(left)


@dataclass(frozen=True)
class SemanticCluster:
    """A representative and the near-duplicates folded into it."""

    representative: Any
    members: Tuple[Any, ...]
    similarity: float


@dataclass(frozen=True)
class DedupReport:
    """The clusters and what was folded away."""

    clusters: Tuple[SemanticCluster, ...]
    unique: int
    removed: int
    threshold: float
    method: str
    detail: str = ""

    @property
    def kept(self) -> Tuple[Any, ...]:
        return tuple(cluster.representative for cluster in self.clusters)

    @property
    def compression(self) -> float:
        total = self.unique + self.removed
        return self.removed / total if total else 0.0


def _fingerprint(text: str, method: str, bits: int, num_perm: int, seed: int) -> Any:
    if method == "simhash":
        return simhash(text, bits=bits)
    if method == "minhash":
        return minhash_signature(text, num_perm=num_perm, seed=seed)
    raise ValueError(f"unknown method {method!r}; use simhash or minhash")


def _similarity(left: Any, right: Any, method: str, bits: int) -> float:
    if method == "simhash":
        return simhash_similarity(left, right, bits=bits)
    return minhash_similarity(left, right)


def dedupe_texts(items: Iterable[Any], *, text: Callable[[Any], str] = str,
                 threshold: float = 0.9, method: str = "simhash",
                 bits: int = 64, num_perm: int = 64, seed: int = 0,
                 max_members: int = 20) -> DedupReport:
    """Greedily cluster near-duplicate texts in input order.

    The first item in a cluster is its representative and is included in
    ``members``. ``threshold`` is in ``[0, 1]``; the default ``0.9`` is
    deliberately conservative (it folds near-copies, not merely related rules).
    ``max_members`` bounds the stored member list per cluster; the counts and
    removal total are still exact. ``SemanticCluster.similarity`` is the
    **minimum** similarity of any member to the representative -- the loosest
    member admitted -- which is what a hygiene audit needs to inspect.
    """
    if method not in ("simhash", "minhash"):
        raise ValueError(f"unknown method {method!r}; use simhash or minhash")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if max_members < 0:
        raise ValueError("max_members must be >= 0")
    rows = list(items)
    if not rows:
        return DedupReport(clusters=(), unique=0, removed=0,
                           threshold=threshold, method=method,
                           detail="empty input")
    clusters: List[Dict[str, Any]] = []
    for item in rows:
        fingerprint = _fingerprint(text(item), method, bits, num_perm, seed)
        best = None
        best_similarity = -1.0
        for cluster in clusters:
            score = _similarity(fingerprint, cluster["fingerprint"], method, bits)
            if score > best_similarity:
                best, best_similarity = cluster, score
        if best is not None and best_similarity >= threshold:
            best["members"].append(item)
            best["min_similarity"] = min(best["min_similarity"], best_similarity)
        else:
            clusters.append({"representative": item, "members": [item],
                             "fingerprint": fingerprint,
                             "min_similarity": 1.0})
    result = []
    for cluster in clusters:
        members = tuple(cluster["members"][:max_members]) if max_members else ()
        result.append(SemanticCluster(representative=cluster["representative"],
                                      members=members,
                                      similarity=float(cluster["min_similarity"])))
    removed = len(rows) - len(clusters)
    detail = (f"{len(clusters)} cluster(s), {removed} near-duplicate(s) folded "
              f"at threshold={threshold:g} via {method}")
    return DedupReport(clusters=tuple(result), unique=len(clusters),
                       removed=removed, threshold=threshold, method=method,
                       detail=detail)
