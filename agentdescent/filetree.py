"""Directories as evolvable state: load a tree, materialise it, serialise it.

An artifact's state is a flat ``{key: value}`` dict, and the engine never
interprets the keys -- it only asks whether two diffs touch the same one
(:func:`~agentdescent.aggregator.diffs_contradict`) and unions the ones that do
not (:func:`~agentdescent.aggregator.fuse_diffs`). So if the **keys are relative
file paths**, the whole aggregation stack acquires file-level semantics for free:

* two workers editing different files -> complementary diffs -> fused;
* two workers editing the same file -> a contradiction -> resolved on held-out
  score, best one wins.

That is the entire trick behind evolving a skill directory, an agent directory,
or a small agent codebase. This module is the translation layer between such a
directory and that dict:

    load_tree(path)        directory      -> {relpath: text}
    materialize(state, ws) {relpath: text} -> a workspace on disk
    canonical(state)       {relpath: text} -> one lossless string
    parse_tree(text)       that string     -> {relpath: text}

``canonical`` / ``parse_tree`` are a pair because :meth:`Strategy.render` is the
*only* channel through which the engine hands an artifact to ``run`` -- and it is
also the evaluation cache key (``EvolvingArtifact._signature``). A lossy render
would therefore make two different trees share a cached score, so the
serialisation has to be exact, and JSON is used rather than a prettier
delimiter-based format precisely because file *contents* cannot break it.
"""

from __future__ import annotations

import json
import os
import re
import stat
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "TreeSpec",
    "TreeError",
    "canonical",
    "parse_tree",
    "load_tree",
    "materialize",
    "match_any",
    "safe_relpath",
    "tree_summary",
]


class TreeError(ValueError):
    """A directory could not be represented as evolvable state, or vice versa.

    Always a caller-facing problem (a binary file in the tree, a path that
    escapes the workspace, a size cap), never a transient -- so it is raised
    rather than absorbed."""


# ---------------------------------------------------------------------------
# Glob matching
# ---------------------------------------------------------------------------
#
# `fnmatch` is not usable here: its `*` matches `/`, so `*.md` would match
# `references/rules.md`, and `**/*.md` would NOT match a top-level `SKILL.md`
# (the `**/` needs a literal slash to match). Both are silent -- one over-selects
# and one under-selects -- and either way the resulting tree is not the directory
# the caller pointed at. A ~20-line translation gets real glob semantics.

def _glob_to_regex(pattern: str) -> "re.Pattern":
    i, n, out = 0, len(pattern), ["(?s:"]
    while i < n:
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")          # zero or more directories
            i += 3
            continue
        if pattern.startswith("**", i):
            out.append(".*")
            i += 2
            continue
        c = pattern[i]
        if c == "*":
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        elif c == "[":
            j = pattern.find("]", i)
            if j < 0:
                out.append(re.escape(c))
            else:
                out.append("[" + pattern[i + 1:j].replace("\\", "\\\\") + "]")
                i = j
        else:
            out.append(re.escape(c))
        i += 1
    out.append(r")\Z")
    return re.compile("".join(out))


_GLOB_CACHE: Dict[str, "re.Pattern"] = {}


def match_any(relpath: str, patterns: Sequence[str]) -> bool:
    """Does ``relpath`` (posix, no leading ``./``) match any of ``patterns``?

    Supports ``*`` (within a path segment), ``**`` (across segments), ``?`` and
    ``[...]``. A bare directory pattern (``tests``) also matches everything under
    it, so ``frozen=["tests"]`` means what a reader expects.
    """
    for p in patterns:
        if not p:
            continue
        for candidate in (p, p.rstrip("/") + "/**") if "*" not in p and "?" not in p else (p,):
            rx = _GLOB_CACHE.get(candidate)
            if rx is None:
                rx = _GLOB_CACHE[candidate] = _glob_to_regex(candidate)
            if rx.match(relpath):
                return True
    return False


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------


def safe_relpath(path: str) -> str:
    """Normalise ``path`` to a workspace-relative posix path, or raise.

    Every key in the state is a **path proposed by a model**, so this is a
    security boundary, not tidying: an absolute path or a ``..`` segment would
    have :func:`materialize` write outside the workspace it was given.
    """
    if not isinstance(path, str) or not path.strip():
        raise TreeError(f"empty or non-string path: {path!r}")
    p = path.replace("\\", "/").strip()
    if p.startswith("/") or re.match(r"^[A-Za-z]:", p):
        raise TreeError(f"absolute path is not allowed in a file tree: {path!r}")
    if "\x00" in p:
        raise TreeError(f"path contains a NUL byte: {path!r}")
    parts: List[str] = []
    for seg in p.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            raise TreeError(
                f"path escapes the tree with '..': {path!r}. Paths are relative to "
                "the artifact root and may not traverse upwards.")
        parts.append(seg)
    if not parts:
        raise TreeError(f"path resolves to nothing: {path!r}")
    return "/".join(parts)


# ---------------------------------------------------------------------------
# What belongs in the tree
# ---------------------------------------------------------------------------


#: The aggregator's default trust region caps a single op value at
#: ``AggregatorConfig.trust_region_chars`` (32_000). A file above that can be
#: *loaded* but never *changed*: every diff touching it is rejected as
#: ``oversized``. Defaulting below the cap keeps the tree and the optimizer
#: consistent; :meth:`TreeSpec.validate_against` is the escape hatch when a
#: caller genuinely raises both.
DEFAULT_MAX_FILE_BYTES = 28_000


@dataclass(frozen=True)
class TreeSpec:
    """Which files make up an evolvable tree, and how big it may get."""

    include: Sequence[str] = (
        "**/*.md", "**/*.txt", "**/*.py", "**/*.json", "**/*.yaml", "**/*.yml",
        "**/*.toml", "**/*.sh", "**/*.cfg", "**/*.ini",
    )
    exclude: Sequence[str] = (
        "**/.git/**", "**/__pycache__/**", "**/node_modules/**", "**/.venv/**",
        "**/*.egg-info/**", "**/.pytest_cache/**", "**/.DS_Store",
    )
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
    max_files: int = 200
    max_total_bytes: int = 2_000_000

    def selects(self, relpath: str) -> bool:
        return (match_any(relpath, self.include)
                and not match_any(relpath, self.exclude))

    def validate_against(self, trust_region_chars: int) -> None:
        """Fail now if the loader admits files the optimizer can never accept.

        Without this the mismatch shows up as a pile of ``oversized`` entries in
        ``result.outcomes()`` five rounds in, which reads as "my reflector emits
        junk" rather than "this file was never editable"."""
        if self.max_file_bytes > trust_region_chars:
            raise TreeError(
                f"TreeSpec(max_file_bytes={self.max_file_bytes:,}) exceeds the "
                f"aggregator's trust region ({trust_region_chars:,} chars per op), "
                "so any diff touching a file above that size would always be "
                "rejected as 'oversized'. Lower max_file_bytes, or raise it with "
                "agg_config=AggregatorConfig(trust_region_chars=...).")


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

_MAGIC = "agentdescent.filetree/1"


def canonical(state: Mapping[str, str]) -> str:
    """A lossless, stable serialisation of a file tree.

    Stable because the engine caches evaluations on it, lossless because two
    different trees must never collide in that cache, and JSON because a file's
    *content* must not be able to forge the delimiter of its own container."""
    return json.dumps({"_": _MAGIC, "files": dict(sorted(state.items()))},
                      ensure_ascii=False, sort_keys=True)


def parse_tree(rendered: str) -> Dict[str, str]:
    """The inverse of :func:`canonical`.

    ``run`` receives ``Strategy.render(state)``, so this is how a runner gets the
    tree back in order to write it to disk. Also accepts a plain
    ``{path: content}`` object, which is what a hand-written test tends to pass.
    """
    try:
        data = json.loads(rendered)
    except Exception as e:  # noqa: BLE001 - the caller gets a precise message
        raise TreeError(
            f"not a serialised file tree ({type(e).__name__}); expected the output "
            "of agentdescent.filetree.canonical()") from e
    if isinstance(data, dict) and data.get("_") == _MAGIC:
        files = data.get("files") or {}
    elif isinstance(data, dict):
        files = data
    else:
        raise TreeError(f"serialised file tree must be an object, got {type(data).__name__}")
    out: Dict[str, str] = {}
    for k, v in files.items():
        if not isinstance(v, str):
            raise TreeError(f"file {k!r} has non-text content ({type(v).__name__})")
        out[safe_relpath(k)] = v
    return out


def tree_summary(state: Mapping[str, str], limit: int = 40) -> str:
    """A human/LLM-readable listing (paths + sizes), for prompts and logs.

    Deliberately *not* the render: a reflector that needs the whole tree in its
    prompt pays for the whole tree, and most reflections only need to know what
    exists (see :func:`agentdescent.treestrategy.tree_reflector`)."""
    items = sorted(state.items())
    lines = [f"{p}  ({len(c):,} chars)" for p, c in items[:limit]]
    if len(items) > limit:
        lines.append(f"... and {len(items) - limit} more file(s)")
    return "\n".join(lines) or "(empty tree)"


# ---------------------------------------------------------------------------
# Disk <-> state
# ---------------------------------------------------------------------------


def load_tree(path: str, spec: Optional[TreeSpec] = None) -> Dict[str, str]:
    """Read a directory into ``{relpath: text}``.

    Files that match ``spec.include`` but cannot be represented -- binary, or
    over ``max_file_bytes`` -- **raise** rather than being skipped. Skipping
    silently would mean the evolved tree is not the tree the caller pointed at,
    and :meth:`~agentdescent.evolution.EvolutionResult.write_to` would then
    delete-by-omission or write back a truncated directory. Widen ``exclude`` to
    say "this file is not part of the artifact" on purpose.
    """
    spec = spec or TreeSpec()
    root = os.path.abspath(os.path.expanduser(path))
    if not os.path.isdir(root):
        raise TreeError(f"not a directory: {path!r}")
    state: Dict[str, str] = {}
    total = 0
    rejected: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root).replace(os.sep, "/")
        rel_dir = "" if rel_dir == "." else rel_dir
        # prune excluded directories in place -- cheaper, and keeps .git out of
        # the walk entirely rather than filtering thousands of paths afterwards.
        dirnames[:] = [d for d in sorted(dirnames)
                       if not match_any(f"{rel_dir}/{d}".lstrip("/") + "/x", spec.exclude)]
        for fname in sorted(filenames):
            rel = f"{rel_dir}/{fname}".lstrip("/")
            full = os.path.join(dirpath, fname)
            if os.path.islink(full):
                continue                    # never follow links out of the tree
            if not spec.selects(rel):
                continue
            size = os.path.getsize(full)
            if size > spec.max_file_bytes:
                rejected.append(f"{rel} ({size:,} bytes > max_file_bytes="
                                f"{spec.max_file_bytes:,})")
                continue
            with open(full, "rb") as fh:
                raw = fh.read()
            if b"\x00" in raw:
                rejected.append(f"{rel} (binary)")
                continue
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                rejected.append(f"{rel} (not utf-8)")
                continue
            state[safe_relpath(rel)] = text
            total += len(raw)
    if rejected:
        raise TreeError(
            f"{len(rejected)} file(s) under {path!r} match the include patterns but "
            f"cannot be evolved as text:\n  " + "\n  ".join(rejected[:10])
            + ("\n  ..." if len(rejected) > 10 else "")
            + "\nAdd them to TreeSpec(exclude=...) to declare them outside the "
              "artifact, or raise max_file_bytes.")
    if len(state) > spec.max_files:
        raise TreeError(
            f"{len(state)} files exceed TreeSpec(max_files={spec.max_files}); "
            "narrow `include` or raise the cap.")
    if total > spec.max_total_bytes:
        raise TreeError(
            f"{total:,} bytes exceed TreeSpec(max_total_bytes="
            f"{spec.max_total_bytes:,}); every ledger commit stores the whole tree.")
    if not state:
        raise TreeError(
            f"no files under {path!r} matched TreeSpec.include={list(spec.include)}. "
            "An empty artifact cannot be evolved.")
    return state


#: Files made executable on materialisation. A skill that ships a helper script
#: is useless if the agent cannot run it, and the mode bit is not part of the
#: evolvable state (only content is), so it is derived from the path.
_EXEC_PATTERNS = ("**/*.sh", "scripts/**", "**/bin/**")


def materialize(state: Mapping[str, str], dest: str, *, prefix: str = "",
                exec_patterns: Sequence[str] = _EXEC_PATTERNS) -> List[str]:
    """Write a tree into ``dest`` (optionally under ``prefix``); return the paths.

    ``dest`` is created if missing. Every path is re-validated here even though
    the strategy validated it on the way in -- this is the last point before a
    model-authored string becomes a filesystem write, and a resumed ledger or a
    custom aggregator can put state in front of it that never passed through
    :class:`~agentdescent.treestrategy.FileTree`.
    """
    root = os.path.abspath(dest)
    os.makedirs(root, exist_ok=True)
    written: List[str] = []
    for raw_path, content in sorted(state.items()):
        rel = safe_relpath(f"{prefix}/{raw_path}" if prefix else raw_path)
        full = os.path.join(root, rel.replace("/", os.sep))
        # belt and braces: safe_relpath already forbids `..`, but a symlinked
        # parent directory inside dest could still redirect the write.
        real_parent = os.path.realpath(os.path.dirname(full))
        if not (real_parent == os.path.realpath(root)
                or real_parent.startswith(os.path.realpath(root) + os.sep)):
            raise TreeError(f"refusing to write {rel!r} outside {dest!r}")
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as fh:
            fh.write(content)
        if match_any(rel, exec_patterns):
            os.chmod(full, os.stat(full).st_mode | stat.S_IXUSR | stat.S_IXGRP)
        written.append(rel)
    return written
