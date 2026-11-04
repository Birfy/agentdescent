"""Shared helper: peel a trailing rendered block back out of an applied target.

ACE (``# Playbook (ACE)``) and DGM (``## Capabilities``) both append a block
to the caller's prose seed. When that applied artifact is re-opened as the
next run's target, :func:`split_applied_block` finds the block, returns the
prose before it, the block's lines, and any user text *after* it (``tail``)
so a re-render can put that text back instead of silently dropping it (#743).
"""

from __future__ import annotations

from typing import Callable, List, NamedTuple, Optional


class AppliedBlock(NamedTuple):
    prose: str
    lines: List[str]
    tail: str


def _find_header(seed: str, header: str) -> Optional[int]:
    if seed.startswith(header + "\n") or seed.rstrip("\n") == header:
        return 0
    pos = seed.rfind("\n\n" + header + "\n")
    if pos >= 0:
        return pos + 2
    trimmed = seed.rstrip("\n")
    alt = "\n\n" + header
    if trimmed.endswith(alt):
        return len(trimmed) - len(header)
    return None


def split_applied_block(seed: str, header: str,
                        is_item: Callable[[str], bool],
                        is_continuation: Optional[Callable[[str], bool]] = None,
                        is_subheading: Optional[Callable[[str], bool]] = None,
                        ) -> Optional[AppliedBlock]:
    """Locate a trailing ``header`` block in ``seed``.

    A rendered block is contiguous, so the body is the run of lines directly
    under ``header`` that are items (``is_item``), wrapped item lines
    (``is_continuation``, only right after an item; whitespace-only ones count
    only when a non-blank continuation follows), or sub-headings
    (``is_subheading``, only when the next line is an item). The first line
    that is none of these -- a blank line, user prose, a user ``## `` heading
    with its own bullets -- starts ``tail``. Returns ``None`` when there is no
    header, or the header is followed by prose rather than items (a mid-text
    mention).
    """
    if not seed or not header:
        return None
    start = _find_header(seed, header)
    if start is None:
        return None
    after = seed[start + len(header):]
    if after.startswith("\n"):
        after = after[1:]
    raw = after.splitlines(keepends=True)
    texts = [ln.rstrip("\n") for ln in raw]
    last_item = -1
    in_item = False
    for idx, text in enumerate(texts):
        if is_item(text):
            last_item, in_item = idx, True
        elif in_item and is_continuation is not None and is_continuation(text):
            if text.strip():
                last_item = idx
        elif (is_subheading is not None and is_subheading(text)
              and idx + 1 < len(texts) and is_item(texts[idx + 1])):
            in_item = False
        else:
            break
    if last_item < 0 and after.strip():
        return None
    prose = seed[:start]
    if prose.endswith("\n\n"):
        prose = prose[:-1]
    return AppliedBlock(prose, texts[:last_item + 1], "".join(raw[last_item + 1:]))


def append_tail(body: str, tail: str) -> str:
    """Re-attach user ``tail`` text after a rendered block (one blank line)."""
    rest = tail.lstrip("\n")
    if not rest:
        return body
    if not body.endswith("\n"):
        body += "\n"
    return body + "\n" + rest
