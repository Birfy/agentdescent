"""``(v, p)``: the two coordinates Genesis situates an agent by, and the log of
what the finite-lived agents did while they existed.

The paper's whole model is one pair::

    w = (v, p)          accepted version, repository-relative path

``v`` is already the engine's: a ledger snapshot with a version vector. ``p`` has
no counterpart at all -- a worker here holds the whole artifact and may write any
key -- so it is defined in this module, and everything downstream (the spatial
contract in :mod:`examples.genesis._spatial`, the recursion in
:mod:`examples.genesis._delegation`) is written against it rather than against a
path string passed around by hand.

Two operations act on the pair, and they differ in exactly one way:

``(v, p) ⇝ (v, q)``   recursive delegation -- the path moves, the version does not
``(v, p) → (v′, p)``  an accepted event -- the version moves

The first is the proposal policy's business and never touches the ledger; the
second is the aggregator's commit. Keeping them apart is the reason this file
exists: a delegation that advanced the version would make the recursion a
sequence of commits, which is the serial system Genesis is defined against.

:class:`WorldLog` is the other half. Agents are finite-lived by construction here
-- a rollout ends and its Python objects are collected -- so the only way depth,
episode counts and a parent's verdict survive the episode is to write them
somewhere the run can read afterwards. That is what the archive is upstream, and
the log is deliberately *not* part of the artifact: a record the agents could
edit is not evidence about them.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "CONTEXT_FILE",
    "EpisodeRecord",
    "LocalWorld",
    "ROUTING_HEADING",
    "SKILLS_DIR",
    "TRUNCATED",
    "WorldLog",
    "child_paths",
    "normalise",
    "owns",
    "parse_routing",
    "resolve_edit_path",
    "routing_entry",
]

#: The per-directory context record. Version-controlled, so it is part of ``v``
#: and a later agent inherits it -- which is the point: it is how a rejected
#: attempt leaves something behind (paper, appendix 1.4).
CONTEXT_FILE = "CONTEXT.md"

#: The section of a ``CONTEXT.md`` that says which child nodes exist and what
#: each is for. Upstream this is not documentation: it is **how a manager knows
#: where it may delegate**, and a node created without an entry at its parent is
#: a node later agents cannot find. Matched case-insensitively because upstream's
#: own tree writes both "Routing Table" and "Routing table".
ROUTING_HEADING = "## Routing Table"

#: Per-node reusable knowledge. The paper lists it among what an accepted version
#: carries -- "source files, path-specific context, constraints, validation
#: results, reusable skills and provenance records" (3.1) -- and upstream loads it
#: the same way it loads Context: along the chain, so a node inherits its
#: ancestors' skills (``EvoGit.Skills.hierarchical_skill_names/2``).
SKILLS_DIR = ".agents/skills"

#: Upstream's exact marker. It is not decoration: a CONTEXT.md that comes back
#: carrying it is telling the agent the file exceeded the per-file limit and
#: needs pruning, which is the maintenance obligation that keeps these files
#: "current state, not history".
TRUNCATED = "... [Content Truncated] ..."

_ROUTE_LINE = re.compile(
    r"""^\s*[-*]\s*          # a markdown list item
        `?\s*(?P<path>\.?/?[A-Za-z0-9._\-/]+?)\s*/?`?\s*   # the path, backticks optional
        (?:$|[-=]+>|\u2192|:)  # end of line, '->', an arrow, or a colon
    """, re.VERBOSE)


def parse_routing(body: str) -> List[str]:
    """The child paths a ``CONTEXT.md`` routes to, in the order it lists them.

    Only the routing section is read: a path mentioned in prose is a mention,
    and treating it as a route would let any sentence create authority.
    """
    if not body:
        return []
    lines, out, inside = body.splitlines(), [], False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("##"):
            inside = stripped.lower().startswith(ROUTING_HEADING.lower())
            continue
        if not inside:
            continue
        match = _ROUTE_LINE.match(line)
        if match:
            path = normalise(match.group("path"))
            if path and path not in out:
                out.append(path)
    return out


def routing_entry(body: str, path: str, note: str) -> Optional[str]:
    """``body`` with ``path`` added to its routing table, or ``None`` if listed.

    Idempotent on purpose. Upstream's rule for these files is *current state, not
    history* -- a routing table that grows an entry every time a manager passes
    through is a transcript, which is the one thing it must not become.
    """
    path = normalise(path)
    if not path or path in parse_routing(body):
        return None
    line = f"- `./{path}/` -> {note}"
    lines = (body or "").splitlines()
    for i, existing in enumerate(lines):
        if existing.strip().lower().startswith(ROUTING_HEADING.lower()):
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("##"):
                j += 1
            while j > i + 1 and not lines[j - 1].strip():
                j -= 1
            lines.insert(j, line)
            return "\n".join(lines) + "\n"
    tail = "" if not lines or not lines[-1].strip() else "\n"
    return "\n".join(lines) + tail + f"\n{ROUTING_HEADING}\n\n{line}\n"


def normalise(path: str) -> str:
    """Canonical prefix form: ``"./"``, ``"."``, ``"src/"`` and ``"src"`` agree.

    The root is the empty string rather than ``"."`` so that ``owns`` and
    ``startswith`` need no special case, and so a stray ``"./"`` from a model
    reply cannot become a directory literally named ``.``.
    """
    p = (path or "").strip().replace("\\", "/").strip("/")
    while p.startswith("./"):
        p = p[2:]
    return "" if p in ("", ".") else p


def owns(owner: str, path: str) -> bool:
    """Does an agent situated at ``owner`` have authority over ``path``?

    The root owns everything; any other node owns its own subtree and nothing
    else. This is the whole spatial contract -- everything in ``_spatial.py`` is
    the enforcement of this one predicate.
    """
    owner = normalise(owner)
    path = normalise(path)
    return owner == "" or path == owner or path.startswith(owner + "/")


def child_paths(owner: str, paths: Sequence[str]) -> List[str]:
    """The immediate sub-directories of ``owner`` present in ``paths``.

    What a manager can delegate *to* without inventing a decomposition: the
    routing table of the node it is standing on.
    """
    owner = normalise(owner)
    prefix = f"{owner}/" if owner else ""
    out = set()
    for path in paths:
        path = normalise(path)
        if not owns(owner, path) or path == owner:
            continue
        tail = path[len(prefix):]
        if "/" in tail:
            out.add(prefix + tail.split("/", 1)[0])
    return sorted(out)


def resolve_edit_path(state: Mapping[str, str], owner: str,
                      path: str) -> "Tuple[str, bool]":
    """``(path, was_node_relative)`` -- which path an agent at ``owner`` meant.

    An agent situated at ``src/frontend`` that writes ``lexer.py`` means
    ``src/frontend/lexer.py``; one that writes ``src/__init__.py`` means exactly
    that, and is asking for a change outside its own node. Both are ordinary, and
    telling them apart is the difference between a file landing where it belongs
    and a second copy of the project appearing at the repository root.

    Measured, before this existed: a `deepseek-v4-flash` run produced
    ``__init__.py`` and ``lexer.py`` from an agent at ``src/frontend``; read as
    repository-relative they fell outside its subtree, became requests, and the
    root -- which has authority everywhere -- wrote them at the top level. Four
    files of real work landed in the wrong place and the suite stayed at 0.000.

    The rule, in order, so that a genuine cross-node request is never mangled
    into a nested copy of its own path:

    1. already inside ``owner`` -- take it as written;
    2. ``owner/path`` exists in the tree -- the agent is editing its own file;
    3. ``path`` exists, or its first segment is an existing top-level entry --
       repository-relative, and therefore a request about someone else's node;
    4. ``owner/path`` is new and ``path``'s first segment is unknown -- a bare
       filename or a path below this node, so node-relative;
    5. anything else -- repository-relative.
    """
    owner, path = normalise(owner), normalise(path)
    if not path:
        return path, False
    if owns(owner, path) or not owner:
        return path, False
    nested = f"{owner}/{path}"
    if nested in state:
        return nested, True
    head = path.split("/", 1)[0]
    top = {key.split("/", 1)[0] for key in state}
    if path in state or head in top:
        return path, False
    return nested, True


@dataclass(frozen=True)
class LocalWorld:
    """One ``w = (v, p)``.

    ``version`` is carried for the record only -- the engine stamps the real
    version vector onto the diff -- but a world without it is not the paper's
    object, and the delegation log is unreadable without knowing which version a
    depth-3 executor was standing on.
    """

    version: int
    path: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", normalise(self.path))

    def delegate(self, q: str) -> "LocalWorld":
        """``(v, p) ⇝ (v, q)`` -- the path moves, the version does not.

        Refuses to leave the subtree: a parent may only situate a child *below*
        itself, or the contract its own parent is enforcing is unenforceable one
        level down.
        """
        q = normalise(q)
        if not owns(self.path, q):
            raise ValueError(
                f"delegation must stay inside the parent's subtree: "
                f"{self.path or './'} cannot situate a child at {q or './'}")
        return LocalWorld(version=self.version, path=q)

    def routing(self, state: Mapping[str, str]) -> List[str]:
        """Where this node says work may be delegated to.

        Its own ``CONTEXT.md`` routing table first -- upstream that table is the
        mechanism, not documentation: a manager delegates to the paths its node
        routes to. Falling back to the sub-directories that actually exist keeps
        a node without a table usable rather than sterile, which matters on the
        first episodes of a repository that has almost no ``CONTEXT.md`` yet.
        """
        key = f"{self.path}/{CONTEXT_FILE}" if self.path else CONTEXT_FILE
        declared = [p for p in parse_routing(state.get(key, "")) if owns(self.path, p)
                    and normalise(p) != self.path]
        return declared or child_paths(self.path, list(state))

    def skills(self, state: Mapping[str, str]) -> List[str]:
        """Skill files this node inherits, nearest ancestor last.

        Names only reach the brief, not bodies: upstream hands an agent the skill
        *names* available at its node and lets it read the ones it wants, because
        pasting every ancestor's skills into every episode is how a context
        window is spent on things nobody asked for.
        """
        out: List[str] = []
        for node in self._chain():
            prefix = f"{node}/{SKILLS_DIR}/" if node else f"{SKILLS_DIR}/"
            out += sorted(k for k in state if k.startswith(prefix))
        return out

    def context_key(self) -> str:
        """This node's own ``CONTEXT.md`` path."""
        return f"{self.path}/{CONTEXT_FILE}" if self.path else CONTEXT_FILE

    def situate(self, state: Mapping[str, str], *,
                contracts: Sequence[str] = (), max_chars: int = 8_000) -> str:
        """The context an agent entering this world is given.

        Upstream (``EvoGit.Core.ContextNode.build_context/2``) walks root → path
        and concatenates each directory's ``CONTEXT.md``; the agent inherits the
        chain, not just its own node. Reproduced here, plus a listing of the
        files it is responsible for -- an executor cannot edit what it cannot
        see, and handing it the whole tree is what the path coordinate exists to
        avoid.

        ``contracts`` is the part that is *not* optional, and leaving it out was a
        straight misreading of the paper: "An agent may inspect the complete
        project represented by v, but it begins from p" (3.1). The chain is where
        an agent **begins**, not a wall around what it may read. Upstream it would
        open the specification with a read tool; an agent here has no tools, so
        the human-supplied contract travels in the brief or it is invisible.

        Measured before this existed: the language specification sat at
        ``spec/CONTEXT.md`` -- a sibling of ``src/``, so on nobody's chain -- and
        every agent inferred the whole language from one failing input. They built
        a coherent toolchain with ``('NUMBER', '1')`` tokens and a ``parse(tokens)``
        signature, against a specification that says ``("num", 1)`` and
        ``parse(source)``. Five correct files, every one of them to the wrong
        contract, and a suite stuck at 0.000.
        """
        parts: List[str] = []
        for node in self._chain():
            key = f"{node}/{CONTEXT_FILE}" if node else CONTEXT_FILE
            body = state.get(key)
            if body:
                parts.append(f"--- {key} ---\n{body.strip()}")
        if contracts:
            from agentdescent.filetree import match_any
            given = [k for k in sorted(state) if match_any(k, contracts)]
            if given:
                parts.append("--- the contract you are building against "
                             "(human-supplied, read-only) ---\n"
                             + "\n\n".join(f"# {k}\n{state[k].strip()}" for k in given))
        skills = self.skills(state)
        if skills:
            parts.append("--- skills available here (read one before using it) ---\n"
                         + "\n".join(f"  {k}" for k in skills))
        mine = sorted(p for p in state if owns(self.path, p))
        listing = "\n".join(f"  {p} ({len(state[p])} bytes)" for p in mine) or "  (empty)"
        parts.append(f"--- files under {self.path or './'} ---\n{listing}")
        text = "\n\n".join(parts)
        return text if len(text) <= max_chars else text[:max_chars] + "\n" + TRUNCATED

    def _chain(self) -> List[str]:
        """Root first, this node last -- the inheritance order upstream uses."""
        if not self.path:
            return [""]
        nodes, acc = [""], ""
        for part in self.path.split("/"):
            acc = f"{acc}/{part}" if acc else part
            nodes.append(acc)
        return nodes


@dataclass
class EpisodeRecord:
    """One finite-lived agent, after it has ceased to exist.

    ``verdict`` is the responsible parent's, not the gate's: upstream the parent
    decides accept / reject / request-more-work on the returned contribution, and
    only what survives that is offered to the version history at all.
    """

    agent_id: str
    role: str                    # "manager" | "executor"
    path: str
    depth: int
    version: int
    objective: str = ""
    verdict: str = "accepted"    # "accepted" | "rejected" | "rework"
    n_edits: int = 0
    reason: str = ""


class WorldLog:
    """The archive: episodes, and the rework a parent asked for and did not get.

    Thread-safe because the engine's workers are threads and every one of them
    runs its own recursion into the same log.

    ``rework`` is how "request more work" survives a boundary the engine has no
    third outcome for. :class:`~agentdescent.policies.AcceptDecision` is a
    boolean, so a parent's *not yet* is recorded here and read by the next
    round's manager, which re-delegates to that path instead of choosing freely.
    That is the upstream behaviour reconstructed out of parts the engine already
    has -- not the engine growing a third verdict.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._episodes: List[EpisodeRecord] = []
        self._rework: Dict[str, str] = {}
        self._contract_violations = 0
        self._shape_violations = 0
        self._counter = 0

    # -- episodes ----------------------------------------------------------

    def next_id(self, role: str) -> str:
        with self._lock:
            self._counter += 1
            return f"{role[0].upper()}{self._counter:04d}"

    def record(self, episode: EpisodeRecord) -> EpisodeRecord:
        with self._lock:
            self._episodes.append(episode)
        return episode

    @property
    def episodes(self) -> List[EpisodeRecord]:
        with self._lock:
            return list(self._episodes)

    def observed_depth(self) -> int:
        """The deepest episode actually run -- the number upstream reports (4/5/8).

        Observed, not configured: a ``--depth 8`` run whose managers never
        delegate that far reports what happened, which is the only version of
        this number worth comparing with a paper.
        """
        with self._lock:
            return max((e.depth for e in self._episodes), default=0)

    def verdicts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for episode in self.episodes:
            out[episode.verdict] = out.get(episode.verdict, 0) + 1
        return out

    # -- rework ------------------------------------------------------------

    def request_rework(self, path: str, reason: str) -> None:
        with self._lock:
            self._rework[normalise(path)] = reason

    def take_rework(self) -> Optional[Tuple[str, str]]:
        """Pop one outstanding request, oldest first. ``None`` when there is none."""
        with self._lock:
            if not self._rework:
                return None
            path = next(iter(self._rework))
            return path, self._rework.pop(path)

    @property
    def pending_rework(self) -> Dict[str, str]:
        with self._lock:
            return dict(self._rework)

    # -- the spatial contract ---------------------------------------------

    def note_contract_violation(self, n: int = 1) -> None:
        with self._lock:
            self._contract_violations += n

    def note_shape_violation(self, n: int = 1) -> None:
        with self._lock:
            self._shape_violations += n

    @property
    def shape_violations(self) -> int:
        """Edits dropped for making the key space stop being a tree.

        Separate from ``contract_violations`` because they need opposite fixes:
        one is an agent writing outside its authority, the other is an agent
        mistaking a file for a node.
        """
        with self._lock:
            return self._shape_violations

    @property
    def contract_violations(self) -> int:
        """Edits dropped for escaping their author's subtree.

        Counted rather than swallowed, for the reason the engine counts
        ``section-violation``: a run whose proposals were all discarded and a run
        whose agents had nothing to say look identical without it, and they need
        opposite fixes.
        """
        with self._lock:
            return self._contract_violations

    def summary(self) -> str:
        verdicts = self.verdicts()
        return (f"episodes={len(self.episodes)}  depth={self.observed_depth()}  "
                f"accepted={verdicts.get('accepted', 0)}  "
                f"rejected={verdicts.get('rejected', 0)}  "
                f"rework={verdicts.get('rework', 0)}  "
                f"contract_violations={self.contract_violations}  "
                f"shape_violations={self.shape_violations}")
