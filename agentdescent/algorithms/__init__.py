"""Named algorithms a spec, the CLI or an MCP host can ask for by name (#384).

``EvolveSpec`` is JSON, so until now the only way to run something other than
the default recipe for a ``kind`` was to write Python. This registry is the
other way in: an *algorithm* is a name, a one-line summary, the ``kind`` s it
applies to, a declared fidelity, a typed parameter schema with defaults, and a
``build`` function that turns validated parameters into the pieces the spec
composer already knows how to place (``Policies`` slots, ``evolve()`` knobs,
``AggregatorConfig`` defaults).

    {"kind": "text", ..., "algorithm": "rrsi"}
    {"kind": "text", ..., "algorithm": {"name": "rrsi", "params": {"b_max": 3}}}

and, with no spec at all::

    agentdescent algorithms list [--kind text] [--json]
    agentdescent algorithms show rrsi

Everything an algorithm contributes is a *default*: an explicit ``policies``
slot, ``agg_config`` field or ``evolve`` knob in the same spec wins, exactly as
it wins over the defaults of a ``kind``.

The registry lives in the package (and so in the wheel) on purpose: the
``examples/`` ports are not shipped, and an algorithm a user can pick from a
spec has to be importable where the spec runs. A mechanism that an example also
uses is implemented here once and imported by the example.

Fidelity is a label, not a promise of a number: it says what the entry
reproduces, so a result is never read as more than it is.

* ``benchmark_faithful`` -- reproduces the published setup and its numbers.
* ``mechanism_microport`` -- the published mechanism, wired into this engine
  and checked against pinned fixtures, not the paper's benchmark.
* ``preset`` -- an engine component under an algorithm's name; no claim to be
  the paper's whole method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = ["Algorithm", "AlgorithmError", "BuildContext", "ENTRY_POINT_GROUP",
           "FIDELITY", "Overrides", "Param", "external_errors", "get", "names",
           "kind_mismatch_message", "register", "resolve_params", "listing"]

FIDELITY = ("benchmark_faithful", "mechanism_microport", "preset")


class AlgorithmError(ValueError):
    """An algorithm request that cannot be honoured, and what is wrong with it."""


_TYPES = {"int": int, "float": float, "bool": bool, "str": str, "list[str]": list}


@dataclass(frozen=True)
class Param:
    """One parameter: a JSON value with a type, a default and (optionally) bounds."""

    name: str
    type: str                      # one of _TYPES
    default: Any
    help: str = ""
    choices: Optional[Tuple[Any, ...]] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None

    def __post_init__(self) -> None:
        if self.type not in _TYPES:
            raise ValueError(f"param {self.name!r}: type must be one of {sorted(_TYPES)}")

    def coerce(self, value: Any, *, where: str) -> Any:
        """``value`` as this parameter's type, or :class:`AlgorithmError`.

        Strict about type -- ``"3"`` is not an ``int`` and ``true`` is not a
        number -- except that an ``int`` is a valid ``float``, which JSON does
        not distinguish.
        """
        t = self.type
        if t == "bool":
            ok = isinstance(value, bool)
        elif t == "int":
            ok = isinstance(value, int) and not isinstance(value, bool)
        elif t == "float":
            ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        elif t == "str":
            ok = isinstance(value, str)
        else:  # list[str]
            ok = isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value)
        if not ok:
            raise AlgorithmError(f"{where}: expected {t}, got {value!r}")
        if t == "float":
            value = float(value)
        if t == "list[str]":
            value = list(value)
        if self.choices is not None and value not in self.choices:
            raise AlgorithmError(f"{where}: must be one of {list(self.choices)}, got {value!r}")
        if self.minimum is not None and isinstance(value, (int, float)) and value < self.minimum:
            raise AlgorithmError(f"{where}: must be >= {self.minimum}, got {value!r}")
        if self.maximum is not None and isinstance(value, (int, float)) and value > self.maximum:
            raise AlgorithmError(f"{where}: must be <= {self.maximum}, got {value!r}")
        return value

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"name": self.name, "type": self.type, "default": self.default,
                               "help": self.help}
        if self.choices is not None:
            out["choices"] = list(self.choices)
        if self.minimum is not None:
            out["minimum"] = self.minimum
        if self.maximum is not None:
            out["maximum"] = self.maximum
        return out


@dataclass
class BuildContext:
    """What ``build`` may read: the spec, and the models it already resolved.

    ``reflect`` is the reflection model (the spec's ``reflect``, or its ``agent``
    when none is named) -- what ``propose`` is driven by.
    """

    spec: Any
    reflect: Any = None


@dataclass
class Overrides:
    """What an algorithm contributes; every field is a default the spec can override.

    ``policies`` maps a ``Policies`` slot name to the built object, ``knobs`` are
    ``evolve()`` keyword defaults and ``agg_config`` are ``AggregatorConfig``
    fields. ``notes`` are shown by ``plan``.

    ``strategy_from`` is for an algorithm over a directory kind that needs its
    own strategy *class*: ``compose`` calls it with the ``FileTree`` it built for
    the kind (loaded with the kind's include list, carrying the kind's frozen
    globs and validators) and uses the result. Prefer it to a ``strategy`` knob,
    which replaces that ``FileTree`` and silently drops its safety rails.
    """

    policies: Dict[str, Any] = field(default_factory=dict)
    knobs: Dict[str, Any] = field(default_factory=dict)
    agg_config: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    strategy_from: Optional[Callable[[Any], Any]] = None


@dataclass(frozen=True)
class Algorithm:
    name: str
    summary: str
    kinds: Tuple[str, ...]
    fidelity: str
    params: Tuple[Param, ...]
    build: Callable[[BuildContext, Dict[str, Any]], Overrides]
    #: Where the method comes from, for a reader deciding whether it is the one they mean.
    source: str = ""
    #: What it adds to the cost of a run, in words ("none", "one reviewer call per candidate").
    extra_calls: str = "none"
    #: A page in ``docs/`` with the details, when there is one.
    doc: str = ""

    def __post_init__(self) -> None:
        if self.fidelity not in FIDELITY:
            raise ValueError(f"{self.name}: fidelity must be one of {FIDELITY}")

    def describe(self) -> Dict[str, Any]:
        """The JSON-able card ``show`` prints and ``plan`` echoes."""
        return {"name": self.name, "summary": self.summary, "kinds": list(self.kinds),
                "fidelity": self.fidelity, "source": self.source,
                "extra_calls": self.extra_calls, "doc": self.doc,
                "params": [p.to_dict() for p in self.params]}


_REGISTRY: Dict[str, Algorithm] = {}
#: Names registered by the shipped ``builtin`` module, for the generated catalog.
_BUILTIN_NAMES: set = set()

#: Entry-point group a third-party package can register an :class:`Algorithm` under.
ENTRY_POINT_GROUP = "agentdescent.algorithms"
_EXTERNAL_ERRORS: Dict[str, str] = {}
_external_loaded = False


def register(algorithm: Algorithm, *, replace: bool = False) -> Algorithm:
    """Add an algorithm. A second registration of a name is an error, not a shadow."""
    if algorithm.name in _REGISTRY and not replace:
        raise ValueError(f"algorithm {algorithm.name!r} is already registered")
    _REGISTRY[algorithm.name] = algorithm
    return algorithm


def names(*, builtin_only: bool = False) -> List[str]:
    _load_builtin()
    _load_external()
    return sorted(name for name in _REGISTRY
                  if not builtin_only or name in _BUILTIN_NAMES)


def external_errors() -> Dict[str, str]:
    """Entry-point algorithms that failed to load, ``name -> error``.

    A broken third-party plugin must not take the shipped registry down; it is
    recorded here instead, and `get` raises when that exact name is requested.
    """
    _load_builtin()
    _load_external()
    return dict(_EXTERNAL_ERRORS)


def get(name: Any, *, where: str = "algorithm") -> Algorithm:
    """The named algorithm, or an :class:`AlgorithmError` listing the valid names."""
    _load_builtin()
    _load_external()
    if (isinstance(name, str) and name in _EXTERNAL_ERRORS
            and name not in _REGISTRY):
        raise AlgorithmError(
            f"{where}: algorithm {name!r} failed to load: {_EXTERNAL_ERRORS[name]}")
    if not isinstance(name, str) or name not in _REGISTRY:
        raise AlgorithmError(
            f"{where}: unknown algorithm {name!r}; available: {sorted(_REGISTRY)} "
            "(see `agentdescent algorithms list`)")
    return _REGISTRY[name]


def listing(kind: Optional[str] = None, *,
            builtin_only: bool = False) -> List[Algorithm]:
    """Registered algorithms, by name; only those that apply to ``kind`` when given.

    ``builtin_only`` excludes algorithms a third-party entry point installed in
    this environment registered -- what the generated catalog and its check use
    so an installed plugin cannot rewrite our docs.
    """
    _load_builtin()
    _load_external()
    return [a for name, a in sorted(_REGISTRY.items())
            if (kind is None or kind in a.kinds)
            and (not builtin_only or name in _BUILTIN_NAMES)]


def kind_mismatch_message(algorithm: Algorithm, kind: str, *,
                          target: Optional[str] = None,
                          prefix: str = "algorithm") -> str:
    """One error text for "``algorithm`` does not apply to ``kind``".

    Names the algorithms that do apply (computed from the registry, so a
    plugin-registered algorithm shows up too) and, when the algorithm runs on
    ``text`` and ``target`` is a skill directory holding ``SKILL.md`` (or a
    single file), how to run it on that one file instead.
    """
    import os

    name = algorithm.name
    msg = (f"{prefix} {name!r} applies to kind {list(algorithm.kinds)}, "
           f"not {kind!r}")
    usable = [a.name for a in listing(kind)]
    if usable:
        msg += (f"; for kind {kind!r}: {usable} "
                f"(agentdescent algorithms list --kind {kind})")
    else:
        msg += (f"; no named algorithm applies to kind {kind!r}; omit "
                "--algorithm for the default loop")
    if kind != "text" and "text" in algorithm.kinds and target:
        path = os.path.expanduser(str(target))
        one = None
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "SKILL.md")):
            one = os.path.join(str(target), "SKILL.md")
        elif os.path.isfile(path):
            one = str(target)
        if one is not None:
            msg += (f"; to run {name} on one file, target it as text: "
                    f"agentdescent init {one} --kind text ... --algorithm {name}")
    return msg


def resolve_params(algorithm: Algorithm, params: Optional[Mapping[str, Any]], *,
                   where: str = "algorithm.params") -> Dict[str, Any]:
    """Defaults, overridden by ``params``; unknown names and wrong types are errors."""
    params = dict(params or {})
    known = {p.name: p for p in algorithm.params}
    unknown = sorted(set(params) - set(known))
    if unknown:
        raise AlgorithmError(
            f"{where}: {algorithm.name!r} has no parameter(s) {unknown}; "
            f"parameters: {sorted(known) or 'none'}")
    out = {p.name: p.default for p in algorithm.params}
    for key, value in params.items():
        out[key] = known[key].coerce(value, where=f"{where}.{key}")
    return out


_loaded = False
_load_error: Optional[BaseException] = None


def _load_builtin() -> None:
    """Register the shipped algorithms once.

    A failure is remembered and raised again on every later call: retrying would
    hit a half-populated registry (``already registered``) and bury the real
    error under a confusing one."""
    global _loaded, _load_error
    if _load_error is not None:
        raise _load_error
    if not _loaded:
        _loaded = True
        try:
            from . import builtin  # noqa: F401  (registers on import)
            _BUILTIN_NAMES.update(_REGISTRY)
        except BaseException as e:
            _load_error = e
            raise


def _load_external() -> None:
    """Register algorithms advertised by installed packages via entry points.

    One import is one plugin's chance to add a name; a failure is recorded in
    ``_EXTERNAL_ERRORS`` rather than raised, so a broken third-party package
    cannot hide the shipped algorithms. The group is the whole contract: an
    entry point loads to an :class:`Algorithm` (or a zero-argument callable
    returning one) and its name must match the entry point name.
    """
    global _external_loaded
    if _external_loaded:
        return
    _external_loaded = True
    from importlib import metadata

    try:
        points = metadata.entry_points(group=ENTRY_POINT_GROUP)
    except TypeError:       # Python 3.9: no `group=` keyword
        points = metadata.entry_points().get(ENTRY_POINT_GROUP, [])
    for point in points:
        name = getattr(point, "name", repr(point))
        try:
            loaded = point.load()
            algorithm = (loaded() if callable(loaded)
                         and not isinstance(loaded, Algorithm) else loaded)
            if not isinstance(algorithm, Algorithm):
                raise TypeError(
                    f"entry point returned {type(algorithm).__name__}, expected Algorithm")
            if algorithm.name != name:
                raise ValueError(
                    f"entry point is named {name!r} but returned {algorithm.name!r}; "
                    "the names must match")
            register(algorithm)
        except KeyboardInterrupt:
            raise
        except BaseException as e:  # noqa: BLE001 - one plugin must not break the registry
            _EXTERNAL_ERRORS[name] = f"{type(e).__name__}: {e}"
