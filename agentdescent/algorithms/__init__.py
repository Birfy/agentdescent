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

__all__ = ["Algorithm", "AlgorithmError", "BuildContext", "FIDELITY", "Overrides",
           "Param", "get", "names", "register", "resolve_params", "listing"]

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
    """

    policies: Dict[str, Any] = field(default_factory=dict)
    knobs: Dict[str, Any] = field(default_factory=dict)
    agg_config: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


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


def register(algorithm: Algorithm, *, replace: bool = False) -> Algorithm:
    """Add an algorithm. A second registration of a name is an error, not a shadow."""
    if algorithm.name in _REGISTRY and not replace:
        raise ValueError(f"algorithm {algorithm.name!r} is already registered")
    _REGISTRY[algorithm.name] = algorithm
    return algorithm


def names() -> List[str]:
    _load_builtin()
    return sorted(_REGISTRY)


def get(name: Any, *, where: str = "algorithm") -> Algorithm:
    """The named algorithm, or an :class:`AlgorithmError` listing the valid names."""
    _load_builtin()
    if not isinstance(name, str) or name not in _REGISTRY:
        raise AlgorithmError(
            f"{where}: unknown algorithm {name!r}; available: {sorted(_REGISTRY)} "
            "(see `agentdescent algorithms list`)")
    return _REGISTRY[name]


def listing(kind: Optional[str] = None) -> List[Algorithm]:
    """Registered algorithms, by name; only those that apply to ``kind`` when given."""
    _load_builtin()
    return [a for _, a in sorted(_REGISTRY.items()) if kind is None or kind in a.kinds]


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


def _load_builtin() -> None:
    global _loaded
    if not _loaded:
        _loaded = True
        try:
            from . import builtin  # noqa: F401  (registers on import)
        except BaseException:
            _loaded = False
            raise
