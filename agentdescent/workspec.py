"""A rollout as data, so it can be handed to a process that is not this one.

Every attempt to run rollouts elsewhere hits the same wall first, and it is not
the executor. It is that the work itself cannot be sent. Measured, on this
package as it stands:

    Task, Diff, EvidenceCard, AppendRules, SingleSlot   picklable
    rewards.last_number()                               Can't pickle local object
    reflector(model)                                    Can't pickle local object
    LLMAgent(openai_compatible(...))                    Can't pickle local object
    run=lambda rendered, task: ...                      Can't pickle <lambda>

Every factory in the package returns a closure, and so does every example in the
documentation. So swapping `ThreadPoolExecutor` for `ProcessPoolExecutor` fails
on the first submit, and the fix is not a better executor -- it is a way to
describe the work that does not embed the functions.

`Ref` is that: a dotted target plus JSON configuration, resolved on the far side
by the worker's own copy of the code.

**Why not `cloudpickle`.** It sends closures directly, which is less work here
and worse everywhere after. It serialises the *sending* process's code and
executes it on the receiving side, so a version skew between the two becomes a
silent wrong answer rather than an import error; and it is a third-party
dependency in a package whose core has none. Resolving by name means the worker
runs the code it actually has, and a mismatch surfaces as a missing attribute.

**`resolve()` is arbitrary code execution**, and inside one process that is
unremarkable -- it already shares a trust domain with its caller. Across a
process or a machine boundary it is the boundary, so resolution is restricted to
an allowlist of import prefixes and configuration is restricted to JSON scalars.
Without both, anything that can write to a work queue can run code on every
worker.

**References nest.** `reflector(claude())` is the normal shape of an actor here,
so a `Ref` may hold `Ref`s in its config and they are resolved innermost first.
Without that, every composed actor would need a flattening factory whose only
purpose is to make the arguments JSON.

**Secrets do not travel in a spec.** A spec is pickled, logged, cached and
journalled; a spec carrying an API key leaks it into all four. `SandboxSpec`
carries variable *names*, and the values are read on the far side.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from importlib import import_module
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from .evolution import Task
from .policies import SandboxSpec

__all__ = [
    "DEFAULT_ALLOWED_PREFIXES",
    "Ref",
    "RefError",
    "RolloutSpec",
    "resolve_ref",
]

#: Import prefixes a worker will resolve without being told otherwise. Narrow on
#: purpose: this package's own factories, and nothing else. A caller running
#: their own code across processes widens it explicitly, which is the point at
#: which they get to think about who can write to the queue.
DEFAULT_ALLOWED_PREFIXES: Tuple[str, ...] = ("agentdescent.",)

#: JSON scalars, and containers of them. A `Ref` whose config held an arbitrary
#: object would be a closure again, wearing a different hat.
_SCALARS = (str, int, float, bool, type(None))


class RefError(ValueError):
    """A reference could not be resolved, and why -- never a bare ImportError."""


def _check_config(config: Mapping[str, Any], where: str) -> None:
    for key, value in config.items():
        if not isinstance(key, str):
            raise RefError(f"{where}: config keys must be strings, got {key!r}")
        if isinstance(value, Ref):
            continue                      # a nested reference; resolved in order
        if isinstance(value, _SCALARS):
            continue
        if isinstance(value, (list, tuple)):
            for item in value:
                if not isinstance(item, _SCALARS):
                    raise RefError(
                        f"{where}: config[{key!r}] holds {type(item).__name__}; only "
                        "JSON scalars survive the trip, and anything else is a "
                        "closure in disguise")
            continue
        if isinstance(value, dict):
            _check_config(value, f"{where}: config[{key!r}]")
            continue
        raise RefError(
            f"{where}: config[{key!r}] is {type(value).__name__}; only JSON "
            "scalars, lists and dicts of them are allowed")


@dataclass(frozen=True)
class Ref:
    """A callable named rather than sent: ``"module:attribute"`` plus config.

    ``target`` names a *factory*, and ``config`` is what to call it with, so the
    common case -- ``rewards.last_number(gold_key="gold")`` -- round-trips as
    data. Set ``call=False`` when the attribute is already the callable.
    """

    target: str
    config: Dict[str, Any] = field(default_factory=dict)
    #: Whether the resolved attribute is called with ``config`` to produce the
    #: callable (a factory), or already is one.
    call: bool = True

    def __post_init__(self) -> None:
        if ":" not in self.target:
            raise RefError(
                f"target {self.target!r} needs the form 'module:attribute' -- a "
                "bare dotted path is ambiguous about where the module ends")
        _check_config(self.config, f"Ref({self.target!r})")

    def resolve(self, allowed: Sequence[str] = DEFAULT_ALLOWED_PREFIXES) -> Callable:
        """Import and build the callable, on whichever side calls this."""
        return resolve_ref(self, allowed)


def resolve_ref(ref: Ref, allowed: Sequence[str] = DEFAULT_ALLOWED_PREFIXES) -> Callable:
    """Resolve ``ref``, refusing anything outside ``allowed``.

    The refusal happens **before the import**, not after: importing a module to
    discover that it was not allowed has already run its top level.
    """
    module_name, _, attr = ref.target.partition(":")
    if not any(module_name == p.rstrip(".") or module_name.startswith(p)
               for p in allowed):
        raise RefError(
            f"refusing to resolve {ref.target!r}: {module_name!r} is outside the "
            f"allowed prefixes {tuple(allowed)!r}. Widen them deliberately -- "
            "resolution runs whatever it imports, so this is the trust boundary "
            "between a work queue and the machine reading it.")
    try:
        module = import_module(module_name)
    except ImportError as e:
        raise RefError(
            f"{ref.target!r}: cannot import {module_name!r} ({e}). The worker "
            "resolves against its own copy of the code, so this usually means "
            "the two sides do not have the same package installed.") from None
    try:
        obj = getattr(module, attr)
    except AttributeError:
        raise RefError(
            f"{ref.target!r}: {module_name!r} has no attribute {attr!r}") from None
    if not ref.call:
        if not callable(obj):
            raise RefError(f"{ref.target!r} is not callable and call=False")
        return obj
    # Resolve nested references first, innermost outwards. Composition is the
    # normal shape here -- `reflector(claude())`, `LLMAgent(openai_compatible(...))`
    # -- and without it every composed actor would need a bespoke factory whose
    # only job is to flatten the arguments into JSON.
    config = {k: (resolve_ref(v, allowed) if isinstance(v, Ref) else v)
              for k, v in ref.config.items()}
    try:
        return obj(**config)
    except TypeError as e:
        raise RefError(
            f"{ref.target!r}: calling it with {sorted(ref.config)} failed ({e}). "
            "`config` is the factory's keyword arguments; set call=False if the "
            "attribute is already the callable.") from None


@dataclass(frozen=True)
class RolloutSpec:
    """One rollout, described completely enough to run somewhere else.

    Complete matters: whatever is missing here is something the far side has to
    guess, and a guess about which environment a measurement was taken in is how
    two scores stop being comparable.
    """

    rendered: str
    task: Task
    run: Ref
    reward: Ref
    #: What environment this rollout needs. The default asks for nothing special,
    #: which is what a local run has always got.
    sandbox: SandboxSpec = field(default_factory=SandboxSpec)
    #: Identifies this attempt. Re-dispatch after a worker is presumed lost
    #: reuses it, so the merger can tell a retry from a second opinion -- without
    #: that, one wrongly-presumed-dead worker puts two cards for the same task
    #: into the pool.
    lease_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def with_lease(self, lease_id: str) -> "RolloutSpec":
        return replace(self, lease_id=lease_id)

    def resolve(self, allowed: Sequence[str] = DEFAULT_ALLOWED_PREFIXES
                ) -> Tuple[Callable, Callable]:
        """Rebuild ``(run, reward)`` on this side of the boundary."""
        return self.run.resolve(allowed), self.reward.resolve(allowed)
