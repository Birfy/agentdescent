"""Shared plumbing for the product surfaces (CLI + MCP).

The four P1 surfaces (`platform`, `capability`, `ops`, `experience`) all need
the same small things: a schema version, a fail-closed input error, JSON/JSONL
loading, atomic output, and a ``module:attribute`` ref resolver with a message
that names the ref. Keeping them here means the commands cannot drift on the
envelope or on how a bad file is reported.

This module is not a public API; it is imported by the surface modules and by
``cli``/``mcp``. It deliberately has no top-level package export.
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import tempfile
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Type, TypeVar

__all__ = [
    "SURFACE_SCHEMA_VERSION",
    "SurfaceError",
    "atomic_write_json",
    "atomic_write_text",
    "load_json",
    "load_jsonl",
    "parse_csv",
    "parse_floats",
    "parse_ints",
    "require_callable",
    "require_instance",
    "require_mapping",
    "require_text",
    "ensure_write_path",
    "resolve_builder",
    "resolve_ref",
    "surface_allow_from_env",
    "surface_write_dirs_from_env",
]

SURFACE_SCHEMA_VERSION = 1
T = TypeVar("T")


class SurfaceError(ValueError):
    """An input the surface cannot use; the message is safe to print."""


def require_text(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise SurfaceError(f"{name} must be a string")
    if not value.strip():
        raise SurfaceError(f"{name} must be a non-empty string")
    return value


def require_mapping(value: Any, name: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SurfaceError(f"{name} must be a JSON object")
    return dict(value)


def require_callable(value: Any, name: str) -> Any:
    if not callable(value):
        raise SurfaceError(f"{name} must be callable")
    return value


def require_instance(value: Any, cls: Type[T], name: str) -> T:
    if not isinstance(value, cls):
        raise SurfaceError(
            f"{name} must be {cls.__name__}, got {type(value).__name__}")
    return value


def _path(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SurfaceError(f"{name} must be a non-empty path")
    return os.path.abspath(os.path.expanduser(value))


def load_json(value: Any, *, name: str) -> Any:
    """A JSON object/array from a path or an inline mapping/sequence."""
    if isinstance(value, (Mapping, list, tuple)):
        return value
    path = _path(value, name)
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except OSError as error:
        raise SurfaceError(f"{name} {value!r} cannot be read: {error}") from None
    except ValueError as error:
        raise SurfaceError(f"{name} {value!r} is not valid JSON: {error}") from None


def load_jsonl(value: Any, *, name: str) -> Iterator[Dict[str, Any]]:
    """One JSON object per line, or a JSON array of objects."""
    if isinstance(value, list):
        rows = value
    else:
        path = _path(value, name)
        try:
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
        except OSError as error:
            raise SurfaceError(f"{name} {value!r} cannot be read: {error}") from None
        stripped = text.strip()
        if not stripped:
            return iter(())
        if stripped.startswith("["):
            try:
                rows = json.loads(stripped)
            except ValueError as error:
                raise SurfaceError(
                    f"{name} {value!r} is not valid JSON: {error}") from None
            if not isinstance(rows, list):
                raise SurfaceError(f"{name} must be a JSON array")
        else:
            rows = []
            for line_no, line in enumerate(text.splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError as error:
                    raise SurfaceError(
                        f"{name} line {line_no} is not valid JSON: {error}"
                    ) from None
                if not isinstance(row, Mapping):
                    raise SurfaceError(
                        f"{name} line {line_no} must be a JSON object")
                rows.append(dict(row))
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise SurfaceError(f"{name}[{index}] must be a JSON object")
        yield dict(row)


def atomic_write_text(path: str, text: str) -> str:
    """Write text atomically; returns the absolute path."""
    resolved = os.path.abspath(os.path.expanduser(path))
    directory = os.path.dirname(resolved)
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".agentdescent-surface-", dir=directory, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, resolved)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return resolved


def atomic_write_json(path: str, payload: Any) -> str:
    """Write ``payload`` as pretty JSON, atomically; returns the absolute path."""
    return atomic_write_text(
        path, json.dumps(payload, indent=2, sort_keys=True,
                         ensure_ascii=False, default=str) + "\n")


def surface_allow_from_env(environ: Optional[Mapping[str, str]] = None
                            ) -> tuple:
    """The operator's ref allowlist for the MCP surfaces (empty = refuse)."""
    raw = (environ if environ is not None else os.environ).get(
        "AGENTDESCENT_SURFACE_ALLOW", "")
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def surface_write_dirs_from_env(environ: Optional[Mapping[str, str]] = None
                                 ) -> tuple:
    """The operator's write-directory allowlist for the MCP surfaces.

    Unset defaults to the server's cwd and ``$AGENTDESCENT_HOME``; an empty
    value refuses every MCP write path.
    """
    env = environ if environ is not None else os.environ
    raw = env.get("AGENTDESCENT_SURFACE_WRITE_DIRS")
    if raw is None:
        home = env.get("AGENTDESCENT_HOME") or "~/.agentdescent"
        return (os.getcwd(), os.path.abspath(os.path.expanduser(home)))
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _allowed_module(module_name: str, allow: Optional[Sequence[str]]) -> bool:
    if allow is None:
        return True
    if not allow:
        return False
    for prefix in allow:
        prefix = str(prefix).strip()
        if not prefix:
            continue
        if module_name == prefix.rstrip(".") or module_name.startswith(
                prefix if prefix.endswith(".") else prefix + "."):
            return True
    return False


def ensure_write_path(path: Any, *, name: str,
                      allow: Optional[Sequence[str]] = None) -> Optional[str]:
    """Resolve a write path and enforce the operator's allowlist when set."""
    if path is None:
        return None
    resolved = os.path.abspath(os.path.expanduser(require_text(path, name)))
    if allow is not None:
        roots = [os.path.abspath(os.path.expanduser(str(root)))
                 for root in allow]
        if not roots or not any(
                resolved == root or resolved.startswith(root + os.sep)
                for root in roots):
            raise SurfaceError(
                f"{name} {path!r} is outside the operator's write allowlist "
                f"(AGENTDESCENT_SURFACE_WRITE_DIRS)")
    return resolved


def resolve_ref(ref: Any, *, name: str,
                allow: Optional[Sequence[str]] = None) -> Any:
    """Resolve ``module:attribute``; the message names the ref on failure."""
    text = require_text(ref, name)
    if ":" not in text:
        raise SurfaceError(
            f"{name} must be 'module:attribute', got {text!r}")
    module_name, attribute = text.rsplit(":", 1)
    if not _allowed_module(module_name, allow):
        raise SurfaceError(
            f"{name} {text!r} is outside the operator's ref allowlist "
            f"(AGENTDESCENT_SURFACE_ALLOW)")
    try:
        module = importlib.import_module(module_name)
    except Exception as error:  # noqa: BLE001 - the spec's import is user code
        raise SurfaceError(
            f"{name} {text!r} cannot import {module_name!r}: "
            f"{type(error).__name__}: {error}") from None
    try:
        return getattr(module, attribute)
    except AttributeError:
        raise SurfaceError(
            f"{name} {text!r}: module {module_name!r} has no attribute "
            f"{attribute!r}") from None


def resolve_builder(ref: Any, *, name: str,
                    allow: Optional[Sequence[str]] = None) -> Any:
    """Resolve a ref and call it when it is a function/class builder.

    Specs usually point at a factory (``mypkg.wiring:build_pipeline``), but a
    module may also expose a ready object. A function, builtin, method or class
    is called once; a callable instance is returned as-is (it is already the
    object the spec wanted). A failure names the ref instead of leaking a raw
    traceback.
    """
    value = resolve_ref(ref, name=name, allow=allow)
    if (inspect.isfunction(value) or inspect.isbuiltin(value)
            or inspect.ismethod(value) or inspect.isclass(value)):
        try:
            value = value()
        except Exception as error:  # noqa: BLE001 - the builder is user code
            raise SurfaceError(
                f"{name} builder {ref!r} failed: "
                f"{type(error).__name__}: {error}") from None
    return value


def parse_csv(value: Any, *, name: str, cast=str) -> tuple:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parts = list(value)
    else:
        raise SurfaceError(f"{name} must be a comma-separated string or a list")
    try:
        return tuple(cast(part) for part in parts)
    except (TypeError, ValueError) as error:
        raise SurfaceError(f"{name} has an invalid value: {error}") from None


def parse_floats(value: Any, *, name: str) -> tuple:
    return parse_csv(value, name=name, cast=float)


def parse_ints(value: Any, *, name: str) -> tuple:
    return parse_csv(value, name=name, cast=int)
