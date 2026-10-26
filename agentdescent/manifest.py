"""A reproducible manifest for one RSI run.

A run is reproducible only if the spec, data, model configuration and code that
produced it are identifiable. ``RunManifest`` hashes those inputs into one
stable id and a field-by-field record; ``verify_manifest`` says which input
changed. Secrets are redacted before hashing and never stored.

    manifest = build_manifest(spec, data_paths=[spec.data["path"]],
                              model_config={"model": "x", "api_key": "..."})
    manifest.manifest_id
    verify_manifest(spec, manifest, data_paths=[...])  # [] means identical
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

__all__ = ["RunManifest", "build_manifest", "diff_manifest", "verify_manifest"]

_SECRET_NAMES = {
    "api_key", "apikey", "access_token", "auth_token", "authorization",
    "bearer", "password", "secret", "token", "client_secret", "private_key",
}


def _is_secret_key(key: Any) -> bool:
    name = str(key).strip().casefold()
    if name in _SECRET_NAMES:
        return True
    return name.endswith(("_key", "_token", "_secret", "_password"))


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def redact(value: Any) -> Any:
    """Recursively replace secret-looking values with ``"<redacted>"``."""
    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            out[key] = ("<redacted>" if _is_secret_key(key)
                        else redact(item))
        return out
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if isinstance(value, str):
        text = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._\-]+", r"\1<redacted>", value)
        text = re.sub(r"(?i)(api[_-]?key=)[^&\s]+", r"\1<redacted>", text)
        text = re.sub(r"\bsk-[A-Za-z0-9]{6,}\b", "<redacted>", text)
        return text
    return value


def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _tree_hash(path: str) -> str:
    entries = []
    for root, _dirs, files in os.walk(path):
        _dirs.sort()
        for name in sorted(files):
            full = os.path.join(root, name)
            rel = os.path.relpath(full, path).replace(os.sep, "/")
            entries.append((rel, _file_hash(full)))
    entries.sort()
    return _sha(_canonical(entries))


def _path_hash(path: str) -> str:
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isdir(path):
        return _tree_hash(path)
    if os.path.isfile(path):
        return _file_hash(path)
    return _sha(f"<missing:{path}>")


@dataclass(frozen=True)
class RunManifest:
    """The hashed inputs of one run and the resulting stable id."""

    manifest_id: str
    spec_hash: str
    data_hash: str
    model_hash: str
    code_version: str
    seeds: tuple = ()
    algorithm: Optional[str] = None
    files: Dict[str, str] = field(default_factory=dict)
    missing: Tuple[str, ...] = ()
    model_config: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "manifest_id": self.manifest_id,
            "spec_hash": self.spec_hash,
            "data_hash": self.data_hash,
            "model_hash": self.model_hash,
            "code_version": self.code_version,
            "seeds": list(self.seeds),
            "algorithm": self.algorithm,
            "files": dict(self.files),
            "missing": list(self.missing),
            "model_config": dict(self.model_config),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunManifest":
        if int(payload.get("schema_version", 1)) != 1:
            raise ValueError(f"unsupported RunManifest schema_version {payload.get('schema_version')!r}")
        return cls(
            manifest_id=str(payload.get("manifest_id", "")),
            spec_hash=str(payload.get("spec_hash", "")),
            data_hash=str(payload.get("data_hash", "")),
            model_hash=str(payload.get("model_hash", "")),
            code_version=str(payload.get("code_version", "")),
            seeds=tuple(payload.get("seeds") or ()),
            algorithm=payload.get("algorithm"),
            files=dict(payload.get("files") or {}),
            missing=tuple(payload.get("missing") or ()),
            model_config=dict(payload.get("model_config") or {}),
            extra=dict(payload.get("extra") or {}),
        )


def build_manifest(spec: Any, *, data_paths: Sequence[str] = (),
                   model_config: Optional[Mapping[str, Any]] = None,
                   code_version: str = "", extra: Optional[Mapping[str, Any]] = None
                   ) -> RunManifest:
    """Hash a spec, its data files/directories, model config and code version.

    ``data_paths`` defaults to the spec's ``data.path`` when it names a real
    file/directory. The manifest id is a hash of the canonical field record, so
    two runs with identical inputs share it.
    """
    spec_dict = spec.to_dict() if hasattr(spec, "to_dict") else dict(spec)
    spec_hash = _sha(_canonical(redact(spec_dict)))
    paths = list(data_paths)
    if not paths:
        data = spec_dict.get("data") or {}
        candidate = data.get("path") if isinstance(data, Mapping) else None
        if candidate:
            paths = [str(candidate)]
    missing = []
    files = {}
    for path in paths:
        absolute = str(os.path.abspath(os.path.expanduser(path)))
        files[absolute] = _path_hash(path)
        if not os.path.exists(os.path.expanduser(path)):
            missing.append(absolute)
    # Content-addressed: the set of content hashes, not the absolute paths.
    data_hash = _sha(_canonical(sorted(files.values())))
    safe_model = redact(dict(model_config or {}))
    model_hash = _sha(_canonical(safe_model))
    evolve = spec_dict.get("evolve") or {}
    seeds = tuple(evolve.get("seeds") or
                  ((evolve["seed"],) if "seed" in evolve else ())) \
        if isinstance(evolve, Mapping) else ()
    algorithm = spec_dict.get("algorithm")
    if isinstance(algorithm, Mapping):
        algorithm = algorithm.get("name")
    fields = {
        "spec_hash": spec_hash,
        "data_hash": data_hash,
        "model_hash": model_hash,
        "code_version": str(code_version),
        "seeds": list(seeds),
        "algorithm": algorithm,
        "extra": redact(dict(extra or {})),
    }
    return RunManifest(
        manifest_id=_sha(_canonical(fields)), spec_hash=spec_hash,
        data_hash=data_hash, model_hash=model_hash,
        code_version=str(code_version), seeds=seeds, algorithm=algorithm,
        files=files, missing=tuple(missing), model_config=safe_model,
        extra=redact(dict(extra or {})))


def diff_manifest(a: RunManifest, b: RunManifest) -> Dict[str, Dict[str, Any]]:
    """Field-level differences between two manifests (empty when identical)."""
    out = {}
    for key in ("spec_hash", "data_hash", "model_hash", "code_version",
                "seeds", "algorithm", "files", "missing", "model_config",
                "extra"):
        left, right = getattr(a, key), getattr(b, key)
        if left != right:
            out[key] = {"left": left, "right": right}
    return out


def verify_manifest(spec: Any, manifest: RunManifest, *, data_paths: Sequence[str] = (),
                    model_config: Optional[Mapping[str, Any]] = None,
                    code_version: str = "", extra: Optional[Mapping[str, Any]] = None
                    ) -> List[str]:
    """Rebuild the manifest from current inputs; return changed field names."""
    rebuilt = build_manifest(spec, data_paths=data_paths,
                             model_config=model_config,
                             code_version=code_version, extra=extra)
    return sorted(diff_manifest(manifest, rebuilt))
