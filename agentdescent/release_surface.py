"""The ``agentdescent release`` surface: evidence -> decision -> signed receipt.

The library already has the three pieces a release needs:

* :func:`~agentdescent.trust_evidence.build_trust_evidence` turns paired task
  scores into eval/bootstrap, canary, conformal and sequential evidence;
* :func:`~agentdescent.promotion.promote` makes the fail-closed
  promote/defer/reject/rollback decision;
* :class:`~agentdescent.promotion_ledger.PromotionLedger` signs the accepted
  decision and writes its lineage node.

``release_pipeline`` composes those three in Python. This module is the same
composition for the two product transports -- the CLI and the MCP server -- with
the file formats, secret handling, lineage persistence and JSON envelope a
person or a host agent needs. It adds no statistics and no promotion rules.

The surface is deliberately fail-closed:

* a reject or defer is a normal report, and the lineage is not touched;
* an accepted release requires an artifact hash, a key id and a non-empty
  secret read from the environment or a file (never from argv);
* a malformed scores/rules/lineage file is an input error, never a partial
  release;
* the signed manifest is written only after the ledger verifies its signature.

``agentdescent release --json`` and the MCP ``release`` tool return the same
object, with ``schema_version: 1``.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import fields, replace
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .lineage import LineageGraph, build_lineage
from .policy_release import (
    ArtifactManifest, SignedEnvelope, verify_manifest,
)
from .promotion import PromotionRules
from .promotion_ledger import PromotionLedger
from .release_pipeline import ReleaseReport, release_evidence
from .trust_evidence import build_trust_evidence

__all__ = [
    "RELEASE_SCHEMA_VERSION",
    "ReleaseSurfaceError",
    "release_payload",
    "release_text",
    "verify_release",
    "load_scores",
    "load_rules",
    "load_lineage",
    "save_lineage",
    "resolve_secret",
]

RELEASE_SCHEMA_VERSION = 1

_DEFAULT_SECRET_ENV = "AGENTDESCENT_RELEASE_KEY"


class ReleaseSurfaceError(ValueError):
    """A release input the surface cannot use; the message is safe to print."""


def _mapping(value: Any, name: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ReleaseSurfaceError(f"{name} must be a JSON object")
    return dict(value)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ReleaseSurfaceError(f"{name} must be a number, not bool")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ReleaseSurfaceError(f"{name} must be a number") from None
    if not math.isfinite(number):
        raise ReleaseSurfaceError(f"{name} must be finite")
    return number


def load_scores(value: Any, *, name: str) -> Dict[str, float]:
    """A task -> score mapping from a mapping, a JSON file or ``-`` (stdin)."""
    if isinstance(value, Mapping):
        payload = dict(value)
    elif isinstance(value, str):
        text = value
        if value == "-":
            import sys

            text = sys.stdin.read()
        elif os.path.isfile(os.path.expanduser(value)):
            path = os.path.abspath(os.path.expanduser(value))
            try:
                with open(path, encoding="utf-8") as handle:
                    text = handle.read()
            except OSError as error:
                raise ReleaseSurfaceError(
                    f"{name} file {value!r} cannot be read: {error}") from None
        else:
            if value.strip().startswith("{") or value.strip().startswith("["):
                pass
            else:
                raise ReleaseSurfaceError(
                    f"{name} file {value!r} does not exist; pass a JSON "
                    "object, an existing path, or '-'")
        try:
            payload = json.loads(text)
        except ValueError as error:
            raise ReleaseSurfaceError(
                f"{name} is not valid JSON: {error}") from None
    else:
        raise ReleaseSurfaceError(
            f"{name} must be a JSON object, a path or '-'")
    payload = _mapping(payload, name)
    if not payload:
        raise ReleaseSurfaceError(f"{name} must not be empty")
    scores: Dict[str, float] = {}
    for key, raw in payload.items():
        if not isinstance(key, str) or not key.strip():
            raise ReleaseSurfaceError(f"{name} task ids must be non-empty strings")
        scores[key] = _finite(raw, f"{name}[{key!r}]")
    return scores


def _rules_from_mapping(payload: Mapping[str, Any]) -> PromotionRules:
    allowed = {field.name for field in fields(PromotionRules)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ReleaseSurfaceError(
            "unknown promotion rule(s): " + ", ".join(unknown)
            + "; allowed: " + ", ".join(sorted(allowed)))
    try:
        return PromotionRules(**dict(payload))
    except (TypeError, ValueError) as error:
        raise ReleaseSurfaceError(f"invalid promotion rules: {error}") from None


def load_rules(value: Any, *, min_effect: float = 0.0,
               max_risk: float = 1.0) -> PromotionRules:
    """Rules from a JSON file/mapping, or the surface's two common flags."""
    if value is None:
        try:
            return PromotionRules(min_effect=min_effect, max_risk=max_risk)
        except (TypeError, ValueError) as error:
            raise ReleaseSurfaceError(f"invalid promotion rules: {error}") from None
    if isinstance(value, Mapping):
        return _rules_from_mapping(value)
    if not isinstance(value, str):
        raise ReleaseSurfaceError("rules must be a JSON object or a path")
    path = os.path.abspath(os.path.expanduser(value))
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except OSError as error:
        raise ReleaseSurfaceError(
            f"rules file {value!r} cannot be read: {error}") from None
    except ValueError as error:
        raise ReleaseSurfaceError(
            f"rules file {value!r} is not valid JSON: {error}") from None
    return _rules_from_mapping(_mapping(payload, "rules"))


def _float_list(value: Any, name: str) -> Optional[Tuple[float, ...]]:
    if value is None:
        return None
    if isinstance(value, str):
        path = os.path.abspath(os.path.expanduser(value))
        try:
            with open(path, encoding="utf-8") as handle:
                value = json.load(handle)
        except OSError as error:
            raise ReleaseSurfaceError(
                f"{name} file {path!r} cannot be read: {error}") from None
        except ValueError as error:
            raise ReleaseSurfaceError(
                f"{name} file {path!r} is not valid JSON: {error}") from None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ReleaseSurfaceError(f"{name} must be a JSON array of numbers")
    return tuple(_finite(item, f"{name}[{index}]")
                 for index, item in enumerate(value))


def resolve_secret(*, secret_env: Optional[str] = None,
                   secret_file: Optional[str] = None) -> bytes:
    """The signing secret from the environment or a file, never argv.

    The default environment variable is ``AGENTDESCENT_RELEASE_KEY``. A caller
    that supplies both is refused, because which one won would be ambiguous.
    """
    if secret_env is not None and secret_file is not None:
        raise ReleaseSurfaceError(
            "pass secret_env= or secret_file=, not both")
    if secret_file is not None:
        path = os.path.abspath(os.path.expanduser(secret_file))
        try:
            with open(path, "rb") as handle:
                secret = handle.read()
        except OSError as error:
            raise ReleaseSurfaceError(
                f"secret file {secret_file!r} cannot be read: {error}") from None
        if not secret:
            raise ReleaseSurfaceError(
                f"secret file {secret_file!r} is empty")
        return secret
    name = secret_env or _DEFAULT_SECRET_ENV
    if not isinstance(name, str) or not name.strip():
        raise ReleaseSurfaceError("secret_env must be a non-empty string")
    raw = os.environ.get(name)
    if raw is None or not raw:
        raise ReleaseSurfaceError(
            f"release secret is not set; export {name}=... or pass "
            f"secret_file= (the secret is never read from argv)")
    return raw.encode("utf-8")


def load_lineage(path: Optional[str]) -> LineageGraph:
    """Load a lineage graph from JSON or JSONL; a missing file starts empty."""
    if path is None:
        return LineageGraph()
    resolved = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(resolved):
        return LineageGraph()
    try:
        with open(resolved, encoding="utf-8") as handle:
            text = handle.read()
    except OSError as error:
        raise ReleaseSurfaceError(
            f"lineage {path!r} cannot be read: {error}") from None
    if not text.strip():
        return LineageGraph()
    records: Any
    try:
        payload = json.loads(text)
        if isinstance(payload, Mapping):
            if "nodes" not in payload:
                raise ReleaseSurfaceError(
                    f"lineage {path!r} is a JSON object without a 'nodes' "
                    "list")
            records = payload["nodes"]
        elif isinstance(payload, list):
            records = payload
        else:
            raise ReleaseSurfaceError(
                f"lineage {path!r} must be a JSON object with 'nodes' or "
                "a list")
        if not isinstance(records, list):
            raise ReleaseSurfaceError(
                f"lineage {path!r}: 'nodes' must be a list")
    except json.JSONDecodeError:
        records = []
        for line_no, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError as error:
                raise ReleaseSurfaceError(
                    f"lineage {path!r} line {line_no} is not valid JSON: {error}"
                ) from None
            if not isinstance(record, Mapping):
                raise ReleaseSurfaceError(
                    f"lineage {path!r} line {line_no} is not an object")
            records.append(dict(record))
    try:
        return build_lineage(records)
    except (TypeError, ValueError, KeyError) as error:
        raise ReleaseSurfaceError(f"lineage {path!r} is malformed: {error}") from None


def _receipt_release(receipt: Any) -> Dict[str, Any]:
    """The signed ``release`` object from a receipt, or from itself."""
    if isinstance(receipt, Mapping):
        payload = dict(receipt)
    elif isinstance(receipt, str):
        path = os.path.abspath(os.path.expanduser(receipt))
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except OSError as error:
            raise ReleaseSurfaceError(
                f"receipt {receipt!r} cannot be read: {error}") from None
        except ValueError as error:
            raise ReleaseSurfaceError(
                f"receipt {receipt!r} is not valid JSON: {error}") from None
        payload = _mapping(payload, "receipt")
    else:
        raise ReleaseSurfaceError("receipt must be a JSON object or a path")
    release = payload.get("release") if "release" in payload else payload
    if release is None:
        raise ReleaseSurfaceError(
            "receipt has no signed release; a deferred or rejected run cannot "
            "be verified")
    return _mapping(release, "receipt release")


def _receipt_manifest(value: Any) -> ArtifactManifest:
    data = _mapping(value, "receipt release.envelope.manifest")
    allowed = {
        "artifact_id", "content_hash", "kind", "created_at", "parents",
        "evidence", "metadata",
    }
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ReleaseSurfaceError(
            "unknown manifest field(s): " + ", ".join(unknown))
    parents = data.get("parents") or ()
    if isinstance(parents, str):
        parents = (parents,)
    elif isinstance(parents, Sequence) and not isinstance(
            parents, (str, bytes)):
        parents = tuple(str(parent) for parent in parents)
    else:
        raise ReleaseSurfaceError(
            "receipt release.envelope.manifest.parents must be a list or a string")
    evidence = data.get("evidence") or {}
    metadata = data.get("metadata") or {}
    if not isinstance(evidence, Mapping):
        raise ReleaseSurfaceError(
            "receipt release.envelope.manifest.evidence must be a JSON object")
    if not isinstance(metadata, Mapping):
        raise ReleaseSurfaceError(
            "receipt release.envelope.manifest.metadata must be a JSON object")
    try:
        created_at = float(data.get("created_at") or 0.0)
    except (TypeError, ValueError):
        raise ReleaseSurfaceError(
            "receipt release.envelope.manifest.created_at must be a number"
        ) from None
    try:
        return ArtifactManifest(
            artifact_id=_required_text(
                data.get("artifact_id"),
                "receipt release.envelope.manifest.artifact_id"),
            content_hash=_required_text(
                data.get("content_hash"),
                "receipt release.envelope.manifest.content_hash"),
            kind=_required_text(
                data.get("kind"), "receipt release.envelope.manifest.kind"),
            created_at=created_at,
            parents=parents,
            evidence=dict(evidence),
            metadata=dict(metadata))
    except (TypeError, ValueError) as error:
        raise ReleaseSurfaceError(
            f"receipt release.envelope.manifest is invalid: {error}") from None


def _receipt_envelope(release: Mapping[str, Any]) -> SignedEnvelope:
    envelope = _mapping(
        release.get("envelope"), "receipt release.envelope")
    manifest = _receipt_manifest(
        envelope.get("manifest") or release.get("manifest"))
    signed_at = envelope.get("signed_at") or 0.0
    try:
        signed_at = float(signed_at)
    except (TypeError, ValueError):
        raise ReleaseSurfaceError(
            "receipt release.envelope.signed_at must be a number") from None
    return SignedEnvelope(
        manifest=manifest,
        key_id=_required_text(
            envelope.get("key_id"), "receipt release.envelope.key_id"),
        signature=_required_text(
            envelope.get("signature"), "receipt release.envelope.signature"),
        algorithm=str(envelope.get("algorithm") or "hmac-sha256"),
        signed_at=signed_at)


def verify_release(receipt: Any, *, key_id: Optional[str] = None,
                   secret_env: Optional[str] = None,
                   secret_file: Optional[str] = None,
                   lineage: Optional[str] = None) -> Dict[str, Any]:
    """Verify a signed release receipt without trusting its JSON.

    The receipt is what ``agentdescent release --json`` wrote (or its ``release``
    sub-object). The signature is recomputed from the embedded manifest, so an
    edited ``content_hash``, evidence or signature fails. ``key_id`` pins the
    expected signer: a receipt signed by another key is invalid, not accepted.
    ``lineage`` additionally requires the artifact's node to carry the same
    manifest hash.
    """
    release = _receipt_release(receipt)
    envelope = _receipt_envelope(release)
    secret = resolve_secret(secret_env=secret_env, secret_file=secret_file)
    expected_key = (None if key_id is None
                    else _required_text(key_id, "key_id"))
    if expected_key is not None and expected_key != envelope.key_id:
        # Do not consult a keyring under the receipt's own key here: the point
        # of pinning is that a receipt naming another signer is refused.
        check = None
        valid = False
        reason = (f"receipt is signed by {envelope.key_id!r}, expected "
                  f"{expected_key!r}")
    else:
        keyring = {expected_key or envelope.key_id: secret}
        check = verify_manifest(envelope, keyring)
        valid = check.valid
        reason = check.reason
    manifest = envelope.manifest
    evidence = manifest.evidence
    decision_hash = str(evidence.get("decision_hash") or "")
    if valid and evidence.get("action") != "promote":
        valid = False
        reason = "the manifest's decision action is not 'promote'"
    if valid and not (decision_hash and evidence.get("evidence_hash")):
        valid = False
        reason = "the manifest is missing decision_hash or evidence_hash"
    lineage_checked = False
    if valid and lineage is not None:
        graph = load_lineage(lineage)
        if manifest.artifact_id not in graph:
            valid = False
            reason = (f"lineage has no node for {manifest.artifact_id!r}")
        else:
            node = graph[manifest.artifact_id]
            recorded = node.metadata.get("manifest_hash")
            if recorded != manifest.manifest_hash():
                valid = False
                reason = "lineage manifest_hash does not match the receipt"
            else:
                lineage_checked = True
    return {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "ok": True,
        "valid": valid,
        "reason": reason,
        "artifact_id": manifest.artifact_id,
        "manifest_hash": manifest.manifest_hash(),
        "decision_hash": decision_hash,
        "key_id": ((check.key_id if check is not None else "")
                   or envelope.key_id),
        "lineage_checked": lineage_checked,
    }


def _atomic_write(path: str, text: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".agentdescent-release-", dir=directory, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def save_lineage(path: str, graph: LineageGraph) -> None:
    """Persist a lineage graph as one JSON object, atomically."""
    _atomic_write(path, json.dumps(
        {"schema_version": RELEASE_SCHEMA_VERSION,
         "nodes": graph.to_dict()["nodes"]},
        indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def _contamination(value: Any) -> Optional[bool]:
    if value is None or value == "unknown":
        return None
    if isinstance(value, bool):
        return value
    if value == "clean":
        return False
    if value == "contaminated":
        return True
    raise ReleaseSurfaceError(
        "contamination must be 'unknown', 'clean' or 'contaminated'")


def _approval(value: Any) -> Optional[bool]:
    if value is None or value == "unknown":
        return None
    if isinstance(value, bool):
        return value
    if value == "approved":
        return True
    if value == "rejected":
        return False
    raise ReleaseSurfaceError(
        "approval must be 'unknown', 'approved' or 'rejected'")


def _parents(value: Any) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parts = [str(part) for part in value]
    else:
        raise ReleaseSurfaceError("parents must be a comma-separated string or a list")
    if any(not part.strip() for part in parts):
        raise ReleaseSurfaceError("parent ids must be non-empty")
    return tuple(parts)


def release_payload(
        before: Mapping[str, float], after: Mapping[str, float], *,
        artifact_id: str,
        artifact_hash: str,
        key_id: str,
        secret: bytes,
        rules: Optional[PromotionRules] = None,
        lineage: Optional[str] = None,
        parents: Any = (),
        contaminated: Any = None,
        approved: Any = None,
        risk: Optional[float] = None,
        min_effect: float = 0.0,
        alpha: float = 0.05,
        seed: int = 0,
        proxy_gains: Any = None,
        oracle_gains: Any = None,
        issuer: str = "agentdescent",
        run_id: str = "",
        kind: str = "skill",
        metadata: Optional[Mapping[str, Any]] = None,
        out: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the release gate and return the CLI/MCP JSON object.

    ``before``/``after`` are task -> score mappings (or paths/JSON strings,
    which :func:`load_scores` resolves). A reject or defer returns
    ``ok: true, released: false``; only an accepted promotion is signed. The
    lineage file is written only after a release, atomically.
    """
    before_scores = load_scores(before, name="before")
    after_scores = load_scores(after, name="after")
    artifact_id = _required_text(artifact_id, "artifact_id")
    artifact_hash = _required_text(artifact_hash, "artifact_hash")
    key_id = _required_text(key_id, "key_id")
    if not isinstance(secret, (bytes, bytearray)) or not secret:
        raise ReleaseSurfaceError("secret must be non-empty bytes")
    policy = rules if rules is not None else PromotionRules()
    if not isinstance(policy, PromotionRules):
        raise ReleaseSurfaceError("rules must be a PromotionRules")
    parent_ids = _parents(parents)
    proxy = _float_list(proxy_gains, "proxy_gains")
    oracle = _float_list(oracle_gains, "oracle_gains")
    if (proxy is None) != (oracle is None):
        raise ReleaseSurfaceError(
            "proxy_gains and oracle_gains must be supplied together")
    contamination = _contamination(contaminated)
    approval = _approval(approved)
    effective_risk = None if risk is None else _finite(risk, "risk")
    if effective_risk is not None and not 0.0 <= effective_risk <= 1.0:
        raise ReleaseSurfaceError("risk must be in [0, 1]")
    min_effect = _finite(min_effect, "min_effect")
    if min_effect < 0:
        raise ReleaseSurfaceError("min_effect must be >= 0")
    # The rules file owns the decision threshold; using the flag here too would
    # let evidence and decision disagree.
    min_effect = policy.min_effect
    alpha = _finite(alpha, "alpha")
    if not 0.0 < alpha < 1.0:
        raise ReleaseSurfaceError("alpha must be in (0, 1)")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ReleaseSurfaceError("seed must be an int")
    if not isinstance(run_id, str):
        raise ReleaseSurfaceError("run_id must be a string")
    kind = _required_text(kind, "kind")
    if not isinstance(issuer, str) or not issuer.strip():
        raise ReleaseSurfaceError("issuer must be a non-empty string")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise ReleaseSurfaceError("metadata must be a JSON object or None")

    lineage_created = False
    if lineage is not None:
        lineage_created = not os.path.exists(os.path.expanduser(lineage))
    graph = load_lineage(lineage)
    ledger = PromotionLedger(graph, issuer=issuer)
    kwargs: Dict[str, Any] = {}
    if contamination is not None:
        kwargs["contaminated"] = contamination
    if approval is not None:
        kwargs["approved"] = approval
    if effective_risk is not None:
        kwargs["risk"] = effective_risk
    if proxy is not None:
        kwargs["proxy_gains"] = proxy
        kwargs["oracle_gains"] = oracle
    evidence = build_trust_evidence(
        before_scores, after_scores, alpha=alpha, seed=seed,
        min_effect=min_effect, **kwargs)
    # Optional statistical reports only participate when the rules ask for
    # them. A ``continue`` sequential report or an ``insufficient`` conformal
    # report would otherwise defer a release the caller never asked to gate on
    # those tests (``promote`` judges any report that is present).
    # Drop only a non-decision. A conformal/sequential *reject* must still
    # hard-fail even when that test was not required; otherwise the surface
    # would release a candidate the same evidence rejects.
    if (not policy.require_conformal
            and evidence.conformal is not None
            and evidence.conformal.decision == "insufficient"):
        evidence = replace(evidence, conformal=None)
    if (not policy.require_sequential
            and evidence.sequential is not None
            and evidence.sequential.decision == "continue"):
        evidence = replace(evidence, sequential=None)
    report: ReleaseReport = release_evidence(
        evidence, artifact_id=artifact_id, artifact_hash=artifact_hash,
        key_id=key_id, secret=bytes(secret), rules=policy, ledger=ledger,
        parents=parent_ids, run_id=run_id, kind=kind, metadata=metadata)

    if report.released and lineage is not None:
        save_lineage(lineage, graph)

    payload: Dict[str, Any] = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "ok": True,
        "released": report.released,
        "artifact_id": artifact_id,
        "artifact_hash": artifact_hash,
        "decision": report.decision.to_dict(),
        "evidence": report.evidence.to_dict(),
        "release": None if report.release is None else report.release.to_dict(),
        "detail": report.detail,
        "lineage": None if lineage is None else os.path.abspath(
            os.path.expanduser(lineage)),
        "lineage_created": lineage_created,
    }
    if report.released and report.release is not None:
        payload["next"] = [
            f"agentdescent audit {artifact_id} --archive <archive>",
            "keep the signed manifest with the release artifact",
        ]
    else:
        payload["next"] = [
            "inspect decision.missing / decision.reasons",
            "re-run with --clean/--approved or stronger paired evidence",
        ]
    if out is not None:
        _atomic_write(out, json.dumps(
            payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
        payload["out"] = os.path.abspath(os.path.expanduser(out))
    return payload


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ReleaseSurfaceError(f"{name} must be a string")
    if not value.strip():
        raise ReleaseSurfaceError(f"{name} must be a non-empty string")
    return value


def release_text(payload: Mapping[str, Any]) -> str:
    """A short human summary of a :func:`release_payload` result."""
    decision = payload.get("decision") or {}
    evidence = payload.get("evidence") or {}
    release = payload.get("release")
    lines = [
        f"released: {'yes' if payload.get('released') else 'no'}",
        f"artifact: {payload.get('artifact_id')}",
        f"decision: {decision.get('action')} "
        f"({'accepted' if decision.get('accepted') else 'not accepted'})",
    ]
    reasons = list(decision.get("reasons") or [])
    missing = list(decision.get("missing") or [])
    if reasons:
        lines.append("reasons: " + "; ".join(reasons))
    if missing:
        lines.append("missing: " + ", ".join(missing))
    evaluation = evidence.get("eval") or {}
    if evaluation:
        delta = evaluation.get("delta")
        delta_text = ("?" if delta is None else f"{float(delta):+.4f}")
        lines.append(
            f"eval: delta={delta_text} "
            f"ci={evaluation.get('ci')} passed={evaluation.get('passed')}")
    if release:
        lines += [
            f"manifest: {release.get('manifest_hash', '')}",
            f"key: {release.get('envelope', {}).get('key_id', '')}",
            f"lineage: {(release.get('lineage') or {}).get('artifact_id', '')}"
            + (" (new file)" if payload.get("lineage_created") else ""),
        ]
        lines.append(f"detail: {payload.get('detail', '')}")
    else:
        lines.append(f"detail: {payload.get('detail', '')}")
    next_steps = payload.get("next") or []
    if next_steps:
        lines.append("next:")
        lines.extend("  " + step for step in next_steps)
    return "\n".join(lines)
