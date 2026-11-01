"""Diagnose a run: turn show's evidence into one actionable verdict.

`show` returns the samples, rejected candidates and outcome histogram; a user
still has to decide whether the agent, the reflector, the gate or the scorer is
the problem. This module makes that decision explicit and names the next step.
It reads the same `show_payload`, so the CLI and MCP surfaces see one answer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["DiagnosisReport", "adopted_count", "diagnose_payload"]


@dataclass(frozen=True)
class DiagnosisReport:
    """One run's failure mode, the evidence, and what to try next."""

    run_id: str
    verdict: str
    summary: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    recommendations: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {"run_id": self.run_id, "verdict": self.verdict,
                "summary": self.summary, "evidence": self.evidence,
                "recommendations": list(self.recommendations)}


def _samples(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(payload.get("samples") or [])


def _rejected(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(payload.get("rejected") or [])


def _reward(sample: Dict[str, Any]) -> Optional[float]:
    try:
        value = sample.get("reward")
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _matches_gold(sample: Dict[str, Any]) -> bool:
    if sample.get("gold") is None:
        return False
    if sample.get("output_truncated") or sample.get("gold_truncated"):
        return False
    return str(sample.get("output")) == str(sample.get("gold"))


# Outcome categories that mean a candidate was adopted. Named algorithms
# (GEPA / SkillOpt / DGM) record their own category instead of
# ``MergeOutcome.COMMITTED`` ("committed"), so diagnose and the decision log
# must recognise them too.
_ADOPT_OUTCOME_KEYS = frozenset({
    "committed",
    "gepa-select",
    "skillopt-accept",
    "dgm-select",
})


def adopted_count(
    *,
    history: Optional[Any] = None,
    outcomes: Optional[Dict[str, Any]] = None,
) -> int:
    """How many candidates a run adopted.

    Prefers the per-round ``committed`` totals (the same source as the CLI's
    ``+N/-M`` lines). When there is no history, falls back to outcome
    categories that mean adoption: the default ``committed`` plus the named
    algorithm categories ``gepa-select``, ``skillopt-accept`` and
    ``dgm-select``.
    """
    if history:
        total = 0
        for row in history:
            try:
                if isinstance(row, dict):
                    value = row.get("committed", 0)
                else:
                    value = getattr(row, "committed", 0)
                total += int(value or 0)
            except (TypeError, ValueError):
                continue
        return total
    outcomes = outcomes or {}
    total = 0
    for key in _ADOPT_OUTCOME_KEYS:
        try:
            total += int(outcomes.get(key, 0) or 0)
        except (TypeError, ValueError):
            continue
    return total



def _has_log(run_id: str, store: Optional[str]) -> bool:
    """Whether the run left a ``log.txt``; only a detached run writes one."""
    try:
        from . import runstore
        return os.path.exists(runstore.get(run_id, store=store).log_path())
    except Exception:  # noqa: BLE001 - an unknown run has no log to read
        return False


def _run_failed_next(run_id: str, error: str, store: Optional[str]) -> Tuple[str, ...]:
    """Next steps for a failed run that point somewhere real (#635).

    A foreground run has no ``log.txt``, and ``doctor`` never checks a model
    name, so for a provider refusal both old suggestions led nowhere."""
    first = ("read log.txt / status.json for the first failure"
             if _has_log(run_id, store) else
             "read status.json (`agentdescent show <run_id>`) for the first failure")
    if "QuotaExhausted" in error or "quota for this model is used up" in error:
        return ("the account's quota for this model is used up; wait for the "
                "reset the provider names, or point api_key_env / model at "
                "another plan",
                first.replace("<run_id>", run_id))
    if "ProviderConfigError" in error or " -- check that " in error \
            or "does not serve model" in error:
        return ("the provider refused the model or key; fix the hint in the "
                "error (model name, base URL, or API key variable) and rerun",
                first.replace("<run_id>", run_id))
    return (first.replace("<run_id>", run_id),
            "run `agentdescent doctor` before retrying")


def diagnose_payload(run_id: str, *, store: Optional[str] = None) -> DiagnosisReport:
    """Classify why a run ended where it did.

    The verdicts are deliberately few: ``solved``, ``partial-improvement``,
    ``running``, ``cancelled``, ``run-failed``, ``reflector-error``,
    ``scoring-mismatch``, ``agent-failures``, ``gate-rejected``,
    ``no-improvement`` and ``unknown``. Evidence is bounded to a few items so a
    host can read it.
    """
    from .cli import show_payload

    payload = show_payload(run_id, store=store, diff=False)
    state = str(payload.get("state") or "")
    if state == "cancelled":
        return DiagnosisReport(
            run_id, "cancelled", "the run was cancelled before it finished",
            evidence={"stop_reason": payload.get("stop_reason"),
                      "outcomes": payload.get("outcomes") or {}},
            recommendations=("resume the run if the answer is still wanted",))
    if payload.get("error"):
        evidence = {"error": payload["error"]}
        if payload.get("propose_error"):
            evidence["propose_error"] = payload["propose_error"]
        return DiagnosisReport(
            run_id, "run-failed", f"the run reported an error: {payload['error']}",
            evidence=evidence,
            recommendations=_run_failed_next(run_id, str(payload["error"]), store))
    if state == "running":
        return DiagnosisReport(
            run_id, "running", "the run is still running; there is no verdict yet",
            evidence={"stop_reason": payload.get("stop_reason"),
                      "history": payload.get("history") or []},
            recommendations=("use `watch`/`status` and diagnose again when done",))

    final = payload.get("final_reward")
    final_value = float(final) if final is not None else None
    outcomes = payload.get("outcomes") or {}
    committed = adopted_count(history=payload.get("history") or None,
                              outcomes=outcomes)
    target_hit = payload.get("stop_reason") == "target_reward"
    if final_value is not None and (target_hit or final_value >= 0.999):
        return DiagnosisReport(run_id, "solved",
                               f"held-out reward is {final_value:.3f}",
                               evidence={"final_reward": final_value,
                                         "outcomes": outcomes},
                               recommendations=("review and apply the artifact",))

    perr = payload.get("propose_errors") or 0
    if perr and not committed:
        detail = payload.get("propose_error") or "no first error recorded"
        return DiagnosisReport(
            run_id, "reflector-error",
            f"{perr} reflector call(s) failed; the loop was not learning",
            evidence={"propose_errors": perr, "propose_error": detail,
                      "outcomes": outcomes},
            recommendations=("check the reflector model, endpoint and key "
                             "(`agentdescent doctor`)",))

    samples = _samples(payload)
    rejected = _rejected(payload)
    low = [sample for sample in samples
           if (_reward(sample) is not None and _reward(sample) < 0.999)]
    all_match = bool(low) and all(_matches_gold(sample) for sample in low)
    if all_match:
        worst = low[0]
        return DiagnosisReport(
            run_id, "scoring-mismatch",
            "the agent's answers match gold but the run scored them below 1",
            evidence={"sample": worst, "outcomes": outcomes},
            recommendations=("inspect the scorer/normaliser; re-run `plan` "
                             "with the same data",))

    if committed:
        return DiagnosisReport(
            run_id, "partial-improvement",
            f"committed {committed} candidate(s); held-out reward "
            f"{final_value:.3f} is below the target",
            evidence={"final_reward": final_value, "outcomes": outcomes,
                      "committed": committed},
            recommendations=("raise rounds or n_workers for another attempt",
                             "inspect the remaining failures in `show --json`"))

    if rejected:
        top = rejected[-1]
        return DiagnosisReport(
            run_id, "gate-rejected",
            "candidates were proposed but the held-out gate refused them",
            evidence={"rejected": len(rejected), "last": top,
                      "outcomes": outcomes},
            recommendations=("inspect `rejected` diffs and their p_improve",
                             "if every proposal is a regression, improve the reflector "
                             "prompt or data",))

    failing = [sample for sample in low if not _matches_gold(sample)]
    if failing and len(low) >= max(1, (len(samples) + 1) // 2):
        worst = low[0]
        return DiagnosisReport(
            run_id, "agent-failures",
            "the agent is answering wrongly; the reflector has failures to learn from",
            evidence={"sample": worst, "samples": len(samples),
                      "failing": len(failing), "outcomes": outcomes},
            recommendations=("read the samples' output vs gold",
                             "fix the skill/prompt or add cases for this failure class",))

    if final_value is not None and final_value >= 0.5:
        return DiagnosisReport(
            run_id, "no-improvement",
            f"held-out reward {final_value:.3f} but no candidate was adopted",
            evidence={"final_reward": final_value, "outcomes": outcomes},
            recommendations=("raise rounds or n_workers", "inspect the outcome histogram",))
    return DiagnosisReport(
        run_id, "unknown",
        "not enough evidence to classify the run",
        evidence={"outcomes": outcomes, "final_reward": final_value},
        recommendations=("run `show --json` and read samples/rejected",))
