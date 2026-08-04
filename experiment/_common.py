"""Shared runtime helpers for the local TextGrad and OpenEvolve experiments."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from agentdescent.agents import Usage, claude, openai_compatible


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"


def add_model_args(
    parser: argparse.ArgumentParser,
    *,
    max_tokens: int = 2048,
    temperature: float = 0.0,
) -> argparse.ArgumentParser:
    parser.add_argument("--provider", choices=("openai", "glm", "claude"), default="openai")
    parser.add_argument("--model", default="glm-5.2")
    parser.add_argument("--max-tokens", type=int, default=max_tokens)
    parser.add_argument("--temperature", type=float, default=temperature)
    parser.add_argument(
        "--thinking",
        choices=("disabled", "enabled", "default"),
        default="disabled",
        help="GLM thinking mode; 'default' omits the provider-specific field",
    )
    parser.add_argument("--api-timeout", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true", help="skip the paid-run confirmation")
    return parser


def require_api_environment(provider: str) -> None:
    if provider in ("openai", "glm"):
        missing = [name for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL") if not os.getenv(name)]
        if missing:
            raise RuntimeError(f"missing environment variable(s): {', '.join(missing)}")


def confirm_paid_run(args: argparse.Namespace, description: str) -> None:
    if args.yes:
        return
    answer = input(
        f"{description}\n"
        f"provider={args.provider}, model={args.model}. This makes paid API calls. Continue? [y/N] "
    )
    if answer.strip().lower() not in ("y", "yes"):
        raise SystemExit("cancelled")


def make_completion(args: argparse.Namespace, usage: Usage, *, temperature: float | None = None):
    chosen_temperature = args.temperature if temperature is None else temperature
    if args.provider in ("openai", "glm"):
        provider_options = {}
        if args.thinking != "default":
            provider_options["thinking"] = {"type": args.thinking}
        return openai_compatible(
            model=args.model,
            usage=usage,
            max_tokens=args.max_tokens,
            timeout=args.api_timeout,
            temperature=chosen_temperature,
            **provider_options,
        )
    return claude(
        model=args.model,
        usage=usage,
        max_tokens=args.max_tokens,
        temperature=chosen_temperature,
    )


def usage_dict(usage: Usage) -> Dict[str, Any]:
    return {
        "calls": usage.calls,
        "failures": usage.failures,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "model_seconds": round(usage.seconds, 6),
    }


def sum_usage(items: Dict[str, Usage]) -> Dict[str, Any]:
    snapshots = {name: usage_dict(usage) for name, usage in items.items()}
    totals = {
        key: sum(snapshot[key] for snapshot in snapshots.values())
        for key in ("calls", "failures", "prompt_tokens", "completion_tokens", "total_tokens")
    }
    totals["model_seconds"] = round(
        sum(snapshot["model_seconds"] for snapshot in snapshots.values()), 6
    )
    return {"by_phase": snapshots, "total": totals}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)
