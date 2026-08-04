"""Shared command-line arguments for the faithful algorithm ports.

Algorithm-specific vocabulary stays in each port. In particular, iteration
flags such as ``--rounds``, ``--generations``, ``--iterations``, and ``--steps``
must not be normalised here: they are part of the upstream algorithm's language.
"""

from __future__ import annotations

import argparse


PROVIDER_CHOICES = ("claude", "openai", "glm")
DEFAULT_MODEL = "claude-haiku-4-5"


def add_standard_args(
    parser: argparse.ArgumentParser,
    *,
    model_default=DEFAULT_MODEL,
    max_seconds_default: float = 30.0,
) -> argparse.ArgumentParser:
    """Add the provider/runtime flags shared by every algorithm port.

    Defaults that describe an algorithm's measured setup remain caller-owned.
    DGM, for example, deliberately defaults ``model`` to ``None`` because its
    surrogate can run without an API, while async wall-clock budgets differ by
    workload.
    """
    parser.add_argument(
        "--provider",
        default="claude",
        choices=PROVIDER_CHOICES,
        help=("claude, or any OpenAI-compatible endpoint (DeepSeek, GLM, "
              "vLLM, ...) via OPENAI_BASE_URL + OPENAI_API_KEY; 'glm' is a "
              "legacy alias"),
    )
    parser.add_argument(
        "--model", default=model_default,
        help="model id (DGM may omit it for deterministic proposals)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--async",
        dest="asynchronous",
        action="store_true",
        help="run barrier-free (async_evolve)",
    )
    parser.add_argument(
        "--async-ratio", type=int, default=3, help="staleness lag budget")
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=max_seconds_default,
        help="async wall-clock budget",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show the plan without loading data or accessing the network",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip confirmation before real model API calls",
    )
    return parser
