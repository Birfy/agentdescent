"""The command-line contract shared by the faithful algorithm ports.

Algorithm-specific vocabulary stays in each port. In particular, iteration
flags such as ``--rounds``, ``--generations``, ``--iterations``, and ``--steps``
must not be normalised here: they are part of the upstream algorithm's language.

Declaring a flag in one place is only half of a contract -- the code that
*honours* it has to live here too, or a port can grow a ``--yes`` it never reads
and every test still passes. So the three behaviours behind the shared flags are
functions, not prose: ``confirm`` for ``--yes``, ``completion_for`` for
``--provider``/``--model``, and the early ``--dry-run`` return that each port's
``main`` performs before touching data.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from agentdescent.agents import Usage, claude, openai_compatible


PROVIDER_CHOICES = ("claude", "openai", "glm")
# Providers served by the OpenAI-compatible adapter; 'glm' is a legacy alias.
OPENAI_COMPATIBLE = ("openai", "glm")
DEFAULT_MODEL = "claude-haiku-4-5"


def add_standard_args(
    parser: argparse.ArgumentParser,
    *,
    model_default: Optional[str] = DEFAULT_MODEL,
    model_help: str = "model id",
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
    parser.add_argument("--model", default=model_default, help=model_help)
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


def is_openai_compatible(args: argparse.Namespace) -> bool:
    """Whether ``--provider`` selects the OpenAI-compatible adapter."""
    return args.provider in OPENAI_COMPATIBLE


def confirm(args: argparse.Namespace) -> bool:
    """Whether the run may proceed to real model API calls.

    Honours ``--yes``, and treats a non-interactive stdin as consent so a port
    stays scriptable in CI. Prints ``aborted.`` when the answer is no, so the
    caller only has to ``return``.
    """
    if args.yes or not sys.stdin.isatty():
        return True
    if input("\nProceed with real API calls? [y/N] ").strip().lower() in ("y", "yes"):
        return True
    print("aborted.")
    return False


def completion_for(args: argparse.Namespace, *, usage: Optional[Usage] = None,
                   **kwargs):
    """Build the ``Completion`` that ``--provider`` and ``--model`` select.

    Extra keyword arguments reach whichever factory is chosen, so pass only
    options both accept; branch in the caller for genuinely one-sided ones (ADAS
    does this for the OpenAI-only ``--timeout``).
    """
    if is_openai_compatible(args):
        return openai_compatible(model=args.model, usage=usage, **kwargs)
    return claude(model=args.model, usage=usage, **kwargs)
