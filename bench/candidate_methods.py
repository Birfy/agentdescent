"""Compatibility entry point for the candidate-method live benchmark."""

from examples.candidate_methods.benchmark import *  # noqa: F401,F403
from examples.candidate_methods.benchmark import main


if __name__ == "__main__":
    raise SystemExit(main())
