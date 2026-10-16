"""Fixture test runner -- frozen, never shown to the agent, never edited by it.

Usage: ``python -I runner.py <workspace-root>``. Runs every ``test_*`` function
under ``tests/`` and prints one JSON object ``{"module::name": message}`` where
the message is ``""`` for a pass and the assertion text for a failure. Written
by hand rather than pytest so the offline suite does not depend on it.
"""

import importlib
import json
import pkgutil
import sys
import traceback


def main(root):
    sys.path.insert(0, root)
    results = {}
    tests = importlib.import_module("tests")
    for info in sorted(pkgutil.iter_modules(tests.__path__), key=lambda i: i.name):
        if not info.name.startswith("test_"):
            continue
        try:
            module = importlib.import_module(f"tests.{info.name}")
        except Exception as error:  # noqa: BLE001 - an import failure fails its tests
            results[f"{info.name}::<import>"] = f"{type(error).__name__}: {error}"
            continue
        for name in sorted(dir(module)):
            if not name.startswith("test_"):
                continue
            key = f"{info.name}::{name}"
            try:
                getattr(module, name)()
                results[key] = ""
            except AssertionError as error:
                # A bare `assert` has no message; show the failing line the way
                # pytest does, so the failure says *what* did not hold.
                line = traceback.extract_tb(error.__traceback__)[-1].line or ""
                results[key] = f"AssertionError: {error}".strip() + (
                    f"  [{line.strip()}]" if not str(error) else "")
            except Exception as error:  # noqa: BLE001
                results[key] = f"{type(error).__name__}: {error}"
    print(json.dumps(results))


if __name__ == "__main__":
    main(sys.argv[1])
