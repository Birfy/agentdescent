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


def _limits():
    """Cap CPU, address space and file size before any candidate code is imported.

    Done here, in the child, rather than with ``preexec_fn`` in the parent: that
    hook is unsafe when the parent has threads (the engine's workers do) and can
    deadlock before ``exec``, so the parent's timeout would never even start.
    Best effort -- absent on Windows, and some platforms refuse RLIMIT_AS.
    """
    try:
        import resource
    except ImportError:
        return
    for name, value in (("RLIMIT_CPU", 10), ("RLIMIT_AS", 1 << 30),
                        ("RLIMIT_FSIZE", 1 << 22)):
        limit = getattr(resource, name, None)
        if limit is not None:
            try:
                resource.setrlimit(limit, (value, value))
            except (ValueError, OSError):
                pass


def main(root):
    _limits()
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
