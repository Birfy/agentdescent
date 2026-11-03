"""Read-only module reachability audit for the AgentDescent package.

The package's product surface is the CLI and the MCP server; everything else is
either a library dependency of those, a top-level export, or an isolated module
that only its own tests/docs use. This tool makes that distinction measurable
without changing a single import:

    python -m tools.module_audit
    python -m tools.module_audit --json
    python -m tools.module_audit --limit 15

It is deliberately **report-only**: it never fails a build, moves a file or
changes an export. The output is the input to a cleanup decision (which module
to quarantine or merge), not the decision itself.

Categories:

``cli/mcp``
    Reachable from ``cli.py``, ``mcp.py``, ``evolvespec.py``, ``demo.py`` or
    ``__main__.py`` through intra-package imports.
``exported-only``
    Not on the CLI/MCP path, but reachable from the package ``__init__``
    (so it is part of the public Python surface).
``isolated``
    Neither: only its own tests/docs/examples reference it.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import warnings
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = ["analyze", "main"]

SCHEMA_VERSION = 1
_ROOT_MODULES = ("cli", "mcp", "evolvespec", "demo", "__main__")


def _package_of(key: str, packages: Mapping[str, bool]) -> str:
    if key == "__init__":
        return ""
    if packages.get(key, False):
        return key
    return key.rsplit(".", 1)[0] if "." in key else ""


def _resolve_relative(key: str, level: int, module: Optional[str],
                      packages: Mapping[str, bool]) -> str:
    package = _package_of(key, packages)
    parts = package.split(".") if package else []
    up = level - 1
    if up:
        parts = parts[:-up] if up <= len(parts) else []
    if module:
        parts.extend(module.split("."))
    return ".".join(parts)


def _parse(path: Path) -> Optional[ast.AST]:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            return ast.parse(source)
        except SyntaxError:
            return None


def _imports(path: Path, key: str, packages: Mapping[str, bool],
             package_name: str) -> Tuple[str, ...]:
    tree = _parse(path)
    if tree is None:
        return ()
    found = set()
    ref = re.compile(
        rf"{re.escape(package_name)}\.([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*):")
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for module in ref.findall(node.value):
                found.add(module)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == package_name:
                    found.add("__init__")
                elif alias.name.startswith(package_name + "."):
                    found.add(alias.name[len(package_name) + 1:])
        elif isinstance(node, ast.ImportFrom):
            base = ""
            if node.level:
                base = _resolve_relative(key, node.level, node.module, packages)
                if base:
                    found.add(base)
            elif node.module and (
                    node.module == package_name
                    or node.module.startswith(package_name + ".")):
                base = ("" if node.module == package_name
                        else node.module[len(package_name) + 1:])
                if base:
                    found.add(base)
            for alias in node.names:
                if alias.name == "*":
                    continue
                found.add((base + "." + alias.name) if base else alias.name)
    return tuple(sorted(found))


def _normalise(target: str, modules: Mapping[str, Path]) -> Optional[str]:
    if target in modules:
        return target
    prefixes = [key for key in modules if key.startswith(target + ".")]
    if not prefixes:
        return None
    return min(prefixes, key=len)


def _graph(root: Path, package_name: str) -> Tuple[Dict[str, Path], Dict[str, bool],
                                                   Dict[str, Tuple[str, ...]]]:
    files: Dict[str, Path] = {}
    packages: Dict[str, bool] = {}
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(root)
        parts = list(relative.with_suffix("").parts)
        is_package = parts[-1] == "__init__"
        if is_package:
            parts = parts[:-1]
        key = ".".join(parts) if parts else "__init__"
        files[key] = path
        packages[key] = is_package
    modules = {key: path for key, path in files.items() if key != "__init__"}
    raw = {key: _imports(path, key, packages, package_name)
           for key, path in files.items()}
    graph: Dict[str, Tuple[str, ...]] = {}
    for key, targets in raw.items():
        resolved = []
        for target in targets:
            normal = _normalise(target, modules)
            if normal is None:
                continue
            resolved.append(normal)
            # Importing ``a.b.c`` executes ``a/__init__`` and ``a/b/__init__``
            # too, so a reached leaf also reaches every known ancestor package.
            parts = normal.split(".")
            for index in range(1, len(parts)):
                ancestor = ".".join(parts[:index])
                if ancestor in modules:
                    resolved.append(ancestor)
        graph[key] = tuple(sorted(set(resolved)))
    return modules, packages, graph


def _reach(graph: Mapping[str, Sequence[str]], roots: Iterable[str]) -> set:
    seen = set()
    queue = deque(roots)
    while queue:
        key = queue.popleft()
        if key in seen or key not in graph:
            continue
        seen.add(key)
        for target in graph.get(key, ()):
            if target not in seen:
                queue.append(target)
    return seen


def _loc(path: Path) -> int:
    try:
        return len(path.read_text(encoding="utf-8").splitlines())
    except (OSError, UnicodeDecodeError):
        return 0


def _top_level_exports(root: Path) -> int:
    tree = _parse(root / "__init__.py")
    if tree is None:
        return 0
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "__all__"
                   for target in node.targets):
            continue
        value = node.value
        if isinstance(value, (ast.List, ast.Tuple)):
            return len(value.elts)
    return 0


def _cli_commands(root: Path) -> int:
    tree = _parse(root / "cli.py")
    if tree is None:
        return 0
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_parser"):
            continue
        if not (isinstance(func.value, ast.Name) and func.value.id == "sub"):
            continue
        if node.args and isinstance(node.args[0], ast.Constant) and isinstance(
                node.args[0].value, str):
            count += 1
    return count


def _mcp_tools(root: Path) -> int:
    tree = _parse(root / "mcp.py")
    if tree is None:
        return 0
    for node in tree.body:
        if isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        else:
            continue
        if not (isinstance(target, ast.Name)
                and target.id == "TOOL_DESCRIPTIONS"):
            continue
        if isinstance(value, ast.Dict):
            return len(value.keys)
    return 0


def _subpackages(modules: Mapping[str, Path],
                 packages: Mapping[str, bool]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[str]] = defaultdict(list)
    for key in modules:
        if "." in key:
            group = key.split(".", 1)[0]
        elif packages.get(key, False):
            group = key
        else:
            group = "(top-level)"
        groups[group].append(key)
    rows = []
    for group, keys in groups.items():
        rows.append({
            "name": group,
            "modules": len(keys),
            "loc": sum(_loc(modules[key]) for key in keys),
        })
    return sorted(rows, key=lambda row: (-row["loc"], row["name"]))


def analyze(root: Path, *, package_name: str = "agentdescent") -> Dict[str, Any]:
    """Measure one package; returns a JSON-safe report."""
    root = Path(root)
    modules, _packages, graph = _graph(root, package_name)
    roots = [key for key in _ROOT_MODULES if key in graph]
    cli_reach = _reach(graph, roots)
    init_reach = _reach(graph, ["__init__"]) if "__init__" in graph else set()

    inbound: Dict[str, List[str]] = defaultdict(list)
    for source, targets in graph.items():
        for target in targets:
            if target in modules:
                inbound[target].append(source)

    def category(key: str) -> str:
        if key in cli_reach:
            return "cli/mcp"
        if key in init_reach:
            return "exported-only"
        return "isolated"

    rows = []
    for key, path in modules.items():
        rows.append({
            "module": key,
            "loc": _loc(path),
            "category": category(key),
            "inbound": sorted(inbound.get(key, ())),
        })
    rows.sort(key=lambda row: row["module"])
    categories: Dict[str, Dict[str, int]] = {
        name: {"modules": 0, "loc": 0}
        for name in ("cli/mcp", "exported-only", "isolated")
    }
    for row in rows:
        bucket = categories[row["category"]]
        bucket["modules"] += 1
        bucket["loc"] += row["loc"]

    isolated = [row for row in rows if row["category"] == "isolated"]
    exported_only = [row for row in rows if row["category"] == "exported-only"]
    isolated.sort(key=lambda row: (-row["loc"], row["module"]))
    exported_only.sort(key=lambda row: (-row["loc"], row["module"]))

    return {
        "schema_version": SCHEMA_VERSION,
        "package": package_name,
        "root": str(root.resolve()),
        "modules": len(rows),
        "loc": sum(row["loc"] for row in rows),
        "top_level_exports": _top_level_exports(root),
        "cli_commands": _cli_commands(root),
        "mcp_tools": _mcp_tools(root),
        "categories": categories,
        "isolated": isolated,
        "exported_only": exported_only,
        "subpackages": _subpackages(modules, _packages),
    }


def _print_human(report: Mapping[str, Any], *, limit: int) -> None:
    print(f"package: {report['package']}  root: {report['root']}")
    print(f"modules: {report['modules']}  loc: {report['loc']:,}  "
          f"top-level exports: {report['top_level_exports']}  "
          f"CLI commands: {report['cli_commands']}  "
          f"MCP tools: {report['mcp_tools']}")
    print("\ncategories:")
    for name in ("cli/mcp", "exported-only", "isolated"):
        row = report["categories"][name]
        print(f"  {name:<14} {row['modules']:>3} modules  {row['loc']:>7,} loc")
    print(f"\nisolated (top {limit}, largest first):")
    for row in report["isolated"][:limit]:
        inbound = ",".join(row["inbound"][:4]) or "-"
        print(f"  {row['module']:<28} {row['loc']:>5}  inbound={inbound}")
    print(f"\nexported-only (top {limit}, largest first):")
    for row in report["exported_only"][:limit]:
        inbound = ",".join(row["inbound"][:4]) or "-"
        print(f"  {row['module']:<28} {row['loc']:>5}  inbound={inbound}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.module_audit",
        description="Read-only reachability audit of the AgentDescent package.")
    parser.add_argument(
        "--root", default=None,
        help="package directory (default: <repo>/agentdescent)")
    parser.add_argument("--package", default="agentdescent",
                        help="import package name (default: agentdescent)")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--limit", type=int, default=30,
                        help="rows per list in the human output (default 30)")
    parser.add_argument("--fail-on-isolated", action="store_true",
                        help="exit 1 when any isolated module exists (opt-in)")
    args = parser.parse_args(argv)
    root = Path(args.root) if args.root else (
        Path(__file__).resolve().parents[1] / "agentdescent")
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 2
    report = analyze(root, package_name=args.package)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        _print_human(report, limit=args.limit)
    if args.fail_on_isolated and report["isolated"]:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
