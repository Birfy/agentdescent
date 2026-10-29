"""A self-modification must not escape the authority it was granted."""

import pytest

from agentdescent.safe_selfmod import (
    ModificationSpec,
    SafeSelfModifier,
)


def _spec(source="def f(x):\n    return x + 1\n", *, caps=(), risk=0.0, mid="m"):
    return ModificationSpec(mid, source, "skills/f.py", caps, risk)


def test_safe_low_risk_source_is_allowed():
    gate = SafeSelfModifier({"files:write"})
    decision = gate.review(_spec(caps=("files:write",), risk=0.01))
    assert decision.allowed
    assert decision.action == "allow"
    assert decision.capabilities == ("files:write",)


def test_forbidden_import_call_and_dunder_are_denied():
    gate = SafeSelfModifier(set())
    assert not gate.review(_spec("import os\n")).allowed
    assert "import:os" in gate.review(_spec("import os\n")).violations
    assert "call:eval" in gate.review(_spec("eval('1')\n")).violations
    assert "private-attr:__class__" in gate.review(_spec("x.__class__\n")).violations
    assert "dunder:__builtins__" in gate.review(_spec("__builtins__\n")).violations


def test_missing_capability_and_high_risk_are_denied():
    gate = SafeSelfModifier({"files:write"}, max_risk=0.4)
    missing = gate.review(_spec(caps=("network:outbound",)))
    assert not missing.allowed
    assert "capability:network:outbound" in missing.violations
    high = gate.review(_spec(risk=0.9))
    assert not high.allowed
    assert "max_risk" in high.reason


def test_canary_threshold_routes_to_canary():
    gate = SafeSelfModifier(set(), max_risk=0.5, canary_risk=0.2)
    decision = gate.review(_spec(risk=0.3))
    assert decision.allowed and decision.needs_canary
    assert decision.action == "canary"


def test_promote_hold_and_rollback():
    gate = SafeSelfModifier(set(), max_risk=0.5, canary_risk=0.2)
    low = _spec(risk=0.0)
    assert gate.promote(low, before=0.7, after=0.8).action == "promote"
    assert gate.promote(low, before=0.7, after=0.6).action == "rollback"
    high = _spec(risk=0.3)
    held = gate.promote(high, before=0.7, after=0.8)
    assert held.action == "hold" and not held.allowed
    assert gate.promote(high, before=0.7, after=0.8, canary=0.75).action == "promote"
    assert gate.promote(high, before=0.7, after=0.8, canary=0.6).action == "rollback"


def test_denied_modification_never_promotes():
    gate = SafeSelfModifier(set())
    decision = gate.promote(_spec("import os\n"), before=0.1, after=1.0)
    assert decision.action == "deny"
    assert not decision.allowed


def test_rollback_plan_is_only_for_rollback_or_deny():
    gate = SafeSelfModifier(set())
    denied = gate.review(_spec("import os\n"))
    assert len(gate.rollback_plan(denied)) == 2
    assert gate.rollback_plan(gate.review(_spec())) == ()


def test_invalid_specs_and_gate_arguments_are_refused():
    with pytest.raises(ValueError):
        ModificationSpec("", "x", "t")
    with pytest.raises(ValueError):
        ModificationSpec("m", "", "t")
    with pytest.raises(ValueError):
        ModificationSpec("m", "x", "")
    with pytest.raises(ValueError):
        ModificationSpec("m", "x", "t", risk=1.2)
    with pytest.raises(ValueError):
        ModificationSpec("m", "x", "t", risk=True)
    with pytest.raises(ValueError):
        ModificationSpec("m", "x", "t", requested_capabilities="files:write")
    assert ModificationSpec("m", "x", "t", risk="0.3").risk == 0.3
    with pytest.raises(ValueError):
        SafeSelfModifier(set(), max_risk=2)
    with pytest.raises(ValueError):
        SafeSelfModifier(set(), canary_risk=0.5, max_risk=0.2)
    with pytest.raises(ValueError):
        SafeSelfModifier(set(), tolerance=-0.1)
    gate = SafeSelfModifier(set())
    with pytest.raises(TypeError):
        gate.review("not a spec")
    with pytest.raises(ValueError):
        gate.promote(_spec(), before=float("nan"), after=1.0)


def test_aliasing_and_module_renames_do_not_escape_the_gate():
    gate = SafeSelfModifier(set())
    for source in (
        "e = eval\ne('1')\n",
        "import builtins\nf = builtins.exec\nf('1')\n",
        "import io\nw = io.open\nw('/etc/passwd')\n",
        "import posix\nposix.system('id')\n",
        "import runpy\nrunpy.run_path('x.py')\n",
    ):
        decision = gate.review(_spec(source))
        assert not decision.allowed, source
        assert decision.violations


def test_file_and_network_use_require_declared_capabilities():
    gate = SafeSelfModifier({"files:write", "network:outbound"})
    undeclared = gate.review(_spec("import pathlib\npathlib.Path('x').write_text('p')\n"))
    assert not undeclared.allowed
    assert any("capability-required" in item for item in undeclared.violations)
    declared = gate.review(_spec(
        "import pathlib\npathlib.Path('x').write_text('p')\n",
        caps=("files:write",)))
    assert declared.allowed

    network = gate.review(_spec("import urllib.request\nurllib.request.urlopen('http://x')\n"))
    assert not network.allowed
    assert any("capability-required" in item for item in network.violations)


def test_allowed_modules_cannot_re_export_process_spawn_or_dunder_traversal():
    gate = SafeSelfModifier(set())
    probes = [
        "import random\nrandom._os.system('echo PWNED')\n",
        "import typing\ntyping.sys.modules['os'].system('echo PWNED')\n",
        "import dataclasses\ndataclasses.sys.modules['subprocess'].run(['echo','PWNED'])\n",
        "import collections\ncollections._sys.modules['os'].system('echo PWNED')\n",
        "import string\nF = string.Formatter()\n",
    ]
    for source in probes:
        decision = gate.review(_spec(source))
        assert not decision.allowed, source
        assert decision.violations
        assert not any(item.startswith("syntax") for item in decision.violations), (source, decision.violations)
