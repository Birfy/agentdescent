"""Flags this runner accepts, and whether it reads them.

`--reflective-merge` sat in the shared parser for all eleven MethodPolicy ports
and was never read: `run_port` used `policy.reflective`, a per-method constant.
A run could pass the flag, print `reflective_merge=True` because the *policy*
said so, and be byte-identical to a run that did not pass it.

That is not only a dead flag. A control experiment in this series varied exactly
that flag, found the two arms identical, and concluded the merge layer was not
the cause of a failure -- a refutation built on a no-op.
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from examples import _method_runner as mr


def _args(**kw):
    base = dict(reflective_merge=False, no_reflective_merge=False)
    base.update(kw)
    return SimpleNamespace(**base)


def _policy(reflective: bool):
    from examples.promptbreeder import promptbreeder_genetic_prompts as pb
    from dataclasses import replace
    return replace(pb.build(0), reflective=reflective)


def test_no_flag_keeps_the_methods_own_declaration():
    """`reflective` is a fidelity statement, not a knob: AFlow's contested
    workflow fields are model-merged because its optimizer rewrites a whole
    workflow, and Voyager's are not because its library overwrites a key."""
    assert mr._reflective_override(_args(), _policy(True)) is None
    assert mr._reflective_override(_args(), _policy(False)) is None


def test_the_flag_turns_it_on_and_the_negation_turns_it_off():
    assert mr._reflective_override(_args(reflective_merge=True), _policy(False)) is True
    assert mr._reflective_override(_args(no_reflective_merge=True), _policy(True)) is False


def test_both_flags_at_once_is_refused_rather_than_resolved():
    """A run that asked for the merge and against it has not said what it wants,
    and picking one is how a control arm ends up measuring the other."""
    with pytest.raises(SystemExit, match="contradictory"):
        mr._reflective_override(
            _args(reflective_merge=True, no_reflective_merge=True), _policy(True))


def test_run_port_reads_the_override_rather_than_the_policy_field():
    source = inspect.getsource(mr.run_port)
    assert "policy.reflective if reflective is None else" in source, (
        "the override is not threaded, so the flag is decorative again")
    assert "if policy.reflective:" not in source, (
        "the policy field is still deciding on its own")


def test_the_recorded_configuration_reports_what_actually_ran():
    """`reflective_merge` in the payload used to echo `policy.reflective`, so a
    run's own record could not distinguish an override from a declaration."""
    source = inspect.getsource(mr.run_port)
    assert '"reflective_merge": use_reflective' in source


def test_the_banner_says_when_it_was_overridden():
    source = inspect.getsource(mr.standard_main)
    assert "(overridden)" in source
