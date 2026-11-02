"""Every platform cycle is a reserved, authorized and audited tenant action."""

import json

import pytest

from agentdescent.compute_policy import (
    ComputeAction, ComputeController, ComputeOption,
)
from agentdescent.durable_runs import SqliteRunStore
from agentdescent.loop_kernel import LoopObservation
from agentdescent.model_router import ModelRouter, ModelSpec
from agentdescent.platform import PlatformConfig, PlatformReport, RSIPlatform
from agentdescent.tenant_platform import TenantCycleReport, TenantPlatform
from agentdescent.workspaces import (
    PermissionDenied,
    Principal,
    Quota,
    Role,
    Workspace,
    WorkspaceRegistry,
)

_TENANT = "tenant-a"
_WORKSPACE = "ws-1"


def _registry(*, quota=None):
    reg = WorkspaceRegistry(clock=lambda: 100.0)
    reg.add_workspace(Workspace(
        _TENANT, _WORKSPACE, "Research",
        quota or Quota(max_tokens=10000, max_cost=100, max_concurrent=2)))
    people = {
        "operator": Principal(
            "operator", _TENANT, {_WORKSPACE: (Role.OPERATOR,)}),
        "viewer": Principal(
            "viewer", _TENANT, {_WORKSPACE: (Role.VIEWER,)}),
        "owner": Principal("owner", _TENANT, {"*": (Role.OWNER,)}),
        "outsider": Principal("outsider", "tenant-b", {"*": (Role.OWNER,)}),
    }
    for principal in people.values():
        reg.add_principal(principal)
    return reg, people


def _platform(tmp_path, **overrides):
    router = ModelRouter([ModelSpec("m", cost=0.5)],
                         cost_weight=0.0, exploration=0.0)
    compute = ComputeController([
        ComputeOption("mcts", ComputeAction.MCTS, calls=1,
                      quality_prior=(5, 2)),
    ], router, exploration=0.0)
    values = dict(max_generations=1, target=0.85)
    values.update(overrides)
    return RSIPlatform(
        SqliteRunStore(str(tmp_path / "runs.sqlite3")),
        PlatformConfig(_TENANT, _WORKSPACE, **values), compute=compute)


def _step(context, decision):
    return LoopObservation(
        0.5, cost=decision.expected_cost, artifact_id="a0",
        evidence={"action": decision.action,
                  "rewards": {"mcts": 0.5, "direct": 0.4}})


def _actions(reg):
    return [event.action for event in
            reg.audit_events("owner", tenant_id=_TENANT,
                             workspace_id=_WORKSPACE)]


def test_happy_path_reserves_commits_and_audits(tmp_path):
    platform = _platform(tmp_path)
    reg, people = _registry()
    tenant = TenantPlatform(platform, reg, people["operator"])
    run_id = platform.submit()

    result = tenant.run_cycle(
        run_id, "worker-1", tokens=100, cost=1.5, step=_step)

    assert isinstance(result, TenantCycleReport)
    assert isinstance(result.report, PlatformReport)
    assert result.report.complete
    assert result.report.loop.run_id == run_id
    assert result.reservation.active is False
    assert result.reservation.committed is True
    assert result.usage.active == 0
    assert result.usage.tokens == 100
    assert result.usage.cost == 1.5
    assert result.detail.startswith("reservation=")

    data = result.to_dict()
    assert json.loads(json.dumps(data))["usage"] == {
        "tokens": 100, "cost": 1.5, "active": 0}
    assert data["reservation_id"] == result.reservation.reservation_id
    assert data["reservation"]["committed"] is True
    assert data["loop"]["run_id"] == run_id
    assert data["promotion"] is None
    assert data["foundry"] is None
    assert data["experience"] is None

    events = reg.audit_events(
        "owner", tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert any(event.action == "quota.reserve" and event.allowed
               for event in events)
    assert any(event.action == "quota.commit" and event.allowed
               for event in events)


def test_viewer_is_denied_before_the_platform_runs(tmp_path):
    platform = _platform(tmp_path)
    reg, people = _registry()
    tenant = TenantPlatform(platform, reg, people["viewer"])

    with pytest.raises(PermissionDenied):
        tenant.run_cycle(
            "viewer-run", "worker-1", tokens=10, cost=0.1, step=_step)

    with pytest.raises(KeyError):
        platform.kernel.snapshot("viewer-run")
    assert reg.usage("owner", _TENANT, _WORKSPACE).active == 0
    assert "quota.commit" not in _actions(reg)


def test_quota_exceeded_denies_without_leaking_a_reservation(tmp_path):
    platform = _platform(tmp_path)
    reg, people = _registry(
        quota=Quota(max_tokens=10, max_cost=1, max_concurrent=1))
    tenant = TenantPlatform(platform, reg, people["operator"])
    before = reg.usage("owner", _TENANT, _WORKSPACE)

    with pytest.raises(PermissionDenied, match="token quota exceeded"):
        tenant.run_cycle(
            "quota-run", "worker-1", tokens=100, cost=0.5, step=_step)

    after = reg.usage("owner", _TENANT, _WORKSPACE)
    assert after.active == before.active == 0
    assert after.tokens == before.tokens == 0
    assert after.cost == before.cost == 0
    with pytest.raises(KeyError):
        platform.kernel.snapshot("quota-run")


def test_platform_failure_releases_the_reservation_and_propagates(
        tmp_path, monkeypatch):
    platform = _platform(tmp_path)
    reg, people = _registry()
    tenant = TenantPlatform(platform, reg, people["operator"])
    run_id = platform.submit()

    def boom(*args, **kwargs):
        raise RuntimeError("platform exploded")

    monkeypatch.setattr(platform, "run_cycle", boom)
    with pytest.raises(RuntimeError, match="platform exploded"):
        tenant.run_cycle(
            run_id, "worker-1", tokens=100, cost=1.0, step=_step)

    usage = reg.usage("owner", _TENANT, _WORKSPACE)
    assert usage.active == 0
    assert usage.tokens == 0
    assert usage.cost == 0
    assert "quota.release" in _actions(reg)
    assert "quota.commit" not in _actions(reg)


def test_step_error_releases_the_reservation_and_propagates(tmp_path):
    platform = _platform(tmp_path)
    reg, people = _registry()
    tenant = TenantPlatform(platform, reg, people["operator"])
    run_id = platform.submit()

    def boom(context, decision):
        raise RuntimeError("step exploded")

    with pytest.raises(RuntimeError, match="step exploded"):
        tenant.run_cycle(
            run_id, "worker-1", tokens=100, cost=1.0, step=boom)

    usage = reg.usage("owner", _TENANT, _WORKSPACE)
    assert usage.active == 0
    assert usage.tokens == 0
    assert usage.cost == 0
    assert "quota.release" in _actions(reg)
    assert "quota.commit" not in _actions(reg)


def test_usage_of_supplies_the_committed_actuals(tmp_path):
    platform = _platform(tmp_path)
    reg, people = _registry()
    seen = {}

    def usage_of(report):
        seen["report"] = report
        return (37.0, 0.75)

    tenant = TenantPlatform(
        platform, reg, people["operator"], usage_of=usage_of)
    run_id = platform.submit()

    result = tenant.run_cycle(
        run_id, "worker-1", tokens=100, cost=1.5, step=_step)

    assert seen["report"] is result.report
    assert result.reservation.tokens == 100
    assert result.reservation.cost == 1.5
    assert result.usage.tokens == 37
    assert result.usage.cost == 0.75
    assert result.usage.active == 0
    assert result.reservation.committed is True


def test_usage_over_the_reservation_fails_closed_and_releases(tmp_path):
    platform = _platform(tmp_path)
    reg, people = _registry()
    tenant = TenantPlatform(
        platform, reg, people["operator"],
        usage_of=lambda report: (11.0, 1.0))
    run_id = platform.submit()

    with pytest.raises(PermissionDenied, match="commit exceeds reservation"):
        tenant.run_cycle(
            run_id, "worker-1", tokens=10, cost=1.0, step=_step)

    usage = reg.usage("owner", _TENANT, _WORKSPACE)
    assert usage.active == 0
    assert usage.tokens == 0
    assert usage.cost == 0
    reservation = next(iter(reg._reservations.values()))
    assert reservation.active is False
    assert reservation.committed is False
    assert "quota.release" in _actions(reg)
    assert not any(
        event.action == "quota.commit" and event.allowed
        for event in reg.audit_events(
            "owner", tenant_id=_TENANT, workspace_id=_WORKSPACE))


def test_usage_of_non_finite_actuals_is_refused_and_released(tmp_path):
    platform = _platform(tmp_path)
    reg, people = _registry()
    tenant = TenantPlatform(
        platform, reg, people["operator"],
        usage_of=lambda report: (float("nan"), 1.0))
    run_id = platform.submit()

    with pytest.raises(ValueError, match="used tokens"):
        tenant.run_cycle(
            run_id, "worker-1", tokens=10, cost=1.0, step=_step)

    usage = reg.usage("owner", _TENANT, _WORKSPACE)
    assert usage.active == 0
    assert usage.tokens == 0
    assert usage.cost == 0
    assert "quota.release" in _actions(reg)


def test_invalid_principal_workspace_and_types_are_refused(tmp_path):
    platform = _platform(tmp_path)
    reg, people = _registry()
    operator = people["operator"]

    with pytest.raises(TypeError):
        TenantPlatform(object(), reg, operator)
    with pytest.raises(TypeError):
        TenantPlatform(platform, object(), operator)
    with pytest.raises(TypeError):
        TenantPlatform(platform, reg, object())
    with pytest.raises(TypeError):
        TenantPlatform(platform, reg, operator, usage_of=42)
    with pytest.raises(ValueError):
        TenantPlatform(platform, reg, people["outsider"])
    with pytest.raises(ValueError):
        TenantPlatform(platform, reg, operator, workspace_id="missing")
    with pytest.raises(ValueError):
        TenantPlatform(platform, reg, operator, workspace_id="")


@pytest.mark.parametrize(
    "field,value",
    [
        ("tokens", float("nan")),
        ("tokens", float("inf")),
        ("tokens", -1),
        ("cost", float("nan")),
        ("cost", float("inf")),
        ("cost", -0.5),
    ],
)
def test_non_finite_or_negative_reservation_is_refused(
        tmp_path, field, value):
    platform = _platform(tmp_path)
    reg, people = _registry()
    tenant = TenantPlatform(platform, reg, people["operator"])
    values = {"tokens": 10.0, "cost": 1.0}
    values[field] = value

    with pytest.raises(ValueError):
        tenant.run_cycle(
            "bad-usage", "worker-1", step=_step, **values)

    usage = reg.usage("owner", _TENANT, _WORKSPACE)
    assert usage.active == 0
    assert usage.tokens == 0
    assert usage.cost == 0
    with pytest.raises(KeyError):
        platform.kernel.snapshot("bad-usage")
