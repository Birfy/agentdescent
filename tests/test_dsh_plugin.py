"""Evolving a dsh plugin -- the artifact that can rewrite the rules.

Two things are worth testing here and neither is the happy path: that the
governance surface stays frozen no matter what the caller passes, and that a
plugin which cannot be meaningfully evolved is refused with the reason rather
than half-evolved.
"""

from __future__ import annotations

import json
import os

import pytest

from agentdescent.dsh import find_plugin, list_plugins, profile_dir
from agentdescent.dsh.plugin import (
    FROZEN_PLUGIN_PATHS,
    NotEvolvable,
    evolve_dsh_plugin,
    resolve_evolvable_plugin,
)


def make_profile(home, name="web", dependencies=None, bundles=("@deepseek-ai/dsh-base",)):
    path = os.path.join(str(home), "profiles", name)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "package.json"), "w", encoding="utf-8") as fh:
        json.dump({"name": f"dsh-profile-{name}", "private": True,
                   "dependencies": dependencies or {},
                   "dsh": {"profile": {"bundles": list(bundles)}}}, fh)
    return path


def make_package(root, name, *, bundle=True):
    path = os.path.join(str(root), name)
    os.makedirs(path, exist_ok=True)
    manifest = {"name": name, "version": "0.1.0"}
    if bundle:
        manifest["dsh"] = {"bundle": {"patch": "./cordis.patch.yml"}}
    with open(os.path.join(path, "package.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)
    return path


def install(profile, name, target, *, link=True):
    """Put `target` where Node would resolve `name` from `profile`."""
    modules = os.path.join(profile, "node_modules")
    os.makedirs(modules, exist_ok=True)
    entry = os.path.join(modules, name)
    if link:
        os.symlink(target, entry)
    else:
        os.makedirs(entry, exist_ok=True)
        with open(os.path.join(target, "package.json"), encoding="utf-8") as src, \
                open(os.path.join(entry, "package.json"), "w", encoding="utf-8") as dst:
            dst.write(src.read())
    return entry


# ---------------------------------------------------------------------------
# Finding plugins
# ---------------------------------------------------------------------------


def test_the_profile_directory_follows_the_harness_home(tmp_path):
    assert profile_dir("demo", environ={"DSH_HOME": str(tmp_path)}) == \
        os.path.join(str(tmp_path), "profiles", "demo")


def test_plugins_come_from_the_manifest_not_from_scanning_node_modules(tmp_path):
    """A transitive package a plugin pulls in is not a thing to evolve."""
    checkout = make_package(tmp_path / "src", "dsh-plugin-mine")
    profile = make_profile(tmp_path, dependencies={"dsh-plugin-mine": f"link:{checkout}"})
    install(profile, "dsh-plugin-mine", checkout)
    make_package(os.path.join(profile, "node_modules"), "some-transitive-dep")

    found = list_plugins("web", environ={"DSH_HOME": str(tmp_path)})
    assert [p.name for p in found] == ["dsh-plugin-mine"]
    assert found[0].linked and found[0].is_bundle


def test_a_linked_plugin_resolves_through_the_symlink_to_its_checkout(tmp_path):
    checkout = make_package(tmp_path / "src", "dsh-plugin-mine")
    profile = make_profile(tmp_path, dependencies={"dsh-plugin-mine": f"link:{checkout}"})
    install(profile, "dsh-plugin-mine", checkout)
    plugin = find_plugin("dsh-plugin-mine", "web", environ={"DSH_HOME": str(tmp_path)})
    assert plugin is not None
    assert plugin.path == os.path.realpath(checkout)


def test_a_library_dependency_is_not_reported_as_a_bundle(tmp_path):
    """A package without `dsh.bundle` installs but activates no layer."""
    checkout = make_package(tmp_path / "src", "plain-lib", bundle=False)
    profile = make_profile(tmp_path, dependencies={"plain-lib": f"link:{checkout}"})
    install(profile, "plain-lib", checkout)
    plugin = find_plugin("plain-lib", "web", environ={"DSH_HOME": str(tmp_path)})
    assert plugin is not None and not plugin.is_bundle


def test_a_profile_with_no_plugins_is_empty_not_an_error(tmp_path):
    make_profile(tmp_path)
    assert list_plugins("web", environ={"DSH_HOME": str(tmp_path)}) == []


def test_an_absent_profile_is_empty_rather_than_a_crash(tmp_path):
    assert list_plugins("nope", environ={"DSH_HOME": str(tmp_path)}) == []


# ---------------------------------------------------------------------------
# What may be evolved
# ---------------------------------------------------------------------------


def test_a_registry_installed_plugin_is_refused_with_the_fix(tmp_path):
    """Built output from a tarball has no source behind it, and the next install
    erases any edit -- silently, at a time unrelated to the run."""
    checkout = make_package(tmp_path / "src", "dsh-plugin-published")
    profile = make_profile(tmp_path, dependencies={"dsh-plugin-published": "^1.0.0"})
    install(profile, "dsh-plugin-published", checkout, link=False)
    with pytest.raises(NotEvolvable, match="built output"):
        resolve_evolvable_plugin("dsh-plugin-published", "web",
                                 environ={"DSH_HOME": str(tmp_path)})


def test_an_unknown_plugin_names_the_command_that_installs_one(tmp_path):
    make_profile(tmp_path)
    with pytest.raises(NotEvolvable, match="dsh plugin"):
        resolve_evolvable_plugin("absent", "web", environ={"DSH_HOME": str(tmp_path)})


def test_a_linked_checkout_resolves(tmp_path):
    checkout = make_package(tmp_path / "src", "dsh-plugin-mine")
    profile = make_profile(tmp_path, dependencies={"dsh-plugin-mine": f"link:{checkout}"})
    install(profile, "dsh-plugin-mine", checkout)
    plugin = resolve_evolvable_plugin("dsh-plugin-mine", "web",
                                      environ={"DSH_HOME": str(tmp_path)})
    assert plugin.linked


# ---------------------------------------------------------------------------
# The frozen set -- the part with no visible symptom when it is wrong
# ---------------------------------------------------------------------------


def test_the_governance_surface_is_frozen_by_default():
    """Each of these can either rewrite the rules or move the goalposts."""
    assert "cordis.patch.yml" in FROZEN_PLUGIN_PATHS      # decides which rows mount
    assert "package.json" in FROZEN_PLUGIN_PATHS          # install-time scripts
    assert any(p.startswith("test") for p in FROZEN_PLUGIN_PATHS)


def test_a_callers_frozen_list_adds_to_the_default_it_cannot_replace_it(tmp_path, monkeypatch):
    """The one mistake here has no symptom: the run still passes and still
    commits, having quietly been allowed to edit the rows that decide whether
    anything reviews it."""
    checkout = make_package(tmp_path / "src", "dsh-plugin-mine")
    profile = make_profile(tmp_path, dependencies={"dsh-plugin-mine": f"link:{checkout}"})
    install(profile, "dsh-plugin-mine", checkout)

    seen = {}

    def fake_evolve_agent_code(path, data, **kwargs):
        seen.update(kwargs)
        seen["path"] = path
        raise SystemExit(0)                # stop before doing any real work

    monkeypatch.setattr("agentdescent.dsh.plugin.evolve_agent_code", fake_evolve_agent_code)
    with pytest.raises(SystemExit):
        evolve_dsh_plugin("dsh-plugin-mine", [{"prompt": "q", "gold": "a"}],
                          entrypoint=["echo"], reflect_with=lambda p: "",
                          frozen=("src/secret.ts",),
                          environ={"DSH_HOME": str(tmp_path)})

    assert "src/secret.ts" in seen["frozen"]              # the caller's addition
    assert "cordis.patch.yml" in seen["frozen"]           # and the defaults survive
    assert "package.json" in seen["frozen"]


def test_a_plugin_is_L1_so_every_merge_goes_through_the_oracle(tmp_path, monkeypatch):
    checkout = make_package(tmp_path / "src", "dsh-plugin-mine")
    profile = make_profile(tmp_path, dependencies={"dsh-plugin-mine": f"link:{checkout}"})
    install(profile, "dsh-plugin-mine", checkout)
    seen = {}

    def fake_evolve_agent_code(path, data, **kwargs):
        seen.update(kwargs)
        raise SystemExit(0)

    monkeypatch.setattr("agentdescent.dsh.plugin.evolve_agent_code", fake_evolve_agent_code)
    with pytest.raises(SystemExit):
        evolve_dsh_plugin("dsh-plugin-mine", [{"prompt": "q", "gold": "a"}],
                          entrypoint=["echo"], reflect_with=lambda p: "",
                          environ={"DSH_HOME": str(tmp_path)})
    assert seen["blast_radius"] >= 0.5


def test_setup_installs_with_scripts_off(tmp_path, monkeypatch):
    """A candidate that could add a postinstall would otherwise run code on this
    machine before a single test did -- and the gate is what must run first."""
    checkout = make_package(tmp_path / "src", "dsh-plugin-mine")
    profile = make_profile(tmp_path, dependencies={"dsh-plugin-mine": f"link:{checkout}"})
    install(profile, "dsh-plugin-mine", checkout)
    seen = {}

    def fake_evolve_agent_code(path, data, **kwargs):
        seen.update(kwargs)
        raise SystemExit(0)

    monkeypatch.setattr("agentdescent.dsh.plugin.evolve_agent_code", fake_evolve_agent_code)
    with pytest.raises(SystemExit):
        evolve_dsh_plugin("dsh-plugin-mine", [{"prompt": "q", "gold": "a"}],
                          entrypoint=["echo"], reflect_with=lambda p: "",
                          environ={"DSH_HOME": str(tmp_path)})
    assert "--ignore-scripts" in seen["setup_cmd"]
    assert seen["test_cmd"] == ("npm", "test")


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


def test_the_runner_lists_which_plugins_can_be_evolved(tmp_path, capsys, monkeypatch):
    from examples.dsh.evolve_dsh_plugin import main

    linked = make_package(tmp_path / "src", "dsh-plugin-linked")
    published = make_package(tmp_path / "src", "dsh-plugin-published")
    profile = make_profile(tmp_path, dependencies={
        "dsh-plugin-linked": f"link:{linked}", "dsh-plugin-published": "^1.0.0"})
    install(profile, "dsh-plugin-linked", linked)
    install(profile, "dsh-plugin-published", published, link=False)
    monkeypatch.setenv("DSH_HOME", str(tmp_path))

    assert main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "dsh-plugin-linked" in out and "linked" in out
    assert "not evolvable" in out


def test_the_runner_refuses_a_registry_plugin_before_loading_anything(tmp_path, capsys,
                                                                     monkeypatch):
    from examples.dsh.evolve_dsh_plugin import main

    published = make_package(tmp_path / "src", "dsh-plugin-published")
    profile = make_profile(tmp_path, dependencies={"dsh-plugin-published": "^1.0.0"})
    install(profile, "dsh-plugin-published", published, link=False)
    monkeypatch.setenv("DSH_HOME", str(tmp_path))

    code = main(["--plugin", "dsh-plugin-published", "--tasks", "/nonexistent.jsonl"])
    assert code == 1
    assert "built output" in capsys.readouterr().err


def test_the_dry_run_shows_the_frozen_set_and_calls_nothing(tmp_path, capsys, monkeypatch):
    from examples.dsh.evolve_dsh_plugin import main

    checkout = make_package(tmp_path / "src", "dsh-plugin-mine")
    profile = make_profile(tmp_path, dependencies={"dsh-plugin-mine": f"link:{checkout}"})
    install(profile, "dsh-plugin-mine", checkout)
    tasks = os.path.join(str(tmp_path), "t.jsonl")
    with open(tasks, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"prompt": "q", "gold": "a"}) + "\n")
    monkeypatch.setenv("DSH_HOME", str(tmp_path))

    assert main(["--plugin", "dsh-plugin-mine", "--tasks", tasks, "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "cordis.patch.yml" in out and "package.json" in out
    assert "oracle" in out
    assert "nothing was written" in out
