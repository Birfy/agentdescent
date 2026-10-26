"""A manifest must identify inputs exactly, and never leak a secret."""

from __future__ import annotations

import json
from pathlib import Path

from agentdescent import (RunManifest, build_manifest, diff_manifest,
                          verify_manifest)
from agentdescent.evolvespec import EvolveSpec


def _spec(tmp_path, *, prompt="q"):
    data = tmp_path / "cases.jsonl"
    data.write_text(json.dumps({"prompt": prompt, "gold": "a"}) + "\n")
    return EvolveSpec(kind="text", target=str(tmp_path / "prompt.txt"),
                      data={"path": str(data)}, agent="echo"), data


def test_identical_inputs_share_a_manifest(tmp_path):
    spec, data = _spec(tmp_path)
    a = build_manifest(spec, data_paths=[str(data)], code_version="1")
    b = build_manifest(spec, data_paths=[str(data)], code_version="1")
    assert a == b and a.manifest_id == b.manifest_id
    assert verify_manifest(spec, a, data_paths=[str(data)], code_version="1") == []


def test_spec_data_and_code_changes_are_attributed(tmp_path):
    spec, data = _spec(tmp_path)
    base = build_manifest(spec, data_paths=[str(data)], code_version="1")
    data.write_text(json.dumps({"prompt": "different", "gold": "a"}) + "\n")
    changed = verify_manifest(spec, base, data_paths=[str(data)],
                              code_version="1")
    assert "data_hash" in changed and "files" in changed
    current = build_manifest(spec, data_paths=[str(data)], code_version="1")
    assert verify_manifest(spec, current, data_paths=[str(data)],
                           code_version="2") == ["code_version"]


def test_secrets_are_redacted_before_hashing_and_storage(tmp_path):
    spec, data = _spec(tmp_path)
    manifest = build_manifest(spec, data_paths=[str(data)],
                              model_config={"model": "m", "api_key": "sk-secret",
                                            "nested": {"token": "abc"}})
    assert manifest.model_config["api_key"] == "<redacted>"
    assert manifest.model_config["nested"]["token"] == "<redacted>"
    assert "sk-secret" not in json.dumps(manifest.to_dict())


def test_max_tokens_is_not_treated_as_a_secret(tmp_path):
    spec, data = _spec(tmp_path)
    a = build_manifest(spec, data_paths=[str(data)],
                       model_config={"model": "m", "max_tokens": 100})
    b = build_manifest(spec, data_paths=[str(data)],
                       model_config={"model": "m", "max_tokens": 4000})
    assert a.model_hash != b.model_hash and a.manifest_id != b.manifest_id


def test_identical_content_in_different_directories_has_the_same_data_hash(tmp_path):
    spec, _ = _spec(tmp_path)
    one, two = tmp_path / "one", tmp_path / "two"
    one.mkdir(); two.mkdir()
    (one / "cases.jsonl").write_text("same")
    (two / "cases.jsonl").write_text("same")
    a = build_manifest(spec, data_paths=[str(one / "cases.jsonl")])
    b = build_manifest(spec, data_paths=[str(two / "cases.jsonl")])
    assert a.data_hash == b.data_hash


def test_a_missing_path_is_explicit(tmp_path):
    spec, _ = _spec(tmp_path)
    manifest = build_manifest(spec, data_paths=[str(tmp_path / "missing")])
    assert manifest.missing


def test_secret_looking_values_are_scrubbed(tmp_path):
    spec, data = _spec(tmp_path)
    manifest = build_manifest(
        spec, data_paths=[str(data)],
        model_config={"headers": ["Authorization: Bearer sk-live-123456"],
                      "url": "https://x?api_key=sk-live"})
    blob = json.dumps(manifest.to_dict())
    assert "sk-live" not in blob and "Bearer <redacted>" in blob


def test_directory_inputs_are_hashed_by_content(tmp_path):
    spec, _ = _spec(tmp_path)
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "a.md").write_text("one")
    one = build_manifest(spec, data_paths=[str(tree)])
    (tree / "a.md").write_text("two")
    two = build_manifest(spec, data_paths=[str(tree)])
    assert one.data_hash != two.data_hash


def test_manifest_roundtrip_and_diff(tmp_path):
    spec, data = _spec(tmp_path)
    a = build_manifest(spec, data_paths=[str(data)], code_version="1")
    assert RunManifest.from_dict(a.to_dict()) == a
    b = build_manifest(spec, data_paths=[str(data)], code_version="2")
    assert "code_version" in diff_manifest(a, b)
