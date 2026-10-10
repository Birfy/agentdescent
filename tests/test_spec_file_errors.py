"""User-edited input files fail at the boundary, not with a CLI traceback."""
import json
import subprocess
import sys

import pytest

from agentdescent import EvolveSpec, SpecError, load_spec
from agentdescent.evolvespec import load_rows


@pytest.mark.parametrize("payload", [None, 7, "text", [], ["kind"]])
def test_spec_requires_object(payload):
    with pytest.raises(SpecError, match="spec must be a JSON object"):
        EvolveSpec.from_dict(payload)


def test_spec_json_location(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text('{\n "kind": "text",\n}\n')
    with pytest.raises(SpecError, match="line 3, column 1") as exc:
        load_spec(str(path))
    assert str(path) in str(exc.value)


@pytest.mark.parametrize("kind", ["missing", "directory", "encoding"])
def test_spec_read_errors(tmp_path, kind):
    path = tmp_path / "spec.json"
    if kind == "directory":
        path.mkdir()
    elif kind == "encoding":
        path.write_bytes(b"\xff")
    with pytest.raises(SpecError, match="cannot read UTF-8 file") as exc:
        load_spec(str(path))
    assert str(path) in str(exc.value)


@pytest.mark.parametrize("ext,contents,line", [
    ("json", '[\n {"prompt": "ok"},\n]', 3),
    ("jsonl", '\n{"prompt": "ok"}\n\n{"prompt": }\n', 4),
])
def test_data_json_location(tmp_path, ext, contents, line):
    path = tmp_path / ("cases." + ext)
    path.write_text(contents)
    with pytest.raises(SpecError, match=f"line {line}, column") as exc:
        load_rows({"path": str(path)})
    assert str(path) in str(exc.value)


@pytest.mark.parametrize("ext", ["json", "jsonl", "csv", "tsv"])
@pytest.mark.parametrize("kind", ["directory", "encoding"])
def test_data_read_errors(tmp_path, ext, kind):
    path = tmp_path / ("cases." + ext)
    if kind == "directory":
        path.mkdir()
    else:
        path.write_bytes(b"\xff")
    with pytest.raises(SpecError, match="cannot read UTF-8 file"):
        load_rows({"path": str(path)})


@pytest.mark.parametrize("source", ["missing", "spec", "data"])
def test_cli_plan_bad_file_has_no_traceback(tmp_path, source):
    path = tmp_path / "spec.json"
    if source == "spec":
        path.write_text('{"kind": }')
    elif source == "data":
        data = tmp_path / "cases.jsonl"
        data.write_text('\n{"prompt": }\n')
        path.write_text(json.dumps({"kind": "text", "target": "instruction",
                                    "data": {"path": str(data)},
                                    "agent": "openai_compatible"}))
    result = subprocess.run([sys.executable, "-m", "agentdescent.cli", "plan", str(path)],
                            capture_output=True, text=True)
    assert result.returncode == 2
    output = result.stdout + result.stderr
    assert "Traceback" not in output
    assert str(path if source != "data" else data) in output
    if source == "data":
        assert json.loads(result.stdout)["ok"] is False
        assert "line 2, column" in output
    else:
        assert result.stderr.startswith("error: ")


@pytest.mark.parametrize("ext", ["json", "jsonl", "csv", "tsv"])
def test_valid_data_still_loads(tmp_path, ext):
    rows = [{"prompt": "hello", "gold": "world"}]
    contents = {"json": json.dumps(rows), "jsonl": '\n' + json.dumps(rows[0]) + '\n\n',
                "csv": 'prompt,gold\nhello,world\n',
                "tsv": 'prompt\tgold\nhello\tworld\n'}
    path = tmp_path / ("cases." + ext)
    path.write_text(contents[ext])
    assert load_rows({"path": str(path)}) == rows


def test_valid_spec_still_loads(tmp_path):
    payload = {"kind": "text", "target": "instruction", "data": {"inline": []}}
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(payload))
    spec = load_spec(str(path), absolutise=False)
    assert spec.kind == "text"
    assert spec.target == "instruction"
    assert spec.data == payload["data"]
