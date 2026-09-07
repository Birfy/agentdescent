"""The read-only HTTP view of the run store, and what it refuses to serve."""

import json
import threading
import urllib.error
import urllib.request

import pytest

from agentdescent import runstore

from tests.test_evolvespec import _dir_spec  # noqa: F401


@pytest.fixture
def served(tmp_path, monkeypatch):
    """A live server on an ephemeral port, over a store with one finished run."""
    monkeypatch.setenv("AGENTDESCENT_HOME", str(tmp_path / "home"))
    store = str(tmp_path / "runs")
    rd = runstore.create(_dir_spec(str(tmp_path)).to_dict(), store=store)
    runstore.execute(rd)
    httpd = runstore.serve_http(host="127.0.0.1", port=0, store=store, serve_forever=False)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield base, rd
    httpd.shutdown()
    httpd.server_close()


def _get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.status, r.headers, r.read()


def test_the_panel_page_is_self_contained(served):
    base, _ = served
    status, headers, body = _get(base + "/")
    assert status == 200 and headers["Content-Type"].startswith("text/html")
    text = body.decode()
    # no external anything: a host embeds this, often with no network
    assert "http://" not in text.replace("http://127.0.0.1", "") or "src=" not in text
    assert "<script" in text and "api/runs" in text
    assert headers["Cache-Control"] == "no-store"


def test_runs_endpoint_lists_the_store(served):
    base, rd = served
    status, headers, body = _get(base + "/api/runs")
    assert status == 200 and headers["Content-Type"] == "application/json"
    runs = json.loads(body)
    assert [r["run_id"] for r in runs] == [rd.run_id]
    assert runs[0]["state"] == "done" and runs[0]["best_reward"] == 1.0


def test_one_run_carries_its_rounds(served):
    base, rd = served
    _, _, body = _get(f"{base}/api/runs/{rd.run_id}")
    payload = json.loads(body)
    assert payload["run_id"] == rd.run_id
    assert payload["rounds"] and payload["rounds"][0]["held_out_reward"] >= 0


def test_an_unknown_run_is_404_not_a_traceback(served):
    base, _ = served
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(base + "/api/runs/nope")
    assert e.value.code == 404 and b"error" in e.value.read()


@pytest.mark.parametrize("suffix", ["/api/runs/..", "/api/runs/%2e%2e", "/etc/passwd", "/api/other"])
def test_it_serves_nothing_but_the_two_endpoints(served, suffix):
    base, _ = served
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(base + suffix)
    assert e.value.code in (400, 404)


def _get_with_origin(url, origin):
    req = urllib.request.Request(url, headers={"Origin": origin})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.headers


def test_cors_is_echoed_only_for_loopback_origins(served):
    """A host panel on another loopback port must read it; a website must not."""
    base, _ = served
    for good in ("http://127.0.0.1:3080", "http://localhost:3080", "http://[::1]:9"):
        headers = _get_with_origin(base + "/api/runs", good)
        assert headers["Access-Control-Allow-Origin"] == good, good
        assert headers["Vary"] == "Origin"
    for bad in ("https://evil.com", "http://127.0.0.1.evil.com", "http://evil.com:3080",
                "null", "file://"):
        headers = _get_with_origin(base + "/api/runs", bad)
        assert "Access-Control-Allow-Origin" not in headers, bad


def test_no_cors_header_without_an_origin(served):
    base, _ = served
    _, headers, _ = _get(base + "/api/runs")
    assert "Access-Control-Allow-Origin" not in headers


def test_the_cli_exposes_serve():
    from agentdescent import cli

    parser = cli.build_parser()
    choices = set(next(a for a in parser._actions if a.dest == "cmd").choices)
    assert "serve" in choices
    args = parser.parse_args(["serve", "--port", "9999"])
    assert args.port == 9999 and args.host == runstore.DEFAULT_HTTP_HOST
