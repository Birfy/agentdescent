from lib import call_with_retry

def test_succeeds_first():
    assert call_with_retry(lambda: 1) == 1

def test_uses_all_attempts():
    calls = {"n": 0}
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("boom")
        return "ok"
    assert call_with_retry(flaky, attempts=3) == "ok"
