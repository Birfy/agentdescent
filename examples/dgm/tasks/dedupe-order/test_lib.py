from lib import dedupe

def test_single():
    assert dedupe(["beta"]) == ["beta"]

def test_keeps_order():
    assert dedupe(["alpha", "beta", "gamma", "beta", "delta"]) == [
        "alpha", "beta", "gamma", "delta"]
