from lib import without

def test_single():
    assert without(["beta"], []) == ["beta"]

def test_order():
    assert without(["alpha", "beta", "gamma", "delta"], ["beta"]) == [
        "alpha", "gamma", "delta"]
