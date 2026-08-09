from lib import round_half_up

def test_below_half():
    assert round_half_up(1.4) == 1

def test_exact_half():
    assert round_half_up(0.5) == 1
