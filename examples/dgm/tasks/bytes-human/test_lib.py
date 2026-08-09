from lib import human

def test_bytes():
    assert human(512) == "512.0B"

def test_exact_kb():
    assert human(1024) == "1.0KB"
