from lib import LRU

def test_under_capacity():
    c = LRU(2); c.put("a", 1); c.put("b", 2)
    assert set(c.data) == {"a", "b"}

def test_evicts_oldest():
    c = LRU(2); c.put("a", 1); c.put("b", 2); c.put("c", 3)
    assert set(c.data) == {"b", "c"}
