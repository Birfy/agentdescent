from lib import Queue

def test_single():
    q = Queue(); q.push(1)
    assert q.pop() == 1

def test_order():
    q = Queue(); q.push(1); q.push(2)
    assert q.pop() == 1
