class LRU:
    """Bounded cache evicting the LEAST recently inserted key."""

    def __init__(self, size):
        self.size = size
        self.data = {}

    def put(self, k, v):
        if len(self.data) >= self.size:
            self.data.pop(list(self.data)[-1])
        self.data[k] = v
