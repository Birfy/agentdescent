def total(d):
    """Sum every numeric leaf in a nested dict."""
    return sum(v for v in d.values() if isinstance(v, (int, float)))
