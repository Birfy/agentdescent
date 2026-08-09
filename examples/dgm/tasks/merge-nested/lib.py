def deep_merge(a, b):
    """Merge b into a, recursing into nested dicts."""
    out = dict(a)
    out.update(b)
    return out
