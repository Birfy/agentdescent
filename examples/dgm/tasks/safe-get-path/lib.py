def get_in(d, path, default=None):
    """Walk a dotted path, returning default if any step is missing."""
    for part in path.split("."):
        d = d[part]
    return d
