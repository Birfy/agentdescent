def expand(spec):
    """\"1-4\" -> [1,2,3,4]; a bare number -> [n]."""
    if "-" not in spec:
        return [int(spec)]
    lo, hi = spec.split("-")
    return list(range(int(lo), int(hi)))
