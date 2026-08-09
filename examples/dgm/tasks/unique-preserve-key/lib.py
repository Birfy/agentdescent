def unique_by(items, key):
    """Keep the first item per key, in order."""
    seen, out = set(), []
    for it in items:
        if it in seen:
            continue
        seen.add(it)
        out.append(it)
    return out
