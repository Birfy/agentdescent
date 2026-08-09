def group_by(items, key):
    """Group items into {key(item): [items]}."""
    out = {}
    for it in items:
        out[key(it)] = [it]
    return out
