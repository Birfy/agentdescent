def lookup(mapping, key):
    return mapping.get(key)


def first_or_none(items):
    return items[0] if items else None


def dig(mapping, path):
    current = mapping
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current
