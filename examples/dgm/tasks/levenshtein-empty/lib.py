def distance(a, b):
    """Levenshtein edit distance."""
    prev = list(range(1, len(b) + 1))
    for i, ca in enumerate(a):
        cur = [i + 1]
        for j, cb in enumerate(b):
            cur.append(min(prev[j] + (ca != cb), cur[j] + 1, prev[j] + 1))
        prev = cur[1:]
    return prev[-1]
