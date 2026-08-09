def transpose(m):
    """Transpose a rectangular matrix."""
    n = len(m)
    return [[m[j][i] for j in range(n)] for i in range(n)]
