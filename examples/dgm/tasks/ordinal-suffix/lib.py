def ordinal(n):
    """1 -> 1st, 2 -> 2nd, 11 -> 11th."""
    last = n % 10
    return f"{n}" + {1: "st", 2: "nd", 3: "rd"}.get(last, "th")
