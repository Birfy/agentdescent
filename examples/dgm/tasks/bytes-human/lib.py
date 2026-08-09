UNITS = ["B", "KB", "MB", "GB"]

def human(n):
    """Format a byte count, e.g. 1024 -> \"1.0KB\"."""
    i = 0
    while n > 1024 and i < len(UNITS) - 1:
        n /= 1024
        i += 1
    return f"{n:.1f}{UNITS[i]}"
