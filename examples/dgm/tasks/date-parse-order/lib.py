def parse_date(s):
    """Parse YYYY-MM-DD into (year, month, day)."""
    parts = s.split("-")
    return (int(parts[0]), int(parts[2]), int(parts[1]))
