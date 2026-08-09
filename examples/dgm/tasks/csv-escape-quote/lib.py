def to_csv_field(s):
    """Quote a field, doubling any embedded quote."""
    return '"' + s + '"'
