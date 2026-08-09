def strip_comment(line):
    """Remove a trailing # comment, ignoring # inside quotes."""
    return line.split("#")[0].rstrip()
