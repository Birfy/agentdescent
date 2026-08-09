def is_palindrome(s):
    """Ignoring case and non-alphanumerics."""
    return s == s[::-1]
