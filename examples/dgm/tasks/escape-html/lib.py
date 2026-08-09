def escape(s):
    """Escape &, < and > for HTML."""
    return s.replace("<", "&lt;").replace(">", "&gt;").replace("&", "&amp;")
