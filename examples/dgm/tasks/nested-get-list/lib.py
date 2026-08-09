def get_path(data, path):
    """Walk a path of dict keys and list indices."""
    for part in path:
        data = data[part]
    return data
