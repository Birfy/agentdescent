from .errors import ConfigError

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


def parse_port(text):
    try:
        port = int(text)
    except (TypeError, ValueError):
        raise ConfigError(f"port is not a number: {text!r}")
    if not 1 <= port <= 65535:
        raise ConfigError(f"port out of range: {port}")
    return port


def parse_bool(text):
    value = str(text).strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ConfigError(f"not a boolean: {text!r}")


def parse_ratio(text):
    left, sep, right = str(text).partition(":")
    if not sep:
        raise ConfigError(f"ratio needs a colon: {text!r}")
    try:
        num, den = int(left), int(right)
    except ValueError:
        raise ConfigError(f"ratio parts are not numbers: {text!r}")
    if den == 0:
        raise ConfigError("ratio denominator is zero")
    return num, den
