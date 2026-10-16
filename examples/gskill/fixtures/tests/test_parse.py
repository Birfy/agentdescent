from kvtools.errors import ConfigError
from kvtools.parse import parse_bool, parse_port, parse_ratio

from ._util import raises


def test_parse_port_ok():
    assert parse_port("8080") == 8080


def test_parse_port_not_a_number():
    assert raises(ConfigError, parse_port, "http")


def test_parse_port_zero():
    assert raises(ConfigError, parse_port, "0")


def test_parse_port_too_big():
    assert raises(ConfigError, parse_port, "70000")


def test_parse_bool_true_words():
    assert parse_bool(" Yes ") is True


def test_parse_bool_false_words():
    assert parse_bool("off") is False


def test_parse_bool_unknown():
    assert raises(ConfigError, parse_bool, "maybe")


def test_parse_bool_empty():
    assert raises(ConfigError, parse_bool, "")


def test_parse_ratio_ok():
    assert parse_ratio("3:4") == (3, 4)


def test_parse_ratio_no_colon():
    assert raises(ConfigError, parse_ratio, "34")


def test_parse_ratio_not_numbers():
    assert raises(ConfigError, parse_ratio, "a:b")


def test_parse_ratio_zero_denominator():
    assert raises(ConfigError, parse_ratio, "1:0")
