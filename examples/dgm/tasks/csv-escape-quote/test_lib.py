from lib import to_csv_field

def test_plain():
    assert to_csv_field("ab") == '"ab"'

def test_embedded_quote():
    assert to_csv_field('a"b') == '"a""b"'
