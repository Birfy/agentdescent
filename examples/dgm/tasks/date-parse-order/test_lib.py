from lib import parse_date

def test_symmetric():
    assert parse_date("2024-07-07") == (2024, 7, 7)

def test_month_day():
    assert parse_date("2024-03-15") == (2024, 3, 15)
