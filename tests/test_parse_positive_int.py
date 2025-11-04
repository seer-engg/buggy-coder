import pytest

from src.validators import parse_positive_int


def test_parse_positive_int_accepts_positive_integers():
    assert parse_positive_int(5) == 5
    assert parse_positive_int("42") == 42


def test_parse_positive_int_rejects_non_numeric_inputs():
    with pytest.raises(ValueError):
        parse_positive_int("abc")
    with pytest.raises(ValueError):
        parse_positive_int(None)


def test_parse_positive_int_rejects_non_positive_values():
    with pytest.raises(ValueError):
        parse_positive_int(0)
    with pytest.raises(ValueError):
        parse_positive_int(-10)


def test_parse_positive_int_rejects_boolean_values():
    with pytest.raises(ValueError):
        parse_positive_int(True)
    with pytest.raises(ValueError):
        parse_positive_int(False)
