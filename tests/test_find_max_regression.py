"""Regression test for the `find_max` helper scenario.

This captures the expected behaviour that motivated the recent agent prompt
update.  The goal is to guard against fixes that forget to handle empty input.
"""

def find_max(numbers):
    """Reference implementation used by the regression test."""
    if not numbers:
        return None
    max_num = numbers[0]
    for num in numbers:
        if num > max_num:
            max_num = num
    return max_num


def test_find_max_handles_empty_list():
    assert find_max([]) is None


def test_find_max_handles_negative_numbers():
    assert find_max([-5, -2, -9]) == -2


def test_find_max_handles_mixed_numbers():
    assert find_max([0, -1, 3, 2]) == 3
