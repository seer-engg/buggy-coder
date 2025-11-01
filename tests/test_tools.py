import pytest

from src.graph import (
    add_import_buggy,
    rename_first_occurrence,
    bump_indices_off_by_one,
    stub_function_singleline,
)


def test_add_import_inserts_at_top_without_duplication():
    snippet = "print('hello')\n"
    updated = add_import_buggy.invoke({"snippet": snippet, "module": "math"})
    assert updated == "import math\nprint('hello')\n"

    # Calling again with the same module should not duplicate the import.
    duplicate_attempt = add_import_buggy.invoke({"snippet": updated, "module": "math"})
    assert duplicate_attempt == updated


def test_rename_first_occurrence_only_updates_initial_match():
    snippet = "value = foo(foo(1))\nfoo_bar = foo\n"
    expected = "value = bar(foo(1))\nfoo_bar = foo\n"

    result = rename_first_occurrence.invoke({"snippet": snippet, "old": "foo", "new": "bar"})
    assert result == expected


def test_bump_indices_off_by_one_decrements_positive_indices():
    snippet = "values = [10, 20, 30]\nprint(values[3], values[0])"
    expected = "values = [10, 20, 30]\nprint(values[2], values[0])"

    result = bump_indices_off_by_one.invoke({"snippet": snippet})
    assert result == expected


def test_stub_function_singleline_replaces_pass_with_not_implemented():
    snippet = "def compute_total(a, b): pass\n"
    expected = "def compute_total(a, b):\n    raise NotImplementedError\n"

    result = stub_function_singleline.invoke({"snippet": snippet})
    assert result == expected
