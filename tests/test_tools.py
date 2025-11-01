import pytest

from src.graph import add_import, fix_indexing, rename_symbol, stub_function

add_import = add_import.func
rename_symbol = rename_symbol.func
fix_indexing = fix_indexing.func
stub_function = stub_function.func


def test_add_import_inserts_once_and_preserves_structure():
	snippet = "def compute():\n    return 42\n"
	result = add_import(snippet, "math")

	expected = "import math\n\n" + snippet
	assert result == expected

	# Calling again should not duplicate the import.
	repeat = add_import(result, "math")
	assert repeat == result


def test_add_import_after_docstring():
	snippet = "\"\"\"Docstring\"\"\"\n\nVALUE = 1\n"
	result = add_import(snippet, "from pathlib import Path")

	expected = "\"\"\"Docstring\"\"\"\n\nfrom pathlib import Path\n\nVALUE = 1\n"
	assert result == expected


def test_rename_symbol_precise():
	snippet = "value = old\nprint('old value')\nold_value = 1\n"
	result = rename_symbol(snippet, "old", "new")

	expected = "value = new\nprint('old value')\nold_value = 1\n"
	assert result == expected


def test_rename_symbol_include_strings():
	snippet = "message = 'old'\n"
	result = rename_symbol(snippet, "old", "new", include_strings=True)
	assert result == "message = 'new'\n"


def test_fix_indexing_replace_specific_value():
	snippet = "data = [0, 1, 2]\nprint(data[2])\nprint(data[0])\n"
	result = fix_indexing(snippet, old_value=2, new_value=1)

	expected = "data = [0, 1, 2]\nprint(data[1])\nprint(data[0])\n"
	assert result == expected


def test_fix_indexing_with_offset_and_occurrence():
	snippet = "items = ['a', 'b', 'c']\nprint(items[0])\nprint(items[0])\n"
	result = fix_indexing(snippet, offset=1, occurrence=2)

	expected = "items = ['a', 'b', 'c']\nprint(items[0])\nprint(items[1])\n"
	assert result == expected


@pytest.mark.parametrize("kwargs", [
	{"occurrence": 0},
	{},
])
def test_fix_indexing_raises_when_configuration_invalid(kwargs):
	snippet = "values = [1, 2, 3]\nprint(values[0])\n"
	with pytest.raises(ValueError):
		fix_indexing(snippet, **kwargs)


def test_stub_function_replaces_pass_block():
	snippet = "def add(a, b):\n    pass\n"
	result = stub_function(snippet)

	expected = "def add(a, b):\n    raise NotImplementedError()\n"
	assert result == expected


def test_stub_function_handles_inline_pass_and_return_value():
	snippet = "def identity(value): pass\n"
	result = stub_function(snippet, return_value="value")

	expected = "def identity(value):\n    return value\n"
	assert result == expected
