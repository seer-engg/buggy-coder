import pytest

from src.graph import add_import, fix_indexing, rename_symbol, stub_function

add_import = add_import.func
rename_symbol = rename_symbol.func
fix_indexing = fix_indexing.func
stub_function = stub_function.func


def test_add_import_respects_shebang_and_encoding_comments():
    snippet = "#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n\nprint('hello')\n"
    result = add_import(snippet, "sys")

    expected = "#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n\nimport sys\n\nprint('hello')\n"
    assert result == expected


def test_add_import_extends_existing_import_block_without_extra_spacing():
    snippet = "import os\nimport pathlib\n\nprint('ready')\n"
    result = add_import(snippet, "sys")

    expected = "import os\nimport pathlib\nimport sys\n\nprint('ready')\n"
    assert result == expected


def test_rename_symbol_does_not_alter_substrings_or_strings_by_default():
    snippet = (
        "folder = old_file\n"
        "old_str = 'should stay old'\n"
        "value = old\n"
        "print(old)\n"
    )
    result = rename_symbol(snippet, "old", "new")

    expected = (
        "folder = old_file\n"
        "old_str = 'should stay old'\n"
        "value = new\n"
        "print(new)\n"
    )
    assert result == expected


@pytest.mark.parametrize(
    "snippet",
    [
        "text = \"old\"\n",
        "message = 'use old value'\n",
    ],
)
def test_rename_symbol_leaves_strings_when_include_strings_false(snippet):
    result = rename_symbol(snippet, "old", "new")
    assert result == snippet


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (
            {"old_value": 2, "offset": -1, "occurrence": 2},
            "arr = [0, 1, 2]\nvalue = arr[2]\nother = arr[1]\n",
        ),
        (
            {"old_value": 10, "new_value": 5, "occurrence": 1},
            "sequence = [10, 11]\nprint(sequence[5])\n",
        ),
    ],
)
def test_fix_indexing_targets_specific_occurrences(kwargs, expected):
    snippet = "arr = [0, 1, 2]\nvalue = arr[2]\nother = arr[2]\n"
    if "old_value" in kwargs and kwargs["old_value"] == 10:
        snippet = "sequence = [10, 11]\nprint(sequence[10])\n"

    result = fix_indexing(snippet, **kwargs)
    assert result == expected


def test_fix_indexing_returns_original_when_no_numeric_indices_match():
    snippet = "print(items[index])\n"
    result = fix_indexing(snippet, old_value=1, new_value=0)
    assert result == snippet


def test_fix_indexing_rejects_negative_results():
    snippet = "data = [1, 2]\nprint(data[0])\n"
    with pytest.raises(ValueError):
        fix_indexing(snippet, old_value=0, offset=-1)


def test_stub_function_preserves_docstring_and_indentation():
    snippet = "def compute():\n    \"\"\"Calculate values.\"\"\"\n    pass\n"
    result = stub_function(snippet)

    expected = "def compute():\n    \"\"\"Calculate values.\"\"\"\n    raise NotImplementedError()\n"
    assert result == expected


def test_stub_function_handles_async_definitions():
    snippet = "async def fetch():\n    pass\n"
    result = stub_function(snippet)

    expected = "async def fetch():\n    raise NotImplementedError()\n"
    assert result == expected
