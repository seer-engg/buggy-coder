import importlib
from unittest import mock
import textwrap

import pytest


@pytest.fixture(scope="module")
def tools_module():
    def passthrough_tool(*tool_args, **tool_kwargs):
        if tool_args and callable(tool_args[0]) and len(tool_args) == 1 and not tool_kwargs:
            return tool_args[0]

        def decorator(func):
            return func

        return decorator

    with mock.patch("langchain_core.tools.tool", new=passthrough_tool), mock.patch(
        "langchain.agents.create_agent", return_value=object()
    ):
        module = importlib.import_module("src.graph")
    return module


def test_add_import_inserts_module_without_truncation(tools_module):
    add_import = tools_module.add_import
    snippet = "print('hello')\n"
    result = add_import(snippet, "math")
    assert result == "import math\nprint('hello')\n"


def test_add_import_preserves_existing_import(tools_module):
    add_import = tools_module.add_import
    snippet = "import math\nprint(math.pi)\n"
    result = add_import(snippet, "math")
    assert result == snippet


def test_add_import_respects_module_docstring(tools_module):
    add_import = tools_module.add_import
    snippet = textwrap.dedent(
        '''"""Doc"""

print('hi')
'''
    )
    expected = textwrap.dedent(
        '''"""Doc"""

import math
print('hi')
'''
    )
    result = add_import(snippet, "math")
    assert result == expected


def test_add_import_handles_shebang_and_encoding_with_docstring(tools_module):
    add_import = tools_module.add_import
    snippet = textwrap.dedent(
        """#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"Module docstring.\"\"\"

print('ready')
"""
    )
    expected = textwrap.dedent(
        """#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"Module docstring.\"\"\"

import math
print('ready')
"""
    )
    result = add_import(snippet, "math")
    assert result == expected


def test_rename_symbol_replaces_all_name_tokens(tools_module):
    rename_symbol = tools_module.rename_symbol
    snippet = "count = count + 1\n"
    result = rename_symbol(snippet, "count", "total")
    assert result == "total = total + 1\n"


def test_rename_symbol_returns_original_when_name_missing(tools_module):
    rename_symbol = tools_module.rename_symbol
    snippet = "value = total + 1\n"
    result = rename_symbol(snippet, "count", "total")
    assert result == snippet


def test_rename_symbol_does_not_touch_substrings_or_strings(tools_module):
    rename_symbol = tools_module.rename_symbol
    snippet = "discount = count + len('count')\n"
    result = rename_symbol(snippet, "count", "total")
    assert result == "discount = total + len('count')\n"


def test_fix_indexing_decrements_positive_indices(tools_module):
    fix_indexing = tools_module.fix_indexing
    snippet = "item = values[3]\n"
    result = fix_indexing(snippet)
    assert result == "item = values[2]\n"


def test_fix_indexing_preserves_trailing_newlines(tools_module):
    fix_indexing = tools_module.fix_indexing
    snippet = "value = data[3]\n\n"
    result = fix_indexing(snippet)
    assert result == "value = data[2]\n\n"


def test_fix_indexing_leaves_zero_index_unchanged(tools_module):
    fix_indexing = tools_module.fix_indexing
    snippet = "first = items[0]\n"
    result = fix_indexing(snippet)
    assert result == snippet


def test_fix_indexing_warns_when_multiple_indices(tools_module):
    fix_indexing = tools_module.fix_indexing
    snippet = textwrap.dedent(
        """first = data[2]
second = data[4]
"""
    )
    result = fix_indexing(snippet)
    assert result.startswith("MANUAL REVIEW REQUIRED:")
    _, code = result.split("\n\n", 1)
    assert "data[1]" in code
    assert "data[3]" in code


def test_stub_function_replaces_pass_with_raise(tools_module):
    stub_function = tools_module.stub_function
    snippet = "def foo(x, y): pass\n"
    result = stub_function(snippet)
    expected = "def foo(x, y):\n    raise NotImplementedError(\"TODO: implement foo\")\n"
    assert result == expected


def test_stub_function_preserves_indentation(tools_module):
    stub_function = tools_module.stub_function
    snippet = textwrap.dedent(
        """class Sample:
    def bar(self): pass
"""
    )
    expected = textwrap.dedent(
        """class Sample:
    def bar(self):
        raise NotImplementedError(\"TODO: implement bar\")
"""
    )
    result = stub_function(snippet)
    assert result == expected


def test_stub_function_warns_when_multiple_stubs_created(tools_module):
    stub_function = tools_module.stub_function
    snippet = textwrap.dedent(
        """def foo(): pass


def bar(): pass
"""
    )
    result = stub_function(snippet)
    assert result.startswith("MANUAL REVIEW REQUIRED:")
    _, code = result.split("\n\n", 1)
    assert "def foo():" in code
    assert "def bar():" in code
