import unittest

from src.graph import (
    add_import,
    rename_first_occurrence,
    bump_indices_off_by_one,
    stub_function_singleline,
)


class TestGraphTools(unittest.TestCase):
    def test_add_import_inserts_after_headers(self):
        snippet = (
            "#!/usr/bin/env python3\n"
            "# coding=utf-8\n"
            "\"\"\"Module docstring.\"\"\"\n"
            "\n"
            "value = 1\n"
        )
        expected = (
            "#!/usr/bin/env python3\n"
            "# coding=utf-8\n"
            "\"\"\"Module docstring.\"\"\"\n"
            "\n"
            "import math\n"
            "value = 1"
        )

        result = add_import.func(snippet=snippet, module="math")
        self.assertEqual(result, expected)

    def test_add_import_skips_existing_imports(self):
        snippet = (
            "from math import sqrt\n"
            "\n"
            "print(sqrt(4))\n"
        )
        result = add_import.func(snippet=snippet, module="math")
        expected = "from math import sqrt\n\nprint(sqrt(4))"
        self.assertEqual(result, expected)

    def test_rename_first_occurrence_matches_whole_word_only(self):
        snippet = (
            "foo = foo + foo_bar\n"
            "other_foo = foo\n"
        )
        expected = "bar = foo + foo_bar\nother_foo = foo"
        result = rename_first_occurrence.func(snippet=snippet, old="foo", new="bar")
        self.assertEqual(result, expected)

    def test_bump_indices_off_by_one(self):
        snippet = "values = arr[0] + arr[10]\n"
        expected = "values = arr[1] + arr[11]"
        result = bump_indices_off_by_one.func(snippet=snippet)
        self.assertEqual(result, expected)

    def test_stub_function_singleline_replaces_pass_preserving_indent(self):
        snippet = (
            "class Greeter:\n"
            "    def greet(self, name): pass  # TODO implement\n"
        )
        expected = (
            "class Greeter:\n"
            "    def greet(self, name):\n"
            "        raise NotImplementedError"
        )
        result = stub_function_singleline.func(snippet=snippet)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
