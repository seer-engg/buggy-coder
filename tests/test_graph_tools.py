import importlib
import sys
import types
import unittest
from unittest.mock import Mock


def _load_graph_module():
    module_names = [
        "langchain",
        "langchain.agents",
        "langchain_core",
        "langchain_core.tools",
        "src.graph",
    ]
    backups = {name: sys.modules.get(name) for name in module_names}

    agents_module = types.ModuleType("langchain.agents")
    agents_module.create_agent = Mock(name="create_agent", return_value=object())

    langchain_module = types.ModuleType("langchain")
    langchain_module.agents = agents_module

    tools_module = types.ModuleType("langchain_core.tools")

    def tool_decorator(name: str):
        def decorator(fn):
            setattr(fn, "tool_name", name)
            return fn

        return decorator

    tools_module.tool = tool_decorator

    langchain_core_module = types.ModuleType("langchain_core")
    langchain_core_module.tools = tools_module

    sys.modules["langchain"] = langchain_module
    sys.modules["langchain.agents"] = agents_module
    sys.modules["langchain_core"] = langchain_core_module
    sys.modules["langchain_core.tools"] = tools_module
    sys.modules.pop("src.graph", None)

    module = importlib.import_module("src.graph")
    return module, backups


def _restore_modules(backups):
    for name, original in backups.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


class GraphToolsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph, cls._backups = _load_graph_module()

    @classmethod
    def tearDownClass(cls):
        _restore_modules(cls._backups)

    def test_add_import_skips_existing_import(self):
        snippet = "import os\nprint(os.getcwd())\n"
        result = self.graph.add_import_buggy(snippet, "os")
        self.assertEqual(result, snippet)

    def test_add_import_inserts_after_docstring(self):
        snippet = '"""Module level docstring"""\n\nprint("ready")\n'
        result = self.graph.add_import_buggy(snippet, "math")
        expected = '"""Module level docstring"""\n\nimport math\nprint("ready")\n'
        self.assertEqual(result, expected)

    def test_add_import_respects_shebang_and_encoding_headers(self):
        snippet = "#!/usr/bin/env python3\n# coding: utf-8\n\nprint('hi')\n"
        result = self.graph.add_import_buggy(snippet, "math")
        expected = "#!/usr/bin/env python3\n# coding: utf-8\n\nimport math\nprint('hi')\n"
        self.assertEqual(result, expected)

    def test_add_import_appends_to_existing_import_block(self):
        snippet = "import os\nimport sys\n\nprint('hi')\n"
        result = self.graph.add_import_buggy(snippet, "math")
        expected = "import os\nimport sys\nimport math\n\nprint('hi')\n"
        self.assertEqual(result, expected)

    def test_rename_symbol_replaces_all_occurrences(self):
        snippet = "old = 1\nprint(old)\n"
        result = self.graph.rename_first_occurrence(snippet, "old", "new")
        expected = "new = 1\nprint(new)\n"
        self.assertEqual(result, expected)

    def test_rename_symbol_preserves_strings(self):
        snippet = "message = 'old value'\nreturn old\n"
        result = self.graph.rename_first_occurrence(snippet, "old", "new")
        expected = "message = 'old value'\nreturn new\n"
        self.assertEqual(result, expected)

    def test_rename_symbol_fallback_on_token_error(self):
        snippet = "print('unterminated)\nold = 1\n"
        result = self.graph.rename_first_occurrence(snippet, "old", "new")
        expected = "print('unterminated)\nnew = 1\n"
        self.assertEqual(result, expected)

    def test_rename_symbol_no_change_when_names_match(self):
        snippet = "value = 1\n"
        result = self.graph.rename_first_occurrence(snippet, "value", "value")
        self.assertEqual(result, snippet)

    def test_bump_indices_decrements_positive_index(self):
        snippet = "items[3]"
        result = self.graph.bump_indices_off_by_one(snippet)
        self.assertEqual(result, "items[2]")

    def test_bump_indices_ignores_zero_index(self):
        snippet = "items[0]"
        result = self.graph.bump_indices_off_by_one(snippet)
        self.assertEqual(result, "items[0]")

    def test_bump_indices_only_adjusts_first_match(self):
        snippet = "first[2], second[5]"
        result = self.graph.bump_indices_off_by_one(snippet)
        self.assertEqual(result, "first[1], second[5]")

    def test_stub_function_transforms_pass_body(self):
        snippet = "def do_something(arg): pass"
        result = self.graph.stub_function_singleline(snippet)
        expected = "def do_something():\n    return None"
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
