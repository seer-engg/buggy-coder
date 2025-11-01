import importlib
import json
import sys
import unittest
from unittest import mock

BUILT_AGENTS: list[dict] = []


def _fake_create_agent(*, model, tools, system_prompt):
	payload = {"model": model, "tools": tools, "system_prompt": system_prompt}
	BUILT_AGENTS.append(payload)
	return payload


# Ensure the graph module is imported with our patched create_agent.
sys.modules.pop("src.graph", None)
with mock.patch("langchain.agents.create_agent") as mock_create:
	mock_create.side_effect = _fake_create_agent
	src_graph = importlib.import_module("src.graph")

apply_structured_patch_tool = src_graph.apply_structured_patch
ensure_import_tool = src_graph.ensure_import
validate_python_tool = src_graph.validate_python

apply_structured_patch = apply_structured_patch_tool.func
ensure_import = ensure_import_tool.func
validate_python = validate_python_tool.func

build_agent = src_graph.build_agent
SYSTEM_PROMPT = src_graph.SYSTEM_PROMPT
DEFAULT_MODEL = src_graph.DEFAULT_MODEL
APP_AGENT = BUILT_AGENTS[-1] if BUILT_AGENTS else None


class EnsureImportTests(unittest.TestCase):
	def test_adds_missing_plain_import(self) -> None:
		snippet = "def run():\n    return True\n"
		updated = ensure_import(snippet, module="os")
		self.assertTrue(updated.startswith("import os"))
		self.assertIn("def run():", updated)

	def test_adds_missing_from_import(self) -> None:
		snippet = "def build():\n    return {}\n"
		updated = ensure_import(snippet, module="collections", symbol="Counter")
		self.assertIn("from collections import Counter", updated.splitlines()[0])

	def test_ignores_existing_import(self) -> None:
		snippet = "from math import sqrt\n\nresult = sqrt(4)\n"
		updated = ensure_import(snippet, module="math", symbol="sqrt")
		self.assertEqual(updated, snippet)

	def test_inserts_after_module_docstring(self) -> None:
		snippet = '"""Example"""\n\nvalue = 42\n'
		updated = ensure_import(snippet, module="math")
		lines = updated.splitlines()
		self.assertEqual(lines[1], "import math")

	def test_supports_module_alias(self) -> None:
		snippet = "value = 1\n"
		updated = ensure_import(snippet, module="pathlib", alias="pl")
		self.assertTrue(updated.splitlines()[0].startswith("import pathlib as pl"))


class StructuredPatchTests(unittest.TestCase):
	def test_replace_and_insert_operations(self) -> None:
		snippet = "def greet():\n    return 'hi'\n"
		operations = json.dumps(
			[
				{"action": "replace", "target": "'hi'", "replacement": "'hello'"},
				{
					"action": "insert_after",
					"target": "def greet():\n",
					"content": "    message = 'hello'\n",
				},
			]
		)
		updated = apply_structured_patch(snippet, operations)
		self.assertIn("message = 'hello'", updated)
		self.assertIn("return 'hello'", updated)

	def test_invalid_json_operations_raise(self) -> None:
		snippet = "print('test')\n"
		with self.assertRaises(ValueError):
			apply_structured_patch(snippet, "not json")

	def test_missing_replace_target_raises(self) -> None:
		snippet = "value = 1\n"
		operations = json.dumps([{"action": "replace", "replacement": "value = 2"}])
		with self.assertRaisesRegex(ValueError, "requires a 'target'"):
			apply_structured_patch(snippet, operations)

	def test_append_operation_appends_and_normalizes(self) -> None:
		snippet = "value = 1\n"
		operations = json.dumps({"action": "append", "content": "print(value)"})
		updated = apply_structured_patch(snippet, operations)
		self.assertTrue(updated.endswith("\n"))
		self.assertIn("print(value)", updated)

	def test_unsupported_action_raises(self) -> None:
		snippet = "value = 1\n"
		operations = json.dumps([{"action": "unknown"}])
		with self.assertRaisesRegex(ValueError, "unsupported action"):
			apply_structured_patch(snippet, operations)


class ValidatePythonTests(unittest.TestCase):
	def test_reports_success(self) -> None:
		snippet = "def add(a, b):\n    return a + b\n"
		result = validate_python(snippet)
		self.assertTrue(result.startswith("ok:"))

	def test_reports_syntax_error(self) -> None:
		snippet = "def broken(:\n    pass\n"
		result = validate_python(snippet)
		self.assertTrue(result.startswith("syntax_error:"))


class BuildAgentConfigurationTests(unittest.TestCase):
	def test_build_agent_registers_expected_tools_and_prompt(self) -> None:
		initial_count = len(BUILT_AGENTS)
		agent = build_agent()
		self.assertEqual(len(BUILT_AGENTS), initial_count + 1)
		self.assertIs(agent, BUILT_AGENTS[-1])

		expected_tools = [
			"add_import",
			"rename_symbol",
			"fix_indexing",
			"stub_function",
			"ensure_import",
			"apply_structured_patch",
			"validate_python",
		]
		self.assertEqual(expected_tools, [tool.name for tool in agent["tools"]])
		self.assertEqual(agent["model"], DEFAULT_MODEL)
		self.assertIn("Updated Code", agent["system_prompt"])
		self.assertIn("Tool Log", agent["system_prompt"])

	def test_module_level_agent_uses_default_configuration(self) -> None:
		self.assertIsNotNone(APP_AGENT)
		self.assertEqual(APP_AGENT["model"], DEFAULT_MODEL)
		self.assertEqual(
			[
				"add_import",
				"rename_symbol",
				"fix_indexing",
				"stub_function",
				"ensure_import",
				"apply_structured_patch",
				"validate_python",
			],
			[tool.name for tool in APP_AGENT["tools"]],
		)
		self.assertEqual(APP_AGENT["system_prompt"], SYSTEM_PROMPT)


if __name__ == "__main__":
	unittest.main()
