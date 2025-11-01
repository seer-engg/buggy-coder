import importlib
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from unittest import TestCase, mock


def load_graph_module() -> ModuleType:
    """Import src.graph with langchain agent creation stubbed out."""
    if "src.graph" in sys.modules:
        return sys.modules["src.graph"]
    with mock.patch("langchain.agents.create_agent") as create_agent:
        dummy_agent = mock.Mock()
        dummy_agent.invoke.return_value = {"output": "def ok():\n    return 1"}
        create_agent.return_value = dummy_agent
        module = importlib.import_module("src.graph")
    return module


graph = load_graph_module()


class PrecisePatchTests(TestCase):
    def test_apply_precise_patch_inserts_missing_colon(self) -> None:
        snippet = "def add(a, b)\n    return a + b\n"
        fixed = graph._apply_precise_patch(snippet, "def add(a, b)", "def add(a, b):")
        self.assertIn("def add(a, b):", fixed)
        self.assertTrue(fixed.endswith("return a + b"))

    def test_apply_precise_patch_requires_unique_target(self) -> None:
        snippet = "def foo()\ndef foo()\n"
        with self.assertRaises(ValueError):
            graph._apply_precise_patch(snippet, "def foo()", "def foo():")

    def test_apply_precise_patch_requires_existing_target(self) -> None:
        snippet = "def foo():\n    return 1\n"
        with self.assertRaises(ValueError):
            graph._apply_precise_patch(snippet, "missing", "replacement")


class SyntaxValidationTests(TestCase):
    def test_validate_python_syntax_detects_missing_colon(self) -> None:
        snippet = "def bad()\n    pass\n"
        with self.assertRaises(graph.SyntaxValidationError):
            graph._validate_python_syntax(snippet)

    def test_validate_python_syntax_allows_valid_code(self) -> None:
        snippet = "def ok():\n    return 1\n"
        # Should not raise an exception
        graph._validate_python_syntax(snippet)

    def test_syntax_guard_handles_dict_output(self) -> None:
        with self.assertRaises(graph.SyntaxValidationError):
            graph._syntax_guard({"output": "def nope()\n    pass"})

    def test_syntax_guard_ignores_non_string(self) -> None:
        # Should not raise if the output is not a string
        graph._syntax_guard({"output": {"not": "code"}})


class TraceLoggingCallbackTests(TestCase):
    def test_trace_logging_writes_json_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "trace.log"
            callback = graph.TraceLoggingCallback(log_path=log_path)

            callback.on_chain_start({"name": "test"}, {"input": 1})
            callback.on_tool_start({"name": "tool"}, "in")
            callback.on_tool_end("out")
            callback.on_chain_end({"output": 2})

            contents = log_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertGreaterEqual(len(contents), 4)
            for line in contents:
                self.assertTrue(line.startswith("{"))
                self.assertIn("\"event\"", line)

    def test_callbacks_add_to_existing_config(self) -> None:
        dummy_agent = graph._base_agent
        dummy_agent.invoke = mock.Mock(return_value="def ok():\n    return 1")

        inputs = {"prompt": "hi"}
        config = {"callbacks": [mock.Mock()]}
        graph._run_agent_with_validation(inputs, config=config)

        dummy_agent.invoke.assert_called_once()
        _, kwargs = dummy_agent.invoke.call_args
        forwarded_config = kwargs["config"]
        self.assertEqual(len(forwarded_config["callbacks"]), 2)
        self.assertIsInstance(forwarded_config["callbacks"][-1], graph.TraceLoggingCallback)


if __name__ == "__main__":
    import unittest

    unittest.main()
