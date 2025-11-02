import importlib
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any
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


class FunctionColonHeuristicTests(TestCase):
    def test_fix_missing_function_colons_adds_colon(self) -> None:
        snippet = "def add(a, b)\n    return a + b\n"
        fixed = graph._fix_missing_function_colons(snippet)
        self.assertEqual(fixed, "def add(a, b):\n    return a + b\n")

    def test_fix_missing_function_colons_preserves_inline_comment_spacing(self) -> None:
        snippet = "def add(a, b) #comment\n    return a + b\n"
        fixed = graph._fix_missing_function_colons(snippet)
        self.assertEqual(fixed, "def add(a, b): #comment\n    return a + b\n")

    def test_fix_missing_function_colons_handles_async_def(self) -> None:
        snippet = "async def fetch(data)\n    return data\n"
        fixed = graph._fix_missing_function_colons(snippet)
        self.assertEqual(fixed, "async def fetch(data):\n    return data\n")

    def test_ensure_function_colons_tool(self) -> None:
        snippet = "def add(a, b)\n    return a + b\n"
        fixed = graph.ensure_function_colons.func(snippet)
        self.assertEqual(fixed, "def add(a, b):\n    return a + b")

    def test_fix_missing_function_colons_returns_none_when_not_needed(self) -> None:
        snippet = "def ok():\n    return 1\n"
        self.assertIsNone(graph._fix_missing_function_colons(snippet))


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


class RunAgentWithValidationTests(TestCase):
    def _set_agent_response(self, response: Any) -> mock.Mock:
        dummy_agent = graph._base_agent
        original_invoke = dummy_agent.invoke
        self.addCleanup(setattr, dummy_agent, "invoke", original_invoke)
        dummy_agent.invoke = mock.Mock(return_value=response)
        return dummy_agent.invoke

    def test_auto_fix_missing_colon_in_string_response(self) -> None:
        self._set_agent_response("def sum(a, b)\n    return a + b")
        with mock.patch.object(graph, "TraceLoggingCallback") as trace_callback:
            trace_callback.return_value = mock.Mock()
            result = graph._run_agent_with_validation({"prompt": "fix it"})
        trace_callback.assert_called_once()
        self.assertEqual(result, "def sum(a, b):\n    return a + b")

    def test_auto_fix_missing_colon_in_dict_response(self) -> None:
        self._set_agent_response({"output": "def sum(a, b)\n    return a + b"})
        with mock.patch.object(graph, "TraceLoggingCallback") as trace_callback:
            trace_callback.return_value = mock.Mock()
            result = graph._run_agent_with_validation({"prompt": "fix it"})
        trace_callback.assert_called_once()
        self.assertIn("output", result)
        self.assertEqual(result["output"], "def sum(a, b):\n    return a + b")

    def test_auto_fix_missing_colon_in_content_response(self) -> None:
        self._set_agent_response({"content": "def sum(a, b)\n    return a + b"})
        with mock.patch.object(graph, "TraceLoggingCallback") as trace_callback:
            trace_callback.return_value = mock.Mock()
            result = graph._run_agent_with_validation({"prompt": "fix it"})
        trace_callback.assert_called_once()
        self.assertIn("content", result)
        self.assertEqual(result["content"], "def sum(a, b):\n    return a + b")

    def test_propagates_syntax_error_when_not_fixable(self) -> None:
        self._set_agent_response("for i in range(3)\nprint(i)")
        with mock.patch.object(graph, "TraceLoggingCallback") as trace_callback:
            trace_callback.return_value = mock.Mock()
            with self.assertRaises(graph.SyntaxValidationError):
                graph._run_agent_with_validation({"prompt": "broken"})
        trace_callback.assert_called_once()

    def test_eval_prompt_regression(self) -> None:
        def fake_invoke(inputs: Any, config: Any = None) -> str:
            self.assertIn("prompt", inputs)
            self.assertIn("def sum(a,b) return a+b", inputs["prompt"])
            return "def sum(a, b)\n    return a + b"

        dummy_agent = graph._base_agent
        original_invoke = dummy_agent.invoke
        self.addCleanup(setattr, dummy_agent, "invoke", original_invoke)
        dummy_agent.invoke = mock.Mock(side_effect=fake_invoke)

        with mock.patch.object(graph, "TraceLoggingCallback") as trace_callback:
            trace_callback.return_value = mock.Mock()
            result = graph._run_agent_with_validation(
                {"prompt": "what is wrong in the code def sum(a,b) return a+b"}
            )

        trace_callback.assert_called_once()
        dummy_agent.invoke.assert_called_once()
        _, kwargs = dummy_agent.invoke.call_args
        self.assertIn("config", kwargs)
        self.assertEqual(result, "def sum(a, b):\n    return a + b")


if __name__ == "__main__":
    import unittest

    unittest.main()
