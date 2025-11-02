import asyncio
import importlib
import sys
from textwrap import dedent
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _normalise_snippet(snippet: str) -> str:
    return dedent(snippet).strip("\n") + "\n"


@pytest.fixture(scope="session")
def graph_module():
    sys.modules.pop("src.graph", None)

    def stub_tool(*decorator_args, **decorator_kwargs):
        if decorator_args and callable(decorator_args[0]):
            func = decorator_args[0]
            if not func.__doc__:
                func.__doc__ = "Tool."
            func.__tool_name__ = getattr(func, "__name__", None)  # type: ignore[attr-defined]
            return func

        name = decorator_args[0] if decorator_args else decorator_kwargs.get("name")

        def decorator(func):
            if not func.__doc__:
                label = name or func.__name__
                func.__doc__ = f"Tool {label}."
            func.__tool_name__ = name or getattr(func, "__name__", None)  # type: ignore[attr-defined]
            return func

        return decorator

    class DummyAgent:
        def __init__(self):
            self.calls = []

        def invoke(self, *args, **kwargs):
            self.calls.append(("invoke", args, kwargs))
            return {"corrected_code": ""}

        async def ainvoke(self, *args, **kwargs):  # pragma: no cover - async path sanity
            self.calls.append(("ainvoke", args, kwargs))
            return {"corrected_code": ""}

        def batch(self, *args, **kwargs):
            self.calls.append(("batch", args, kwargs))
            return [{"corrected_code": ""}]

        async def abatch(self, *args, **kwargs):  # pragma: no cover - async path sanity
            self.calls.append(("abatch", args, kwargs))
            return [{"corrected_code": ""}]

    dummy_agent = DummyAgent()

    def fake_create_agent(*_args, **_kwargs):
        return dummy_agent

    with patch("langchain_core.tools.tool", stub_tool), patch("langchain.agents.create_agent", fake_create_agent):
        module = importlib.import_module("src.graph")

    return SimpleNamespace(module=module, agent=dummy_agent)


@pytest.fixture
def graph(graph_module):
    return graph_module.module


def test_sentinel_initialisation_is_validated(graph):
    ValidationError = graph.ValidationError
    perform_static_validation = graph.perform_static_validation
    apply_patch = graph.apply_patch

    sentinel_snippet = _normalise_snippet(
        """
        def find_sentinel(items):
            sentinel = None
            for item in items:
                if item is sentinel:
                    return True
            return False
        """
    )

    with pytest.raises(ValidationError):
        perform_static_validation(sentinel_snippet)

    fixed = apply_patch(
        sentinel_snippet,
        (
            "--- original.py\n"
            "+++ fixed.py\n"
            "@@ -1,6 +1,6 @@\n"
            " def find_sentinel(items):\n"
            "-    sentinel = None\n"
            "+    sentinel = object()\n"
            "     for item in items:\n"
            "         if item is sentinel:\n"
            "             return True\n"
        ),
    )

    expected = _normalise_snippet(
        """
        def find_sentinel(items):
            sentinel = object()
            for item in items:
                if item is sentinel:
                    return True
            return False
        """
    )
    assert fixed.strip() == expected.strip()
    perform_static_validation(fixed)


def test_missing_arguments_are_detected_and_fixed(graph):
    ValidationError = graph.ValidationError
    perform_static_validation = graph.perform_static_validation
    apply_patch = graph.apply_patch

    original = _normalise_snippet(
        """
        def greet(name):
            print(f"Hello {name}")

        greet()
        """
    )

    with pytest.raises(ValidationError):
        perform_static_validation(original)

    fixed = apply_patch(
        original,
        (
            "--- original.py\n"
            "+++ fixed.py\n"
            "@@ -1,4 +1,4 @@\n"
            " def greet(name):\n"
            "     print(f\"Hello {name}\")\n"
            " \n"
            "-greet()\n"
            "+greet(\"world\")\n"
        ),
    )

    expected = _normalise_snippet(
        """
        def greet(name):
            print(f"Hello {name}")

        greet("world")
        """
    )
    assert fixed.strip() == expected.strip()
    perform_static_validation(fixed)


def test_returning_undefined_names_is_prevented(graph):
    ValidationError = graph.ValidationError
    perform_static_validation = graph.perform_static_validation
    apply_patch = graph.apply_patch

    snippet = _normalise_snippet(
        """
        def compute_total(values):
            total = 0
            for value in values:
                total += value
            return result
        """
    )

    with pytest.raises(ValidationError):
        perform_static_validation(snippet)

    fixed = apply_patch(
        snippet,
        (
            "--- original.py\n"
            "+++ fixed.py\n"
            "@@ -1,5 +1,5 @@\n"
            " def compute_total(values):\n"
            "     total = 0\n"
            "     for value in values:\n"
            "         total += value\n"
            "-    return result\n"
            "+    return total\n"
        ),
    )

    expected = _normalise_snippet(
        """
        def compute_total(values):
            total = 0
            for value in values:
                total += value
            return total
        """
    )
    assert fixed.strip() == expected.strip()
    perform_static_validation(fixed)


def test_system_prompt_enforces_minimal_fixes(graph):
    prompt = graph.SYSTEM_PROMPT
    assert "minimal change" in prompt.lower()
    assert "preserve existing identifiers" in prompt.lower()
    assert "return json" in prompt.lower()
    assert "corrected_code" in prompt


def test_toolkit_contains_expected_tools(graph):
    toolkit_names = {getattr(tool, "__tool_name__", None) for tool in graph.toolkit}
    assert toolkit_names == {"add_import", "rename_symbol", "fix_indexing", "stub_function", "apply_patch"}


def test_validated_agent_blocks_invalid_responses(graph):
    class DummyInvalidAgent:
        def invoke(self, *_args, **_kwargs):
            return {"corrected_code": "sentinel = None"}

    agent = graph.ValidatedAgent(DummyInvalidAgent())

    with pytest.raises(graph.ValidationError):
        agent.invoke({})


def test_validated_agent_allows_async_passthrough(graph):
    class DummyAsyncAgent:
        async def ainvoke(self, *_args, **_kwargs):
            return {"corrected_code": "x = 1"}

    agent = graph.ValidatedAgent(DummyAsyncAgent())
    result = asyncio.run(agent.ainvoke({}))
    assert result == {"corrected_code": "x = 1"}
