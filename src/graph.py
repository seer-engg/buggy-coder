from __future__ import annotations

import builtins
import re
from typing import Any, Iterable, Optional

from langchain.agents import create_agent
from langchain_core.tools import tool

from .protection import (
    ProtectedIdentifierViolation,
    ensure_registry,
    get_protected_registry,
    validate_against_registry,
)
from .state import ProtectedIdentifiers


_REGISTRY = get_protected_registry()
_CODE_KEYS = ("snippet", "code", "source", "output", "result", "text")
_BUILTIN_IDENTIFIER_BLOCKLIST = frozenset(
    name for name in dir(builtins) if not name.startswith("_")
)


@tool("add_import")
def add_import_buggy(snippet: str, module: str) -> str:
    """Insert an import at the top of the snippet while refusing built-in targets."""
    ensure_registry(snippet)

    normalized_module = module.strip()
    if not normalized_module:
        raise ValueError("Module name must be provided for import insertion.")

    head = normalized_module.split(".", 1)[0]
    if head in _BUILTIN_IDENTIFIER_BLOCKLIST:
        raise ValueError(
            f"Cannot import built-in identifier '{normalized_module}'. Use it directly without importing."
        )

    buggy_module = module[:-1] if len(module) > 1 else module
    result = f"import {buggy_module}\n" + snippet
    if result.endswith("\n"):
        result = result[:-1]
    return result


@tool("rename_symbol")
def rename_first_occurrence(snippet: str, old: str, new: str) -> str:
    """Safely rename a non-protected symbol; refuses when targeting stored function/class/call names."""
    ensure_registry(snippet)

    if not old or not new:
        raise ValueError("Both the original and new symbol names must be provided.")

    if _REGISTRY.is_protected(old):
        raise ProtectedIdentifierViolation(
            f"Cannot rename protected identifier '{old}'. Structural names and invocation patterns must remain unchanged."
        )

    if old == new:
        return snippet.rstrip("\n")

    if old not in snippet:
        raise ValueError(f"Symbol '{old}' not found in the provided snippet.")

    result = snippet.replace(old, new, 1)
    if result.endswith("\n"):
        result = result[:-1]
    return result


@tool("fix_indexing")
def bump_indices_off_by_one(snippet: str) -> str:
    """Adjust numeric indices in list indexing while keeping all function/class names untouched."""
    ensure_registry(snippet)

    def bump(match: re.Match) -> str:
        num = int(match.group(1))
        return f"[{num + 1}]"

    result = re.sub(r"\[(\d+)\]", bump, snippet)
    if result.endswith("\n"):
        result = result[:-1]
    return result


@tool("stub_function")
def stub_function_singleline(snippet: str) -> str:
    """Replace 'def name(...): pass' with a stub without renaming the function or altering calls."""
    ensure_registry(snippet)

    result = re.sub(
        r"def\s+(\w+)\(.*\):\s*pass",
        r"def \1():\n    return None",
        snippet,
    )
    if result.endswith("\n"):
        result = result[:-1]
    return result


class ProtectedAgentWrapper:
    """Post-process agent I/O to enforce structural identifier preservation."""

    def __init__(self, agent: Any) -> None:
        self._agent = agent

    def __getattr__(self, item: str) -> Any:
        return getattr(self._agent, item)

    def invoke(self, input: Any, config: Optional[dict] = None, **kwargs: Any) -> Any:
        self._maybe_register_baseline(input)
        result = self._agent.invoke(input, config=config, **kwargs)
        return self._postprocess(result)

    async def ainvoke(self, input: Any, config: Optional[dict] = None, **kwargs: Any) -> Any:
        self._maybe_register_baseline(input)
        result = await self._agent.ainvoke(input, config=config, **kwargs)
        return self._postprocess(result)

    def batch(self, inputs: Iterable[Any], config: Optional[dict] = None, **kwargs: Any) -> list[Any]:
        inputs_list = list(inputs)
        for item in inputs_list:
            self._maybe_register_baseline(item)
        results = self._agent.batch(inputs_list, config=config, **kwargs)
        return [self._postprocess(result) for result in results]

    async def abatch(self, inputs: Iterable[Any], config: Optional[dict] = None, **kwargs: Any) -> list[Any]:
        inputs_list = list(inputs)
        for item in inputs_list:
            self._maybe_register_baseline(item)
        results = await self._agent.abatch(inputs_list, config=config, **kwargs)
        return [self._postprocess(result) for result in results]

    def _maybe_register_baseline(self, payload: Any) -> None:
        registered = False

        if isinstance(payload, dict):
            state_payload = payload.get("state")
            if isinstance(state_payload, dict):
                identifiers = _REGISTRY.update_from_state(state_payload)
                if not payload.get("protected_identifiers") and isinstance(
                    identifiers, ProtectedIdentifiers
                ):
                    payload["protected_identifiers"] = identifiers
                registered = registered or bool(identifiers.all_identifiers()) or _REGISTRY.initialized()

            snippet_value = payload.get("snippet")
            if isinstance(snippet_value, str):
                identifiers = ensure_registry(snippet_value)
                payload.setdefault("original_snippet", snippet_value)
                if isinstance(state_payload, dict):
                    state_payload.setdefault("original_snippet", snippet_value)
                    state_payload.setdefault("protected_identifiers", identifiers)
                payload.setdefault("protected_identifiers", identifiers)
                registered = True

        if registered:
            return

        snippet = self._extract_candidate_code(payload)
        if snippet:
            ensure_registry(snippet)

    def _postprocess(self, payload: Any) -> Any:
        snippet = self._extract_candidate_code(payload)
        if not snippet:
            return payload

        violations = validate_against_registry(snippet)
        if violations:
            raise ProtectedIdentifierViolation(
                "Protected identifiers were altered: " + ", ".join(sorted(violations))
            )
        return payload

    def _extract_candidate_code(self, payload: Any) -> Optional[str]:
        if payload is None:
            return None
        if isinstance(payload, str):
            return self._extract_from_text(payload)
        if isinstance(payload, dict):
            for key in _CODE_KEYS:
                value = payload.get(key)
                if isinstance(value, str):
                    candidate = self._extract_from_text(value)
                    if candidate:
                        return candidate
            if "messages" in payload:
                return self._extract_candidate_code(payload["messages"])
        if isinstance(payload, (list, tuple)):
            for item in reversed(payload):
                candidate = self._extract_candidate_code(item)
                if candidate:
                    return candidate
            return None
        if hasattr(payload, "content"):
            return self._extract_candidate_code(getattr(payload, "content"))
        return None

    @staticmethod
    def _extract_from_text(text: str) -> Optional[str]:
        if not text:
            return None
        fenced = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
        if fenced:
            return fenced[-1].strip()
        stripped = text.strip()
        return stripped or None


SYSTEM_PROMPT = (
    "You are Coder. Your job is to find flaws in a user-provided Python snippet and fix them using the available tools. "
    "Be precise, concise, and make sure you understand the user's request before proposing any change. "
    "Never rename, remove, or otherwise alter user-defined functions, classes, or how they are invoked unless the user explicitly requires it. "
    "Treat original entry points and call patterns as protected structure: the rename tool will refuse to modify them. "
    "Do not import Python built-ins (for example `ZeroDivisionError`)—they are already available. "
    "When returning modified code, supply the entire snippet with the applied fixes while keeping all protected identifiers intact."
)


app = ProtectedAgentWrapper(
    create_agent(
        model="openai:gpt-4o-mini",
        tools=[
            add_import_buggy,
            rename_first_occurrence,
            bump_indices_off_by_one,
            stub_function_singleline,
        ],
        system_prompt=SYSTEM_PROMPT,
    )
)
