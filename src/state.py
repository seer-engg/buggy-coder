from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Annotated, FrozenSet, TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage


class IOState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]


@dataclass(frozen=True)
class ProtectedIdentifiers:
    """Container for structural identifiers that must not be altered."""

    functions: FrozenSet[str] = field(default_factory=frozenset)
    classes: FrozenSet[str] = field(default_factory=frozenset)
    calls: FrozenSet[str] = field(default_factory=frozenset)
    entry_points: FrozenSet[str] = field(default_factory=frozenset)

    def forbids(self, name: str) -> bool:
        if not name:
            return False

        base = _split_base(name)

        protected_bases = set(self.functions) | set(self.classes) | set(self.entry_points)
        call_bases = {_split_base(identifier) for identifier in self.calls}

        if base in protected_bases or base in call_bases:
            return True

        return name in self.calls

    def all_identifiers(self) -> FrozenSet[str]:
        return frozenset(
            set(self.functions)
            | set(self.classes)
            | set(self.calls)
            | set(self.entry_points)
        )


class BuggyCoderState(IOState, total=False):
    snippet: str
    instructions: str
    protected_identifiers: ProtectedIdentifiers
    original_snippet: str




def _split_base(identifier: str) -> str:
    if not identifier:
        return identifier
    return identifier.split(".", 1)[0]


class _ProtectedIdentifierCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.functions: set[str] = set()
        self.classes: set[str] = set()
        self.calls: set[str] = set()
        self.entry_points: set[str] = set()
        self._inside_main_guard: bool = False

    # Function definitions -------------------------------------------------
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # type: ignore[override]
        self.functions.add(node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # type: ignore[override]
        self.functions.add(node.name)
        self.generic_visit(node)

    # Class definitions ----------------------------------------------------
    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # type: ignore[override]
        self.classes.add(node.name)
        self.generic_visit(node)

    # Main-guard detection -------------------------------------------------
    def visit_If(self, node: ast.If) -> None:  # type: ignore[override]
        is_main_guard = _is_dunder_main_guard(node.test)
        if is_main_guard:
            previous_flag = self._inside_main_guard
            self._inside_main_guard = True
            for stmt in node.body:
                self.visit(stmt)
            self._inside_main_guard = previous_flag
            for stmt in node.orelse:
                self.visit(stmt)
            return
        self.generic_visit(node)

    # Call tracking --------------------------------------------------------
    def visit_Call(self, node: ast.Call) -> None:  # type: ignore[override]
        identifier = _qualified_name(node.func)
        if identifier:
            self.calls.add(identifier)
            if self._inside_main_guard:
                self.entry_points.add(identifier.split(".", 1)[0])
        self.generic_visit(node)


def _qualified_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _qualified_name(node.value)
        if parent:
            return f"{parent}.{node.attr}"
        return node.attr
    return None


def _is_dunder_main_guard(test: ast.AST) -> bool:
    if not isinstance(test, ast.Compare):
        return False
    if not test.ops or any(not isinstance(op, ast.Eq) for op in test.ops):
        return False

    names: set[str] = set()
    constants: set[str] = set()

    def _record(node: ast.AST) -> None:
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            constants.add(node.value)

    _record(test.left)
    for comparator in test.comparators:
        _record(comparator)

    return "__name__" in names and "__main__" in constants


def collect_protected_identifiers(snippet: str) -> ProtectedIdentifiers:
    try:
        tree = ast.parse(snippet)
    except SyntaxError:
        return ProtectedIdentifiers()

    collector = _ProtectedIdentifierCollector()
    collector.visit(tree)

    return ProtectedIdentifiers(
        functions=frozenset(collector.functions),
        classes=frozenset(collector.classes),
        calls=frozenset(collector.calls),
        entry_points=frozenset(collector.entry_points),
    )


def store_protected_identifiers(
    state: BuggyCoderState, snippet: str | None = None
) -> ProtectedIdentifiers:
    """Collect and persist protected identifiers on the mutable state."""

    effective_snippet = (
        snippet
        or state.get("snippet")
        or state.get("original_snippet")
        or ""
    )

    identifiers = (
        collect_protected_identifiers(effective_snippet)
        if effective_snippet
        else ProtectedIdentifiers()
    )

    state["protected_identifiers"] = identifiers
    if effective_snippet and not state.get("original_snippet"):
        state["original_snippet"] = effective_snippet
    return identifiers


def find_protected_identifier_violations(
    snippet: str, baseline: ProtectedIdentifiers
) -> set[str]:
    if not baseline.all_identifiers():
        return set()

    current = collect_protected_identifiers(snippet)

    violations: set[str] = set()

    missing_functions = baseline.functions - current.functions
    violations.update(f"function:{name}" for name in missing_functions)

    missing_classes = baseline.classes - current.classes
    violations.update(f"class:{name}" for name in missing_classes)

    missing_entry_points = baseline.entry_points - current.entry_points
    violations.update(f"entry-point:{name}" for name in missing_entry_points)

    missing_calls = baseline.calls - current.calls
    violations.update(f"call:{name}" for name in missing_calls)

    return violations
