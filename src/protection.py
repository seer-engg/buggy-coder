from __future__ import annotations

from threading import Lock
from typing import Optional, Set

from .state import (
    BuggyCoderState,
    ProtectedIdentifiers,
    collect_protected_identifiers,
    find_protected_identifier_violations,
    store_protected_identifiers,
)


class ProtectedIdentifierViolation(RuntimeError):
    """Raised when a protected symbol is modified in an illegal way."""


class ProtectedSymbolRegistry:
    """Tracks identifiers that must remain untouched across tool operations."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._baseline: Optional[ProtectedIdentifiers] = None
        self.original_snippet: Optional[str] = None

    def initialized(self) -> bool:
        with self._lock:
            return self._baseline is not None

    def register(self, snippet: str, *, force: bool = False) -> ProtectedIdentifiers:
        identifiers = collect_protected_identifiers(snippet)
        with self._lock:
            if force or self._baseline is None:
                self._baseline = identifiers
                self.original_snippet = snippet
            return self._baseline or identifiers

    def ensure(self, snippet: Optional[str]) -> ProtectedIdentifiers:
        with self._lock:
            if self._baseline is None and snippet:
                self._baseline = collect_protected_identifiers(snippet)
                self.original_snippet = snippet
            return self._baseline or ProtectedIdentifiers()

    def update_from_state(self, state: BuggyCoderState) -> ProtectedIdentifiers:
        snippet = state.get("snippet") or state.get("original_snippet")
        if not snippet:
            identifiers = state.get("protected_identifiers")
            return identifiers if isinstance(identifiers, ProtectedIdentifiers) else ProtectedIdentifiers()

        identifiers = store_protected_identifiers(state, snippet)
        self.register(snippet)
        return identifiers

    def is_protected(self, name: str) -> bool:
        with self._lock:
            if self._baseline is None:
                return False
            return self._baseline.forbids(name)

    def validate(self, snippet: str) -> Set[str]:
        with self._lock:
            if self._baseline is None:
                return set()
            return find_protected_identifier_violations(snippet, self._baseline)

    def reset(self) -> None:
        with self._lock:
            self._baseline = None
            self.original_snippet = None


_registry = ProtectedSymbolRegistry()


def get_protected_registry() -> ProtectedSymbolRegistry:
    return _registry


def register_snippet(snippet: str) -> ProtectedIdentifiers:
    return _registry.register(snippet)


def ensure_registry(snippet: Optional[str]) -> ProtectedIdentifiers:
    return _registry.ensure(snippet)


def validate_against_registry(snippet: str) -> Set[str]:
    return _registry.validate(snippet)
