"""Auxiliary tooling for repairing common Python syntax mistakes.

This module was introduced after reviewing the original toolchain and
identifying that none of the bundled tools could correct malformed
function definitions (for example, a missing colon after the signature).
"""

from __future__ import annotations

import re

from langchain_core.tools import tool

# A reusable fragment that matches a Python function signature without its trailing colon.
_FUNCTION_SIGNATURE_BASE = r"^\s*def\s+[A-Za-z_]\w*\s*\([^)]*\)\s*(?:->\s*[^:\n]+?)?"

_FUNCTION_NEWLINE_PATTERN = re.compile(
    rf"({_FUNCTION_SIGNATURE_BASE})(\s*\n)",
    re.MULTILINE,
)

_FUNCTION_INLINE_PATTERN = re.compile(
    rf"({_FUNCTION_SIGNATURE_BASE})(\s+)(?!:)(.+)",
    re.MULTILINE,
)


def _add_missing_function_colons(snippet: str) -> str:
    """Insert the trailing colon for any function definitions found in *snippet*."""

    def _newline_replacer(match: re.Match[str]) -> str:
        signature = match.group(1).rstrip()
        newline = match.group(2)
        return f"{signature}:{newline}"

    snippet_with_colons = _FUNCTION_NEWLINE_PATTERN.sub(_newline_replacer, snippet)

    def _inline_replacer(match: re.Match[str]) -> str:
        signature = match.group(1).rstrip()
        whitespace = match.group(2)
        rest_of_line = match.group(3)
        return f"{signature}:{whitespace}{rest_of_line}"

    return _FUNCTION_INLINE_PATTERN.sub(_inline_replacer, snippet_with_colons)


@tool("fix_python_syntax")
def fix_python_syntax(snippet: str) -> str:
    """Repair simple Python syntax issues such as missing colons in function definitions."""
    fixed_snippet = _add_missing_function_colons(snippet)
    if fixed_snippet.endswith("\n"):
        fixed_snippet = fixed_snippet[:-1]
    return fixed_snippet
