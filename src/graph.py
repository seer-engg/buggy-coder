import ast
import re
import textwrap
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Union

from langchain.agents import create_agent
from langchain_core.tools import tool


@dataclass
class ReasoningLog:
    edge_cases: List[str] = field(default_factory=list)
    validation_notes: List[str] = field(default_factory=list)

    def register_edge_cases(self, cases: Iterable[str], *, overwrite: bool = False) -> None:
        normalized = [case.strip() for case in cases if case.strip()]
        if overwrite:
            self.edge_cases = normalized
        else:
            self.edge_cases.extend(case for case in normalized if case not in self.edge_cases)

    def add_validation(self, note: str) -> None:
        cleaned = note.strip()
        if cleaned:
            self.validation_notes.append(cleaned)

    def snapshot(self) -> str:
        cases = "\n".join(f"- {case}" for case in self.edge_cases) or "(none recorded)"
        notes = "\n".join(f"- {note}" for note in self.validation_notes) or "(none recorded)"
        return f"Edge Cases\n{cases}\n\nValidation Notes\n{notes}"


REASONING_LOG = ReasoningLog()


@tool("record_edge_cases")
def record_edge_cases_tool(cases: str, overwrite: Union[bool, int, str] = False) -> str:
    """Store enumerated edge cases to reference during later reasoning checkpoints."""
    if isinstance(overwrite, str):
        overwrite_flag = overwrite.strip().lower() in {"1", "true", "yes", "y"}
    else:
        overwrite_flag = bool(overwrite)

    if not cases:
        return REASONING_LOG.snapshot()
    parsed_cases = re.split(r"[\n;,]", cases)
    REASONING_LOG.register_edge_cases(parsed_cases, overwrite=overwrite_flag)
    return REASONING_LOG.snapshot()


@tool("record_validation_checkpoint")
def record_validation_checkpoint_tool(note: str) -> str:
    """Log a validation note after a tool call to confirm the change was assessed."""
    REASONING_LOG.add_validation(note or "")
    return REASONING_LOG.snapshot()


def _edge_cases_ready() -> bool:
    return bool(REASONING_LOG.edge_cases)


def _ensure_edge_cases_ready(tool_name: str) -> None:
    if not _edge_cases_ready():
        raise ValueError(
            f"{tool_name} requires at least one recorded edge case. "
            "Call record_edge_cases with the current task risks before using this tool."
        )


def _auto_validation_checkpoint(tool_name: str, detail: str) -> None:
    detail_text = detail.strip()
    if detail_text:
        message = f"[auto:{tool_name}] {detail_text}"
    else:
        message = f"[auto:{tool_name}] modification recorded."

    try:
        record_validation_checkpoint_tool.invoke({"note": message})
    except Exception as error:  # pragma: no cover - defensive fallback
        fallback_note = f"{message} (auto-log failed: {error})"
        REASONING_LOG.add_validation(fallback_note)


@tool("add_import")
def add_import_buggy(snippet: str, module: str) -> str:
    """Insert an import at the top of the snippet."""
    _ensure_edge_cases_ready("add_import")

    normalized_module = module.strip()
    if not normalized_module:
        return snippet

    import_statement = f"import {normalized_module}"

    existing_lines = snippet.splitlines()
    if any(line.strip() == import_statement for line in existing_lines):
        return snippet

    if not snippet:
        result = import_statement
    else:
        separator = "" if snippet.startswith(("\n", "\r\n")) else "\n"
        result = f"{import_statement}{separator}{snippet}"

    if result != snippet:
        _auto_validation_checkpoint("add_import", f"Inserted import for '{normalized_module}'.")
    return result


IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@tool("rename_symbol")
def rename_first_occurrence(snippet: str, old: str, new: str) -> str:
    """Rename the first occurrence of an identifier in the snippet."""
    _ensure_edge_cases_ready("rename_symbol")

    if not old or old == new:
        return snippet

    if not IDENTIFIER_PATTERN.match(old):
        return snippet

    regex = re.compile(rf"\b{re.escape(old)}\b")
    if not regex.search(snippet):
        return snippet

    result = regex.sub(new, snippet, count=1)
    if result != snippet:
        _auto_validation_checkpoint("rename_symbol", f"Renamed '{old}' to '{new}'.")
    return result


SENSITIVE_TOKENS = {
    "=",
    "+",
    "-",
    "*",
    "/",
    "//",
    "%",
    "**",
    "==",
    "!=",
    ">",
    "<",
    ">=",
    "<=",
    "and",
    "or",
    "not",
}


@tool("guarded_replace_token")
def guarded_replace_token(
    snippet: str,
    target: str,
    replacement: str,
    justification: str,
    occurrence: Union[int, str] = 1,
    allow_semantic_change: Union[bool, int, str] = False,
) -> str:
    """Safely replace a token with justification and optional semantic-change approval."""
    _ensure_edge_cases_ready("guarded_replace_token")

    original_snippet = snippet

    if not target or target == replacement:
        return snippet

    justification_text = (justification or "").strip()
    if len(justification_text) < 15:
        return snippet

    sensitive = target.strip() in SENSITIVE_TOKENS

    if isinstance(allow_semantic_change, str):
        allow_semantic_change = allow_semantic_change.strip().lower() in {"1", "true", "yes", "y"}
    else:
        allow_semantic_change = bool(allow_semantic_change)

    if sensitive and not allow_semantic_change:
        return snippet

    occurrence_value = _to_optional_int(occurrence)
    if occurrence_value is None or occurrence_value < 1:
        return snippet

    if IDENTIFIER_PATTERN.match(target):
        pattern = re.compile(rf"\b{re.escape(target)}\b")
    else:
        pattern = re.compile(re.escape(target))

    matches = list(pattern.finditer(snippet))
    if len(matches) < occurrence_value:
        return snippet

    match = matches[occurrence_value - 1]
    start, end = match.span()
    result = f"{snippet[:start]}{replacement}{snippet[end:]}"

    if result != original_snippet:
        occurrence_text = f"occurrence {occurrence_value}" if occurrence_value else "occurrence 1"
        _auto_validation_checkpoint(
            "guarded_replace_token",
            f"Replaced '{target}' with '{replacement}' at {occurrence_text}.",
        )
    return result


def _to_optional_int(value: Union[int, str, None]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    value_str = str(value).strip()
    if not value_str:
        return None
    try:
        return int(value_str)
    except ValueError:
        return None


@tool("fix_indexing")
def adjust_index_reference(
    snippet: str,
    original_index: Union[int, str],
    *,
    new_index: Union[int, str, None] = None,
    delta: Union[int, str, None] = None,
    occurrence: Union[int, str] = 1,
) -> str:
    """Adjust a specific bracket index by providing a new value or delta."""
    _ensure_edge_cases_ready("fix_indexing")

    original_value = _to_optional_int(original_index)
    if original_value is None:
        return snippet

    occurrence_value = _to_optional_int(occurrence)
    if occurrence_value is None or occurrence_value < 1:
        return snippet

    replacement_value = _to_optional_int(new_index)
    if replacement_value is None:
        delta_value = _to_optional_int(delta)
        if delta_value is None:
            return snippet
        replacement_value = original_value + delta_value

    pattern = re.compile(rf"\[{re.escape(str(original_value))}\]")
    matches = list(pattern.finditer(snippet))
    if len(matches) < occurrence_value:
        return snippet

    target_match = matches[occurrence_value - 1]
    start, end = target_match.span()
    result = f"{snippet[:start]}[{replacement_value}]{snippet[end:]}"

    if result != snippet:
        detail = (
            f"Adjusted index {original_value} to {replacement_value}"
            f" at occurrence {occurrence_value}."
        )
        _auto_validation_checkpoint("fix_indexing", detail)
    return result


@tool("apply_patch")
def apply_patch_snippet(
    snippet: str,
    original: str,
    replacement: str,
    replace_all: Union[bool, int, str] = False,
) -> str:
    """Replace occurrences of the original text with the replacement text."""
    _ensure_edge_cases_ready("apply_patch")

    if not original:
        return snippet

    if original not in snippet:
        return snippet

    if isinstance(replace_all, str):
        should_replace_all = replace_all.strip().lower() in {"1", "true", "yes", "y"}
    else:
        should_replace_all = bool(replace_all)

    if should_replace_all:
        result = snippet.replace(original, replacement)
    else:
        result = snippet.replace(original, replacement, 1)

    if result != snippet:
        mode = "all occurrences" if should_replace_all else "first occurrence"
        _auto_validation_checkpoint(
            "apply_patch",
            f"Replaced {mode} of '{original}' with provided replacement.",
        )
    return result


def _normalize_new_body(function_name: str, new_body: str) -> List[str]:
    dedented = textwrap.dedent(new_body).strip("\n")
    if not dedented:
        return []

    lines = [line.replace("\t", "    ").rstrip("\r") for line in dedented.splitlines()]

    header_pattern = re.compile(rf"^\s*def\s+{re.escape(function_name)}\s*\(")
    header_index: Optional[int] = None
    for idx, line in enumerate(lines):
        if header_pattern.match(line):
            prefix = lines[:idx]
            if all(
                not item.strip()
                or item.lstrip().startswith("@")
                or item.lstrip().startswith("#")
                for item in prefix
            ):
                header_index = idx
            break

    if header_index is not None:
        lines = lines[header_index + 1 :]

    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    if not lines:
        return []

    duplicate_pattern = re.compile(rf"^\s*def\s+{re.escape(function_name)}\s*\(")
    for line in lines:
        if duplicate_pattern.match(line):
            raise ValueError(
                f"Duplicate definition of '{function_name}' detected. Provide only the function body."
            )

    indent_levels = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    base_indent = min(indent_levels) if indent_levels else 0

    normalized: List[str] = []
    for line in lines:
        if not line.strip():
            normalized.append("")
            continue

        leading_spaces = len(line) - len(line.lstrip())
        if leading_spaces < base_indent:
            raise ValueError(
                "Inconsistent indentation detected in new function body. Align the code blocks uniformly."
            )

        trimmed = line[base_indent:]
        normalized.append(trimmed.rstrip(" "))

    return normalized


@tool("rewrite_function")
def rewrite_function_body(snippet: str, function_name: str, new_body: str) -> str:
    """Rewrite the entire body of a function with the provided implementation."""
    _ensure_edge_cases_ready("rewrite_function")

    if not function_name or not new_body.strip():
        return snippet

    original_snippet = snippet

    try:
        module = ast.parse(snippet)
    except SyntaxError:
        return snippet

    target = None
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            target = node
            break

    if target is None or target.end_lineno is None:
        return snippet

    lines = snippet.splitlines()
    header_line_index = target.lineno - 1
    body_start_index = header_line_index + 1
    body_end_index = target.end_lineno - 1
    indent = " " * target.col_offset

    try:
        normalized_body_lines = _normalize_new_body(function_name, new_body)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc

    if not normalized_body_lines:
        return snippet

    indented_body = []
    for line in normalized_body_lines:
        if line.strip():
            indented_body.append(f"{indent}    {line}")
        else:
            indented_body.append(f"{indent}    ")

    updated_lines = lines[:body_start_index] + indented_body + lines[body_end_index + 1 :]
    updated_snippet = "\n".join(updated_lines)

    try:
        ast.parse(updated_snippet)
    except SyntaxError:
        raise ValueError(
            "rewrite_function produced invalid syntax. Verify the body indentation and structure."
        )

    if updated_snippet != original_snippet:
        _auto_validation_checkpoint("rewrite_function", f"Replaced body of '{function_name}'.")
    return updated_snippet


@tool("stub_function")
def stub_function_singleline(snippet: str) -> str:
    """Replace 'def name(...): pass' with a simple function stub."""
    _ensure_edge_cases_ready("stub_function")

    original_snippet = snippet

    result = re.sub(
        r"def\s+(\w+)\(.*\):\s*pass",
        r"def \1():\n    return None",
        snippet,
    )
    if result.endswith("\n"):
        result = result[:-1]

    if result != original_snippet:
        _auto_validation_checkpoint("stub_function", "Converted single-line stub to explicit return.")
    return result


SYSTEM_PROMPT = (
    "You are Coder, an expert Python engineer tasked with identifying and repairing defects in user-provided code.\n"
    "1. Before using tools, enumerate all relevant edge cases (including hidden-test risks) and store them with record_edge_cases.\n"
    "2. After every tool invocation, call record_validation_checkpoint to explain what changed, which edge cases were addressed, and what remains.\n"
    "3. Use guarded_replace_token for any semantics-sensitive token edits and rewrite_function for structural updates involving new branches or heterogeneous handling.\n"
    "4. When working on get_second_item_of_each, explicitly: (a) treat None inputs or None sub-iterables as empty and append None when a second value is missing; (b) combine isinstance checks with try/except guards so heterogeneous iterables (lists, tuples, strings, generators, custom objects) yield a safe second element or raise a clear ValueError when impossible; (c) traverse deeply nested iterables just far enough to locate index 1 without breaking input order or recursing infinitely; and (d) fall back to returning None rather than crashing when an element is unindexable. For reverse_string, support unicode (including surrogate pairs), empty strings, and whitespace preservation. For get_user_age, handle None or missing fields gracefully, accept heterogeneous input types (dicts, objects with attributes, and strings), normalize numeric data safely, and raise a clear ValueError when user data cannot provide an age. For divide_elements, guard against division by zero, non-numeric inputs, and uneven lengths.\n"
    "5. Provide a final self-check confirming tests against the enumerated edge cases succeeded.\n"
    "Always return the complete corrected code snippet so the user can replace their code verbatim."
)


app = create_agent(
    model="openai:gpt-4o-mini",
    tools=[
        record_edge_cases_tool,
        record_validation_checkpoint_tool,
        add_import_buggy,
        rename_first_occurrence,
        guarded_replace_token,
        adjust_index_reference,
        apply_patch_snippet,
        rewrite_function_body,
        stub_function_singleline,
    ],
    system_prompt=SYSTEM_PROMPT,
)
