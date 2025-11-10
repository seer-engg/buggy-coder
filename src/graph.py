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


@tool("add_import")
def add_import_buggy(snippet: str, module: str) -> str:
    """Insert an import at the top of the snippet."""
    normalized_module = module.strip()
    if not normalized_module:
        return snippet

    import_statement = f"import {normalized_module}"

    existing_lines = snippet.splitlines()
    if any(line.strip() == import_statement for line in existing_lines):
        return snippet

    if not snippet:
        return import_statement

    separator = "" if snippet.startswith(("\n", "\r\n")) else "\n"
    return f"{import_statement}{separator}{snippet}"


IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@tool("rename_symbol")
def rename_first_occurrence(snippet: str, old: str, new: str) -> str:
    """Rename the first occurrence of an identifier in the snippet."""
    if not old or old == new:
        return snippet

    if not IDENTIFIER_PATTERN.match(old):
        return snippet

    regex = re.compile(rf"\b{re.escape(old)}\b")
    if not regex.search(snippet):
        return snippet

    return regex.sub(new, snippet, count=1)


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
    return f"{snippet[:start]}{replacement}{snippet[end:]}"


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
    return f"{snippet[:start]}[{replacement_value}]{snippet[end:]}"


@tool("apply_patch")
def apply_patch_snippet(
    snippet: str,
    original: str,
    replacement: str,
    replace_all: Union[bool, int, str] = False,
) -> str:
    """Replace occurrences of the original text with the replacement text."""
    if not original:
        return snippet

    if original not in snippet:
        return snippet

    if isinstance(replace_all, str):
        should_replace_all = replace_all.strip().lower() in {"1", "true", "yes", "y"}
    else:
        should_replace_all = bool(replace_all)

    if should_replace_all:
        return snippet.replace(original, replacement)

    return snippet.replace(original, replacement, 1)


def _normalize_new_body(new_body: str) -> List[str]:
    dedented = textwrap.dedent(new_body).strip("\n")
    if not dedented:
        return []
    return dedented.splitlines()


@tool("rewrite_function")
def rewrite_function_body(snippet: str, function_name: str, new_body: str) -> str:
    """Rewrite the entire body of a function with the provided implementation."""
    if not function_name or not new_body.strip():
        return snippet

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

    normalized_body_lines = _normalize_new_body(new_body)
    if not normalized_body_lines:
        return snippet

    indented_body = [
        f"{indent}    {line}" if line.strip() else f"{indent}"
        for line in normalized_body_lines
    ]

    updated_lines = lines[:body_start_index] + indented_body + lines[body_end_index + 1 :]
    return "\n".join(updated_lines)


@tool("stub_function")
def stub_function_singleline(snippet: str) -> str:
    """Replace 'def name(...): pass' with a simple function stub."""
    result = re.sub(
        r"def\s+(\w+)\(.*\):\s*pass",
        r"def \1():\n    return None",
        snippet,
    )
    if result.endswith("\n"):
        result = result[:-1]
    return result


SYSTEM_PROMPT = (
    "You are Coder, an expert Python engineer tasked with identifying and repairing defects in user-provided code.\n"
    "1. Before using tools, enumerate all relevant edge cases (including hidden-test risks) and store them with record_edge_cases.\n"
    "2. After every tool invocation, call record_validation_checkpoint to explain what changed, which edge cases were addressed, and what remains.\n"
    "3. Use guarded_replace_token for any semantics-sensitive token edits and rewrite_function for structural updates involving new branches or heterogeneous handling.\n"
    "4. When working on get_second_items, ensure missing or None entries and heterogeneous iterables are handled gracefully; for reverse_string, consider unicode, empty strings, and whitespace preservation; for divide_elements, guard against division by zero, non-numeric inputs, and uneven lengths.\n"
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
