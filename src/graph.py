import io
import re
import tokenize
from typing import List, Tuple

from langchain.agents import create_agent
from langchain_core.tools import tool

MANUAL_REVIEW_PREFIX = "MANUAL REVIEW REQUIRED:"
_ENCODING_COMMENT_RE = re.compile(r"#.*coding[:=]\s*[-\w.]+")
_TRIPLE_QUOTE_PREFIXES = ("'''", '"""')
_INDEX_PATTERN = re.compile(r"\[(\d+)\]")
_STUB_PATTERN = re.compile(
    r"^(?P<indent>[ \t]*)def\s+(?P<name>\w+)\((?P<params>[^)]*)\):\s*pass\s*(?P<comment>#.*)?$",
    re.MULTILINE,
)


def _count_trailing_newlines(value: str) -> int:
    count = 0
    for char in reversed(value):
        if char == "\n":
            count += 1
        else:
            break
    return count


def _preserve_trailing_newlines(original: str, updated: str) -> str:
    trailing = _count_trailing_newlines(original)
    core = updated.rstrip("\n")
    if trailing:
        return core + ("\n" * trailing)
    return core


def _compose_tool_result(original: str, updated: str, warning: str | None = None) -> str:
    normalised = _preserve_trailing_newlines(original, updated)
    if warning:
        warning_text = f"{MANUAL_REVIEW_PREFIX} {warning.strip()}"
        if normalised:
            return f"{warning_text}\n\n{normalised}"
        return warning_text
    return normalised


def _find_import_insertion_index(lines: List[str]) -> int:
    idx = 0
    total = len(lines)

    if idx < total and lines[idx].startswith("#!"):
        idx += 1

    if idx < total and _ENCODING_COMMENT_RE.match(lines[idx]):
        idx += 1

    if idx < total:
        stripped = lines[idx].lstrip()
        for quote in _TRIPLE_QUOTE_PREFIXES:
            if stripped.startswith(quote):
                closing_on_same_line = quote in stripped[len(quote):]
                idx += 1
                if closing_on_same_line:
                    break
                while idx < total and quote not in lines[idx]:
                    idx += 1
                if idx < total:
                    idx += 1
                break

    while idx < total and lines[idx].strip() == "":
        idx += 1

    return idx


@tool("add_import")
def add_import(snippet: str, module: str) -> str:
    module_name = module.strip()
    if not module_name:
        return snippet

    import_line = f"import {module_name}"
    lines = snippet.splitlines()

    if any(line.strip() == import_line for line in lines):
        return snippet

    insert_at = _find_import_insertion_index(lines)
    lines.insert(insert_at, import_line)
    updated = "\n".join(lines)
    return _compose_tool_result(snippet, updated)


@tool("rename_symbol")
def rename_symbol(snippet: str, old: str, new: str) -> str:
    original_name = old.strip()
    replacement_name = new.strip()

    if not original_name or original_name == replacement_name:
        return snippet

    tokens = list(tokenize.generate_tokens(io.StringIO(snippet).readline))
    changed = False
    new_tokens: List[tokenize.TokenInfo] = []

    for token_info in tokens:
        if token_info.type == tokenize.NAME and token_info.string == original_name:
            token_info = token_info._replace(string=replacement_name)
            changed = True
        new_tokens.append(token_info)

    if not changed:
        return snippet

    updated = tokenize.untokenize(new_tokens)
    return _compose_tool_result(snippet, updated)


@tool("fix_indexing")
def fix_indexing(snippet: str) -> str:
    adjustments: List[Tuple[int, int]] = []

    def adjust(match: re.Match[str]) -> str:
        value = int(match.group(1))
        if value == 0:
            return match.group(0)
        new_value = max(value - 1, 0)
        adjustments.append((value, new_value))
        return f"[{new_value}]"

    updated = _INDEX_PATTERN.sub(adjust, snippet)

    if not adjustments:
        return snippet

    warning = None
    if len(adjustments) > 1:
        warning = "Multiple indexing adjustments were applied; please verify the result manually."

    return _compose_tool_result(snippet, updated, warning)


@tool("stub_function")
def stub_function(snippet: str) -> str:
    replacements: List[str] = []

    def replace(match: re.Match[str]) -> str:
        indent = match.group("indent") or ""
        name = match.group("name")
        params = match.group("params")
        replacements.append(name)
        signature = f"{indent}def {name}({params}):"
        body_indent = f"{indent}    "
        stub_body = f'{body_indent}raise NotImplementedError("TODO: implement {name}")'
        return f"{signature}\n{stub_body}"

    updated = _STUB_PATTERN.sub(replace, snippet)

    if not replacements:
        return snippet

    warning = None
    if len(replacements) > 1:
        warning = "Multiple function stubs were generated; please ensure each implementation is reviewed."

    return _compose_tool_result(snippet, updated, warning)


SYSTEM_PROMPT = (
    "You are Coder. Your job is finding flaws in a user-glam code and fixing them using the tools that you have. "
    "Always prefer the smallest possible change required to address the bug. "
    "Never introduce stylistic or unrelated edits. "
    "If you cannot safely fix the problem with the available tools, respond with "
    '"MANUAL REVIEW REQUIRED: <reason>" and describe the risk before ending your reply. '
    "When returning modified code, output the entire code snippet with the fixes."
)


app = create_agent(
    model="openai:gpt-4o-mini",
    tools=[add_import, rename_symbol, fix_indexing, stub_function],
    system_prompt=SYSTEM_PROMPT,
)
