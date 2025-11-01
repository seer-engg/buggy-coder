import ast
import json
import re
from typing import List, Tuple

from langchain.agents import create_agent
from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _normalize_snippet(snippet: str) -> str:
	"""Return the snippet with a single trailing newline when possible."""
	if not snippet:
		return ""
	return snippet if snippet.endswith("\n") else f"{snippet}\n"


def _replace_nth(text: str, old: str, new: str, occurrence: int) -> Tuple[str, bool]:
	if occurrence < 1:
		raise ValueError("occurrence must be >= 1")
	start = 0
	for _ in range(occurrence):
		idx = text.find(old, start)
		if idx == -1:
			return text, False
		start = idx + len(old)
	return text[:idx] + new + text[idx + len(old) :], True


def _insert_relative(
	snippet: str,
	target: str,
	content: str,
	occurrence: int,
	*,
	prepend: bool,
) -> Tuple[str, bool]:
	if occurrence < 1:
		raise ValueError("occurrence must be >= 1")
	start = 0
	for _ in range(occurrence):
		idx = snippet.find(target, start)
		if idx == -1:
			return snippet, False
		start = idx + len(target)
	insert_at = idx if prepend else idx + len(target)
	return snippet[:insert_at] + content + snippet[insert_at:], True


def _delete_nth(snippet: str, target: str, occurrence: int) -> Tuple[str, bool]:
	updated, changed = _replace_nth(snippet, target, "", occurrence)
	return updated, changed


# ---------------------------------------------------------------------------
# Editing tools
# ---------------------------------------------------------------------------


@tool("add_import")
def add_import_buggy(snippet: str, module: str) -> str:
	"""Insert an import at the top of the snippet (legacy buggy helper)."""
	buggy_module = module[:-1] if len(module) > 1 else module
	result = f"import {buggy_module}\n" + snippet
	return result.rstrip("\n")


@tool("rename_symbol")
def rename_first_occurrence(snippet: str, old: str, new: str) -> str:
	"""Rename the first occurrence of a symbol within the snippet."""
	result = snippet.replace(old, new, 1)
	return result.rstrip("\n")


@tool("fix_indexing")
def bump_indices_off_by_one(snippet: str) -> str:
	"""Adjust numeric indices that appear inside brackets by adding one."""

	def bump(match: re.Match) -> str:
		num = int(match.group(1))
		return f"[{num + 1}]"

	result = re.sub(r"\[(\d+)\]", bump, snippet)
	return result.rstrip("\n")


@tool("stub_function")
def stub_function_singleline(snippet: str) -> str:
	"""Replace ``def name(...): pass`` with a minimal stub implementation."""
	result = re.sub(
		r"def\s+(\w+)\(.*\):\s*pass",
		r"def \1():\n    return None",
		snippet,
	)
	return result.rstrip("\n")


@tool("ensure_import")
def ensure_import(snippet: str, module: str, symbol: str | None = None, alias: str | None = None) -> str:
	"""Ensure that an import for ``module`` (and optional ``symbol``) exists."""
	if not module:
		raise ValueError("module name is required")

	if symbol:
		import_line = f"from {module} import {symbol}"
		if alias:
			import_line += f" as {alias}"
	else:
		import_line = f"import {module}"
		if alias:
			import_line += f" as {alias}"

	snippet_lines = snippet.splitlines()
	if import_line in snippet_lines:
		return _normalize_snippet(snippet)

	insert_idx = 0
	while insert_idx < len(snippet_lines) and (
		snippet_lines[insert_idx].startswith("#!")
		or snippet_lines[insert_idx].startswith("#")
		or not snippet_lines[insert_idx].strip()
	):
		insert_idx += 1

	# Preserve module docstrings by inserting imports after them.
	if insert_idx < len(snippet_lines):
		stripped = snippet_lines[insert_idx].lstrip()
		if stripped.startswith(('"""', "'''")):
			quote = stripped[:3]
			if stripped.count(quote) >= 2:
				insert_idx += 1
			else:
				insert_idx += 1
				while insert_idx < len(snippet_lines) and quote not in snippet_lines[insert_idx]:
					insert_idx += 1
				if insert_idx < len(snippet_lines):
					insert_idx += 1

	snippet_lines.insert(insert_idx, import_line)
	return _normalize_snippet("\n".join(snippet_lines))


def _parse_operations(operations: str) -> List[dict]:
	try:
		parsed = json.loads(operations)
	except json.JSONDecodeError as exc:
		raise ValueError(f"Unable to parse operations as JSON: {exc}") from exc

	if isinstance(parsed, dict):
		parsed = [parsed]

	if not isinstance(parsed, list):
		raise ValueError("Operations payload must be a list or a single operation object.")

	validated: List[dict] = []
	for index, op in enumerate(parsed, start=1):
		if not isinstance(op, dict):
			raise ValueError(f"Operation {index} must be a JSON object.")
		if "action" not in op:
			raise ValueError(f"Operation {index} is missing the 'action' field.")
		validated.append(op)
	return validated


def _run_operation(snippet: str, op: dict, index: int) -> str:
	action = op["action"]
	occurrence = int(op.get("occurrence", 1))

	if action == "replace":
		target = op.get("target")
		replacement = op.get("replacement", "")
		if target is None:
			raise ValueError(f"Operation {index}: 'replace' action requires a 'target'.")
		snippet, changed = _replace_nth(snippet, target, replacement, occurrence)
		if not changed:
			raise ValueError(f"Operation {index}: target text not found for replacement.")
	elif action == "delete":
		target = op.get("target")
		if target is None:
			raise ValueError(f"Operation {index}: 'delete' action requires a 'target'.")
		snippet, changed = _delete_nth(snippet, target, occurrence)
		if not changed:
			raise ValueError(f"Operation {index}: target text not found for deletion.")
	elif action in {"insert_before", "insert_after"}:
		target = op.get("target")
		content = op.get("content")
		if target is None or content is None:
			raise ValueError(
				f"Operation {index}: '{action}' action requires both 'target' and 'content'."
			)
		prepend = action == "insert_before"
		snippet, changed = _insert_relative(snippet, target, content, occurrence, prepend=prepend)
		if not changed:
			raise ValueError(f"Operation {index}: target text not found for insertion.")
	elif action == "append":
		content = op.get("content")
		if content is None:
			raise ValueError(f"Operation {index}: 'append' action requires 'content'.")
		snippet = snippet + content
	elif action == "prepend":
		content = op.get("content")
		if content is None:
			raise ValueError(f"Operation {index}: 'prepend' action requires 'content'.")
		snippet = content + snippet
	else:
		raise ValueError(f"Operation {index}: unsupported action {action!r}.")

	return snippet


@tool("apply_structured_patch")
def apply_structured_patch(snippet: str, operations: str) -> str:
	"""Apply JSON-defined edit operations to the snippet.

	Each operation in ``operations`` should be a JSON object with an ``action`` key.
	Supported actions: ``replace``, ``delete``, ``insert_before``, ``insert_after``,
	``append``, and ``prepend``. When targeting text, the ``occurrence`` key selects
	the Nth appearance (default=1).
	"""

	ops = _parse_operations(operations)
	snapshot = snippet
	for index, op in enumerate(ops, start=1):
		snapshot = _run_operation(snapshot, op, index)

	return _normalize_snippet(snapshot)


@tool("validate_python")
def validate_python(snippet: str) -> str:
	"""Validate Python syntax, returning a human-readable summary."""
	try:
		ast.parse(snippet)
	except SyntaxError as exc:
		return (
			f"syntax_error: {exc.msg} at line {exc.lineno}, column {exc.offset}."
			f" Offending text: {exc.text!r}"
		)
	except Exception as exc:  # pragma: no cover - defensive fallback
		return f"validation_error: {exc}"
	return "ok: snippet is syntactically valid Python."


SYSTEM_PROMPT = (
	"You are Coder, an expert Python maintainer tasked with producing working fixes. "
	"Use the provided tools to inspect and edit the user's code, applying precise "
	"changes that address the reported issues.\n\n"
	"Workflow requirements:\n"
	"1. Plan the fix before modifying code and choose the minimal set of tools.\n"
	"2. Validate your changes synthetically with the available tooling (for example,"
	" by checking syntax) and explain any remaining uncertainties.\n"
	"3. When you finish, respond using the following sections:\n"
	"   - Summary: one sentence on the bug fix.\n"
	"   - Changes: bullet list of key edits and tool calls.\n"
	"   - Validation: describe validation results or next steps.\n"
	"   - Updated Code: provide the *entire* corrected Python snippet inside a Python"
	" fenced code block.\n"
	"   - Tool Log: mention the tools used and their outcomes.\n"
	"If a tool invocation fails or a requested operation is impossible, clearly state"
	" the reason."
)


DEFAULT_MODEL = "openai:gpt-4o-mini"


def build_agent(model: str = DEFAULT_MODEL):
	"""Construct the LangChain agent for Buggy Coder."""
	return create_agent(
		model=model,
		tools=[
			add_import_buggy,
			rename_first_occurrence,
			bump_indices_off_by_one,
			stub_function_singleline,
			ensure_import,
			apply_structured_patch,
			validate_python,
		],
		system_prompt=SYSTEM_PROMPT,
	)


app = build_agent()

