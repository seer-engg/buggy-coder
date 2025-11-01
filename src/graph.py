import ast
import io
import re
import tokenize
from typing import List, Optional, Tuple

from langchain.agents import create_agent
from langchain_core.tools import tool


def _split_lines_preserve(snippet: str) -> Tuple[List[str], bool]:
	"""Split code into lines while tracking the trailing newline."""
	ends_with_newline = snippet.endswith("\n")
	return snippet.splitlines(), ends_with_newline


def _reconstruct_from_lines(lines: List[str], end_with_newline: bool) -> str:
	"""Join code lines and optionally ensure a trailing newline."""
	joined = "\n".join(lines)
	if end_with_newline and (joined and not joined.endswith("\n")):
		joined += "\n"
	elif not end_with_newline and joined.endswith("\n"):
		joined = joined.rstrip("\n")
	return joined


def _line_offset_map(snippet: str) -> List[int]:
	"""Create a map from 1-based line numbers to absolute string offsets."""
	offsets: List[int] = []
	cursor = 0
	for line in snippet.splitlines(True):
		offsets.append(cursor)
		cursor += len(line)
	return offsets


def _compute_import_insertion_index(lines: List[str], snippet: str) -> int:
	"""Determine the best insertion index for a new import statement."""
	if not lines:
		return 0

	index = 0
	# Preserve a leading shebang line.
	if lines and lines[0].startswith("#!"):
		index = 1

	encoding_pattern = re.compile(r"^#.*coding[:=]")
	while index < len(lines) and encoding_pattern.match(lines[index]):
		index += 1

	# Try to place imports after a module docstring if present.
	try:
		module_ast = ast.parse(snippet)
	except SyntaxError:
		module_ast = None  # Fall back to heuristic below.

	if module_ast and module_ast.body:
		first_stmt = module_ast.body[0]
		if (
			isinstance(first_stmt, ast.Expr)
			and isinstance(first_stmt.value, ast.Constant)
			and isinstance(first_stmt.value.value, str)
		):
			docstring_end = getattr(first_stmt, "end_lineno", first_stmt.lineno)
			index = max(index, docstring_end)

	# Skip over any leading blank lines after docstring/shebang.
	while index < len(lines) and not lines[index].strip():
		index += 1

	# If there is an existing import block, append to it.
	import_block_start = None
	import_block_end = None
	for position in range(index, len(lines)):
		stripped = lines[position].strip()
		if not stripped:
			if import_block_start is not None:
				import_block_end = position
				break
			continue
		if stripped.startswith(("import ", "from ")):
			if import_block_start is None:
				import_block_start = position
			import_block_end = position + 1
		else:
			break

	if import_block_end is not None:
		return import_block_end

	return index


def _ensure_surrounding_blank_lines(lines: List[str], index: int) -> None:
	"""Ensure there is spacing around inserted imports when needed."""
	if index > 0 and lines[index - 1].strip() and not lines[index - 1].strip().startswith(("import ", "from ")):
		lines.insert(index, "")
		index += 1

	if index + 1 < len(lines):
		next_line = lines[index + 1].strip()
		if next_line and not next_line.startswith(("import ", "from ")):
			lines.insert(index + 1, "")


@tool("add_import")
def add_import(snippet: str, module: str) -> str:
	"""Insert an exact import statement at the top of the snippet without corrupting code."""
	module = module.strip()
	if not module:
		return snippet

	if module.startswith(("import ", "from ")):
		import_line = module
	else:
		import_line = f"import {module}"

	lines, ends_with_newline = _split_lines_preserve(snippet)

	normalised_import = re.sub(r"\s+", " ", import_line.strip())
	for existing in lines:
		if re.sub(r"\s+", " ", existing.strip()) == normalised_import:
			return snippet

	insert_index = _compute_import_insertion_index(lines, snippet)
	lines.insert(insert_index, import_line)
	_ensure_surrounding_blank_lines(lines, insert_index)
	return _reconstruct_from_lines(lines, ends_with_newline)


@tool("rename_symbol")
def rename_symbol(snippet: str, old: str, new: str, *, include_strings: bool = False) -> str:
	"""Rename identifiers precisely without touching substrings or altering formatting."""
	if old == new:
		return snippet

	reader = io.StringIO(snippet).readline
	tokens = list(tokenize.generate_tokens(reader))
	updated_tokens = []

	for token in tokens:
		if token.type == tokenize.NAME and token.string == old:
			updated_tokens.append(token._replace(string=new))
		elif include_strings and token.type == tokenize.STRING and old in token.string:
			updated_tokens.append(token._replace(string=token.string.replace(old, new)))
		else:
			updated_tokens.append(token)

	result = tokenize.untokenize(updated_tokens)
	if not snippet.endswith("\n") and result.endswith("\n"):
		result = result[:-1]
	return result


@tool("fix_indexing")
def fix_indexing(
	snippet: str,
	old_value: Optional[int] = None,
	new_value: Optional[int] = None,
	*,
	offset: Optional[int] = None,
	occurrence: int = 1,
) -> str:
	"""Safely adjust a numeric index in square brackets.

	Arguments:
		snippet: The original code snippet.
		old_value: Existing numeric literal to update. If omitted, the nth match is used.
		new_value: Replacement index value. If omitted, `offset` is applied to the matched value.
		offset: Numeric offset to apply to the resolved index.
		occurrence: 1-based occurrence number to modify when multiple indices exist.
	"""
	if occurrence < 1:
		raise ValueError("occurrence must be >= 1")

	matches = list(re.finditer(r"\[(\d+)\]", snippet))
	if not matches:
		return snippet

	filtered: List[re.Match[str]]
	if old_value is not None:
		filtered = [m for m in matches if int(m.group(1)) == old_value]
		if not filtered:
			return snippet
	else:
		filtered = matches

	index = min(len(filtered), occurrence) - 1
	if index < 0:
		return snippet

	target = filtered[index]
	current_value = int(target.group(1))

	if new_value is None:
		if offset is None:
			raise ValueError("Provide either new_value or offset when adjusting indices.")
		new_value = current_value + offset

	if new_value < 0:
		raise ValueError("Indices cannot be negative.")

	start, end = target.span(1)
	result = snippet[:start] + str(new_value) + snippet[end:]
	return result


def _default_stub_body(return_value: Optional[str]) -> List[str]:
	if return_value is not None:
		body = f"return {return_value}" if return_value.strip() else "return None"
	else:
		body = "raise NotImplementedError()"
	return body.splitlines()


def _build_stub_lines(
	snippet: str,
	function_node: ast.AST,
	pass_node: ast.stmt,
	return_value: Optional[str],
) -> Tuple[int, int, str]:
	lines = snippet.splitlines()
	snippet_offsets = _line_offset_map(snippet)

	function_line = lines[function_node.lineno - 1]
	function_indent = re.match(r"[ \t]*", function_line).group()

	pass_line = lines[pass_node.lineno - 1]
	pass_leading = re.match(r"[ \t]*", pass_line).group()

	if pass_leading:
		body_indent = pass_leading
	else:
		indent_unit = "\t" if "\t" in function_indent else "    "
		body_indent = function_indent + indent_unit

	stub_lines = [body_indent + line for line in _default_stub_body(return_value)]

	inline = pass_node.lineno == function_node.lineno
	if inline:
		start_offset = snippet_offsets[pass_node.lineno - 1] + pass_node.col_offset
		# Include any trailing whitespace between the colon and `pass`.
		while start_offset > snippet_offsets[pass_node.lineno - 1] and snippet[start_offset - 1] in " \t":
			start_offset -= 1
		end_offset = snippet_offsets[pass_node.lineno - 1] + getattr(pass_node, "end_col_offset", pass_node.col_offset + 4)
		replacement = "\n" + "\n".join(stub_lines)
	else:
		start_offset = snippet_offsets[pass_node.lineno - 1]
		if pass_node.lineno < len(snippet_offsets):
			end_offset = snippet_offsets[pass_node.lineno]
		else:
			end_offset = len(snippet)
		segment = snippet[start_offset:end_offset]
		replacement = "\n".join(stub_lines)
		if segment.endswith("\n") and not replacement.endswith("\n"):
			replacement += "\n"

	return start_offset, end_offset, replacement


def _apply_replacements(snippet: str, replacements: List[Tuple[int, int, str]]) -> str:
	if not replacements:
		return snippet

	replacements_sorted = sorted(replacements, key=lambda item: item[0], reverse=True)
	modified = snippet
	for start, end, replacement in replacements_sorted:
		modified = modified[:start] + replacement + modified[end:]
	return modified


@tool("stub_function")
def stub_function(snippet: str, return_value: Optional[str] = None) -> str:
	"""Turn `pass`-only function bodies into working stubs that preserve the signature."""
	try:
		tree = ast.parse(snippet)
	except SyntaxError:
		pattern = re.compile(r"(?m)(^[ \t]*def\s+\w+\(.*?\):)(?:[ \t]*pass(?:\s*#.*)?)$")

		def repl(match: re.Match[str]) -> str:
			def_line = match.group(1)
			indent = re.match(r"[ \t]*", def_line).group()
			indent_unit = "\t" if "\t" in def_line else "    "
			body_indent = indent + indent_unit
			body_lines = _default_stub_body(return_value)
			stub = "\n".join(body_indent + line for line in body_lines)
			return f"{def_line}\n{stub}"

		return pattern.sub(repl, snippet)

	replacements: List[Tuple[int, int, str]] = []
	for node in ast.walk(tree):
		if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
			body = list(node.body)
			if not body:
				continue

			if (
				isinstance(body[0], ast.Expr)
				and isinstance(body[0].value, ast.Constant)
				and isinstance(body[0].value.value, str)
			):
				effective_body = body[1:]
			else:
				effective_body = body

			if len(effective_body) != 1 or not isinstance(effective_body[0], ast.Pass):
				continue

			replacements.append(_build_stub_lines(snippet, node, effective_body[0], return_value))

	return _apply_replacements(snippet, replacements)


SYSTEM_PROMPT = (
	"You are Coder. Your job is finding flaws in a user-glam code and fixing them using the tools that you have."
	"Be precise, concise, and always try to understand the user's query before jumping to an answer."
	"When returning modified code, output the entire code snippet with the fixes."
)


app = create_agent(
	model="openai:gpt-4o-mini",
	tools=[add_import, rename_symbol, fix_indexing, stub_function],
	system_prompt=SYSTEM_PROMPT,
)

