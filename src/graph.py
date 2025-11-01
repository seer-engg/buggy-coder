import ast
import io
import re
import tokenize

from langchain.agents import create_agent
from langchain_core.tools import tool



def _docstring_end_line(snippet: str, tree: ast.AST | None) -> int | None:
	"""Return the line number where the module docstring ends (1-indexed)."""
	if tree is None:
		return None

	if not getattr(tree, "body", None):
		return None

	first_stmt = tree.body[0]
	if not isinstance(first_stmt, ast.Expr):
		return None

	value = getattr(first_stmt, "value", None)
	if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
		return None

	end_lineno = getattr(first_stmt, "end_lineno", None)
	if end_lineno is not None:
		return end_lineno

	# Fallback: estimate based on docstring content
	docstring = value.value
	return first_stmt.lineno + docstring.count("\n")


@tool("add_import")
def add_import_buggy(snippet: str, module: str) -> str:
	"""Insert an import at the appropriate location within the snippet."""
	import_statement = f"import {module}"

	try:
		tree = ast.parse(snippet)
	except SyntaxError:
		tree = None

	if tree is not None:
		for node in ast.walk(tree):
			if isinstance(node, ast.Import):
				for alias in node.names:
					if alias.name == module:
						return snippet
			elif isinstance(node, ast.ImportFrom):
				if node.module == module:
					return snippet
	else:
		if re.search(rf"^\s*import\s+{re.escape(module)}(?:\s|$)", snippet, flags=re.MULTILINE):
			return snippet
		if re.search(rf"^\s*from\s+{re.escape(module)}\s+import\b", snippet, flags=re.MULTILINE):
			return snippet

	if not snippet:
		return f"{import_statement}\n"

	lines = snippet.splitlines()
	had_trailing_newline = snippet.endswith("\n")

	def _is_shebang_or_encoding(line: str) -> bool:
		return line.startswith("#!") or bool(re.match(r"^#.*coding[:=]", line))

	insert_idx = 0
	while insert_idx < len(lines) and _is_shebang_or_encoding(lines[insert_idx]):
		insert_idx += 1

	docstring_end = _docstring_end_line(snippet, tree)
	if docstring_end is not None:
		insert_idx = max(insert_idx, docstring_end)

	while insert_idx < len(lines) and lines[insert_idx].strip() == "":
		insert_idx += 1

	# Try to place the new import alongside existing imports
	import_block_start = None
	for idx in range(insert_idx, len(lines)):
		stripped = lines[idx].strip()
		if not stripped or stripped.startswith("#"):
			continue
		if stripped.startswith("import ") or stripped.startswith("from "):
			import_block_start = idx
			break
		break

	if import_block_start is not None:
		last_import_idx = import_block_start - 1
		idx = import_block_start
		while idx < len(lines):
			stripped = lines[idx].strip()
			if stripped.startswith("import ") or stripped.startswith("from "):
				last_import_idx = idx
				idx += 1
				continue
			if not stripped or stripped.startswith("#"):
				idx += 1
				continue
			break
		insert_idx = last_import_idx + 1 if last_import_idx >= import_block_start else import_block_start

	if docstring_end is not None and insert_idx == docstring_end and insert_idx >= 1:
		if lines[insert_idx - 1].strip():
			lines.insert(insert_idx, "")
			insert_idx += 1

	lines.insert(insert_idx, import_statement)

	result = "\n".join(lines)
	if had_trailing_newline:
		result += "\n"

	return result


@tool("rename_symbol")
def rename_first_occurrence(snippet: str, old: str, new: str) -> str:
	"""Rename occurrences of a symbol in the snippet while preserving formatting."""
	if old == new:
		return snippet

	try:
		tokens = list(tokenize.generate_tokens(io.StringIO(snippet).readline))
	except (tokenize.TokenError, IndentationError, SyntaxError):
		pattern = re.compile(rf"\b{re.escape(old)}\b")
		return re.sub(pattern, new, snippet)

	had_trailing_newline = snippet.endswith("\n")
	changed = False
	updated_tokens: list[tuple[int, str]] = []

	for tok_type, tok_string, *_ in tokens:
		if tok_type == tokenize.NAME and tok_string == old:
			tok_string = new
			changed = True
		updated_tokens.append((tok_type, tok_string))

	if not changed:
		return snippet

	result = tokenize.untokenize(updated_tokens)
	if not had_trailing_newline and result.endswith("\n"):
		result = result[:-1]

	return result


@tool("fix_indexing")
def bump_indices_off_by_one(snippet: str) -> str:
	"""Adjust numeric indices in list indexing by moving from 1-based to 0-based access."""

	pattern = re.compile(r"\[(\s*)(\d+)(\s*)\]")

	def adjust(match: re.Match[str]) -> str:
		value = int(match.group(2))
		if value == 0:
			return match.group(0)
		return f"[{match.group(1)}{value - 1}{match.group(3)}]"

	updated, _ = pattern.subn(adjust, snippet, count=1)
	return updated


@tool("stub_function")
def stub_function_singleline(snippet: str) -> str:
	"""Replace 'def name(...): pass' with a simple function stub."""
	return re.sub(
		r"def\s+(\w+)\(.*\):\s*pass",
		r"def \1():\n    return None",
		snippet,
	)

SYSTEM_PROMPT = (
	"You are Coder. Your job is finding flaws in a user-glam code and fixing them using the tools that you have."
	"Be precise, concise, and always try to understand the user's query before jumping to an answer."
	"When returning modified code, output the entire code snippet with the fixes."
)


app = create_agent(
	model="openai:gpt-4o-mini",
	tools=[add_import_buggy, rename_first_occurrence, bump_indices_off_by_one, stub_function_singleline],
	system_prompt=SYSTEM_PROMPT,
)

