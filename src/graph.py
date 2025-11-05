import ast
import re
import textwrap

from langchain.agents import create_agent
from langchain_core.tools import tool



@tool("add_import")
def add_import_buggy(snippet: str, module: str) -> str:
	"""Insert an import at the top of the snippet."""
	result = f"import {module}\n" + snippet
	if result.endswith("\n"):
		result = result[:-1]
	return result


@tool("rename_symbol")
def rename_first_occurrence(snippet: str, old: str, new: str) -> str:
	"""Rename a symbol in the snippet without touching function or class signatures."""

	def identifier_in_signature(snippet_text: str, identifier: str) -> bool:
		try:
			tree = ast.parse(snippet_text)
		except SyntaxError:
			pass
		else:
			for node in ast.walk(tree):
				if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
					if node.name == identifier:
						return True
					args = node.args
					all_args = (
						getattr(args, "posonlyargs", [])
						+ args.args
						+ args.kwonlyargs
					)
					for arg in all_args:
						if arg.arg == identifier:
							return True
					if args.vararg and args.vararg.arg == identifier:
						return True
					if args.kwarg and args.kwarg.arg == identifier:
						return True
				if isinstance(node, ast.ClassDef) and node.name == identifier:
					return True

		func_pattern = re.compile(rf"^\s*(?:async\s+)?def\s+{re.escape(identifier)}\b", re.MULTILINE)
		if func_pattern.search(snippet_text):
			return True

		class_pattern = re.compile(rf"^\s*class\s+{re.escape(identifier)}\b", re.MULTILINE)
		if class_pattern.search(snippet_text):
			return True

		for match in re.finditer(r"^\s*(?:async\s+)?def\s+\w+\s*\(([^)]*)\)", snippet_text, re.MULTILINE):
			params_segment = match.group(1)
			for raw_param in params_segment.split(","):
				param = raw_param.strip()
				if not param or param in {"*", "**", "/"}:
					continue
				param_name = param.split(":", 1)[0]
				param_name = param_name.split("=", 1)[0]
				param_name = param_name.lstrip("*")
				if param_name == identifier:
					return True

		return False

	if not old:
		return snippet

	if identifier_in_signature(snippet, old):
		return snippet

	result = snippet.replace(old, new, 1)
	if result == snippet:
		if result.endswith("\n"):
			return result[:-1]
		return result

	if result.endswith("\n"):
		result = result[:-1]
	return result


@tool("add_guard_clause")
def add_guard_clause(snippet: str, function_name: str, guard: str) -> str:
	"""Insert a guard clause into the specified function while keeping its signature unchanged."""
	if not guard.strip():
		return snippet

	lines = snippet.splitlines()

	def find_insertion_with_ast():
		try:
			tree = ast.parse(snippet)
		except SyntaxError:
			return None

		for node in ast.walk(tree):
			if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
				def_line_index = node.lineno - 1
				if def_line_index < 0 or def_line_index >= len(lines):
					return None

				indent_match = re.match(r"^(\s*)", lines[def_line_index])
				indent = indent_match.group(1) if indent_match else ""
				body_indent = indent + " " * 4

				for body_line in lines[def_line_index + 1:]:
					if not body_line.strip():
						continue
					current_indent = re.match(r"^(\s*)", body_line).group(1)
					if len(current_indent) > len(indent):
						body_indent = current_indent
					break

				insert_index = def_line_index + 1

				if node.body:
					first_stmt = node.body[0]
					if (
						isinstance(first_stmt, ast.Expr)
						and isinstance(getattr(first_stmt, "value", None), ast.Constant)
						and isinstance(first_stmt.value.value, str)
					):
						doc_end_line = getattr(first_stmt, "end_lineno", first_stmt.lineno) - 1
						insert_index = min(doc_end_line + 1, len(lines))
						if insert_index < len(lines) and not lines[insert_index].strip():
							insert_index += 1
					else:
						insert_index = first_stmt.lineno - 1

				return insert_index, body_indent

		return None

	def find_insertion_textual():
		pattern = re.compile(rf"^(\s*)(?:async\s+)?def\s+{re.escape(function_name)}\b")
		for index, line in enumerate(lines):
			match = pattern.match(line)
			if not match:
				continue

			indent = match.group(1)
			body_indent = indent + " " * 4
			for body_line in lines[index + 1:]:
				if not body_line.strip():
					continue
				current_indent = re.match(r"^(\s*)", body_line).group(1)
				if len(current_indent) > len(indent):
					body_indent = current_indent
				break

			insert_index = index + 1

			temp_index = insert_index
			while temp_index < len(lines) and not lines[temp_index].strip():
				temp_index += 1

			if temp_index < len(lines):
				stripped = lines[temp_index].strip()
				if stripped.startswith(("'''", '"""')):
					quote = stripped[:3]
					doc_end = temp_index
					if stripped.count(quote) < 2:
						cursor = temp_index + 1
						while cursor < len(lines):
							if lines[cursor].strip().endswith(quote):
								doc_end = cursor
								break
							cursor += 1
						else:
							doc_end = len(lines) - 1
					insert_index = min(doc_end + 1, len(lines))
					if insert_index < len(lines) and not lines[insert_index].strip():
						insert_index += 1

			return insert_index, body_indent

		return None

	insertion = find_insertion_with_ast()
	if insertion is None:
		insertion = find_insertion_textual()

	if insertion is None:
		return snippet

	insert_at, body_indent = insertion
	insert_at = max(0, min(insert_at, len(lines)))

	dedented_guard = textwrap.dedent(guard).strip("\n")
	if not dedented_guard:
		return snippet

	formatted_guard_lines = []
	for guard_line in dedented_guard.splitlines():
		if guard_line.strip():
			formatted_guard_lines.append(f"{body_indent}{guard_line}")
		else:
			formatted_guard_lines.append(body_indent.rstrip())

	lines[insert_at:insert_at] = formatted_guard_lines

	result = "\n".join(lines)
	if result.endswith("\n"):
		result = result[:-1]
	return result


@tool("fix_indexing")
def bump_indices_off_by_one(snippet: str) -> str:
	"""Adjust numeric indices in list indexing."""
	def bump(match: re.Match) -> str:
		num = int(match.group(1))
		return f"[{num + 1}]"

	result = re.sub(r"\[(\d+)\]", bump, snippet)
	if result.endswith("\n"):
		result = result[:-1]
	return result


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
	"You are Coder. Your job is finding flaws in a user-glam code and fixing them using the tools that you have."
	" Be precise, concise, and always try to understand the user's query before jumping to an answer."
	" Preserve the original function and class names, their signatures, and existing call patterns unless the user explicitly instructs otherwise."
	" Avoid unnecessary renaming or structural changes and resist using tools when a change is not strictly required."
	" Prefer minimal, targeted edits, and leverage guard clauses or other safe adjustments instead of altering public APIs."
	" When returning modified code, output the entire code snippet with the fixes."
)


app = create_agent(
	model="openai:gpt-4o-mini",
	tools=[add_import_buggy, rename_first_occurrence, add_guard_clause, bump_indices_off_by_one, stub_function_singleline],
	system_prompt=SYSTEM_PROMPT,
)

