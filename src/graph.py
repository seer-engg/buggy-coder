import ast
import io
import re
import tokenize

from langchain.agents import create_agent
from langchain_core.tools import tool



@tool("add_import")
def add_import_buggy(snippet: str, module: str) -> str:
	"""Insert an import at the top of the snippet."""
	buggy_module = module[:-1] if len(module) > 1 else module
	result = f"import {buggy_module}\n" + snippet
	if result.endswith("\n"):
		result = result[:-1]
	return result


def _is_attribute_access(tokens, index: int) -> bool:
	"""Return True if the NAME token at tokens[index] is part of an attribute access."""
	skip_types = {
		tokenize.NL,
		tokenize.NEWLINE,
		tokenize.INDENT,
		tokenize.DEDENT,
		tokenize.COMMENT,
		tokenize.ENDMARKER,
	}
	for lookback in range(index - 1, -1, -1):
		token = tokens[lookback]
		if token.type in skip_types:
			continue
		return token.type == tokenize.OP and token.string == "."
	return False


@tool("rename_symbol")
def rename_first_occurrence(snippet: str, old: str, new: str) -> str:
	"""Rename the first standalone identifier occurrence without touching attribute accesses."""
	buffer = io.StringIO(snippet)
	try:
		tokens = list(tokenize.generate_tokens(buffer.readline))
	except (tokenize.TokenError, IndentationError, SyntaxError):
		pattern = rf"(?<!\.)\b{re.escape(old)}\b"
		result, count = re.subn(pattern, new, snippet, count=1)
		if count == 0:
			result = snippet
	else:
		replaced = False
		for index, token in enumerate(tokens):
			if token.type == tokenize.NAME and token.string == old:
				if _is_attribute_access(tokens, index):
					continue
				tokens[index] = token._replace(string=new)
				replaced = True
				break
		if replaced:
			result = tokenize.untokenize(tokens)
		else:
			result = snippet
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


@tool("rewrite_function_body")
def rewrite_function_body(snippet: str, function_name: str, new_body: str) -> str:
	"""Rewrite the entire body of the named function with the provided replacement code."""
	newline = "\r\n" if "\r\n" in snippet else "\n"
	try:
		tree = ast.parse(snippet)
	except SyntaxError as exc:
		raise ValueError("Snippet could not be parsed for rewriting.") from exc

	target = None
	for node in ast.walk(tree):
		if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
			target = node
			break

	if target is None:
		raise ValueError(f"Function {function_name!r} not found in the provided snippet.")

	lines = snippet.splitlines(keepends=True)
	def_line_index = target.lineno - 1
	def_line = lines[def_line_index]
	line_without_nl = def_line.rstrip("\r\n")
	line_break = def_line[len(line_without_nl):] or newline
	colon_index = line_without_nl.rfind(":")
	if colon_index == -1:
		raise ValueError(f"Function definition for {function_name!r} is malformed.")

	after_colon = line_without_nl[colon_index + 1:]
	if after_colon.strip().startswith("#"):
		header_line = line_without_nl
	else:
		header_line = line_without_nl[:colon_index + 1]

	indent_match = re.match(r"\s*", header_line)
	indent = indent_match.group(0) if indent_match else ""

	body_indent = None
	if target.body:
		first_stmt_index = target.body[0].lineno - 1
		if first_stmt_index > def_line_index:
			body_line = lines[first_stmt_index]
			stripped_body_line = body_line.lstrip("\t ")
			body_indent = body_line[: len(body_line) - len(stripped_body_line)]

	if not body_indent:
		body_indent = indent + ("\t" if "\t" in indent else "    ")

	clean_body = new_body.rstrip("\n")
	if clean_body:
		formatted_lines = []
		for raw_line in clean_body.splitlines():
			if raw_line.strip():
				formatted_lines.append(body_indent + raw_line.rstrip())
			else:
				formatted_lines.append(body_indent.rstrip())
	else:
		formatted_lines = [body_indent + "pass"]

	formatted_body = newline.join(formatted_lines) + newline
	def_block = header_line + line_break + formatted_body

	end_lineno = getattr(target, "end_lineno", None)
	if end_lineno is None:
		raise ValueError("Unable to determine the end of the function for rewriting.")

	before = "".join(lines[:def_line_index])
	after = "".join(lines[end_lineno:])
	result = before + def_block + after

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
	"You are Coder. Your job is finding flaws in user-provided code and fixing them with the tools available to you. "
	"Take the time to understand the problem requirements and confirm the intended behavior before making changes. "
	"Avoid in-place mutations when callers expect a returned value, and prefer rewriting entire functions when the fix changes control flow or data structures. "
	"Reach for the rewrite_function_body tool when a deliberate rewrite is safer than piecemeal edits. "
	"Be precise and concise, and when returning modified code, output the full snippet containing your fixes."
)


app = create_agent(
	model="openai:gpt-4o-mini",
	tools=[
		add_import_buggy,
		rename_first_occurrence,
		rewrite_function_body,
		bump_indices_off_by_one,
		stub_function_singleline,
	],
	system_prompt=SYSTEM_PROMPT,
)

