import re

from langchain.agents import create_agent
from langchain_core.tools import tool



@tool("add_import")
def add_import(snippet: str, module: str) -> str:
	"""Insert an import at the top of the snippet."""
	module_name = module.strip()
	if not module_name:
		raise ValueError("Module name must be non-empty.")

	import_line = f"import {module_name}\n"
	return import_line + snippet


@tool("rename_symbol")
def rename_symbol(snippet: str, old: str, new: str) -> str:
	"""Rename all occurrences of an identifier while keeping formatting intact."""
	if not old:
		raise ValueError("Old symbol must be non-empty.")

	if old == new:
		return snippet

	pattern = re.compile(rf"\b{re.escape(old)}\b")
	result, _ = pattern.subn(new, snippet)
	return result


@tool("fix_indexing")
def fix_indexing(snippet: str) -> str:
	"""Attempt to correct common off-by-one mistakes in list indexing."""

	index_pattern = re.compile(r"\[(.*?)\]")

	def adjust(match: re.Match) -> str:
		expr = match.group(1)
		stripped = expr.strip()

		if re.fullmatch(r"-?\d+", stripped):
			num = int(stripped)
			if num > 0:
				num -= 1
			return f"[{num}]"

		len_plus_match = re.fullmatch(r"len\((.+)\)\s*\+\s*1", stripped)
		if len_plus_match:
			inner = len_plus_match.group(1)
			return f"[len({inner})]"

		len_match = re.fullmatch(r"len\((.+)\)", stripped)
		if len_match:
			inner = len_match.group(1)
			return f"[len({inner}) - 1]"

		if re.fullmatch(r"(.+?)\s*\+\s*1", stripped):
			base = re.sub(r"\s*\+\s*1\s*$", "", stripped)
			return f"[{base}]"

		return match.group(0)

	return index_pattern.sub(adjust, snippet)


@tool("stub_function")
def stub_function(snippet: str) -> str:
	"""Replace 'def name(...): pass' with a stub that preserves the signature and indentation."""

	pattern = re.compile(
		r"^(?P<indent>[ \t]*)def\s+(?P<name>\w+)\((?P<params>[^)]*)\):\s*pass\s*$",
		re.MULTILINE,
	)

	def replace(match: re.Match) -> str:
		indent = match.group("indent")
		name = match.group("name")
		params = match.group("params")
		signature = f"{indent}def {name}({params}):"
		body = f"{indent}    raise NotImplementedError"
		return "\n".join([signature, body])

	return pattern.sub(replace, snippet)

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

