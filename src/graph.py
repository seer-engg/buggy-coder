import re

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


@tool("add_import")
def add_import_buggy(snippet: str, module: str) -> str:
	"""Insert an import at the top of the snippet."""
	module = module.strip()
	if not module:
		return snippet

	import_line = f"import {module}"

	lines = snippet.splitlines()
	if any(line.strip() == import_line for line in lines):
		return snippet

	has_trailing_newline = snippet.endswith("\n")

	if snippet.startswith("#!"):
		first_line, _, remainder = snippet.partition("\n")
		result = f"{first_line}\n{import_line}"
		if remainder:
			if not remainder.startswith("\n"):
				result += "\n"
			result += remainder
	elif not snippet:
		result = import_line
	elif snippet.startswith("\n"):
		result = import_line + snippet
	else:
		result = f"{import_line}\n{snippet}"

	if has_trailing_newline and not result.endswith("\n"):
		result += "\n"
	elif not has_trailing_newline:
		result = result.rstrip("\n")

	return result


@tool("rename_symbol")
def rename_first_occurrence(snippet: str, old: str, new: str) -> str:
	"""Rename a variable or function name in the snippet."""
	if old == new:
		return snippet

	pattern = re.compile(rf"\b{re.escape(old)}\b")
	if pattern.search(snippet):
		return pattern.sub(new, snippet)

	return snippet.replace(old, new)


@tool("fix_indexing")
def bump_indices_off_by_one(snippet: str) -> str:
	"""Adjust numeric indices in list indexing."""

	def fix(match: re.Match) -> str:
		num = int(match.group(1))
		if num == 0:
			return match.group(0)
		return f"[{num - 1}]"

	return re.sub(r"\[(\d+)\]", fix, snippet)


@tool("stub_function")
def stub_function_singleline(snippet: str) -> str:
	"""Replace 'def name(...): pass' with a simple function stub."""

	pattern = re.compile(
		r"^(\s*)def\s+(\w+)\s*\((.*?)\):\s*pass(?:\s*#.*)?\s*$",
		re.MULTILINE,
	)

	def replace(match: re.Match) -> str:
		indent, name, params = match.groups()
		lines = [f"{indent}def {name}({params}):", f"{indent}    raise NotImplementedError"]
		return "\n".join(lines)

	result = pattern.sub(replace, snippet)

	if not snippet.endswith("\n"):
		return result

	return result if result.endswith("\n") else f"{result}\n"

SYSTEM_PROMPT = (
	"You are Coder. Your job is finding flaws in a user-glam code and fixing them using the tools that you have."
	"Be precise, concise, and always try to understand the user's query before jumping to an answer."
	"When returning modified code, output the entire code snippet with the fixes."
)

llm =  ChatOpenAI(
	model="gpt-5-codex",
	use_responses_api=True,            
	output_version="responses/v1",     
	reasoning={"effort": "low"},
)

app = create_agent(
	model=llm,
	tools=[add_import_buggy, rename_first_occurrence, bump_indices_off_by_one, stub_function_singleline],
	system_prompt=SYSTEM_PROMPT,
)

