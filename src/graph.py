import re
from typing import Any, Iterable

from langchain.agents import create_agent
from langchain.agents.middleware import AgentState, after_model
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.runtime import Runtime


_MODULE_NAME_PATTERN = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")
_CRITICAL_IMPORTS = {"unittest"}


def _extract_top_level_symbols(snippet: str) -> set[str]:
	return {
		match.group(2)
		for match in re.finditer(r"^(def|class)\s+([A-Za-z_]\w*)", snippet, flags=re.MULTILINE)
	}


def _extract_import_modules(snippet: str) -> set[str]:
	modules: set[str] = set()
	for raw_line in snippet.splitlines():
		code = raw_line.split("#", 1)[0].strip()
		if not code:
			continue
		if code.startswith("import "):
			remainder = code[len("import "):]
			for module_part in remainder.split(","):
				module_name = module_part.strip().split(" as ", 1)[0].strip()
				if module_name:
					modules.add(module_name)
		elif code.startswith("from "):
			parts = code.split()
			if len(parts) >= 4 and parts[2] == "import":
				modules.add(parts[1])
	return modules


def _required_entities_from(snippet: str) -> tuple[set[str], set[str]]:
	symbols = _extract_top_level_symbols(snippet)
	imports = _extract_import_modules(snippet)
	return symbols, imports & _CRITICAL_IMPORTS


def _validate_post_edit(before: str, after: str) -> None:
	required_symbols, required_imports = _required_entities_from(before)
	if not required_symbols and not required_imports:
		return
	updated_symbols, updated_imports = _required_entities_from(after)
	missing_symbols = sorted(required_symbols - updated_symbols)
	missing_imports = sorted(required_imports - updated_imports)
	if missing_symbols or missing_imports:
		missing_parts: list[str] = []
		if missing_symbols:
			missing_parts.append(f"symbols: {', '.join(missing_symbols)}")
		if missing_imports:
			missing_parts.append(f"imports: {', '.join(missing_imports)}")
		raise ValueError(
			"Post-edit validation failed; missing " + " and ".join(missing_parts)
		)


REASONING_CONTEXT_KEY = "_buggy_coder_reasoning_emitted"


def _flatten_ai_content(message: AIMessage) -> str:
	content = message.content
	if isinstance(content, str):
		return content
	if isinstance(content, list):
		parts: list[str] = []
		for block in content:
			if isinstance(block, str):
				parts.append(block)
			elif isinstance(block, dict):
				text_value = block.get("text") if isinstance(block.get("text"), str) else None
				if block.get("type") == "text" and text_value is not None:
					parts.append(text_value)
			else:
				text_attr = getattr(block, "text", None)
				type_attr = getattr(block, "type", None)
				if isinstance(text_attr, str) and type_attr == "text":
					parts.append(text_attr)
		return "".join(parts)
	return str(content)


@after_model()
def enforce_reasoning_before_tools(state: AgentState, runtime: Runtime[Any]) -> dict[str, Any] | None:
	messages = state.get("messages", [])
	if not messages:
		return None

	reasoning_recorded = bool(runtime.context.get(REASONING_CONTEXT_KEY, False))
	last_ai: AIMessage | None = None
	for message in reversed(messages):
		if isinstance(message, AIMessage):
			last_ai = message
			break

	if last_ai is None:
		return None

	if not reasoning_recorded:
		if getattr(last_ai, "tool_calls", None):
			raise ValueError("Provide a `REASONING:` analysis turn before invoking any tools.")
		text = _flatten_ai_content(last_ai).strip()
		if not text.startswith("REASONING:"):
			raise ValueError("The first assistant turn must begin with `REASONING:` and outline your plan before using tools.")
		runtime.context[REASONING_CONTEXT_KEY] = True

	return None


@tool("add_import")
def add_import_buggy(snippet: str, module: str) -> str:
	"""Insert an import at the top of the snippet."""
	original_snippet = snippet
	candidate_module = module.strip()
	if not candidate_module:
		raise ValueError("Module name cannot be empty.")
	if not _MODULE_NAME_PATTERN.match(candidate_module):
		raise ValueError("Invalid module name; expected a dotted Python identifier path.")

	def has_module(target_lines: Iterable[str]) -> bool:
		for raw_line in target_lines:
			code = raw_line.split("#", 1)[0].strip()
			if not code:
				continue
			if code.startswith("import "):
				remainder = code[len("import "):]
				modules = [part.strip().split(" as ", 1)[0].strip() for part in remainder.split(",")]
				if candidate_module in modules:
					return True
			elif code.startswith("from "):
				parts = code.split()
				if len(parts) >= 4 and parts[1] == candidate_module and parts[2] == "import":
					return True
		return False

	if has_module(original_snippet.splitlines()):
		return original_snippet

	snippet_without_trailing_newlines = original_snippet.rstrip("\n")
	extra_newlines = len(original_snippet) - len(snippet_without_trailing_newlines)
	lines = snippet_without_trailing_newlines.splitlines()

	insert_index = 0
	if lines:
		first_line = lines[0]
		if first_line.startswith("#!") or first_line.startswith("# coding") or first_line.startswith("# -*-"):
			insert_index = 1

	new_lines = lines[:insert_index] + [f"import {candidate_module}"] + lines[insert_index:]
	post_import_index = insert_index + 1
	if len(new_lines) > post_import_index:
		next_line = new_lines[post_import_index].strip()
		if next_line and not next_line.startswith("import ") and not next_line.startswith("from "):
			new_lines.insert(post_import_index, "")

	body = "\n".join(new_lines).rstrip("\n")
	newline_count = max(1, extra_newlines)
	result = body + "\n" * newline_count

	_validate_post_edit(original_snippet, result)
	return result


@tool("rename_symbol")
def rename_first_occurrence(snippet: str, old: str, new: str) -> str:
	"""Rename a variable or function name in the snippet."""
	result = snippet.replace(old, new, 1)
	if result.endswith("\n"):
		result = result[:-1]

	_validate_post_edit(snippet, result)
	return result


@tool("apply_patch")
def apply_patch(snippet: str, target: str, replacement: str, count: int | None = None) -> str:
	"""Perform a multi-line text replacement within the snippet."""
	if target not in snippet:
		raise ValueError("The target text was not found in the snippet.")
	if count is not None:
		try:
			replacement_count = int(count)
		except (TypeError, ValueError) as exc:
			raise ValueError("count must be an integer or omitted.") from exc
	else:
		replacement_count = None

	if replacement_count is None or replacement_count < 0:
		updated = snippet.replace(target, replacement)
	else:
		updated = snippet.replace(target, replacement, replacement_count)

	_validate_post_edit(snippet, updated)
	return updated


@tool("append_snippet")
def append_snippet(snippet: str, addition: str) -> str:
	"""Append a code block to the snippet without overwriting existing content."""
	if not addition or not addition.strip():
		raise ValueError("addition must contain non-empty content.")

	original_snippet = snippet
	base = snippet.rstrip("\n")
	base_trailing_newlines = len(snippet) - len(base)
	addition_body = addition.strip("\n")

	if not base:
		result = addition_body + "\n"
	else:
		separator_newlines = max(2, base_trailing_newlines or 1)
		result = base + "\n" * separator_newlines + addition_body + "\n"

	_validate_post_edit(original_snippet, result)
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

	_validate_post_edit(snippet, result)
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

	_validate_post_edit(snippet, result)
	return result

SYSTEM_PROMPT = (
	"You are Coder. Your job is finding flaws in a user-glam code and fixing them using the tools that you have. "
	"Be precise, concise, and always try to understand the user's query before jumping to an answer. "
	"Workflow requirements:\n"
	"- Your first assistant turn MUST be a reasoning-only message prefixed with `REASONING:`. In that turn, restate the bug or request, enumerate edge cases (including non-happy paths and type variations), and outline the strategy you will follow. Do not call any tools before emitting this reasoning turn.\n"
	"- After the reasoning turn, select and sequence tools deliberately to implement the strategy.\n"
	"- Suppress intermediate reasoning from the final user-visible message whenever the user requests code-only output; provide the complete updated code (ideally in code fences) without extra prose.\n"
	"- Ensure that required imports and critical symbols remain present; the environment will validate this before finalizing.\n"
	"When returning modified code, output the entire code snippet with the fixes."
)


app = create_agent(
	model="openai:gpt-4o-mini",
	tools=[
		add_import_buggy,
		rename_first_occurrence,
		apply_patch,
		append_snippet,
		bump_indices_off_by_one,
		stub_function_singleline,
	],
	system_prompt=SYSTEM_PROMPT,
	middleware=[enforce_reasoning_before_tools],
)

