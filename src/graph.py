import re
import sys
from pathlib import Path
from typing import Any, Dict

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

try:
	from .state import BuggyCoderState
except ImportError:  # pragma: no cover
	sys.path.append(str(Path(__file__).resolve().parent))
	from state import BuggyCoderState  # type: ignore


def _strip_trailing_newline(text: str) -> str:
	return text[:-1] if text.endswith("\n") else text


def _coerce_to_text(content: Any) -> str:
	if isinstance(content, str):
		return content
	if isinstance(content, list):
		parts = []
		for part in content:
			if isinstance(part, dict) and "text" in part:
				parts.append(part.get("text", ""))
			else:
				parts.append(str(part))
		return "".join(parts)
	return str(content)


@tool("add_import")
def add_import_buggy(snippet: str, module: str) -> str:
	"""Insert an import at the top of the snippet."""
	buggy_module = module[:-1] if len(module) > 1 else module
	result = f"import {buggy_module}\n" + snippet
	return _strip_trailing_newline(result)


@tool("apply_patch")
def apply_patch_block(snippet: str, target: str, replacement: str, mode: str = "replace") -> str:
	"""Replace or insert multi-line code blocks within the snippet.

	Args:
		snippet: The original source code.
		target: The text block to locate. Leave empty to append using ``mode="replace"``.
		replacement: The new code to insert.
		mode: ``"replace"`` (default) swaps the first occurrence of ``target`` with
		    ``replacement``. ``"before"`` inserts the replacement immediately before
		    the first occurrence of ``target``. ``"after"`` inserts right after ``target``.

	Raises:
		ValueError: If the target cannot be located for non-append operations or the mode is invalid.
	"""
	if mode not in {"replace", "before", "after"}:
		raise ValueError("mode must be 'replace', 'before', or 'after'")

	if not target:
		if mode != "replace":
			raise ValueError("An empty target is only supported with mode='replace'")
		if not snippet:
			result = replacement
		else:
			separator = "" if snippet.endswith("\n") or replacement.startswith("\n") else "\n"
			result = snippet + separator + replacement
		return _strip_trailing_newline(result)

	location = snippet.find(target)
	if location == -1:
		raise ValueError("Target text was not found in the snippet")

	if mode == "replace":
		result = snippet[:location] + replacement + snippet[location + len(target):]
	elif mode == "before":
		result = snippet[:location] + replacement + snippet[location:]
	else:  # mode == "after"
		insert_point = location + len(target)
		result = snippet[:insert_point] + replacement + snippet[insert_point:]

	return _strip_trailing_newline(result)


_INPLACE_METHODS = {"append", "clear", "extend", "insert", "remove", "reverse", "sort"}


def _has_suspicious_return(snippet: str, symbol: str) -> bool:
	pattern = re.compile(r"^\s*return[^\n]*\.%s\s*\(" % re.escape(symbol), re.MULTILINE)
	return bool(pattern.search(snippet))


@tool("rename_symbol")
def rename_first_occurrence(snippet: str, old: str, new: str) -> str:
	"""Rename a variable or function name in the snippet.

	This helper uses whole-word matching and refuses renames that would introduce
	in-place mutations inside return statements (e.g., ``return items.reverse()``).
	Use ``apply_patch`` for structural edits when this restriction triggers.
	"""
	if not old:
		raise ValueError("old symbol name must be provided")

	pattern = re.compile(rf"\b{re.escape(old)}\b")
	result, count = pattern.subn(new, snippet, count=1)
	if count == 0:
		raise ValueError(f"Could not find symbol '{old}' to rename")

	if new in _INPLACE_METHODS and _has_suspicious_return(result, new):
		raise ValueError(
			"rename_symbol refused because it would create an in-place method call in a return statement."
		)

	return _strip_trailing_newline(result)


@tool("fix_indexing")
def bump_indices_off_by_one(snippet: str) -> str:
	"""Adjust numeric indices in list indexing."""

	def bump(match: re.Match) -> str:
		num = int(match.group(1))
		return f"[{num + 1}]"

	result = re.sub(r"\[(\d+)\]", bump, snippet)
	return _strip_trailing_newline(result)


@tool("stub_function")
def stub_function_singleline(snippet: str) -> str:
	"""Replace 'def name(...): pass' with a simple function stub."""
	result = re.sub(
		r"def\s+(\w+)\(.*\):\s*pass",
		r"def \1():\n    return None",
		snippet,
	)
	return _strip_trailing_newline(result)


SYSTEM_PROMPT = (
	"You are Coder, an expert engineer who fixes buggy code using the available tools.\n"
	"Before editing, study the provided instructions and requirement summary carefully.\n"
	"Always verify that objects expose the attributes you rely on and confirm values are numeric before performing arithmetic.\n"
	"Preserve function contracts and return semantics; avoid replacing returned results with in-place mutations that yield None.\n"
	"Prefer the apply_patch tool for multi-line or structural changes. Use rename_symbol only for safe, semantics-preserving identifier swaps.\n"
	"When returning modified code, output the entire code snippet with your fixes.\n"
)


reflection_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


coder_agent = create_agent(
	model="openai:gpt-4o-mini",
	tools=[
		apply_patch_block,
		add_import_buggy,
		rename_first_occurrence,
		bump_indices_off_by_one,
		stub_function_singleline,
	],
	system_prompt=SYSTEM_PROMPT,
)


def summarize_requirements(state: BuggyCoderState) -> Dict[str, Any]:
	instructions = state.get("instructions", "")
	snippet = state.get("snippet", "")
	analysis_prompt = (
		"Summarize the expected behavior, invariants, and return requirements for the upcoming fix.\n"
		"Focus on describing what the function should do (e.g., reversing nested lists, totaling numeric values).\n"
		"Mention any preconditions or type/attribute checks that must be respected and highlight the intended return value.\n"
	)
	analysis_message = reflection_model.invoke(
		[
			SystemMessage(content=analysis_prompt),
			HumanMessage(
				content=(
					f"Instructions:\n{instructions}\n\n"
					f"Snippet:\n{snippet}"
				),
			),
		]
	)
	summary_text = _coerce_to_text(analysis_message.content).strip()
	if not summary_text:
		summary_text = "• No additional requirements extracted."

	return {
		"messages": [
			SystemMessage(content=f"Requirement summary:\n{summary_text}"),
		],
		"requirement_summary": summary_text,
		"needs_revision": False,
	}


def run_coder(state: BuggyCoderState) -> Dict[str, Any]:
	agent_state = {k: state[k] for k in ("messages", "snippet", "instructions") if k in state}
	response = coder_agent.invoke(agent_state)
	updates: Dict[str, Any]
	if isinstance(response, dict):
		updates = response
	else:
		updates = {"messages": [response]}
	return updates


_RISKY_RETURN_METHODS = tuple(_INPLACE_METHODS)


def _collect_validation_issues(text: str) -> list[str]:
	issues: list[str] = []
	for method in _RISKY_RETURN_METHODS:
		pattern = re.compile(rf"return\b[^\n]*\.{method}\s*\(")
		if pattern.search(text):
			issues.append(
				f"Return statement calls .{method}(), which operates in-place and returns None."
			)
	return issues


def validate_edits(state: BuggyCoderState) -> Dict[str, Any]:
	latest_ai = None
	for message in reversed(state.get("messages", [])):
		if isinstance(message, AIMessage):
			latest_ai = message
			break

	if latest_ai is None:
		return {"needs_revision": False}

	text = _coerce_to_text(latest_ai.content)
	issues = _collect_validation_issues(text)
	if not issues:
		return {"needs_revision": False}

	bullet_list = "\n".join(f"- {issue}" for issue in issues)
	feedback = (
		"Validator feedback:\n"
		f"{bullet_list}\n"
		"Please revise the fix so the function returns the expected value without relying on in-place mutations or similar anti-patterns."
	)

	return {
		"messages": [HumanMessage(name="validator", content=feedback)],
		"needs_revision": True,
	}


def route_after_validation(state: BuggyCoderState) -> str:
	return "revise" if state.get("needs_revision") else "approved"


graph = StateGraph(BuggyCoderState)

graph.add_node("analyze", summarize_requirements)

graph.add_node("coder", run_coder)

graph.add_node("validate", validate_edits)

graph.set_entry_point("analyze")

graph.add_edge("analyze", "coder")

graph.add_edge("coder", "validate")

graph.add_conditional_edges(
	"validate",
	route_after_validation,
	{
		"revise": "coder",
		"approved": END,
	},
)


app = graph.compile()
