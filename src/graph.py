import re
from textwrap import dedent

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

try:
	from .state import BuggyCoderState
except ImportError:  # pragma: no cover - fallback when package context missing
	from state import BuggyCoderState  # type: ignore[right-import]


@tool("add_import")
def add_import(snippet: str, module: str) -> str:
	"""Insert an import at the top of the snippet."""
	module_name = module.strip()
	if not module_name:
		return snippet if snippet.endswith("\n") else f"{snippet}\n"

	if not snippet.strip():
		return f"import {module_name}\n"

	lines = snippet.splitlines()
	if any(line.strip() == f"import {module_name}" for line in lines):
		return snippet if snippet.endswith("\n") else f"{snippet}\n"

	insert_at = 0

	if lines and lines[insert_at].startswith("#!"):
		insert_at += 1

	encoding_pattern = re.compile(r"^#.*coding[:=]")
	if insert_at < len(lines) and encoding_pattern.match(lines[insert_at]):
		insert_at += 1

	while insert_at < len(lines) and lines[insert_at].strip() == "":
		insert_at += 1

	if insert_at < len(lines):
		stripped = lines[insert_at].lstrip()
		docstring_match = re.match(r"[rubfRUBF]*('{3}|\"{3})", stripped)
		if docstring_match:
			quote = docstring_match.group(1)
			remaining = stripped[docstring_match.end():]
			if quote in remaining and not stripped.endswith(f"{quote}\\"):
				insert_at += 1
			else:
				insert_at += 1
				while insert_at < len(lines):
					if quote in lines[insert_at]:
						insert_at += 1
						break
					insert_at += 1
			while insert_at < len(lines) and lines[insert_at].strip() == "":
				insert_at += 1

	while insert_at < len(lines) and lines[insert_at].startswith(("import ", "from ")):
		insert_at += 1

	lines.insert(insert_at, f"import {module_name}")

	if (
		insert_at + 1 < len(lines)
		and lines[insert_at + 1].strip()
		and not lines[insert_at + 1].startswith(("import ", "from "))
	):
		lines.insert(insert_at + 1, "")

	return "\n".join(lines) + "\n"


@tool("rename_symbol")
def rename_first_occurrence(snippet: str, old: str, new: str) -> str:
	"""Rename a variable or function name in the snippet."""
	result = snippet.replace(old, new, 1)
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


@tool("add_sanitization_helper")
def add_sanitization_helper(snippet: str, helper_name: str = "sanitize_scores_and_weight") -> str:
	"""Ensure the snippet includes a helper that filters numeric scores and validates a single weight."""
	helper_name = helper_name.strip() or "sanitize_scores_and_weight"
	if f"def {helper_name}(" in snippet:
		return snippet if snippet.endswith("\n") else f"{snippet}\n"

	helper_code = dedent(
		f"""
		def {helper_name}(scores, weight=None):
		    \"\"\"Filter numeric scores and coerce the weight into a positive float.\"\"\"
		    if scores is None:
		        sanitized_scores = []
		    elif not isinstance(scores, (list, tuple)):
		        raise TypeError("scores must be a list or tuple")
		    else:
		        sanitized_scores = []
		        for score in scores:
		            if isinstance(score, bool):
		                continue
		            if isinstance(score, (int, float)):
		                sanitized_scores.append(float(score))

		    numeric_weight = 1.0
		    if weight is not None:
		        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
		            raise TypeError("weight must be numeric or None")
		        numeric_weight = float(weight)
		        if numeric_weight < 0:
		            numeric_weight = abs(numeric_weight)
		        if numeric_weight == 0:
		            raise ValueError("weight must be positive")

		    if not sanitized_scores:
		        return [], numeric_weight

		    return sanitized_scores, float(numeric_weight)
		"""
	).strip("\n")

	snippet_body = snippet.lstrip("\n")
	if snippet_body:
		combined = f"{helper_code}\n\n{snippet_body}"
	else:
		combined = helper_code
	return combined if combined.endswith("\n") else f"{combined}\n"


ANALYSIS_PROMPT = """
You are the analysis planner for Buggy Coder. Examine the provided instructions and Python snippet before any edits.
Return a structured summary with these sections:
1. Goals — enumerate the behaviors or fixes the user requests.
2. Data validation — describe how numeric scores must be filtered, how a single weight should be interpreted as a positive float, and whether helper utilities are required.
3. Edge cases — call out empty inputs, missing weights, non-numeric values, required imports, or other pitfalls.
4. Tooling plan — list which tools (add_import, add_sanitization_helper, etc.) should be used and why.
5. Open questions — note any missing information that should be clarified.

Explicitly restate how compute_weighted_average should accumulate totals: sum(sanitized_scores) * weight and len(sanitized_scores) * weight, skipping only when sanitized_scores is empty.
"""

SYSTEM_PROMPT = """
You are Coder, a meticulous Python repair assistant.
Review the analysis summary message before editing and satisfy every requirement it lists.
When sanitizing score/weight inputs, call sanitize_scores_and_weight (via add_sanitization_helper) and rely on its (numeric_scores, numeric_weight) return contract.
Implement weighted averages as:
    numeric_scores, numeric_weight = sanitize_scores_and_weight(...)
    if numeric_scores:
        total += sum(numeric_scores) * numeric_weight
        weight_total += len(numeric_scores) * numeric_weight
Skip iterations only when numeric_scores is empty.
Use add_import to inject dependencies at column 0 with a trailing newline and without duplicates.
Prefer minimal, test-aligned edits; avoid ad-hoc arithmetic or unsafe type coercion.
Return the entire updated code snippet with your fixes applied.
"""

analysis_model = ChatOpenAI(
	model="gpt-5-codex",
	use_responses_api=True,
	output_version="responses/v1",
	reasoning={"effort": "high"},
	temperature=0,
)

llm = ChatOpenAI(
	model="gpt-5-codex",
	use_responses_api=True,
	output_version="responses/v1",
	reasoning={"effort": "medium"},
	temperature=0,
)

agent = create_agent(
	model=llm,
	tools=[
		add_import,
		rename_first_occurrence,
		bump_indices_off_by_one,
		stub_function_singleline,
		add_sanitization_helper,
	],
	system_prompt=SYSTEM_PROMPT,
)


def perform_analysis(state: BuggyCoderState) -> dict:
	"""Run a dedicated analysis pass so the agent captures key constraints before editing."""
	instructions = state.get("instructions", "").strip()
	snippet = state.get("snippet", "").strip()

	analysis_request = (
		"Produce the structured summary described in the analysis prompt."
		" Highlight the canonical weighted-average accumulation (sum(scores)*weight and len(scores)*weight)"
		" and call out any helper or import requirements the repair must satisfy."
	)

	analysis_messages = [
		SystemMessage(content=ANALYSIS_PROMPT),
		HumanMessage(
			content=(
				f"Instructions:\n{instructions or '<no instructions provided>'}\n\n"
				f"Current snippet:\n{snippet or '<no snippet provided>'}\n\n"
				f"{analysis_request}"
			),
		),
	]

	analysis_response = analysis_model.invoke(analysis_messages)

	return {"messages": [SystemMessage(content=analysis_response.content, name="analysis_summary")]}


workflow = StateGraph(BuggyCoderState)
workflow.add_node("analysis", perform_analysis)
workflow.add_node("coder", agent)
workflow.add_edge(START, "analysis")
workflow.add_edge("analysis", "coder")
workflow.add_edge("coder", END)

app = workflow.compile()
