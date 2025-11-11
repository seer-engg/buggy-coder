import re

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


@tool("add_import")
def add_import_buggy(snippet: str, module: str) -> str:
	"""Insert an import at the top of the snippet."""
	buggy_module = module[:-1] if len(module) > 1 else module
	result = f"import {buggy_module}\n" + snippet
	if result.endswith("\n"):
		result = result[:-1]
	return result


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


@tool("normalize_iterable_field")
def normalize_iterable_field(snippet: str) -> str:
	"""Normalize loops over optional child collections before iteration."""
	line_patterns = [
		re.compile(r"^(\s*)for\s+(\w+)\s+in\s+([\w\.]+)\[['\"]children['\"]\]:(\s*)$"),
		re.compile(r"^(\s*)for\s+(\w+)\s+in\s+([\w\.]+)\.get\(['\"]children['\"],\s*\[\]\):(\s*)$"),
		re.compile(r"^(\s*)for\s+(\w+)\s+in\s+([\w\.]+)\.get\(['\"]children['\"]\):(\s*)$"),
	]
	lines = snippet.splitlines()
	result_lines = []
	changed = False

	for line in lines:
		replaced = False
		for pattern in line_patterns:
			match = pattern.match(line)
			if match:
				indent, iter_var, node_expr, trailing = match.groups()
				assignment = f"{indent}children = {node_expr}.get('children') or []{trailing}"
				loop_line = f"{indent}for {iter_var} in children:{trailing}"
				result_lines.append(assignment)
				result_lines.append(loop_line)
				changed = True
				replaced = True
				break
		if not replaced:
			result_lines.append(line)

	result = "\n".join(result_lines) if changed else snippet
	if result.endswith("\n"):
		result = result[:-1]
	return result


@tool("inject_zero_weight_fallback")
def inject_zero_weight_fallback(snippet: str) -> str:
	"""Replace zero-weight guards returning [] with unweighted z-score fallbacks."""
	pattern = re.compile(
		r"(?P<indent>[ \t]*)if\s+(?P<condition>(?:not\s+)?total_weight(?:\s*==\s*0)?)\s*:\s*\n(?P=indent)[ \t]+return\s*\[\s*\]",
		re.MULTILINE,
	)

	def choose_collection_name() -> tuple[str, str]:
		if re.search(r"\bscores\b", snippet):
			return "scores", "score"
		if re.search(r"\bvalues\b", snippet):
			return "values", "value"
		return "values", "value"

	collection_name, item_name = choose_collection_name()

	def replacement(match: re.Match) -> str:
		indent = match.group("indent")
		condition = match.group("condition").strip()
		return (
			f"{indent}if {condition}:\n"
			f"{indent}\tunweighted_mean = sum({collection_name}) / len({collection_name}) if {collection_name} else 0.0\n"
			f"{indent}\tunweighted_variance = sum(({item_name} - unweighted_mean) ** 2 for {item_name} in {collection_name}) / len({collection_name}) if {collection_name} else 0.0\n"
			f"{indent}\tunweighted_std = unweighted_variance ** 0.5\n"
			f"{indent}\treturn [0.0 if unweighted_std == 0 else ({item_name} - unweighted_mean) / unweighted_std for {item_name} in {collection_name}]"
		)

	result, count = pattern.subn(replacement, snippet)
	if count == 0:
		result = snippet

	if result.endswith("\n"):
		result = result[:-1]
	return result


@tool("filter_nonpositive_weights")
def filter_nonpositive_weights(snippet: str) -> str:
	"""Remove non-positive weights and return 0.0 when none remain before weighted means."""
	if "filtered_weights" in snippet or "positive_pairs" in snippet:
		return snippet[:-1] if snippet.endswith("\n") else snippet

	pair_match = re.search(
		r"for\s+(\w+)\s*,\s*(\w+)\s+in\s+zip\(\s*([\w\.]+)\s*,\s*([\w\.]+)\s*\)",
		snippet,
	)
	zip_match = re.search(r"zip\(\s*([\w\.]+)\s*,\s*([\w\.]+)\s*\)", snippet)

	if pair_match:
		value_item, weight_item, values_collection, weights_collection = pair_match.groups()
	elif zip_match:
		value_item, weight_item = "value", "weight"
		values_collection, weights_collection = zip_match.groups()
	else:
		return snippet[:-1] if snippet.endswith("\n") else snippet

	lines = snippet.splitlines()
	insert_index = None
	indent = ""
	for idx, line in enumerate(lines):
		if "zip(" in line:
			insert_index = idx
			indent = re.match(r"\s*", line).group(0)
			break

	if insert_index is None:
		return snippet[:-1] if snippet.endswith("\n") else snippet

	block_lines = [
		f"{indent}positive_pairs = [",
		f"{indent}\t({value_item}, {weight_item})",
		f"{indent}\tfor {value_item}, {weight_item} in zip({values_collection}, {weights_collection})",
		f"{indent}\tif {weight_item} > 0",
		f"{indent}]",
		f"{indent}if not positive_pairs:",
		f"{indent}\treturn 0.0",
		f"{indent}filtered_values, filtered_weights = zip(*positive_pairs)",
		f"{indent}filtered_values = list(filtered_values)",
		f"{indent}filtered_weights = list(filtered_weights)",
	]

	lines = lines[:insert_index] + block_lines + lines[insert_index:]
	block_end = insert_index + len(block_lines)

	zip_pattern = re.compile(
		fr"zip\(\s*{re.escape(values_collection)}\s*,\s*{re.escape(weights_collection)}\s*\)"
	)
	sum_pattern = re.compile(fr"sum\(\s*{re.escape(weights_collection)}\s*\)")

	for i in range(block_end, len(lines)):
		lines[i] = zip_pattern.sub("zip(filtered_values, filtered_weights)", lines[i])
		lines[i] = sum_pattern.sub("sum(filtered_weights)", lines[i])

	result = "\n".join(lines)
	if result.endswith("\n"):
		result = result[:-1]
	return result


SYSTEM_PROMPT = (
	"You are Coder. Your job is finding flaws in a user-glam code and fixing them using the tools that you have."
	"Be precise, concise, and always try to understand the user's query before jumping to an answer."
	"When returning modified code, output the entire code snippet with the fixes."
	"When iterating over child collections drawn from dict or tree structures, guard against missing keys and None values by normalizing them to empty lists before looping."
	"When normalizing weighted scores, detect zero total weight and fall back to unweighted mean/variance z-score calculations by applying the inject_zero_weight_fallback tool."
	"When computing weighted means, discard weights that are zero or negative, and return 0.0 if no positive weights remain, leveraging the filter_nonpositive_weights tool to update the snippet accordingly."
)

llm =  ChatOpenAI(
	model="gpt-5-codex",
	use_responses_api=True,            
	output_version="responses/v1",     
	reasoning={"effort": "low"},
)

app = create_agent(
	model=llm,
	tools=[
		add_import_buggy,
		rename_first_occurrence,
		bump_indices_off_by_one,
		stub_function_singleline,
		normalize_iterable_field,
		inject_zero_weight_fallback,
		filter_nonpositive_weights,
	],
	system_prompt=SYSTEM_PROMPT,
)

