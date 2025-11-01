import ast
import copy
import io
import re
import tokenize
from typing import Iterable, List, Optional

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from .state import BuggyCoderState


def _normalize_output(snippet: str) -> str:
	"""Match the historical behaviour of returning snippets without trailing newlines."""
	return snippet[:-1] if snippet.endswith("\n") else snippet


def _has_module_docstring(module: ast.Module) -> Optional[ast.stmt]:
	if module.body:
		first = module.body[0]
		if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
			return first
	return None


@tool("add_import")
def add_import_buggy(snippet: str, module: str) -> str:
	"""Insert an import at the top of the snippet if it is not already present."""
	module = module.strip()
	if not module:
		return _normalize_output(snippet)

	import_statement = f"import {module}"
	lines = snippet.splitlines()
	if any(line.strip() == import_statement for line in lines):
		return _normalize_output(snippet)

	insert_idx = 0
	if lines and lines[0].startswith("#!"):
		insert_idx = 1

	try:
		module_ast = ast.parse(snippet)
	except SyntaxError:
		module_ast = None

	if module_ast:
		docstring_node = _has_module_docstring(module_ast)
		if docstring_node and docstring_node.end_lineno is not None:
			insert_idx = max(insert_idx, docstring_node.end_lineno)

	while insert_idx < len(lines) and not lines[insert_idx].strip():
		insert_idx += 1

	lines.insert(insert_idx, import_statement)
	return _normalize_output("\n".join(lines) + ("\n" if snippet.endswith("\n") else ""))


@tool("rename_symbol")
def rename_first_occurrence(snippet: str, old: str, new: str) -> str:
	"""Rename identifier tokens without touching substrings or string literals."""
	if old == new:
		return _normalize_output(snippet)

	replaced = False
	tokens: List[tokenize.TokenInfo] = []
	for token in tokenize.generate_tokens(io.StringIO(snippet).readline):
		if token.type == tokenize.NAME and token.string == old:
			token = tokenize.TokenInfo(token.type, new, token.start, token.end, token.line)
			replaced = True
		tokens.append(token)

	if not replaced:
		return _normalize_output(snippet)

	result = tokenize.untokenize(tokens)
	return _normalize_output(result)


@tool("fix_indexing")
def bump_indices_off_by_one(snippet: str) -> str:
	"""No-op helper to avoid unintended off-by-one mutations."""
	return _normalize_output(snippet)


def _build_stub_replacement(match: re.Match[str]) -> str:
	name, params = match.group(1), match.group(2)
	body = '\n    raise NotImplementedError("Todo: implement function")'
	return f"def {name}({params}):{body}"


@tool("stub_function")
def stub_function_singleline(snippet: str) -> str:
	"""Replace `pass` stubs while preserving the original signature."""
	pattern = re.compile(r"def\s+(\w+)\(([^)]*)\):\s*pass")
	result, count = pattern.subn(_build_stub_replacement, snippet)
	if count == 0:
		return _normalize_output(snippet)
	return _normalize_output(result)


_EMPTY_LIST_KEYWORDS = {"empty list", "empty iterable", "no items"}


def _instructions_contain(instructions: str, keywords: Iterable[str]) -> bool:
	lower = instructions.lower()
	return any(keyword in lower for keyword in keywords)


def _detect_type_name(instructions: str) -> Optional[str]:
	lower = instructions.lower()
	if not any(keyword in lower for keyword in ("convert", "cast", "type conversion", "coerce")):
		return None
	if "float" in lower:
		return "float"
	if "str" in lower or "string" in lower:
		return "str"
	return "int"


def _extract_literal_return(instructions: str, default):
	lower = instructions.lower()
	if "return none" in lower:
		return ast.Constant(value=None)
	if "return true" in lower:
		return ast.Constant(value=True)
	if "return false" in lower:
		return ast.Constant(value=False)
	if re.search(r"return\s+\[\s*\]", instructions, re.IGNORECASE) or "return an empty list" in lower:
		return ast.List(elts=[], ctx=ast.Load())
	match_int = re.search(r"return\s+(-?\d+)", instructions, re.IGNORECASE)
	if match_int:
		return ast.Constant(value=int(match_int.group(1)))
	match_str = re.search(r"return\s+(['"])(.*?)\1", instructions, re.IGNORECASE)
	if match_str:
		return ast.Constant(value=match_str.group(2))
	return ast.Constant(value=default)


def _apply_type_conversions(module: ast.Module, instructions: str) -> bool:
	type_name = _detect_type_name(instructions)
	if not type_name:
		return False

	changed = False
	for node in module.body:
		if not isinstance(node, ast.FunctionDef):
			continue

		arg_names = {arg.arg for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)}
		if node.args.vararg:
			arg_names.add(node.args.vararg.arg)
		if node.args.kwarg:
			arg_names.add(node.args.kwarg.arg)

		numeric_args: set[str] = set()

		class NumericUsageVisitor(ast.NodeVisitor):
			def visit_BinOp(self, binop: ast.BinOp) -> None:
				if isinstance(binop.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)):
					for operand in (binop.left, binop.right):
						if isinstance(operand, ast.Name):
							numeric_args.add(operand.id)
				self.generic_visit(binop)

		NumericUsageVisitor().visit(node)
		targets = sorted(arg_names & numeric_args)
		if not targets:
			continue

		already_converted = set()
		for stmt in node.body:
			if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
				continue
			target = stmt.targets[0]
			if not isinstance(target, ast.Name):
				continue
			if target.id not in targets:
				continue
			value = stmt.value
			if (
				isinstance(value, ast.Call)
				and isinstance(value.func, ast.Name)
				and value.func.id == type_name
				and value.args
				and isinstance(value.args[0], ast.Name)
				and value.args[0].id == target.id
			):
				already_converted.add(target.id)

		pending = [name for name in targets if name not in already_converted]
		if not pending:
			continue

		insert_idx = 0
		if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
			insert_idx = 1

		assignments = []
		for name in pending:
			assignments.append(
				ast.Assign(
					targets=[ast.Name(id=name, ctx=ast.Store())],
					value=ast.Call(
						func=ast.Name(id=type_name, ctx=ast.Load()),
						args=[ast.Name(id=name, ctx=ast.Load())],
						keywords=[],
					),
				)
			)

		node.body[insert_idx:insert_idx] = assignments
		changed = True

	return changed


def _apply_division_guards(module: ast.Module, instructions: str) -> bool:
	lower = instructions.lower()
	if ("divide" not in lower and "division" not in lower) or "zero" not in lower:
		return False

	return_value = _extract_literal_return(instructions, default="Cannot divide by zero")

	changed = False
	for node in module.body:
		if not isinstance(node, ast.FunctionDef):
			continue

		existing_denominators: set[str] = set()
		for stmt in node.body:
			if (
				isinstance(stmt, ast.If)
				and isinstance(stmt.test, ast.Compare)
				and len(stmt.test.ops) == 1
				and isinstance(stmt.test.ops[0], ast.Eq)
				and isinstance(stmt.test.left, ast.Name)
				and len(stmt.test.comparators) == 1
				and isinstance(stmt.test.comparators[0], ast.Constant)
				and stmt.test.comparators[0].value == 0
			):
				existing_denominators.add(stmt.test.left.id)

		denominators: List[str] = []

		class DivisionVisitor(ast.NodeVisitor):
			def visit_BinOp(self, binop: ast.BinOp) -> None:
				if isinstance(binop.op, (ast.Div, ast.FloorDiv, ast.Mod)) and isinstance(binop.right, ast.Name):
					if binop.right.id not in existing_denominators and binop.right.id not in denominators:
						denominators.append(binop.right.id)
				self.generic_visit(binop)

		DivisionVisitor().visit(node)
		if not denominators:
			continue

		insert_idx = 0
		if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
			insert_idx = 1

		guards = []
		for denom in denominators:
			test = ast.Compare(
				left=ast.Name(id=denom, ctx=ast.Load()),
				ops=[ast.Eq()],
				comparators=[ast.Constant(value=0)],
			)
			body = [ast.Return(value=copy.deepcopy(return_value))]
			guards.append(ast.If(test=test, body=body, orelse=[]))

		node.body[insert_idx:insert_idx] = guards
		changed = True

	return changed


def _apply_empty_list_guards(module: ast.Module, instructions: str) -> bool:
	if not _instructions_contain(instructions, _EMPTY_LIST_KEYWORDS):
		return False

	return_value = _extract_literal_return(instructions, default=None)

	changed = False
	for node in module.body:
		if not isinstance(node, ast.FunctionDef):
			continue

		existing_targets: set[str] = set()
		for stmt in node.body:
			if (
				isinstance(stmt, ast.If)
				and isinstance(stmt.test, ast.UnaryOp)
				and isinstance(stmt.test.op, ast.Not)
				and isinstance(stmt.test.operand, ast.Name)
			):
				existing_targets.add(stmt.test.operand.id)

		target_lists: List[str] = []

		class SubscriptVisitor(ast.NodeVisitor):
			def visit_Subscript(self, subscript: ast.Subscript) -> None:
				if isinstance(subscript.value, ast.Name):
					index = subscript.slice
					if isinstance(index, ast.Constant) and index.value == 0:
						name = subscript.value.id
						if name not in existing_targets and name not in target_lists:
							target_lists.append(name)
				self.generic_visit(subscript)

		SubscriptVisitor().visit(node)
		if not target_lists:
			continue

		insert_idx = 0
		if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
			insert_idx = 1

		guards = []
		for name in target_lists:
			test = ast.UnaryOp(op=ast.Not(), operand=ast.Name(id=name, ctx=ast.Load()))
			if isinstance(return_value, ast.List):
				guard_body: List[ast.stmt] = [ast.Return(value=ast.List(elts=[], ctx=ast.Load()))]
			else:
				guard_body = [ast.Return(value=copy.deepcopy(return_value))]
			guards.append(ast.If(test=test, body=guard_body, orelse=[]))

		node.body[insert_idx:insert_idx] = guards
		changed = True

	return changed


def apply_fixes(snippet: str, instructions: str) -> str:
	"""Apply deterministic fixes or surface syntax errors for the provided snippet."""
	snippet = snippet.rstrip("\n") + ("\n" if snippet.endswith("\n") else "")
	try:
		module = ast.parse(snippet)
	except SyntaxError as error:
		location = f"line {error.lineno}, column {error.offset}" if error.lineno and error.offset else "unknown location"
		return f"SyntaxError: {error.msg} ({location})"

	module_copy = copy.deepcopy(module)
	changed = False
	changed |= _apply_type_conversions(module_copy, instructions)
	changed |= _apply_division_guards(module_copy, instructions)
	changed |= _apply_empty_list_guards(module_copy, instructions)

	if not changed:
		return snippet.rstrip("\n")

	ast.fix_missing_locations(module_copy)
	updated = ast.unparse(module_copy)
	return updated.rstrip("\n")


class RuleBasedAgent:
	"""Minimal agent compatible wrapper that emits deterministic responses."""

	def invoke(self, state: BuggyCoderState) -> BuggyCoderState:
		snippet = state.get("snippet", "")
		instructions = state.get("instructions", "")
		response = apply_fixes(snippet, instructions)

		previous_messages = list(state.get("messages", []))
		messages = previous_messages + [AIMessage(content=response)]
		return {
			"messages": messages,
			"snippet": response,
			"instructions": instructions,
		}


app = RuleBasedAgent()
