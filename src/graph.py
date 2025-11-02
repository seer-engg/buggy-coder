import ast
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from langchain.agents import create_agent
from langchain_core.tools import tool


class ValidationError(ValueError):
    """Raised when a static validation rule is violated."""


def _parse_hunk_range(range_expression: str) -> tuple[int, int]:
    start_text, _, length_text = range_expression.partition(",")
    start = int(start_text)
    length = int(length_text) if length_text else 1
    return start, length


def _apply_unified_diff(snippet: str, diff: str) -> str:
    if not diff.strip():
        raise ValueError("Diff content cannot be empty.")

    original_lines = snippet.split("\n")
    diff_lines = diff.splitlines()

    new_lines: List[str] = []
    pointer = 0
    index = 0

    while index < len(diff_lines):
        line = diff_lines[index]

        if line.startswith("---") or line.startswith("+++"):
            index += 1
            continue

        if not line.startswith("@@"):
            raise ValueError(f"Unexpected diff header: {line}")

        header = line.split()
        if len(header) < 3:
            raise ValueError(f"Malformed hunk header: {line}")

        orig_start, _ = _parse_hunk_range(header[1][1:])
        index += 1

        # Copy unchanged lines that appear before the hunk.
        while pointer < orig_start - 1 and pointer < len(original_lines):
            new_lines.append(original_lines[pointer])
            pointer += 1

        while index < len(diff_lines):
            hunk_line = diff_lines[index]
            if hunk_line.startswith("@@"):
                break
            if hunk_line.startswith(" "):
                if pointer >= len(original_lines):
                    raise ValueError("Diff references context beyond original snippet length.")
                new_lines.append(original_lines[pointer])
                pointer += 1
            elif hunk_line.startswith("-"):
                pointer += 1
                if pointer > len(original_lines) + 1:
                    raise ValueError("Diff tried to remove beyond original snippet.")
            elif hunk_line.startswith("+"):
                new_lines.append(hunk_line[1:])
            elif hunk_line.startswith("\\"):
                # `\ No newline at end of file` – safe to ignore.
                pass
            else:
                raise ValueError(f"Unsupported diff line: {hunk_line}")
            index += 1

    # Append any remaining original content.
    new_lines.extend(original_lines[pointer:])

    result = "\n".join(new_lines)
    # Preserve trailing newline if it existed originally.
    if snippet.endswith("\n"):
        return result + "\n"
    return result


def apply_patch(snippet: str, diff: str) -> str:
    """Apply a unified diff to ``snippet`` and return the modified result."""
    return _apply_unified_diff(snippet, diff)


apply_patch_tool = tool("apply_patch")(apply_patch)


def add_import(snippet: str, module: str) -> str:
    module_name = module.strip()
    if not module_name:
        raise ValueError("`module` must be a non-empty string.")

    lines = snippet.split("\n")
    insertion_index = 0

    if lines and lines[0].startswith("#!"):
        insertion_index = 1

    if len(lines) > insertion_index and lines[insertion_index].startswith(("\"\"\"", "'''")):
        closing = lines[insertion_index][:3]
        insertion_index += 1
        while insertion_index < len(lines) and not lines[insertion_index].endswith(closing):
            insertion_index += 1
        if insertion_index < len(lines):
            insertion_index += 1

    lines.insert(insertion_index, f"import {module_name}")
    result = "\n".join(lines)
    if snippet.endswith("\n") or not snippet:
        result += "\n"
    return result


add_import_tool = tool("add_import")(add_import)


def rename_symbol(snippet: str, old: str, new: str) -> str:
    pattern = re.compile(rf"\\b{re.escape(old)}\\b")
    match = pattern.search(snippet)
    if not match:
        return snippet
    start, end = match.span()
    return snippet[:start] + new + snippet[end:]


rename_symbol_tool = tool("rename_symbol")(rename_symbol)


def fix_indexing(snippet: str, target_index: Optional[int] = None, delta: int = -1) -> str:
    def adjust(match: re.Match[str]) -> str:
        index = int(match.group(1))
        if target_index is not None and index != target_index:
            return match.group(0)
        return f"[{index + delta}]"

    return re.sub(r"\[(\d+)\]", adjust, snippet)


fix_indexing_tool = tool("fix_indexing")(fix_indexing)


def stub_function(snippet: str) -> str:
    def rewrite(match: re.Match[str]) -> str:
        name = match.group("name")
        params = match.group("params").strip()
        params_text = params if params else ""
        body = "    raise NotImplementedError\n"
        return f"def {name}({params_text}):\n{body}"

    pattern = re.compile(r"def\s+(?P<name>\w+)\((?P<params>.*?)\):\s*pass")
    return pattern.sub(rewrite, snippet)


stub_function_tool = tool("stub_function")(stub_function)


@dataclass(frozen=True)
class _FunctionSignature:
    required_arguments: int
    has_varargs: bool
    has_varkw: bool


def _collect_function_signatures(tree: ast.AST) -> Dict[str, _FunctionSignature]:
    signatures: Dict[str, _FunctionSignature] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            positional = list(node.args.posonlyargs) + list(node.args.args)
            required_positional = max(len(positional) - len(node.args.defaults), 0)
            required_kwonly = sum(1 for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults) if default is None)
            required_arguments = required_positional + required_kwonly
            signatures[node.name] = _FunctionSignature(
                required_arguments=required_arguments,
                has_varargs=node.args.vararg is not None,
                has_varkw=node.args.kwarg is not None,
            )
    return signatures


def _ensure_sentinel_initialized(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if isinstance(node.value, ast.Constant) and node.value.value is None:
                for target in node.targets:
                    if isinstance(target, ast.Name) and "sentinel" in target.id.lower():
                        raise ValidationError(
                            "Sentinel values must not be initialised to None; use object() instead."
                        )
        elif isinstance(node, ast.AnnAssign):
            if (
                isinstance(node.target, ast.Name)
                and "sentinel" in node.target.id.lower()
                and isinstance(node.value, ast.Constant)
                and node.value.value is None
            ):
                raise ValidationError(
                    "Sentinel values must not be initialised to None; use object() instead."
                )


def _ensure_argument_counts(tree: ast.AST, signatures: Dict[str, _FunctionSignature]) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            signature = signatures.get(node.func.id)
            if not signature:
                continue
            if signature.has_varargs or signature.has_varkw:
                continue
            provided = len(node.args) + sum(1 for keyword in node.keywords if keyword.arg is not None)
            if provided < signature.required_arguments:
                raise ValidationError(
                    f"Call to `{node.func.id}` is missing required arguments (expected at least "
                    f"{signature.required_arguments}, found {provided})."
                )


class _FunctionScopeAnalyzer(ast.NodeVisitor):
    def __init__(self, function: ast.FunctionDef) -> None:
        self._function = function
        args = (
            list(function.args.posonlyargs)
            + list(function.args.args)
            + list(function.args.kwonlyargs)
        )
        self._defined: set[str] = {arg.arg for arg in args}
        if function.args.vararg:
            self._defined.add(function.args.vararg.arg)
        if function.args.kwarg:
            self._defined.add(function.args.kwarg.arg)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # pragma: no cover - not expected
        if node is self._function:
            self.generic_visit(node)
        # Skip nested function definitions.

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # pragma: no cover - not expected
        # Skip nested classes when analysing a function.
        return

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._register_target(target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._register_target(node.target)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._register_target(node.target)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self._register_target(node.target)
        self.visit(node.iter)
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_With(self, node: ast.With) -> None:
        for withitem in node.items:
            if withitem.optional_vars:
                self._register_target(withitem.optional_vars)
        for stmt in node.body:
            self.visit(stmt)

    def visit_Return(self, node: ast.Return) -> None:
        if isinstance(node.value, ast.Name):
            name = node.value.id
            if name not in self._defined:
                raise ValidationError(f"Function `{self._function.name}` returns undefined name `{name}`.")
        self.generic_visit(node)

    def _register_target(self, target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            self._defined.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._register_target(element)


def _ensure_defined_return_values(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            analyzer = _FunctionScopeAnalyzer(node)
            for statement in node.body:
                analyzer.visit(statement)


def perform_static_validation(snippet: str) -> None:
    try:
        tree = ast.parse(snippet)
    except SyntaxError as exc:  # pragma: no cover - defensive programming
        raise ValidationError("Corrected snippet is invalid Python syntax.") from exc

    _ensure_sentinel_initialized(tree)
    signatures = _collect_function_signatures(tree)
    _ensure_argument_counts(tree, signatures)
    _ensure_defined_return_values(tree)


def _extract_code_snippet(response: object) -> Optional[str]:
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        for key in ("corrected_code", "fixed_code", "code", "output", "result"):
            value = response.get(key)
            if isinstance(value, str):
                return value
    return None


class ValidatedAgent:
    def __init__(self, agent: object) -> None:
        self._agent = agent

    def _validate_response(self, response: object) -> None:
        snippet = _extract_code_snippet(response)
        if snippet:
            perform_static_validation(snippet)

    def invoke(self, *args, **kwargs):  # type: ignore[override]
        response = self._agent.invoke(*args, **kwargs)
        self._validate_response(response)
        return response

    async def ainvoke(self, *args, **kwargs):  # type: ignore[override]
        if not hasattr(self._agent, "ainvoke"):
            raise AttributeError("Underlying agent does not support asynchronous invocation.")
        response = await self._agent.ainvoke(*args, **kwargs)
        self._validate_response(response)
        return response

    def batch(self, *args, **kwargs):  # type: ignore[override]
        if not hasattr(self._agent, "batch"):
            raise AttributeError("Underlying agent does not support batch execution.")
        responses = self._agent.batch(*args, **kwargs)
        for response in responses:
            self._validate_response(response)
        return responses

    async def abatch(self, *args, **kwargs):  # type: ignore[override]
        if not hasattr(self._agent, "abatch"):
            raise AttributeError("Underlying agent does not support async batch execution.")
        responses = await self._agent.abatch(*args, **kwargs)
        for response in responses:
            self._validate_response(response)
        return responses

    def __getattr__(self, item: str):  # pragma: no cover - passthrough behaviour
        return getattr(self._agent, item)


SYSTEM_PROMPT = (
    "You are Coder, a meticulous Python repair assistant.\n"
    "Follow these rules strictly:\n"
    "- Understand the bug before making any modification.\n"
    "- Apply the minimal change necessary to satisfy the request.\n"
    "- Preserve existing identifiers unless the user explicitly requests renaming.\n"
    "- Avoid reformatting or rewriting unrelated code.\n"
    "- When responding, return JSON with keys `explanation` (a one-sentence summary of the bug you fixed) "
    "and `corrected_code` (the full updated snippet).\n"
    "- Ensure the updated code passes static validation and introduces no new side effects."
)


toolkit = [
    add_import_tool,
    rename_symbol_tool,
    fix_indexing_tool,
    stub_function_tool,
    apply_patch_tool,
]


app = ValidatedAgent(
    create_agent(
        model="openai:gpt-4o-mini",
        tools=toolkit,
        system_prompt=SYSTEM_PROMPT,
    )
)
