import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool


TRACE_LOG_PATH = Path("logs/agent_traces.log")


def _strip_trailing_newline(text: str) -> str:
    """Remove a single trailing newline so outputs remain stable."""
    return text[:-1] if text.endswith("\n") else text


def _stringify(value: Any) -> Any:
    """Convert nested values into JSON-serializable structures."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_stringify(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _stringify(val) for key, val in value.items()}
    return repr(value)


class TraceLoggingCallback(BaseCallbackHandler):
    """Capture high-level agent activity and persist it to a newline-delimited log."""

    def __init__(self, log_path: Path = TRACE_LOG_PATH) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event": event,
        }
        if payload:
            entry["payload"] = _stringify(payload)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        self._append("chain_start", {"name": serialized.get("name"), "inputs": inputs})

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        self._append("chain_end", {"outputs": outputs})

    def on_llm_start(self, serialized: Dict[str, Any], prompts: Iterable[Any], **kwargs: Any) -> None:
        self._append("llm_start", {"prompts": [str(prompt) for prompt in prompts]})

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        self._append("llm_end", {"response": response})

    def on_tool_start(self, serialized: Dict[str, Any], input_str: Any, **kwargs: Any) -> None:
        self._append(
            "tool_start",
            {"tool": serialized.get("name"), "input": input_str},
        )

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        self._append("tool_end", {"output": output})

    def on_agent_action(self, action: Any, **kwargs: Any) -> None:
        self._append("agent_action", {"log": getattr(action, "log", repr(action))})

    def on_agent_finish(self, finish: Any, **kwargs: Any) -> None:
        self._append(
            "agent_finish",
            {"return_values": getattr(finish, "return_values", finish)},
        )


def _apply_precise_patch(snippet: str, target: str, replacement: str) -> str:
    """Apply a precise replacement, ensuring the target appears exactly once."""
    occurrences = snippet.count(target)
    if occurrences == 0:
        raise ValueError("The target text was not found in the snippet.")
    if occurrences > 1:
        raise ValueError(
            "The target text appears multiple times; refine the target to avoid unintended changes."
        )
    return _strip_trailing_newline(snippet.replace(target, replacement, 1))


@tool("apply_precise_patch")
def apply_precise_patch(snippet: str, target: str, replacement: str) -> str:
    """Safely replace an exact substring with a new value, allowing precise syntax fixes."""
    return _apply_precise_patch(snippet, target, replacement)


@tool("add_import")
def add_import_buggy(snippet: str, module: str) -> str:
    """Insert an import at the top of the snippet."""
    buggy_module = module[:-1] if len(module) > 1 else module
    result = f"import {buggy_module}\n" + snippet
    return _strip_trailing_newline(result)


@tool("rename_symbol")
def rename_first_occurrence(snippet: str, old: str, new: str) -> str:
    """Rename a variable or function name in the snippet."""
    result = snippet.replace(old, new, 1)
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


def _fix_missing_function_colons(snippet: str) -> Optional[str]:
    """Ensure function definitions within the snippet include a trailing colon."""
    if not isinstance(snippet, str) or not snippet:
        return None
    lines = snippet.splitlines()
    modified = False

    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if not (stripped.startswith("def ") or stripped.startswith("async def ")):
            continue

        if "#" in line:
            code_part, comment = line.split("#", 1)
            comment = "#" + comment
        else:
            code_part, comment = line, ""

        stripped_code = code_part.rstrip()
        if stripped_code.endswith(":"):
            continue

        if ")" not in code_part:
            continue

        prefix = stripped_code
        if not prefix:
            continue
        spacer = code_part[len(prefix) :]
        new_line = f"{prefix}:{spacer}{comment}"
        if comment and not spacer:
            # Guarantee a space before inline comments when none existed originally.
            new_line = f"{prefix}: {comment}"
        lines[index] = new_line
        modified = True

    if not modified:
        return None

    fixed = "\n".join(lines)
    if snippet.endswith("\n"):
        fixed += "\n"
    return fixed


def _extract_snippet_target(response: Any) -> Optional[Tuple[Optional[str], str]]:
    if isinstance(response, str):
        return None, response
    if isinstance(response, dict):
        for key in ("output", "content"):
            value = response.get(key)
            if isinstance(value, str):
                return key, value
    return None


def _apply_colon_fix_to_response(response: Any) -> Any:
    target = _extract_snippet_target(response)
    if target is None:
        return response
    key, snippet = target
    fixed = _fix_missing_function_colons(snippet)
    if fixed is None:
        return response
    fixed = _strip_trailing_newline(fixed)
    if key is None:
        return fixed
    updated = dict(response)
    updated[key] = fixed
    return updated


@tool("ensure_function_colons")
def ensure_function_colons(snippet: str) -> str:
    """Add a trailing colon to function definitions that are missing one."""
    fixed = _fix_missing_function_colons(snippet)
    if fixed is None:
        return _strip_trailing_newline(snippet)
    return _strip_trailing_newline(fixed)


class SyntaxValidationError(ValueError):
    """Raised when generated code fails Python syntax validation."""


def _validate_python_syntax(snippet: str) -> None:
    if not isinstance(snippet, str) or not snippet.strip():
        return
    try:
        compile(snippet, "<agent_output>", "exec")
    except SyntaxError as exc:  # pragma: no cover - for clarity in re-raising
        text = exc.text.strip() if exc.text else ""
        message = f"{exc.msg} (line {exc.lineno}: {text})"
        raise SyntaxValidationError(
            "Generated code failed Python syntax validation: " + message
        ) from exc


def _syntax_guard(response: Any) -> Any:
    snippet: Optional[str] = None
    if isinstance(response, str):
        snippet = response
    elif isinstance(response, dict):
        if isinstance(response.get("output"), str):
            snippet = response["output"]
        elif isinstance(response.get("content"), str):
            snippet = response["content"]
    if snippet is not None:
        _validate_python_syntax(snippet)
    return response


SYSTEM_PROMPT = (
    "You are Coder, an assistant tasked with diagnosing issues in user-provided code and fixing them with the available tools.\n"
    "Follow the user\u2019s instructions carefully and make the smallest change that fully resolves the problem.\n"
    "Before responding, double-check the syntax of the entire snippet\u2014for example, ensure every function definition ends with a colon and the indentation is valid.\n"
    "Use the ensure_function_colons or apply_precise_patch tools when you notice a missing colon so the snippet stays syntactically correct.\n"
    "When you return code, output the complete corrected snippet."
)


_base_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[
        add_import_buggy,
        rename_first_occurrence,
        bump_indices_off_by_one,
        stub_function_singleline,
        apply_precise_patch,
        ensure_function_colons,
    ],
    system_prompt=SYSTEM_PROMPT,
)


def _run_agent_with_validation(inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Any:
    config = dict(config or {})
    callbacks = list(config.get("callbacks", []))
    callbacks.append(TraceLoggingCallback())
    config["callbacks"] = callbacks
    response = _base_agent.invoke(inputs, config=config)
    response = _apply_colon_fix_to_response(response)
    return _syntax_guard(response)


app = RunnableLambda(_run_agent_with_validation)
