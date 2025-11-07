import io
import json
import logging
import re
import tokenize
import uuid
from collections.abc import Mapping, Sequence
from typing import Any, AsyncIterator, Iterator, Optional, Sequence as TypingSequence

from langchain.agents import create_agent
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import tool


logger = logging.getLogger("buggy_coder.agent")

MAX_PREVIEW_LENGTH = 240
MAX_COLLECTION_ITEMS = 10
CONFIRMATION_KEYWORDS = (
    "off by one",
    "off-by-one",
    "1-based",
    "one-based",
    "use 1-based",
    "use one-based",
    "increment index",
    "fix index",
    "adjust index",
    "index should be 1-based",
    "should be 1-based",
)


def _truncate_text(text: str, limit: int = MAX_PREVIEW_LENGTH) -> str:
    preview = text.replace("\n", "\\n")
    if len(preview) <= limit:
        return preview
    return f"{preview[: limit - 3]}..."


def _sanitize_for_logging(value: Any, *, limit: int = MAX_PREVIEW_LENGTH, depth: int = 2) -> Any:
    if depth < 0:
        return "<truncated>"

    if isinstance(value, str):
        return _truncate_text(value, limit=limit)

    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for idx, (key, val) in enumerate(value.items()):
            if idx >= MAX_COLLECTION_ITEMS:
                sanitized["..."] = f"{len(value) - MAX_COLLECTION_ITEMS} more"
                break
            sanitized[str(key)] = _sanitize_for_logging(val, limit=limit, depth=depth - 1)
        return sanitized

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        limited_items = list(value[:MAX_COLLECTION_ITEMS])
        sanitized_list = [
            _sanitize_for_logging(item, limit=limit, depth=depth - 1) for item in limited_items
        ]
        if len(value) > MAX_COLLECTION_ITEMS:
            sanitized_list.append(f"... ({len(value) - MAX_COLLECTION_ITEMS} more)")
        return sanitized_list

    return value


def _structured_log(event: str, **payload: Any) -> None:
    record = {"event": event, **payload}
    try:
        message = json.dumps(record, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        fallback = {"event": event, "payload": "<unserializable>"}
        message = json.dumps(fallback, ensure_ascii=False)
    logger.info(message)


def _split_inline_comment(text: str) -> tuple[str, Optional[str]]:
    in_single = False
    in_double = False
    escaped = False
    for idx, char in enumerate(text):
        if char == "\\" and not escaped:
            escaped = True
            continue
        if char == "'" and not in_double and not escaped:
            in_single = not in_single
        elif char == '"' and not in_single and not escaped:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return text[:idx], text[idx:]
        escaped = False
    return text, None


def _should_adjust_from_comments(comments: Sequence[str]) -> bool:
    combined = " ".join(comment.strip().lower() for comment in comments if comment and comment.strip())
    return any(keyword in combined for keyword in CONFIRMATION_KEYWORDS)


def _indent_child(indent: str) -> str:
    if indent and "\t" in indent and indent.strip() == "":
        return indent + "\t"
    return indent + "    "


def _adjust_indices_in_line(body: str) -> tuple[str, int]:
    adjustments = 0
    result_chars: list[str] = []
    i = 0
    in_single = False
    in_double = False
    length = len(body)

    while i < length:
        char = body[i]
        prev_char = body[i - 1] if i > 0 else ""

        if char == "'" and not in_double and prev_char != "\\":
            in_single = not in_single
            result_chars.append(char)
            i += 1
            continue
        if char == '"' and not in_single and prev_char != "\\":
            in_double = not in_double
            result_chars.append(char)
            i += 1
            continue
        if char == "\\" and (in_single or in_double) and i + 1 < length:
            result_chars.append(char)
            result_chars.append(body[i + 1])
            i += 2
            continue

        if not in_single and not in_double and char == "[":
            j = i + 1
            if j < length and body[j].isdigit():
                k = j
                while k < length and body[k].isdigit():
                    k += 1
                if k < length and body[k] == "]":
                    prev_index = i - 1
                    while prev_index >= 0 and body[prev_index].isspace():
                        prev_index -= 1
                    prev_significant = body[prev_index] if prev_index >= 0 else ""
                    if prev_significant and (
                        prev_significant.isalnum()
                        or prev_significant in ")]}_"
                        or prev_significant in "._"
                    ):
                        index_value = int(body[j:k]) + 1
                        result_chars.append(f"[{index_value}]")
                        adjustments += 1
                        i = k + 1
                        continue
        result_chars.append(char)
        i += 1

    return "".join(result_chars), adjustments


def _snippet_metadata(snippet: str) -> dict[str, Any]:
    return {
        "snippet_length": len(snippet),
        "snippet_preview": _truncate_text(snippet),
        "has_trailing_newline": snippet.endswith("\n"),
    }


def _log_tool_event(tool_name: str, phase: str, call_id: str, **extra: Any) -> None:
    _structured_log("tool_event", tool=tool_name, phase=phase, call_id=call_id, **extra)


def _log_tool_start(tool_name: str, *, snippet: str, **extra: Any) -> str:
    call_id = str(uuid.uuid4())
    metadata = {**_snippet_metadata(snippet), **extra}
    _log_tool_event(tool_name, "start", call_id, **metadata)
    return call_id


def _log_tool_end(tool_name: str, call_id: str, *, result: str, **extra: Any) -> None:
    metadata = {
        "result_length": len(result),
        "result_preview": _truncate_text(result),
        "has_trailing_newline": result.endswith("\n"),
    }
    metadata.update(extra)
    _log_tool_event(tool_name, "end", call_id, **metadata)


def _build_agent_context(agent_input: Any) -> dict[str, Any]:
    context: dict[str, Any] = {}
    if isinstance(agent_input, Mapping):
        context["input_keys"] = sorted(str(key) for key in agent_input.keys())
        snippet = agent_input.get("snippet")
        instructions = agent_input.get("instructions")
        if isinstance(snippet, str):
            context["snippet_preview"] = _truncate_text(snippet)
            context["snippet_length"] = len(snippet)
        if isinstance(instructions, str):
            context["instructions_preview"] = _truncate_text(instructions)
            context["instructions_length"] = len(instructions)
        messages = agent_input.get("messages")
        if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes, bytearray)):
            context["message_count"] = len(messages)
    else:
        context["input_preview"] = _sanitize_for_logging(agent_input)
    return context


class StructuredLoggingAgent(Runnable):
    def __init__(self, runnable: Runnable):
        super().__init__()
        self._runnable = runnable

    def __getattr__(self, item: str) -> Any:
        return getattr(self._runnable, item)

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Any:
        run_id = str(uuid.uuid4())
        context = _build_agent_context(input)
        context["run_mode"] = "invoke"
        if config is not None:
            context["config_preview"] = _sanitize_for_logging(config)
        if kwargs:
            context["extra_args"] = _sanitize_for_logging(kwargs)
        _structured_log("agent_run_start", run_id=run_id, **context)
        try:
            result = self._runnable.invoke(input, config=config, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive logging
            _structured_log("agent_run_error", run_id=run_id, run_mode="invoke", error=str(exc))
            raise
        _structured_log(
            "agent_run_end",
            run_id=run_id,
            run_mode="invoke",
            output_preview=_sanitize_for_logging(result),
        )
        return result

    async def ainvoke(self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Any:
        run_id = str(uuid.uuid4())
        context = _build_agent_context(input)
        context["run_mode"] = "ainvoke"
        if config is not None:
            context["config_preview"] = _sanitize_for_logging(config)
        if kwargs:
            context["extra_args"] = _sanitize_for_logging(kwargs)
        _structured_log("agent_run_start", run_id=run_id, **context)
        ainvoke_method = getattr(self._runnable, "ainvoke", None)
        try:
            if callable(ainvoke_method):
                result = await ainvoke_method(input, config=config, **kwargs)
            else:
                result = await super().ainvoke(input, config=config, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive logging
            _structured_log("agent_run_error", run_id=run_id, run_mode="ainvoke", error=str(exc))
            raise
        _structured_log(
            "agent_run_end",
            run_id=run_id,
            run_mode="ainvoke",
            output_preview=_sanitize_for_logging(result),
        )
        return result

    def stream(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        run_id = str(uuid.uuid4())
        context = _build_agent_context(input)
        context["run_mode"] = "stream"
        if config is not None:
            context["config_preview"] = _sanitize_for_logging(config)
        if kwargs:
            context["extra_args"] = _sanitize_for_logging(kwargs)
        _structured_log("agent_run_start", run_id=run_id, **context)

        stream_method = getattr(self._runnable, "stream", None)
        iterator = (
            stream_method(input, config=config, **kwargs)
            if callable(stream_method)
            else super().stream(input, config=config, **kwargs)
        )
        try:
            for chunk in iterator:
                _structured_log(
                    "agent_stream_chunk",
                    run_id=run_id,
                    chunk_preview=_sanitize_for_logging(chunk),
                )
                yield chunk
        except Exception as exc:  # pragma: no cover - defensive logging
            _structured_log("agent_run_error", run_id=run_id, run_mode="stream", error=str(exc))
            raise
        _structured_log("agent_run_end", run_id=run_id, run_mode="stream")

    async def astream(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        run_id = str(uuid.uuid4())
        context = _build_agent_context(input)
        context["run_mode"] = "astream"
        if config is not None:
            context["config_preview"] = _sanitize_for_logging(config)
        if kwargs:
            context["extra_args"] = _sanitize_for_logging(kwargs)
        _structured_log("agent_run_start", run_id=run_id, **context)

        astream_method = getattr(self._runnable, "astream", None)
        async_iterator = (
            astream_method(input, config=config, **kwargs)
            if callable(astream_method)
            else super().astream(input, config=config, **kwargs)
        )
        try:
            async for chunk in async_iterator:
                _structured_log(
                    "agent_stream_chunk",
                    run_id=run_id,
                    chunk_preview=_sanitize_for_logging(chunk),
                )
                yield chunk
        except Exception as exc:  # pragma: no cover - defensive logging
            _structured_log("agent_run_error", run_id=run_id, run_mode="astream", error=str(exc))
            raise
        _structured_log("agent_run_end", run_id=run_id, run_mode="astream")

    def batch(
        self,
        inputs: TypingSequence[Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> TypingSequence[Any]:
        run_id = str(uuid.uuid4())
        items_count = len(inputs) if hasattr(inputs, "__len__") else None
        context = {"run_mode": "batch", "items": items_count}
        if config is not None:
            context["config_preview"] = _sanitize_for_logging(config)
        if kwargs:
            context["extra_args"] = _sanitize_for_logging(kwargs)
        _structured_log("agent_run_start", run_id=run_id, **context)

        batch_method = getattr(self._runnable, "batch", None)
        try:
            if callable(batch_method):
                result = batch_method(inputs, config=config, **kwargs)
            else:
                result = super().batch(inputs, config=config, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive logging
            _structured_log("agent_run_error", run_id=run_id, run_mode="batch", error=str(exc))
            raise
        _structured_log(
            "agent_run_end",
            run_id=run_id,
            run_mode="batch",
            items=items_count,
            output_preview=_sanitize_for_logging(result),
        )
        return result

    async def abatch(
        self,
        inputs: TypingSequence[Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> TypingSequence[Any]:
        run_id = str(uuid.uuid4())
        items_count = len(inputs) if hasattr(inputs, "__len__") else None
        context = {"run_mode": "abatch", "items": items_count}
        if config is not None:
            context["config_preview"] = _sanitize_for_logging(config)
        if kwargs:
            context["extra_args"] = _sanitize_for_logging(kwargs)
        _structured_log("agent_run_start", run_id=run_id, **context)

        abatch_method = getattr(self._runnable, "abatch", None)
        try:
            if callable(abatch_method):
                result = await abatch_method(inputs, config=config, **kwargs)
            else:
                result = await super().abatch(inputs, config=config, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive logging
            _structured_log("agent_run_error", run_id=run_id, run_mode="abatch", error=str(exc))
            raise
        _structured_log(
            "agent_run_end",
            run_id=run_id,
            run_mode="abatch",
            items=items_count,
            output_preview=_sanitize_for_logging(result),
        )
        return result


@tool("add_import")
def add_import_buggy(snippet: str, module: str) -> str:
    call_id = _log_tool_start("add_import", snippet=snippet, module=module)
    normalized_module = module.strip()
    import_line = f"import {normalized_module}"
    if snippet:
        needs_separator = not snippet.startswith("\n")
        result = f"{import_line}\n{snippet}" if needs_separator else f"{import_line}{snippet}"
    else:
        result = f"{import_line}\n"
    _log_tool_end("add_import", call_id, result=result, import_line=import_line)
    return result


@tool("rename_symbol")
def rename_first_occurrence(snippet: str, old: str, new: str) -> str:
    call_id = _log_tool_start("rename_symbol", snippet=snippet, old=old, new=new)
    if not old or old == new:
        _log_tool_end("rename_symbol", call_id, result=snippet, replacements=0)
        return snippet

    replacements = 0
    if old.isidentifier() and new.isidentifier():
        source = snippet if snippet.endswith("\n") else f"{snippet}\n"
        buffer = io.StringIO(source)
        try:
            tokens = tokenize.generate_tokens(buffer.readline)
            new_tokens = []
            for token in tokens:
                token_type, token_string, start, end, line = token
                if token_type == tokenize.NAME and token_string == old:
                    token_string = new
                    replacements += 1
                new_tokens.append((token_type, token_string, start, end, line))
            result = tokenize.untokenize(new_tokens)
            if not snippet.endswith("\n") and result.endswith("\n"):
                result = result[:-1]
        except tokenize.TokenError:
            result, replacements = re.subn(rf"(?<!\w){re.escape(old)}(?!\w)", new, snippet)
    else:
        result, replacements = re.subn(re.escape(old), new, snippet)

    _log_tool_end("rename_symbol", call_id, result=result, replacements=replacements)
    return result


@tool("fix_indexing")
def bump_indices_off_by_one(snippet: str) -> str:
    call_id = _log_tool_start("fix_indexing", snippet=snippet)
    lines = snippet.splitlines(keepends=True)
    amended_lines: list[str] = []
    pending_comments: list[str] = []
    adjustments = 0

    for raw_line in lines:
        stripped = raw_line.lstrip()
        if stripped.startswith("#"):
            pending_comments.append(stripped[1:].strip())
            amended_lines.append(raw_line)
            continue

        line_body = raw_line[:-1] if raw_line.endswith("\n") else raw_line
        body_without_comment, inline_comment = _split_inline_comment(line_body)
        comment_texts = list(pending_comments)
        if inline_comment is not None:
            comment_texts.append(inline_comment[1:].strip())
        if comment_texts and _should_adjust_from_comments(comment_texts):
            updated_body, line_adjustments = _adjust_indices_in_line(body_without_comment)
            if line_adjustments:
                adjustments += line_adjustments
            reconstructed = updated_body
            if inline_comment is not None:
                reconstructed += inline_comment
            if raw_line.endswith("\n"):
                reconstructed += "\n"
            amended_lines.append(reconstructed)
        else:
            amended_lines.append(raw_line)
        pending_comments = []

    result = "".join(amended_lines)
    _log_tool_end("fix_indexing", call_id, result=result, indices_updated=adjustments)
    return result


FUNCTION_SINGLE_LINE_PATTERN = re.compile(
    r"""
    ^(?P<indent>[ \t]*)(?P<async>async\s+)?def\s+
    (?P<name>[A-Za-z_][A-Za-z0-9_]*)
    \((?P<params>[^)]*)\)
    (?P<returns>\s*->\s*[^:]+)?
    :(?P<suffix>[^\n]*)$
    """,
    re.MULTILINE | re.VERBOSE,
)


@tool("stub_function")
def stub_function_singleline(snippet: str) -> str:
    call_id = _log_tool_start("stub_function", snippet=snippet)
    replacements = 0

    def _replacement(match: re.Match[str]) -> str:
        nonlocal replacements
        indent = match.group("indent") or ""
        async_prefix = match.group("async") or ""
        name = match.group("name")
        params = match.group("params")
        returns = match.group("returns") or ""
        suffix = match.group("suffix") or ""

        body_segment, inline_comment = _split_inline_comment(suffix)
        if body_segment.strip() not in {"pass", "...", "Ellipsis"}:
            return match.group(0)

        replacements += 1
        header = f"{indent}{async_prefix}def {name}({params}){returns}:"
        body_indent = _indent_child(indent)
        body_line = f"{body_indent}raise NotImplementedError()"
        if inline_comment:
            body_line += inline_comment
        return f"{header}\n{body_line}"

    result, _ = FUNCTION_SINGLE_LINE_PATTERN.subn(_replacement, snippet)
    _log_tool_end("stub_function", call_id, result=result, stubs_created=replacements)
    return result


SYSTEM_PROMPT = (
    "You are Coder. Your job is finding flaws in a user-glam code and fixing them using the tools that you have."
    "Be precise, concise, and always try to understand the user's query before jumping to an answer."
    "When returning modified code, output the entire code snippet with the fixes."
)


app = StructuredLoggingAgent(
    create_agent(
        model="openai:gpt-4o-mini",
        tools=[add_import_buggy, rename_first_occurrence, bump_indices_off_by_one, stub_function_singleline],
        system_prompt=SYSTEM_PROMPT,
    )
)
