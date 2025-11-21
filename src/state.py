from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class IOState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]


class PRSyncState(TypedDict, total=False):
    """State container for the PR → Asana synchronisation workflow."""

    repository: str
    pull_request_number: int
    pull_request: dict[str, Any]
    asana_task_gids: list[str]
    actions: list[str]
    errors: list[str]
    logs: list[str]
    summary: str
