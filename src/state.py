from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage


class IOState(TypedDict, total=False):
	messages: Annotated[list[AnyMessage], add_messages]


class BuggyCoderState(IOState, total=False):
	snippet: str
	instructions: str
	requirement_summary: str
	needs_revision: bool

