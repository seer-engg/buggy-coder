from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage


class IOState(TypedDict, total=False):
	"""Base IO state for the Buggy Coder agent."""
	thread_id: str
	messages: Annotated[list[AnyMessage], add_messages]


class BuggyCoderState(IOState):
	snippet: str
	instructions: str


