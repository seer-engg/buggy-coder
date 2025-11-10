from typing import Annotated, List, TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage


class IOState(TypedDict, total=False):
	messages: Annotated[list[AnyMessage], add_messages]


class BuggyCoderState(IOState):
	snippet: str
	instructions: str
	edge_cases: List[str]
	validation_notes: List[str]


