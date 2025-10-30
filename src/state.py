from typing import Annotated, TypedDict, Optional

from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage


class IOState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]


class BuggyCoderState(IOState):
    snippet: str
    instructions: str
    # New fields for E2B execution results
    execution_result: Optional[str] = None
    execution_error: Optional[str] = None

