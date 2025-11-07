import re
from langchain.agents import create_agent
from langchain_core.tools import tool

@tool("add_import")
def add_import(snippet: str, module: str) -> str:
    """Insert an import at the top of the snippet."""
    return f"import {module}\n" + snippet

@tool("rename_symbol")
def rename_symbol(snippet: str, old: str, new: str) -> str:
    """Rename a variable or function name in the snippet."""
    return snippet.replace(old, new, 1)

@tool("fix_indexing")
def fix_indexing(snippet: str) -> str:
    """Adjust numeric indices in list indexing."""
    def bump(match: re.Match) -> str:
        num = int(match.group(1))
        return f"[{num + 1}]"
    return re.sub(r"\[(\d+)\]", bump, snippet)

@tool("stub_function")
def stub_function(snippet: str) -> str:
    """Replace 'def name(...): pass' with a simple function stub."""
    return re.sub(r"def\s+(\w+)\(.*\):\s*pass", r"def \1():\n    return None", snippet)

SYSTEM_PROMPT = (
    "You are Coder. Your job is finding flaws in a user-gam code and fixing them using the tools that you have."
    "Be precise, concise, and always try to understand the user's query before jumping to an answer."
    "When returning modified code, output the entire code snippet with the fixes."
)

app = create_agent(
    model="openai:gpt-4o-mini",
    tools=[add_import, rename_symbol, fix_indexing, stub_function],
    system_prompt=SYSTEM_PROMPT,
)
