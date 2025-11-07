import re
from src.tools import add_import, rename_symbol, fix_indexing, stub_function

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
