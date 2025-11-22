import re
from langchain.agents import create_agent
from langchain_core.tools import tool

@tool("detect_common_flaws")
def detect_common_flaws(code: str) -> str:
    flaws = []
    if '==' in code and '!=' in code:
        flaws.append('Use of both == and != might be inconsistent.')
    if 'print(' in code:
        flaws.append('Use of print statements for debugging.')
    # Add more pattern checks as needed
    return '\n'.join(flaws) if flaws else 'No common flaws detected.'

@tool("auto_fix")
def auto_fix_code(code: str, flaw: str) -> str:
    # Implement simple fixes based on flaw description
    if 'print statements' in flaw:
        # Remove print statements
        fixed_code = re.sub(r"\bprint\(.*?\)", "# print statement removed", code)
        return fixed_code
    # Add more fix patterns as needed
    return code

# Create agent with new tools
agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[add_import_buggy, rename_first_occurrence, bump_indices_off_by_one, stub_function_singleline, detect_common_flaws, auto_fix_code],
    system_prompt=SYSTEM_PROMPT,
)
