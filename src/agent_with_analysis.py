from langchain.agents import create_agent
from langchain_core.tools import tool
from src.code_analysis_tools import detect_common_flaws, auto_fix_code
from src.logger import log_message

# Initialize agent with new tools for analysis and fixing
agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[detect_common_flaws, auto_fix_code],
    system_prompt=SYSTEM_PROMPT,
)

# Example usage: analyze code and apply fixes

def analyze_and_fix_code(code: str):
    flaws = detect_common_flaws(code)
    log_message(f"Detected flaws: {flaws}")
    for flaw in flaws.split('\n'):
        if flaw:
            code = auto_fix_code(code, flaw)
            log_message(f"Applied fix for flaw: {flaw}")
    return code
