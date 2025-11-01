# Buggy Coder

- buggy coder is a coding agent with an intentionally buggy implementation.
- the goal of this agent is to find flaws in a user's code and fix them using the tools.
- but the agent is intentionally buggy, so it will return incorrect fixes.
- the meta goal of this agent is testing eval agents to see if they can catch the bugs.

## How to use

```bash
cd agents/buggy-coder
langgraph dev --port 2025
```

## Debugging and trace logs

Every agent run now records a high-level trace of tool calls, LLM prompts, and final outputs. The logs are stored in
`logs/agent_traces.log` as newline-delimited JSON to make it easy to inspect or replay problematic runs. Delete the
file if you want to reset the history before reproducing an issue locally.

## Syntax regression tests

Basic regression tests covering missing-colon syntax errors are located in `tests/test_syntax_guard.py`. Run them with:

```bash
python -m unittest tests/test_syntax_guard.py
```
