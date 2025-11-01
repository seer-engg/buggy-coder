# Buggy Coder

- buggy coder is a coding agent with an intentionally buggy implementation.
- the goal of this agent is to find flaws in a user's code and fix them using the tools.
- but the agent is intentionally buggy, so it will return incorrect fixes.
- the meta goal of this agent is testing eval agents to see if they can catch the bugs.

## Recent improvements

Buggy Coder now ships with general-purpose editing tools (`ensure_import`,
`apply_structured_patch`, and `validate_python`) and an expanded system prompt that
instructs it to produce complete code listings, explanations, validations, and tool
logs. These additions enable the agent to perform multi-line refactors, ensure
correct import management, and surface syntax issues before replying.

## How to use

```bash
cd agents/buggy-coder
langgraph dev --port 2025
```