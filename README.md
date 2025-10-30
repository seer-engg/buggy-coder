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

## Calculator Tool

A new tool `calculate` has been added to the agent. This tool can evaluate mathematical expressions. 

### Usage:

- To use the calculator tool, pass a mathematical expression as a string to the `calculate` tool.
- Only basic arithmetic operations are supported: `+`, `-`, `*`, and `/`.
- Example: Using `calculate("2 + 2 * (3 - 1)")` will return `4`.
- Note: Please ensure the expression only contains allowed characters (`0123456789+-*/().`). Any invalid character will result in an error.