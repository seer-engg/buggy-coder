# Buggy Coder

- buggy coder is a coding agent with an intentionally buggy implementation.
- the goal of this agent is to find flaws in a user's code and fix them using the tools.
- but the agent is intentionally buggy, so it will return incorrect fixes.
- the meta goal of this agent is testing eval agents to see if they can catch the bugs.

## Toolchain

The agent uses a collection of LangChain tools to manipulate user snippets:

- `add_import_buggy`: inserts an import statement but intentionally mangles the module name.
- `rename_symbol`: renames the first occurrence of a symbol in the snippet.
- `fix_indexing`: bumps numeric list indices by one.
- `stub_function`: rewrites `def name(...): pass` bodies into simple stubs.
- `fix_python_syntax`: repairs malformed function definitions by inserting the missing colon.

Originally, no tool could repair syntax errors in a function declaration, which meant
that snippets such as `def sum(a,b) return a+b` could not be corrected. The new
`fix_python_syntax` tool was added to close that gap.

## How to use

```bash
cd agents/buggy-coder
langgraph dev --port 2025
```