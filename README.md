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

## Regression testing

Run the regression test that guards against the previous `parse_positive_int` failure:

```bash
pytest tests/test_parse_positive_int.py
```

## Manual eval verification

After updating the agent prompt and workflow, re-run the problematic eval scenario to
confirm that non-integer and non-positive inputs now raise `ValueError` as expected.
