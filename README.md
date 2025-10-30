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

## Persistent memory

The Buggy Coder agent now keeps conversation state between sessions by using a
SQLite-backed LangGraph checkpointer. When you start a run, provide a
`thread_id` inside the runnable configuration to resume the previous context.
For example:

```python
config = {"configurable": {"thread_id": "user-123"}}
app.invoke({"messages": [{"role": "user", "content": "..."}]}, config)
```

Re-using the same `thread_id` will surface prior exchanges stored in
`storage/buggy_coder_memory.sqlite`.
