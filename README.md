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

The agent now keeps a per-thread conversation history on disk so follow-up
conversations can resume where you left off, even after closing a thread.

- The storage location defaults to `.memory/buggy_coder`. Set the
  `BUGGY_CODER_MEMORY_DIR` environment variable to customise where memories
  are stored.
- When starting a new conversation, pass the same `thread_id` (or
  `session_id`) to the graph to retrieve the previous history.
