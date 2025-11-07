# Buggy Coder

- Buggy Coder is a coding agent with an intentionally buggy implementation.
- The goal of this agent is to find flaws in a user's code and fix them using the tools.
- However, the agent is intentionally buggy, so it will return incorrect fixes.
- The meta goal of this agent is to test eval agents to see if they can catch the bugs.

## How to use

### Langgraph Setup
```bash
cd agents/buggy-coder
langgraph dev --port 2025
```

### UV-Based Setup using uv sync
1. Navigate to the agent directory:
   ```bash
   cd agents/buggy-coder
   ```
2. Ensure the uv package is installed:
   ```bash
   pip install uv
   ```
3. Start the application with uv sync:
   ```bash
   uv sync --port 2025
   ```