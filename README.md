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

## Safety tooling for optional child collections

- The agent now registers a `normalize_iterable_field` tool that rewrites loops consuming `node['children']` or `node.get('children')` into a guarded form. The tool emits:
  ```python
  children = node.get('children') or []
  for child in children:
      ...
  ```
  ensuring that missing keys or `None` values are treated as empty lists before iteration begins.
- When crafting fixes for tree or dictionary structures, normalize optional child collections with this pattern so the agent consistently avoids crashes on absent children.

## Zero-weight normalization fallback

- Hidden evaluation traces (see [`docs/zero_weight_normalization.md`](docs/zero_weight_normalization.md)) revealed that `normalize_scores` implementations must **not** return an empty list when the combined weight is zero. The correct behaviour is to fall back to unweighted z-scores so downstream consumers still receive aligned data.
- The `inject_zero_weight_fallback` tool rewrites zero-weight guards such as `if total_weight == 0: return []` into full mean/variance logic:
  ```python
  if not total_weight:
      unweighted_mean = sum(scores) / len(scores) if scores else 0.0
      unweighted_variance = sum((score - unweighted_mean) ** 2 for score in scores) / len(scores) if scores else 0.0
      unweighted_std = unweighted_variance ** 0.5
      return [0.0 if unweighted_std == 0 else (score - unweighted_mean) / unweighted_std for score in scores]
  ```
- When addressing statistical edge cases, apply this tool (or reproduce its pattern manually) to guarantee the agent preserves results even when weight vectors are degenerate.

## Weighted-mean filtering fallback

- A separate hidden evaluation (documented in [`docs/weighted_mean_handling.md`](docs/weighted_mean_handling.md)) enforces that `weighted_mean` implementations **ignore non-positive weights** before computing aggregates.
- The `filter_nonpositive_weights` tool injects the filtering scaffold and rewrites downstream calculations so only positive-weight pairs contribute:
  ```python
  positive_pairs = [
      (value, weight)
      for value, weight in zip(values, weights)
      if weight > 0
  ]
  if not positive_pairs:
      return 0.0
  filtered_values, filtered_weights = zip(*positive_pairs)
  filtered_values = list(filtered_values)
  filtered_weights = list(filtered_weights)
  ```
- After filtering, subsequent `zip(values, weights)` and `sum(weights)` calls are retargeted to the filtered collections, ensuring the mean remains well-defined. Apply this tool whenever weighted statistics must ignore zero/negative weights.

## Choosing the right fallback

- **Normalization problems** retain all values but switch to unweighted statistics when the total weight is zero.
- **Weighted-mean problems** discard non-positive weights and return `0.0` if nothing remains.
- Use the prompts and tools above to pick the correct fallback pattern for the given task.
