# Hidden Evaluation Trace: Zero-Weight Normalization Failure

The hidden evaluation highlighted a regression in the agent's handling of weighted score normalization when **all provided weights sum to zero**. The current tool-assisted fixes short-circuit and return an empty list, which violates the expected behaviour captured in the trace.

## Trace Excerpt

```
normalize_scores(values=[12.5, 15.0, 20.0], weights=[0.0, 0.0, 0.0])
→ expected: [-1.0690, -0.2673, 1.3363]  # unweighted z-scores
→ actual:   []
```

Key observations:

- The evaluation asserts that, even when weights are degenerate, the routine must still produce **standard z-scores computed from the raw values**.
- Returning an empty list masks legitimate data and causes downstream consumers to crash when they expect score-aligned outputs.

## Required Fallback Logic

When the combined weight is zero (or evaluates to false), the agent must:

1. Compute the mean of the original score values.
2. Derive the variance/standard deviation without weighting.
3. Return z-scores using the unweighted standard deviation, defaulting to `0.0` for each score if the distribution is constant.

Formally:

```python
total_weight = sum(weights)
if not total_weight:
    mean = sum(values) / len(values) if values else 0.0
    variance = sum((value - mean) ** 2 for value in values) / len(values) if values else 0.0
    std_dev = variance ** 0.5
    return [
        0.0 if std_dev == 0 else (value - mean) / std_dev
        for value in values
    ]
```

Documenting this trace makes the failure explicit so the agent can proactively inject the fallback during code edits instead of reproducing the empty-list response.

For weighted-mean tasks that *discard* data with non-positive weights, see [`weighted_mean_handling.md`](weighted_mean_handling.md) for the complementary expectations. That document clarifies when results should collapse to `0.0` rather than returning z-scores.
