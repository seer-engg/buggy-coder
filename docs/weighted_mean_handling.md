# Hidden Evaluation Trace: Weighted-Mean Handling

A separate hidden evaluation targets the agent's behaviour when computing **weighted means**. The test revealed that fixes must remove data points associated with non-positive weights before computing the mean.

## Trace Excerpt

```
weighted_mean(values=[3.0, 7.0, 10.0], weights=[1.0, 0.0, -2.0])
→ expected: 3.0  # Only the first pair counts because the others have non-positive weights
→ actual:   4.0  # Current logic keeps zero & negative weights, producing the wrong mean
```

Key requirements:

1. **Filter non-positive weights**: Pairs where the weight is `<= 0` must be removed prior to computing the weighted sum and total weight.
2. **Fallback when empty**: If no positive-weight pairs remain after filtering, the function must immediately return `0.0` to avoid division-by-zero or stale values.
3. **Maintain alignment**: Subsequent calculations should operate on the filtered collections (e.g., use `zip(filtered_values, filtered_weights)` and `sum(filtered_weights)`).

## Recommended Template

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

weighted_sum = sum(value * weight for value, weight in zip(filtered_values, filtered_weights))
total_weight = sum(filtered_weights)
return weighted_sum / total_weight
```

Referencing this expectation ensures fixes distinguish between **normalization** fallbacks (which operate on all values when weights collapse to zero) and **weighted-mean** fallbacks (which ignore non-positive weights and default to `0.0`).
