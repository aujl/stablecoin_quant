# Analytics Module Guidelines

These instructions cover `stable_yield_lab.analytics` (metrics, performance, portfolio, attribution, risk).

## Design Principles
- Keep analytical routines pure functions with explicit inputs and outputs.
- Document quantitative assumptions (frequency, compounding, currency) in docstrings.
- Optimise for vectorised operations using Pandas/NumPy; avoid Python loops on large datasets.

## Implementation Notes
- Validate shapes and indexes; raise informative errors when expectations are violated.
- Maintain consistent units (APY as decimals, TVL in USD) across functions.
- Provide helper functions that compose cleanly with pipeline and reporting layers.

## Testing Hooks
- Include regression cases for numerical stability (e.g., near-zero volatility, missing values).
- Use `pytest.approx` with tolerances for floating point comparisons.
- Cover risk metrics, attribution breakdowns, and portfolio optimisers with representative scenarios.
