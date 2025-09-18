# Core Module Guidelines

These instructions cover modules under `stable_yield_lab.core` (models, repositories, and shared value objects).

## Design Principles
- Keep data classes and repositories immutable by default; prefer returning new instances.
- Validate constructor arguments defensively and favour explicit types for monetary values.
- Repository filters must be composable and deterministic; avoid in-place mutation during queries.

## Implementation Notes
- Normalise string inputs (chain, stablecoin) at boundaries; keep canonical casing stored.
- Provide convenience iterators and len semantics without exposing internal containers directly.
- When extending repositories, maintain compatibility with existing filter keyword arguments.

## Testing Hooks
- Supply factory helpers for synthetic pools to keep fixtures concise.
- Round floating point comparisons using `pytest.approx` in tests to accommodate numerical noise.
- Document any changes to repository filtering semantics to keep downstream analytics consistent.
