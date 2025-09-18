# Reporting Test Guidelines

Follow the reporting module practices in `src/stable_yield_lab/reporting/AGENTS.md`.

## Fixtures
- Reuse `PoolRepository` fixtures and build synthetic returns DataFrames with explicit UTC indexes.
- Use `tmp_path` for file outputs; assert on written CSV content rather than in-memory side effects.

## Mocking & Determinism
- Avoid mocking Pandas; instead, supply controlled inputs that exercise fee and weighting paths.
- When pipeline data is required, load it via `Pipeline` helpers to keep parity with production flows.

## Coverage Goals
- Validate concentration metrics, warning generation, and realised APY calculations (including edge cases).
- Assert CSV schemas remain stable (column order, names) to prevent breaking downstream consumers.
- Include regression tests for tolerance thresholds (minimum observations, lookback windows).
