# Reporting Module Guidelines

These instructions cover `stable_yield_lab.reporting`.

## Design Principles
- Keep reporting functions pure and file-first: accept repositories/dataframes and return written paths.
- Encapsulate filesystem writes; allow custom directories via parameters.
- Surface informative warnings/messages when metrics cannot be computed.

## Implementation Notes
- Maintain stable CSV schemas; document column changes in release notes and tests.
- Parameterise fee assumptions, lookback windows, and thresholds with sensible defaults.
- Reuse analytics helpers (metrics, weighting) instead of re-implementing calculations.

## Testing Hooks
- Integration tests should call reporting functions with `tmp_path` directories and assert on CSV contents.
- Cover realised APY edge cases (insufficient observations, missing returns).
- When adding new outputs, update reporting tests to include column/filename assertions.
