# Pipeline Module Guidelines

These instructions cover `stable_yield_lab.pipeline` and its helpers.

## Design Principles
- Treat the pipeline as orchestration glue: compose data sources, analytics, and reporting steps.
- Keep configuration parsing separate from execution logic; accept explicit dependencies.
- Expose deterministic behaviour; avoid network access inside pipeline stages.

## Implementation Notes
- Provide clear hooks for injecting mock sources or repositories during tests.
- Ensure historical data pipelines normalise timezone-aware indexes.
- Bubble up errors with context so analysts can trace failing stages quickly.

## Testing Hooks
- Use local CSV fixtures for integration coverage (`sample_pools.csv`, `sample_yields.csv`).
- When adding stages, create focused unit tests alongside end-to-end regression cases.
- Leverage `tmp_path` for filesystem interactions to keep tests isolated.
