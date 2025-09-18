# Pipeline Test Guidelines

Keep tests aligned with `src/stable_yield_lab/pipeline/AGENTS.md`.

## Fixtures
- Use `sample_pools.csv` and `sample_yields.csv` from `src/` for integration smoke tests.
- Provide dummy source classes within tests to exercise pipeline hooks without IO.

## Mocking & Determinism
- Avoid patching the pipeline module globally; inject test doubles via constructor arguments.
- When simulating historical data, ensure timestamps are timezone-aware (`tz="UTC"`).

## Coverage Goals
- Cover configuration loading, repository assembly, and risk-scoring enrichment paths.
- Exercise both snapshot (`Pipeline.run`) and historical (`Pipeline.run_history`) flows.
- Validate error messages and warnings when required configuration keys are missing.
