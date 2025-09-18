# Analytics Test Guidelines

Mirror the expectations in `src/stable_yield_lab/analytics/AGENTS.md` when crafting tests.

## Fixtures
- Use deterministic Pandas objects with explicit indexes to validate vectorised operations.
- When reusing sample numeric series, centralise them in helper fixtures to avoid duplication.

## Mocking & Determinism
- Avoid mocking analytics internals; instead validate end-to-end transformations on controlled inputs.
- For stochastic algorithms, fix seeds or inject deterministic substitutes before asserting.

## Coverage Goals
- Assert numerical stability around edge cases (zero volatility, missing returns, extreme weights).
- Cover attribution decompositions, portfolio allocations, and risk scoring thresholds.
- Ensure docstring examples stay in sync by running doctests where applicable.
