# Testing Guidelines

These instructions govern all tests under `tests/`.

## Practices
- Use `pytest` with files named `test_*.py`.
- Write tests before implementation when possible.
- Prefer parametrised tests and fixtures to reduce duplication.
- Mock network and filesystem interactions; tests must be deterministic.

## Coverage
- Aim for comprehensive coverage of new or changed modules.
- For quantitative logic, include edge cases and numerical stability checks.
- Measure coverage with `pytest --cov` and keep reports clean of missing lines.

## Execution
- Ensure `poetry run pytest` passes locally before committing.
- Keep test data small and checked into version control when feasible.
