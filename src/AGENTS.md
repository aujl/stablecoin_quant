# Implementation Guidelines

These instructions apply to all Python modules under `src/`.

## Coding Conventions
- Follow PEP 8 with 4‑space indents and a 120‑character line limit.
- Type hints are required for all public functions and dataclasses.
- Keep modules small, composable, and documented with clear docstrings.
- Favor pure functions; avoid implicit global state.

## Quantitative Development
- Clearly state mathematical formulas and financial assumptions in comments or docstrings.
- Use vectorised NumPy/Pandas operations when practical to ensure performance on large datasets.
- Validate units and currency denominations; avoid magic numbers.
- Ensure algorithms are numerically stable and reproducible.

## Workflow
1. Define requirements.
2. Write failing tests under `tests/`.
3. Implement or update code until tests pass.
4. Refactor and document.

## Data Handling
- Keep sample datasets small and version‑controlled; never commit secrets.
- Wrap external API calls with adapters that can be mocked in tests.
