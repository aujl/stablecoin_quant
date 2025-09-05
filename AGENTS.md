# Repository Guidelines

## Project Structure & Module Organization
- `source/stable_yield_lab.py`: Core library (data model, sources, metrics, visuals).
- `source/stable_yield_demo.py`: Example usage and quick sanity run.
- `source/sample_pools.csv`: Small dataset used by the demo.
- `create.py`: Script that can (re)generate library/demo/sample files.
- `pyproject.toml`: Python project metadata and dependencies.

## Build, Test, and Development Commands
- Run demo: `python source/stable_yield_demo.py` (renders sample charts, prints counts).
- Quick REPL test: `python -i source/stable_yield_lab.py` then import classes in-session.
- Install deps (local dev): `python -m pip install -U pip && python -m pip install pandas matplotlib`.
- Optional tooling: `pip install ruff black pytest mypy`.

## Coding Style & Naming Conventions
- Style: PEP 8 with type hints; prefer explicit, descriptive names.
- Indentation: 4 spaces; max line length 100 where reasonable.
- Modules: keep core logic in `source/stable_yield_lab.py`; new adapters live under a new `source/adapters/` (e.g., `defillama.py`) and are imported in the demo as needed.
- Formatting: `black source`; Lint: `ruff check source` (fix with `ruff --fix`).

## Testing Guidelines
- Framework: `pytest` (add under `tests/`).
- Naming: files `test_*.py`; tests mirror modules, e.g., `tests/test_metrics.py`.
- Coverage: target core paths (PoolRepository filters, Metrics, Visualizer data prep). Run with `pytest -q`.

## Commit & Pull Request Guidelines
- Commits: small, focused; conventional prefix encouraged (e.g., `feat: add Morpho adapter`, `fix: NaN handling`).
- PRs: include a clear summary, rationale, before/after notes, and screenshots of charts if visuals change.
- Link issues when applicable; check passes for lint and tests.

## Security & Configuration Tips
- Offline by default: network adapters are stubs. When adding real HTTP clients, keep keys in env vars (e.g., `MORPHO_API_KEY`) and never commit secrets.
- Data hygiene: avoid committing large datasets; prefer small samples in `source/` and document sources.
- Reproducibility: keep demo runnable without credentials; guard network calls behind flags.

