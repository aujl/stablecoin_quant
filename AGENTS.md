# Repository Guidelines

## Project Structure & Module Organization
- `src/stable_yield_lab/`: Core Python package (data model, sources, metrics, visuals).
- `src/stable_yield_demo.py`: Example CLI demo (configurable via args/env) and file-first outputs.
- `src/sample_pools.csv`: Small dataset used by the demo.
- `tests/`: Pytest-based tests.
- `pyproject.toml`: Project metadata and dependencies (Poetry-managed).

## Environment & Dependencies (pyenv + Poetry)
- Python: use `pyenv` with 3.12.
  - `pyenv install 3.12.11 && pyenv local 3.12.11`
- Install Poetry (once): follow https://python-poetry.org/docs/
- Project install: `poetry install` (installs runtime + dev dependencies).

## Build, Test, and Development Commands
- Run demo: `poetry run python src/stable_yield_demo.py`.
  - Options: `--csv`, `--min-tvl`, `--min-base-apy`, `--auto-only/--no-auto-only`,
    `--chains`, `--stablecoins`, `--charts`, `--outdir`, `--no-show`.
  - File-first outputs: when `--outdir` is provided, demo writes `pools_filtered.csv`, `by_chain.csv`, `top10.csv`, and saves charts (PNG) instead of showing them.
- REPL: `poetry run python -i -c "import stable_yield_lab as syl; print(dir(syl))"`.
- Pre-commit: `poetry run pre-commit install` then `poetry run pre-commit run -a`.
- Tests: `poetry run pytest -q`.
- Coverage: `poetry run pytest --cov=stable_yield_lab --cov-report=term-missing`.
- Lint/format: `poetry run black . && poetry run flake8 .`.
- Type check: `poetry run mypy .`.

Note: Prefer Poetry over pip. Only use `pip` in constrained environments where Poetry is unavailable.

## Coding Style & Naming Conventions
- Style: PEP 8 with type hints; explicit, descriptive names.
- Indentation: 4 spaces; max line length 120.
- Modules: core logic under `src/stable_yield_lab/`; adapters live under `src/stable_yield_lab/adapters/` (e.g., `defillama.py`).
- Formatting: `black` (enforced via pre-commit). Lint: `flake8`. Types: `mypy`.

## Testing Guidelines
- Framework: `pytest` (files named `test_*.py`).
- Target coverage: core paths (PoolRepository filters, Metrics, Visualizer data prep).
- Local import convenience: `tests/conftest.py` adds `src/` to `sys.path` for local runs. CI installs the package.

## Commit & Pull Request Guidelines
- Commits: small, focused; conventional prefix encouraged (e.g., `feat: add Morpho adapter`, `fix: NaN handling`).
- PRs: include a clear summary, rationale, before/after notes, and screenshots of charts if visuals change.
- Checks: pre-commit must pass; CI runs pytest with coverage.
- pre-commit.ci: enable on the repository to auto-fix PRs and autoupdate hooks weekly.

## Security & Configuration Tips
- Offline by default: network adapters are stubs. When adding real HTTP clients, keep keys in env vars (e.g., `MORPHO_API_KEY`) and never commit secrets.
- Data hygiene: avoid committing large datasets; prefer small samples in `src/` and document sources.
- Reproducibility: keep demo runnable without credentials; guard network calls behind flags.
