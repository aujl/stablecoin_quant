# Visualization Test Guidelines

Follow the rendering guidance in `src/stable_yield_lab/visualization/AGENTS.md`.

## Fixtures
- Use Matplotlib spy fixtures to capture plotting calls without writing files.
- When asserting image output, write to `tmp_path` and validate byte existence/size only.

## Mocking & Determinism
- Monkeypatch `Visualizer._plt` rather than global Matplotlib modules to keep tests isolated.
- Freeze any generated timestamps or random colours if added in the future.

## Coverage Goals
- Cover both interaction-level tests (spy-based) and integration tests that emit PNGs.
- Assert axis labels, titles, legends, and scaling logic to guard against regressions.
- Keep tolerance for numeric conversions (e.g., APY percentages) explicit in assertions.
