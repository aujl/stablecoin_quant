# Visualization Module Guidelines

These instructions apply to `stable_yield_lab.visualization`.

## Design Principles
- Provide thin plotting helpers that accept Pandas objects and return axes/figures where practical.
- Ensure rendering functions accept `show` and `save_path` parameters to support headless environments.
- Keep Matplotlib state isolated; avoid relying on global figures.

## Implementation Notes
- Centralise Matplotlib access through a thin wrapper (`Visualizer._plt`) to enable mocking.
- Validate input columns before plotting and raise helpful errors when required fields are missing.
- Default to non-interactive backends and allow tests to swap in spies.

## Testing Hooks
- Use spies or fixtures to assert Matplotlib interactions without creating windows.
- Integration tests may write PNGs under `tmp_path`; ensure deterministic filenames.
- Capture axis labels, legends, and scaling behaviour to avoid visual regressions.
