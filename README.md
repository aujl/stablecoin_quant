# stablecoin_quant

[![Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/aujl/stablecoin_quant/gh-pages/coverage/badge.json)](https://aujl.github.io/stablecoin_quant/coverage/)

Experimental toolkit for analyzing and visualizing yields on stablecoin pools.

Latest test coverage is published automatically from CI to [GitHub Pages](https://aujl.github.io/stablecoin_quant/coverage/), which hosts the HTML report alongside the badge metadata for the shield above.

## Project Layout

The repository uses a conventional layout to keep code, tests, and
documentation organized:

```
stablecoin_quant/
├── src/        # library source code and demo scripts
├── tests/      # unit tests
├── docs/       # project documentation
└── pyproject.toml
```

Run formatting, linting, and type checks with pre-commit and pytest:

```bash
poetry run pre-commit run -a
poetry run pytest -q
```

### Local Commands & Environment Variables

- Run the demo: `poetry run python src/stable_yield_demo.py configs/demo.toml`
- The demo can plot NAV and cumulative yield trajectories when
  `initial_investment` and `yields_csv` are supplied via the config.
- Override the demo config path with `STABLE_YIELD_CONFIG=/path/to/config.toml`

## Investor Quickstart

Run the demo with sample historical returns to generate Net Asset Value (NAV) and yield curves:

```bash
poetry run python src/stable_yield_demo.py configs/demo.toml
```

The script applies `stable_yield_lab.performance.nav_trajectories` and `performance.yield_trajectories` to compute time-series performance.
`Visualizer.line_chart` renders both the NAV and cumulative yield trajectories with a shared helper.

Interactively explore the same calculations from Python:

```pycon
>>> from pathlib import Path
>>> from tempfile import TemporaryDirectory
>>> import pandas as pd
>>> from stable_yield_lab import Visualizer, performance
>>> yields_df = pd.read_csv("src/sample_yields.csv", parse_dates=["timestamp"])
>>> returns = yields_df.pivot(index="timestamp", columns="name", values="period_return")
>>> nav = performance.nav_trajectories(returns, initial_investment=10_000.0)
>>> cumulative_yield = performance.yield_trajectories(returns) * 100.0
>>> with TemporaryDirectory() as tmp:
...     outdir = Path(tmp)
...     Visualizer.line_chart(
...         nav,
...         title="NAV trajectories",
...         ylabel="Portfolio value (USD)",
...         save_path=str(outdir / "nav_curve.png"),
...         show=False,
...     )
...     Visualizer.line_chart(
...         cumulative_yield,
...         title="Cumulative yield (%)",
...         ylabel="Yield (%)",
...         save_path=str(outdir / "yield_curve.png"),
...         show=False,
...     )
```

A steadily rising NAV indicates compounding growth; falling or flat lines flag underperformance.
For a step-by-step example, see [docs/investor_walkthrough.md](docs/investor_walkthrough.md).

### Demo configuration

Configuration values are supplied via TOML. The `[csv]` section controls the
validated ingestion layer that powers the CLI demo:

| Key | Description |
| --- | --- |
| `path` | Path to the cached CSV of pools. |
| `validation` | One of `"none"`, `"warn"`, or `"strict"` to control schema enforcement. |
| `expected_frequency` | Optional pandas-style frequency string (`"D"`, `"W"`, etc.) checked against inferred cadence. |
| `auto_refresh` | When `true`, call the refresh hook before reading cached data. |
| `refresh_url` | Optional HTTP endpoint that returns the latest CSV; used when `auto_refresh = true`. |

When `auto_refresh` is enabled but no `refresh_url` is provided the demo skips
the refresh and logs a warning. Successful ingestion prints the detected
frequency whenever the timestamp column contains enough information to infer
periodicity.

## Codex Workflows

GitHub workflows tag [@codex](https://github.com/features/copilot) to request automated pull request reviews,
issue triage, and comment follow-ups—no external API credentials are required.

- **Triggers**:
  - `pull_request_target` events (`opened`, `synchronize`, `reopened`) post a single comment asking `@codex` to review for tests, security, and documentation.
  - `issues` events (`opened`, `edited`) tag `@codex` for triage guidance and next steps.
  - `issue_comment` events (`created`) notify `@codex` to summarize the thread and suggest owners/actions.


## Risk Scoring

Each pool is assigned a ``risk_score`` in the range ``1`` (lower risk) to ``3``
using ``src/stable_yield_lab/risk_scoring.py``. The score averages three
normalized factors:

1. **Chain reputation** – established networks such as Ethereum receive a
   higher reputation (lower risk), while lesser known chains start at ``0.5``.
2. **Protocol audits** – more security audits reduce risk. The contribution is
   capped at five audits.
3. **Yield volatility** – unstable historical yields increase risk. Volatility
   is expected as a 0–1 value.

The three components are combined and scaled to the ``[1, 3]`` range. During
``Pipeline.run`` the score is computed for every fetched pool.
