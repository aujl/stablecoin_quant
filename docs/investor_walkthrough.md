# Investor Walkthrough

This guide demonstrates how to evaluate a small set of stablecoin pools using the toolkit's performance utilities and plotting helpers.

## 1. Setup

1. Install dependencies:
   ```bash
   poetry install
   ```
2. Generate charts and reports using the demo:
   ```bash
   poetry run python src/stable_yield_demo.py configs/demo.toml
   ```

## 2. Load sample performance data

```pycon
>>> from pathlib import Path
>>> from tempfile import TemporaryDirectory
>>> import pandas as pd
>>> from stable_yield_lab import Visualizer, performance
>>>
>>> yields_df = pd.read_csv("src/sample_yields.csv", parse_dates=["timestamp"])
>>> returns = yields_df.pivot(index="timestamp", columns="name", values="period_return")
>>> nav = performance.nav_trajectories(returns, initial_investment=10_000.0)
>>> yield_pct = performance.yield_trajectories(returns) * 100.0
>>> with TemporaryDirectory() as tmp:
...     artifact_dir = Path(tmp)
...     Visualizer.line_chart(
...         nav,
...         title="NAV trajectories",
...         ylabel="Portfolio value (USD)",
...         save_path=str(artifact_dir / "nav_curve.png"),
...         show=False,
...     )
...     Visualizer.line_chart(
...         yield_pct,
...         title="Cumulative yield (%)",
...         ylabel="Yield (%)",
...         save_path=str(artifact_dir / "yield_curve.png"),
...         show=False,
...     )
```

The `performance` module transforms the weekly return matrix into Net Asset Value trajectories and cumulative yield series via `nav_trajectories` and `yield_trajectories`. `Visualizer.line_chart` renders both curves and optionally exports PNGs for reporting. Binary outputs are generated locally (and via CI artifacts) so no static charts are committed to the repository. Replace `TemporaryDirectory()` with a persistent `Path` if you want to keep the generated image assets between runs.

## 3. Interpret the charts

### Net Asset Value

Inspect the saved `nav_curve.png` (or the interactive Matplotlib window if `show=True`). A steadily rising NAV indicates compounding growth across the selected pools, while drawdowns or flat stretches flag underperformance periods that merit a deeper dive into pool-specific events.

### Weekly Yield

Open `yield_curve.png` from the same output directory to review the annualised yield profile. Yield spikes or drops highlight weeks where realised returns diverged from the long-run average—use them to annotate catalysts in investor updates or to stress-test assumptions.

## 4. Compare with riskfolio

The portfolio behaviour resembles the [Riskfolio-Lib portfolio optimization example](https://riskfolio-lib.readthedocs.io/en/latest/portfolio.html), which uses `rp.Portfolio` to build efficient frontiers from historical returns. Matching NAV and yield trajectories across both libraries provides credibility that the calculations align with industry-standard techniques.
