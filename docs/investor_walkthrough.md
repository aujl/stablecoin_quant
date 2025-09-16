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

```python
from pathlib import Path

import pandas as pd

from stable_yield_lab import Visualizer, performance


# use bundled weekly returns
yields_df = pd.read_csv("src/sample_yields.csv", parse_dates=["timestamp"])
returns = yields_df.pivot(index="timestamp", columns="name", values="period_return")

nav = performance.nav_curve(returns)
yield_curve = performance.yield_curve(returns)

artifact_dir = Path("artifacts")
artifact_dir.mkdir(exist_ok=True)

Visualizer.plot_nav(nav, save_path=str(artifact_dir / "nav_curve.png"), show=False)
Visualizer.plot_yield(yield_curve, save_path=str(artifact_dir / "yield_curve.png"), show=False)
```

The `performance` module transforms the weekly return matrix into cumulative Net Asset Value and rolling yield series. `Visualizer.plot_nav` and `Visualizer.plot_yield` render the resulting curves and optionally export PNGs for reporting. Binary outputs are generated locally (and via CI artifacts) so no static charts are committed to the repository.

## 3. Interpret the charts

### Net Asset Value

Inspect `artifacts/nav_curve.png` (or the interactive Matplotlib window if `show=True`). A steadily rising NAV indicates compounding growth across the selected pools, while drawdowns or flat stretches flag underperformance periods that merit a deeper dive into pool-specific events.

### Weekly Yield

Open `artifacts/yield_curve.png` to review the annualised yield profile. Yield spikes or drops highlight weeks where realised returns diverged from the long-run average—use them to annotate catalysts in investor updates or to stress-test assumptions.

### Concentration & Risk Table

The demo also emits `concentration.csv`, which now includes per-scope Sharpe, Sortino,
maximum drawdown, and negative-period share metrics whenever historical returns are
available. Filter rows whose `scope` begins with `pool:` to compare individual vault
histories, or look at the aggregate `total` and `chain:<name>` entries to understand
how diversification and platform choice affect realised performance.

## 4. Compare with riskfolio

The portfolio behaviour resembles the [Riskfolio-Lib portfolio optimization example](https://riskfolio-lib.readthedocs.io/en/latest/portfolio.html), which uses `rp.Portfolio` to build efficient frontiers from historical returns. Matching NAV and yield trajectories across both libraries provides credibility that the calculations align with industry-standard techniques.
