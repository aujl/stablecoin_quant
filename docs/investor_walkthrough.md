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
3. Import immutable pool models from the ``stable_yield_lab.core`` namespace when
   scripting your own analytics:
   ```python
   from stable_yield_lab.core import Pool, PoolRepository
   ```

## 2. Load sample performance data

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from stable_yield_lab.analytics import performance
from stable_yield_lab.core import PoolRepository
from stable_yield_lab.visualization import Visualizer

yields_df = pd.read_csv("src/sample_yields.csv", parse_dates=["timestamp"])
returns = yields_df.pivot(index="timestamp", columns="name", values="period_return")
nav = performance.nav_trajectories(returns, initial_investment=10_000.0)
yield_pct = performance.yield_trajectories(returns) * 100.0

with TemporaryDirectory() as tmp:
    artifact_dir = Path(tmp)
    Visualizer.line_chart(
        nav,
        title="NAV trajectories",
        ylabel="Portfolio value (USD)",
        save_path=str(artifact_dir / "nav_curve.png"),
        show=False,
    )
    Visualizer.line_chart(
        yield_pct,
        title="Cumulative yield (%)",
        ylabel="Yield (%)",
        save_path=str(artifact_dir / "yield_curve.png"),
        show=False,
    )
```

The `performance` module transforms the weekly return matrix into Net Asset Value trajectories and cumulative yield series via `nav_trajectories` and `yield_trajectories`. `Visualizer.line_chart` renders both curves and optionally exports PNGs for reporting. Binary outputs are generated locally (and via CI artifacts) so no static charts are committed to the repository. Replace `TemporaryDirectory()` with a persistent `Path` if you want to keep the generated image assets between runs.

## 3. Interpret the charts

### Net Asset Value

Inspect the saved `nav_curve.png` (or the interactive Matplotlib window if `show=True`). A steadily rising NAV indicates compounding growth across the selected pools, while drawdowns or flat stretches flag underperformance periods that merit a deeper dive into pool-specific events.

### Weekly Yield

Open `yield_curve.png` from the same output directory to review the annualised yield profile. Yield spikes or drops highlight weeks where realised returns diverged from the long-run average—use them to annotate catalysts in investor updates or to stress-test assumptions.

### Concentration & Risk Table

The demo also emits `concentration.csv`, which now includes per-scope Sharpe, Sortino,
maximum drawdown, and negative-period share metrics whenever historical returns are
available. Filter rows whose `scope` begins with `pool:` to compare individual vault
histories, or look at the aggregate `total` and `chain:<name>` entries to understand
how diversification and platform choice affect realised performance.

## 4. Compare with riskfolio

The portfolio behaviour resembles the [Riskfolio-Lib portfolio optimization example](https://riskfolio-lib.readthedocs.io/en/latest/portfolio.html), which uses `rp.Portfolio` to build efficient frontiers from historical returns. Matching NAV and yield trajectories across both libraries provides credibility that the calculations align with industry-standard techniques.

## 5. Follow the notebook end-to-end

The repository includes `notebooks/investor_walkthrough.ipynb`, which mirrors the steps
outlined above and adds richer commentary inline. To reproduce the full workflow inside
Jupyter:

1. Install dependencies with Poetry and register the virtualenv as a kernel:
   ```bash
   poetry install
   poetry run python -m ipykernel install --user --name stable-yield --display-name "Stable Yield Lab"
   ```
2. Launch a notebook server from the project root so relative paths resolve cleanly:
   ```bash
   poetry run jupyter lab  # or: poetry run jupyter notebook
   ```
3. Open `notebooks/investor_walkthrough.ipynb` and select the `Stable Yield Lab` kernel if it
   is not chosen automatically.
4. Run the cells sequentially. The notebook loads `src/sample_yields.csv`, computes NAV and
   cumulative yield trajectories, and writes charts into a temporary directory. Adjust the
   `TemporaryDirectory()` block to a persistent `Path` if you want to inspect the saved PNGs
   after shutdown.
5. Modify the configuration cells to point at your own CSV exports or to change reporting
   parameters (for example, `initial_investment` or the subset of pools). Re-run the notebook
   to generate updated tables and charts for investor decks.

Keep the CLI demo (`src/stable_yield_demo.py`) handy while iterating in notebooks—the demo
produces the same artifacts in a non-interactive workflow, which is helpful for regression
testing and automation.

## 6. End-to-end programmatic run

The end-to-end workflow is packaged as `scripts/live_end_to_end.py`. Execute it with
Poetry (or any environment that has the project installed) to fetch live pool data,
generate CSV reports, and create charts. The script now handles sparse DefiLlama
history gracefully by falling back to either `apy` or `apyReward` when `apyBase` is
missing, so historical returns are available whenever DefiLlama serves them.

```bash
poetry run python scripts/live_end_to_end.py
```

Outputs land in `outputs/investor_run_live/` and include cross-section CSVs, optional
NAV/yield curves, and a console summary of top pools plus portfolio metrics.
The bar chart limits to the top 60 pools by net APY, abbreviates long labels, and the TVL/APY scatter annotates only the top performers to avoid clutter.

### Add portfolio risk analytics

Riskfolio-Lib is optional but unlocks the `analytics.risk` helpers. Once the dependency is
installed, append:

```python
try:
    stats = risk.summary_statistics(returns)
    frontier = risk.efficient_frontier(returns)
    stats.to_csv(output_dir / "risk_stats.csv")
    frontier.to_csv(output_dir / "efficient_frontier.csv", index=False)
except RuntimeError as exc:
    print(f"Install riskfolio-lib for advanced risk analytics: {exc}")
```

### Customise adapter mixes

Inject or replace sources in the `snapshot_pipeline` to tailor coverage—e.g. drop `BeefySource`
for an on-chain indexer, add a yearn adapter, or chain a proprietary CSV exporter. As long as
adapters expose `fetch()` returning `Pool` or `PoolReturn` objects, the same Pipeline,
reporting, analytics, and visualisation calls will stitch everything together into a repeatable
end-to-end workflow.

## 7. Notebook catalogue

- `notebooks/investor_walkthrough.ipynb` — guided walkthrough of the sample datasets mirroring the sections above.
- `notebooks/live_end_to_end.ipynb` — step-by-step exploration of the live data pipeline that powers `scripts/live_end_to_end.py`.
- `notebooks/run_demo.ipynb` — notebook front-end for `scripts/run_demo.py`, letting you regenerate the demo artifacts in-place.

Run each notebook from the project root (`poetry run jupyter lab`) so relative paths resolve cleanly, and select the `Stable Yield Lab` kernel created during setup.
