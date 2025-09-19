# stablecoin_quant

[![Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/aujl/stablecoin_quant/gh-pages/coverage/badge.json)](https://aujl.github.io/stablecoin_quant/coverage/)

Experimental toolkit for analyzing and visualizing yields on stablecoin pools.

Latest test coverage is published automatically from CI to [GitHub Pages](https://aujl.github.io/stablecoin_quant/coverage/), which redirects to the full [HTML report](https://aujl.github.io/stablecoin_quant/report/index.html) and hosts the badge metadata for the shield above.

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

Core data models now live under ``stable_yield_lab.core``:

```python
from stable_yield_lab.core import Pool, PoolRepository, ReturnRepository
from stable_yield_lab.visualization import Visualizer
```

The script applies `stable_yield_lab.analytics.performance.nav_trajectories`
and `stable_yield_lab.analytics.performance.yield_trajectories` to compute
time-series performance. `Visualizer.line_chart` and `Visualizer.line_yield`
render the NAV trajectory and annualized yields.
A steadily rising NAV indicates compounding growth; falling or flat lines flag underperformance.
For a step-by-step example, see [docs/investor_walkthrough.md](docs/investor_walkthrough.md).

### Cross-Section Risk Reporting

`stable_yield_lab.reporting.cross_section_report` now enriches the `concentration.csv`
output with realised risk statistics whenever you supply historical returns. Pass a
wide DataFrame of periodic returns (columns correspond to pool names) via the
`returns` argument—`HistoricalCSVSource` and `stable_yield_lab.core.ReturnRepository` make it easy to load
bundled fixtures. The resulting CSV includes:

- `scope`: `total`, `chain:<name>`, `stablecoin:<symbol>`, and `pool:<name>` rows.
- `hhi`: Herfindahl–Hirschman Index based on TVL (unchanged from before).
- `sharpe_ratio`: Sample mean of realised returns divided by sample volatility.
- `sortino_ratio`: Mean return divided by downside deviation (negative periods only).
- `max_drawdown`: Worst peak-to-trough loss computed from a discrete-compounded NAV path.
- `negative_period_share`: Fraction of periods with negative realised returns.

When no return history is provided the new columns still appear but are populated with
`NaN`, preserving backwards compatibility for downstream tooling.

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
