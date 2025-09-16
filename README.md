# stablecoin_quant

Experimental toolkit for analyzing and visualizing yields on stablecoin pools.

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

The script applies `stable_yield_lab.performance.nav_curve` and `performance.yield_curve` to compute time-series performance.
`Visualizer.plot_nav` and `Visualizer.plot_yield` render the NAV trajectory and annualized yields.
A steadily rising NAV indicates compounding growth; falling or flat lines flag underperformance.
For a step-by-step example, see [docs/investor_walkthrough.md](docs/investor_walkthrough.md).

## Codex Workflows

GitHub workflows tag [@codex](https://github.com/features/copilot) to request automated pull request reviews,
issue triage, and comment follow-ups—no external API credentials are required.

- **Triggers**:
  - `pull_request` events (`opened`, `synchronize`, `reopened`) post a comment asking `@codex` to review for tests, security, and documentation.
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
