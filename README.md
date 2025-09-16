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
- Override the demo config path with `STABLE_YIELD_CONFIG=/path/to/config.toml`
- Export `CODEX_API_URL` and `CODEX_API_KEY` to experiment with Codex locally

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

GitHub workflows integrate with [Codex](https://github.com/features/copilot) for automated pull request
reviews and issue analysis.

- **Secrets**: `CODEX_API_URL` and `CODEX_API_KEY` must be set in repository secrets.
- Workflows skip Codex steps if these secrets are unset.
- **Triggers**:
  - `pull_request` events (`opened`, `synchronize`, `reopened`) send the diff to Codex for review and tag `@codex` on the pull request.
  - `issues` events (`opened`, `edited`) forward the issue content for analysis.
  - `issue_comment` events (`created`) send the comment thread for resolution suggestions.


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
