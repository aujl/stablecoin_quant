# Sample Data Provenance

This repository ships with synthetic fixtures so the demo and tests can reason
about historical performance without relying on live market data. The datasets
are intentionally lightweight but cover enough history to exercise stress
handling logic.

## `src/sample_yields.csv`

* Coverage: weekly observations from 2023-01-01 through 2024-07-07 for the
  pools `Morpho USDC (ETH)`, `Aave USDT v3 (Polygon)` and `Curve 3Pool Convex
  (ETH)`.
* Generation process:
  * Start from smooth baseline curves produced by sinusoidal functions with a
    gentle upward drift (roughly 7–12% annualised when compounded weekly).
  * Inject explicit stress regimes at historically inspired dates (e.g.
    March 2023, September 2023, Q1 2024) by overwriting the affected weeks
    with discrete drawdowns between -1% and -1.8%.
  * Clamp non-stressed weeks to a plausible corridor (±0.8% per week) and round
    to 6 decimal places to keep the CSV compact and reproducible.
* The resulting file contains 80 weekly rows per pool (240 rows total) and is
  stored in long format with `timestamp`, `name` and `period_return` columns.

The deterministic generator lives in the task history (see this PR) and can be
replayed with the project environment active: `poetry run python` followed by
the snippet in `tests/test_performance.py::test_nav_and_yield_trajectories`.

## `src/sample_pools.csv` and `src/sample_pools_history.csv`

* `sample_pools.csv` still captures the point-in-time fundamentals (TVL,
  advertised APY, risk score, etc.) but now includes columns describing the
  available realised history for each pool: `history_start`, `history_end`,
  `observations`, `realized_return_52w`, `realized_apy_52w`,
  `realized_volatility_52w`, `max_drawdown_52w`, and `last_period_return`.
  Pools without tracked history leave these fields blank.
* `sample_pools_history.csv` is a companion table that stores the same metrics
  separately to make joins explicit in analytical notebooks.
* Metrics were computed directly from the trailing 52 weeks of
  `sample_yields.csv` using the formulas implemented in
  `tests/test_performance.py::test_sample_history_matches_expected_metrics`:
  
  \[
  R = \prod_{i=1}^n (1 + r_i) - 1,\qquad
  \text{APY} = (1 + R)^{52/n} - 1,\qquad
  \sigma = \operatorname{stdev}(r_i) \sqrt{52}
  \]
  
  Maximum drawdown is measured on the cumulative NAV curve derived from the
  weekly returns.

The JSON fixture `tests/fixtures/sample_history_expected.json` snapshots the
resulting metrics so tests can assert the provenance stays in sync with the
underlying returns.
