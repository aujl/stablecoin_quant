from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from stable_yield_lab.pipeline import Pipeline
from stable_yield_lab.analytics import performance
from stable_yield_lab.analytics.performance import (
    RebalanceScenario,
    ScenarioRunResult,
    cumulative_return,
    nav_series,
    run_rebalance_scenarios,
)
from stable_yield_lab.sources import HistoricalCSVSource
from stable_yield_lab.visualization import Visualizer


def _sample_returns() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "PoolA": [0.012, -0.018, 0.009, -0.004, 0.011, -0.007],
            "PoolB": [0.0, 0.021, -0.012, 0.007, -0.01, 0.016],
        },
        index=dates,
    )


def test_nav_and_yield_trajectories(tmp_path: Path) -> None:
    csv_path = Path(__file__).resolve().parents[2] / "src" / "sample_yields.csv"
    returns = Pipeline([HistoricalCSVSource(str(csv_path))]).run_history()

    nav = performance.nav_trajectories(returns, initial_investment=100.0)
    yield_df = performance.yield_trajectories(returns)

    nav_ts = pd.Timestamp("2023-01-08", tz="UTC")
    yield_ts = pd.Timestamp("2023-01-15", tz="UTC")

    assert nav.loc[nav_ts, "Morpho USDC (ETH)"] == pytest.approx(100.396191, rel=1e-6)
    assert nav.loc[nav_ts, "Aave USDT v3 (Polygon)"] == pytest.approx(100.278093, rel=1e-6)
    assert yield_df.loc[yield_ts, "Curve 3Pool Convex (ETH)"] == pytest.approx(
        0.0046150724699938195, rel=1e-9
    )

    nav_path = tmp_path / "nav.png"
    yield_path = tmp_path / "yield.png"
    Visualizer.line_chart(
        nav, title="NAV over time", ylabel="NAV (USD)", save_path=str(nav_path), show=False
    )
    Visualizer.line_chart(
        yield_df * 100.0,
        title="Yield over time",
        ylabel="Yield (%)",
        save_path=str(yield_path),
        show=False,
    )

    assert nav_path.is_file()
    assert yield_path.is_file()


def test_cumulative_return_matches_manual_compounding() -> None:
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    series = pd.Series([0.1, -0.05, 0.02], index=dates)
    expected = (1.0 + series).cumprod() - 1.0
    result = cumulative_return(series)
    pd.testing.assert_series_equal(result, expected)


def test_cumulative_return_empty_series() -> None:
    empty = pd.Series(dtype=float)
    result = cumulative_return(empty)
    assert result.empty


def test_nav_series_with_weights_and_initial_scaling() -> None:
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    returns = pd.DataFrame(
        {
            "A": [0.1, 0.0, 0.02],
            "B": [0.0, 0.1, -0.01],
        },
        index=dates,
    )
    weights = pd.Series({"A": 0.6, "B": 0.4})
    nav_100 = nav_series(returns, weights, initial=100.0)
    nav_1 = nav_series(returns, weights, initial=1.0)

    clean_weights = weights / weights.sum()
    expected = 100.0 * (1.0 + returns.mul(clean_weights, axis=1).sum(axis=1)).cumprod()

    pd.testing.assert_series_equal(nav_100, expected)
    pd.testing.assert_series_equal(nav_100 / 100.0, nav_1)


def test_nav_series_aligns_and_validates_weights() -> None:
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    returns = pd.DataFrame(
        {
            "A": [0.02, -0.01, 0.015, 0.0],
            "B": [0.005, 0.01, -0.002, 0.008],
        },
        index=dates,
    )
    weights = pd.Series({"A": 0.3, "B": 0.2, "C": 0.5})

    result = nav_series(returns, weights, initial=50.0)

    aligned = weights.reindex(returns.columns).fillna(0.0)
    expected_weights = aligned / aligned.sum()
    expected_nav = (
        50.0 * (1.0 + returns.fillna(0.0).mul(expected_weights, axis=1).sum(axis=1)).cumprod()
    )
    pd.testing.assert_series_equal(result, expected_nav)

    with pytest.raises(ValueError):
        nav_series(returns, pd.Series({"A": 0.0, "B": 0.0}))


def test_nav_series_defaults_and_empty() -> None:
    empty = pd.DataFrame()
    result = nav_series(empty, None, initial=1.0)
    assert result.empty

    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    returns = pd.DataFrame({"A": [0.05, 0.0], "B": [0.0, 0.05]}, index=dates)
    result = nav_series(returns, None, initial=1.0)

    expected_returns = returns.mul(0.5, axis=1).sum(axis=1)
    expected_nav = (1.0 + expected_returns).cumprod()
    pd.testing.assert_series_equal(result, expected_nav)


@pytest.mark.parametrize(
    ("calendar_slice", "expect_tracking_zero"),
    [(slice(None), True), (slice(None, None, 2), False)],
)
def test_run_rebalance_scenarios_tracking_error(calendar_slice, expect_tracking_zero) -> None:
    returns = _sample_returns()
    weights = pd.Series({"PoolA": 0.6, "PoolB": 0.4})

    calendars = {
        "baseline": RebalanceScenario(calendar=returns.index, cost_bps=0.0),
        "candidate": RebalanceScenario(calendar=returns.index[calendar_slice], cost_bps=0.0),
    }

    summary = run_rebalance_scenarios(
        returns,
        weights,
        calendars,
        benchmark="baseline",
        initial_nav=100.0,
    )

    metrics = summary.metrics
    assert set(metrics.columns) >= {"realized_apy", "total_cost", "tracking_error", "terminal_nav"}
    assert list(summary.navs.columns) == ["baseline", "candidate"]
    assert list(summary.returns.columns) == ["baseline", "candidate"]

    baseline_apy = metrics.loc["baseline", "realized_apy"]
    candidate_apy = metrics.loc["candidate", "realized_apy"]
    tracking_error = metrics.loc["candidate", "tracking_error"]

    if expect_tracking_zero:
        assert tracking_error == pytest.approx(0.0, abs=1e-12)
        assert candidate_apy == pytest.approx(baseline_apy)
    else:
        assert tracking_error > 0.0
        assert candidate_apy != pytest.approx(baseline_apy)


@pytest.mark.parametrize("cost_bps", [0.0, 25.0])
def test_run_rebalance_scenarios_costs(cost_bps: float) -> None:
    returns = _sample_returns()
    weights = pd.Series({"PoolA": 0.5, "PoolB": 0.5})

    scenarios = {
        "zero_cost": RebalanceScenario(calendar=returns.index, cost_bps=0.0),
        "test": RebalanceScenario(calendar=returns.index, cost_bps=cost_bps),
    }

    summary = run_rebalance_scenarios(
        returns,
        weights,
        scenarios,
        benchmark="zero_cost",
        initial_nav=100.0,
    )

    metrics = summary.metrics
    zero_cost_apy = metrics.loc["zero_cost", "realized_apy"]
    test_apy = metrics.loc["test", "realized_apy"]
    total_cost = metrics.loc["test", "total_cost"]
    tracking_error = metrics.loc["test", "tracking_error"]

    if cost_bps == 0.0:
        assert total_cost == pytest.approx(0.0)
        assert tracking_error == pytest.approx(0.0, abs=1e-12)
        assert test_apy == pytest.approx(zero_cost_apy)
    else:
        assert total_cost > 0.0
        assert tracking_error > 0.0
        assert test_apy < zero_cost_apy


def test_run_rebalance_scenarios_empty_returns() -> None:
    empty_returns = pd.DataFrame()
    weights = pd.Series(dtype=float)

    result = run_rebalance_scenarios(
        empty_returns,
        weights,
        {"base": RebalanceScenario(calendar=pd.DatetimeIndex([]))},
        initial_nav=1.0,
    )

    assert isinstance(result, ScenarioRunResult)
    assert result.metrics.empty
    assert result.navs.empty
    assert result.returns.empty
