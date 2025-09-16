from pathlib import Path

import pandas as pd
import pytest

from stable_yield_lab import (
    HistoricalCSVSource,
    Pipeline,
    Visualizer,
    cumulative_return,
    nav_series,
    performance,
)

def test_nav_and_yield_trajectories(tmp_path: Path) -> None:
    csv_path = Path(__file__).resolve().parent.parent / "src" / "sample_yields.csv"
    returns = Pipeline([HistoricalCSVSource(str(csv_path))]).run_history()

    nav = performance.nav_trajectories(returns, initial_investment=100.0)
    yield_df = performance.yield_trajectories(returns)

    assert nav.loc[pd.Timestamp("2024-01-08", tz="UTC"), "PoolA"] == pytest.approx(
        102.111, rel=1e-6
    )
    assert yield_df.loc[pd.Timestamp("2024-01-15", tz="UTC"), "PoolB"] == pytest.approx(
        0.015056, rel=1e-6
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


def test_estimate_pool_apy_daily_series_partial_window() -> None:
    dates = pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC")
    returns = pd.Series(
        [0.001, 0.002, 0.0015, -0.0005, 0.002, 0.001],
        index=dates,
        name="PoolX",
    )

    estimate = performance.estimate_pool_apy(
        returns,
        start=pd.Timestamp("2024-01-02", tz="UTC"),
        end=pd.Timestamp("2024-01-05", tz="UTC"),
        frequency="D",
    )

    window = returns.loc["2024-01-02":"2024-01-05"]
    total = (1.0 + window).prod() - 1.0
    periods = float(len(window))
    annual_factor = 365.0 / periods
    expected_gross = (1.0 + total) ** annual_factor - 1.0

    assert estimate.window_start == window.index.min()
    assert estimate.window_end == window.index.max()

    summary = estimate.summary
    assert summary.index.tolist() == ["PoolX"]
    assert summary.loc["PoolX", "periods"] == pytest.approx(periods)
    assert summary.loc["PoolX", "total_return"] == pytest.approx(total)
    assert summary.loc["PoolX", "annualization_factor"] == pytest.approx(annual_factor)
    assert summary.loc["PoolX", "gross_apy"] == pytest.approx(expected_gross)
    assert summary.loc["PoolX", "net_apy"] == pytest.approx(expected_gross)


def test_estimate_pool_apy_weekly_dataframe_with_fees() -> None:
    dates = pd.date_range("2024-01-05", periods=4, freq="W-FRI", tz="UTC")
    returns = pd.DataFrame(
        {
            "PoolA": [0.01, 0.012, 0.008, 0.009],
            "PoolB": [0.015, 0.013, 0.012, 0.011],
        },
        index=dates,
    )

    estimate = performance.estimate_pool_apy(
        returns,
        frequency="W",
        perf_fee_bps=200.0,
        mgmt_fee_bps=100.0,
    )

    summary = estimate.summary
    totals = (1.0 + returns).prod() - 1.0
    periods = float(len(returns))
    annual_factor = 52.0 / periods
    expected_gross = (1.0 + totals) ** annual_factor - 1.0
    fee_frac = 0.03
    expected_net = (expected_gross * (1.0 - fee_frac)).clip(lower=-1.0)

    pd.testing.assert_index_equal(summary.index, returns.columns)
    assert summary["periods"].eq(periods).all()
    pd.testing.assert_series_equal(summary["total_return"], totals, rtol=1e-12, atol=1e-12, check_names=False)
    assert summary["annualization_factor"].eq(annual_factor).all()
    pd.testing.assert_series_equal(summary["gross_apy"], expected_gross, rtol=1e-9, atol=1e-12, check_names=False)
    pd.testing.assert_series_equal(summary["net_apy"], expected_net, rtol=1e-9, atol=1e-12, check_names=False)

    assert estimate.performance_fee_bps == 200.0
    assert estimate.management_fee_bps == 100.0
    assert estimate.window_start == returns.index.min()
    assert estimate.window_end == returns.index.max()
    assert estimate.frequency == "W"
