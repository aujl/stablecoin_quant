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


def test_horizon_apy_diagnostics_reports_missing_and_negative_returns() -> None:
    dates = pd.date_range("2024-01-01", periods=4, freq="W")
    returns = pd.DataFrame(
        {
            "Growth": [0.01, 0.02, None, 0.015],
            "Gap": [0.008, None, 0.007, None],
            "Drawdown": [-0.05, -0.02, None, None],
        },
        index=dates,
    )

    diagnostics = performance.horizon_apy_diagnostics(returns, periods_per_year=52)

    assert list(diagnostics.columns) == ["apy", "periods", "missing_pct", "volatility"]

    growth_filled = pd.Series([0.01, 0.02, 0.0, 0.015])
    expected_growth = float((1.0 + growth_filled).prod())
    expected_growth_apy = expected_growth ** (52 / len(growth_filled)) - 1.0
    assert diagnostics.loc["Growth", "apy"] == pytest.approx(expected_growth_apy)
    assert diagnostics.loc["Growth", "periods"] == 3
    assert diagnostics.loc["Growth", "missing_pct"] == pytest.approx(0.25)
    expected_growth_vol = pd.Series([0.01, 0.02, 0.015]).std(ddof=0)
    assert diagnostics.loc["Growth", "volatility"] == pytest.approx(expected_growth_vol)

    drawdown_filled = pd.Series([-0.05, -0.02, 0.0, 0.0])
    drawdown_growth = float((1.0 + drawdown_filled).prod())
    expected_drawdown_apy = drawdown_growth ** (52 / len(drawdown_filled)) - 1.0
    assert diagnostics.loc["Drawdown", "apy"] == pytest.approx(expected_drawdown_apy)
    assert diagnostics.loc["Drawdown", "apy"] < 0
    assert diagnostics.loc["Drawdown", "periods"] == 2
    assert diagnostics.loc["Drawdown", "missing_pct"] == pytest.approx(0.5)
    expected_drawdown_vol = pd.Series([-0.05, -0.02]).std(ddof=0)
    assert diagnostics.loc["Drawdown", "volatility"] == pytest.approx(expected_drawdown_vol)

    assert diagnostics.loc["Gap", "missing_pct"] == pytest.approx(0.5)


def test_horizon_apy_diagnostics_respects_window_and_validates_input() -> None:
    dates = pd.date_range("2024-01-01", periods=4, freq="W")
    returns = pd.DataFrame({"Growth": [0.01, 0.02, None, 0.015]}, index=dates)

    short_window = performance.horizon_apy_diagnostics(returns, periods_per_year=52, horizon=2)
    assert short_window.loc["Growth", "periods"] == 1
    assert short_window.loc["Growth", "missing_pct"] == pytest.approx(0.5)
    filled = pd.Series([0.0, 0.015])
    expected_apy = float((1.0 + filled).prod()) ** (52 / len(filled)) - 1.0
    assert short_window.loc["Growth", "apy"] == pytest.approx(expected_apy)

    with pytest.raises(ValueError):
        performance.horizon_apy_diagnostics(returns, periods_per_year=52, horizon=0)
