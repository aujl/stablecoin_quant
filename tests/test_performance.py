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


def test_horizon_apys_from_nav_and_yield() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="7D", tz="UTC")
    nav = pd.DataFrame(
        {
            "PoolA": [100.0, 101.0, 104.0, 108.0, 110.0],
            "PoolB": [100.0, 98.0, 99.0, 100.0, 101.0],
        },
        index=idx,
    )
    lookbacks = {
        "last 2 weeks": "14D",
        "last 2 periods": 2,
        "last 52 weeks": "364D",
    }

    apy_nav = performance.horizon_apys(nav, lookbacks=lookbacks, value_type="nav")

    latest = nav.iloc[-1]
    base_2w = nav.loc[: idx[-1] - pd.Timedelta(days=14)].iloc[-1]
    years_2w = pd.Timedelta(days=14) / pd.Timedelta(days=365)
    expected_2w = (latest / base_2w) ** (1.0 / years_2w) - 1.0
    expected_2w.name = "last 2 weeks"

    base_2p = nav.iloc[-(2 + 1)]
    periods_per_year = pd.Timedelta(days=365) / (idx[1] - idx[0])
    years_2p = 2 / periods_per_year
    expected_2p = (latest / base_2p) ** (1.0 / years_2p) - 1.0
    expected_2p.name = "last 2 periods"

    pd.testing.assert_series_equal(apy_nav["last 2 weeks"], expected_2w)
    pd.testing.assert_series_equal(apy_nav["last 2 periods"], expected_2p)
    assert pd.isna(apy_nav.loc["PoolA", "last 52 weeks"])
    assert pd.isna(apy_nav.loc["PoolB", "last 52 weeks"])

    yields = nav / nav.iloc[0] - 1.0
    apy_yield = performance.horizon_apys(yields, lookbacks=lookbacks, value_type="yield")
    pd.testing.assert_frame_equal(apy_nav, apy_yield)


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
