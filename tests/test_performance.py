from pathlib import Path

import pandas as pd
import pytest

from stable_yield_lab import (
    HistoricalCSVSource,
    Pipeline,
    Visualizer,
    cumulative_return,
    nav_series,
    nav_with_rebalance,
    performance,
)


def _manual_nav_with_schedule(
    returns: pd.DataFrame,
    weights: pd.Series | None,
    schedule: pd.DatetimeIndex,
    *,
    initial: float,
) -> pd.Series:
    clean_returns = returns.fillna(0.0)
    if clean_returns.empty:
        return pd.Series(dtype=float)

    if weights is None:
        target = pd.Series(1.0 / clean_returns.shape[1], index=clean_returns.columns)
    else:
        target = weights.reindex(clean_returns.columns).fillna(0.0)

    total = float(target.sum())
    if total == 0.0:
        raise ValueError("weights sum to zero")

    target = target / total
    current = target.copy()

    schedule_index = pd.DatetimeIndex(schedule).intersection(clean_returns.index)
    if clean_returns.index[0] not in schedule_index:
        schedule_index = schedule_index.insert(0, clean_returns.index[0])
    schedule_set = set(schedule_index)

    nav_value = float(initial)
    nav_values: list[float] = []

    for timestamp, period_returns in clean_returns.iterrows():
        if timestamp in schedule_set:
            current = target.copy()

        period_portfolio_return = float(period_returns.mul(current).sum())
        nav_value *= 1.0 + period_portfolio_return
        nav_values.append(nav_value)

        gross_weights = current * (1.0 + period_returns)
        total_gross = float(gross_weights.sum())
        if total_gross == 0.0:
            current = pd.Series(0.0, index=current.index)
        else:
            current = gross_weights / total_gross

    return pd.Series(nav_values, index=clean_returns.index)

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


def test_nav_with_rebalance_weekly_calendar() -> None:
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    returns = pd.DataFrame(
        {
            "A": [
                0.01,
                0.015,
                -0.02,
                0.03,
                -0.005,
                0.01,
                0.0,
                0.02,
                -0.015,
                0.025,
            ],
            "B": [
                -0.005,
                0.0,
                0.01,
                -0.02,
                0.015,
                -0.01,
                0.02,
                -0.005,
                0.01,
                -0.02,
            ],
        },
        index=dates,
    )
    weights = pd.Series({"A": 0.6, "B": 0.4})

    nav = nav_with_rebalance(returns, calendar="W-MON", weights=weights, initial=1.0)

    weekly_calendar = pd.date_range(start=dates[0], end=dates[-1], freq="W-MON")
    expected = _manual_nav_with_schedule(
        returns,
        weights,
        weekly_calendar,
        initial=1.0,
    )

    pd.testing.assert_series_equal(nav, expected)
    assert nav.iloc[-1] == pytest.approx(1.0398719648091561, rel=1e-12)


def test_nav_with_rebalance_monthly_calendar_missing_assets() -> None:
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    returns = pd.DataFrame(
        {
            "A": [
                0.01,
                0.015,
                -0.02,
                0.03,
                -0.005,
                0.01,
                0.0,
                0.02,
                -0.015,
                0.025,
            ],
            "B": [
                -0.005,
                0.0,
                0.01,
                -0.02,
                0.015,
                -0.01,
                0.02,
                -0.005,
                0.01,
                -0.02,
            ],
            "C": [
                0.005,
                -0.002,
                float("nan"),
                -0.001,
                0.0,
                0.002,
                0.001,
                -0.003,
                0.002,
                0.0,
            ],
        },
        index=dates,
    )
    weights = pd.Series({"A": 0.6, "B": 0.4})

    calendar = pd.date_range(start=dates[0], end=dates[-1], freq="MS")
    nav = nav_with_rebalance(returns, calendar=calendar, weights=weights, initial=100.0)

    expected = _manual_nav_with_schedule(
        returns,
        weights,
        calendar,
        initial=100.0,
    )

    pd.testing.assert_series_equal(nav, expected)
    assert nav.iloc[-1] == pytest.approx(104.02009957496077, rel=1e-9)
