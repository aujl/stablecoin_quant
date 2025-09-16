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


def _simulate_rebalanced_nav(
    returns: pd.DataFrame,
    weights: pd.Series,
    *,
    initial: float,
    cost_bps: float,
    fixed_fee: float,
) -> pd.Series:
    """Manual simulation of a rebalanced NAV path with costs."""

    weights = weights.reindex(returns.columns).fillna(0.0)
    total = float(weights.sum())
    if total == 0:
        raise ValueError("weights sum to zero")
    norm_weights = weights / total

    nav = float(initial)
    cost_rate = cost_bps / 10_000.0
    nav_path: list[float] = []

    for _, row in returns.fillna(0.0).iterrows():
        holdings_after = nav * norm_weights * (1.0 + row)
        nav_before_rebalance = float(holdings_after.sum())

        if nav_before_rebalance <= 0.0:
            nav = 0.0
            nav_path.append(nav)
            continue

        weights_after = holdings_after / nav_before_rebalance
        turnover = 0.5 * float((weights_after - norm_weights).abs().sum())
        cost = 0.0
        if turnover > 1e-12:
            cost += nav_before_rebalance * turnover * cost_rate
            cost += fixed_fee
        nav = max(nav_before_rebalance - cost, 0.0)
        nav_path.append(nav)

    return pd.Series(nav_path, index=returns.index, dtype=float)


def _nav_to_period_returns(nav: pd.Series, *, initial: float) -> pd.Series:
    """Convert a NAV path into periodic simple returns."""

    if nav.empty:
        return pd.Series(dtype=float)

    prev = nav.shift(1)
    prev.iloc[0] = initial
    return nav / prev - 1.0

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


def test_nav_series_rebalance_costs_reduce_nav_and_apy() -> None:
    dates = pd.date_range("2024-01-01", periods=4, freq="W")
    returns = pd.DataFrame(
        {
            "A": [0.03, -0.015, 0.02, 0.01],
            "B": [0.008, 0.027, -0.004, 0.017],
        },
        index=dates,
    )
    weights = pd.Series({"A": 0.55, "B": 0.45})
    initial = 1_000.0
    cost_bps = 20.0
    fixed_fee = 1.25

    nav_with_cost = nav_series(
        returns,
        weights,
        initial=initial,
        rebalance_cost_bps=cost_bps,
        rebalance_fixed_fee=fixed_fee,
    )
    expected_nav = _simulate_rebalanced_nav(
        returns,
        weights,
        initial=initial,
        cost_bps=cost_bps,
        fixed_fee=fixed_fee,
    )
    pd.testing.assert_series_equal(
        nav_with_cost, expected_nav, check_exact=False, rtol=1e-12, atol=1e-12
    )

    nav_without_cost = nav_series(returns, weights, initial=initial)
    assert nav_with_cost.iloc[-1] < nav_without_cost.iloc[-1]

    net_returns_cost = _nav_to_period_returns(nav_with_cost, initial=initial)
    net_returns_no_cost = _nav_to_period_returns(nav_without_cost, initial=initial)

    growth_cost = float((1.0 + net_returns_cost).prod())
    growth_no_cost = float((1.0 + net_returns_no_cost).prod())
    assert growth_cost < growth_no_cost

    periods_per_year = 52
    apy_cost = growth_cost ** (periods_per_year / len(net_returns_cost)) - 1.0
    apy_no_cost = growth_no_cost ** (periods_per_year / len(net_returns_no_cost)) - 1.0
    assert apy_cost < apy_no_cost


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
