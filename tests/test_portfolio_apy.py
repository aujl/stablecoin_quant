import math

import pandas as pd
import pytest

from stable_yield_lab import performance, portfolio


def synthetic_nav_inputs() -> tuple[pd.DataFrame, pd.Series]:
    """Generate a small synthetic return panel and portfolio weights."""

    dates = pd.date_range("2024-01-01", periods=4, freq="W")
    returns = pd.DataFrame(
        {
            "PoolA": [0.01, 0.015, -0.005, 0.012],
            "PoolB": [0.008, 0.006, 0.004, 0.007],
        },
        index=dates,
    )
    weights = pd.Series({"PoolA": 0.6, "PoolB": 0.4})
    return returns, weights


def test_apy_performance_summary_expected_vs_realised() -> None:
    returns, weights = synthetic_nav_inputs()
    freq = 52
    initial_nav = 100.0

    metrics, nav = portfolio.apy_performance_summary(
        returns,
        weights,
        freq=freq,
        initial_nav=initial_nav,
    )

    manual_nav = performance.nav_series(returns, weights, initial=initial_nav)
    pd.testing.assert_series_equal(nav, manual_nav)

    expected = portfolio.expected_apy(returns, weights, freq=freq)

    total_return = manual_nav.iloc[-1] / initial_nav - 1.0
    realised_apy = (1.0 + total_return) ** (freq / len(manual_nav)) - 1.0

    assert metrics["expected_apy"] == pytest.approx(expected)
    assert metrics["realized_apy"] == pytest.approx(realised_apy)
    assert metrics["realized_total_return"] == pytest.approx(total_return)
    assert metrics["active_apy"] == pytest.approx(realised_apy - expected)

    portfolio_returns = returns.mul(weights, axis=1).sum(axis=1)
    expected_periodic = (1.0 + expected) ** (1.0 / freq) - 1.0
    active_returns = portfolio_returns - expected_periodic
    manual_te_periodic = active_returns.std(ddof=1)
    manual_te_annualised = manual_te_periodic * math.sqrt(freq)

    assert metrics["tracking_error_periodic"] == pytest.approx(manual_te_periodic)
    assert metrics["tracking_error_annualized"] == pytest.approx(manual_te_annualised)
    assert metrics["horizon_periods"] == pytest.approx(len(manual_nav))
    assert metrics["horizon_years"] == pytest.approx(len(manual_nav) / freq)
    assert metrics["final_nav"] == pytest.approx(manual_nav.iloc[-1])


def test_tracking_error_accepts_dataframe_and_series() -> None:
    returns, weights = synthetic_nav_inputs()
    freq = 12

    portfolio_returns = returns.mul(weights, axis=1).sum(axis=1)
    target = float(portfolio_returns.mean() + 0.001)

    te_series = portfolio.tracking_error(
        portfolio_returns,
        freq=freq,
        target_periodic_return=target,
    )
    te_dataframe = portfolio.tracking_error(
        returns,
        weights,
        freq=freq,
        target_periodic_return=target,
    )

    manual = (portfolio_returns - target).std(ddof=1)
    assert te_series[0] == pytest.approx(manual)
    assert te_series[1] == pytest.approx(manual * math.sqrt(freq))
    assert te_dataframe == te_series


def test_apy_performance_summary_respects_external_nav() -> None:
    returns, weights = synthetic_nav_inputs()
    nav = performance.nav_series(returns, weights, initial=1.0)

    metrics_from_computed, nav_auto = portfolio.apy_performance_summary(returns, weights)
    metrics_from_external, nav_external = portfolio.apy_performance_summary(
        returns,
        weights,
        nav=nav,
    )

    pd.testing.assert_series_equal(nav, nav_external)
    pd.testing.assert_series_equal(nav_auto, nav_external)
    pd.testing.assert_series_equal(metrics_from_computed, metrics_from_external)


def test_tracking_error_requires_positive_frequency() -> None:
    returns, weights = synthetic_nav_inputs()

    with pytest.raises(ValueError):
        portfolio.tracking_error(returns, weights, freq=0)

    with pytest.raises(ValueError):
        portfolio.apy_performance_summary(returns, weights, freq=0)
