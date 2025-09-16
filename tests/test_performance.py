import pandas as pd
import pytest

from stable_yield_lab import cumulative_return, nav_series


def test_cumulative_return_basic():
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    r = pd.Series([0.1, -0.05, 0.02], index=dates)
    expected = (1 + r).cumprod() - 1
    result = cumulative_return(r)
    pd.testing.assert_series_equal(result, expected)


def test_cumulative_return_empty():
    r = pd.Series(dtype=float)
    result = cumulative_return(r)
    assert result.empty


def test_nav_series_with_weights_and_initial_scaling():
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

    portfolio_ret = returns.mul(weights / weights.sum(), axis=1).sum(axis=1)
    expected = 100.0 * (1 + portfolio_ret).cumprod()

    pd.testing.assert_series_equal(nav_100, expected)
    pd.testing.assert_series_equal(nav_100 / 100.0, nav_1)


def test_nav_series_defaults_and_empty():
    empty = pd.DataFrame()
    result = nav_series(empty, None, initial=1.0)
    assert result.empty

    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    returns = pd.DataFrame({"A": [0.05, 0.0], "B": [0.0, 0.05]}, index=dates)
    result = nav_series(returns, None, initial=1.0)
    equal = pd.Series(0.5, index=["A", "B"])
    expected = (1 + returns.mul(equal, axis=1).sum(axis=1)).cumprod()
    pd.testing.assert_series_equal(result, expected)
