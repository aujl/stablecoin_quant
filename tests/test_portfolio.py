import numpy as np
import pandas as pd
import pytest

from stable_yield_lab import portfolio

rp = pytest.importorskip("riskfolio")


def synthetic_returns() -> pd.DataFrame:
    np.random.seed(0)
    mu = [0.01, 0.012, 0.008]
    cov = [[0.0001, 0.00002, 0.000015], [0.00002, 0.00008, 0.000025], [0.000015, 0.000025, 0.00009]]
    data = np.random.multivariate_normal(mu, cov, size=100)
    return pd.DataFrame(data, columns=["A", "B", "C"])


def test_allocate_mean_variance_with_bounds():
    returns = synthetic_returns()
    bounds = {col: (0.1, 0.8) for col in returns.columns}
    w = portfolio.allocate_mean_variance(returns, bounds=bounds)

    assert isinstance(w, pd.Series)
    assert w.sum() == pytest.approx(1.0)
    for asset, (lo, hi) in bounds.items():
        assert lo - 1e-6 <= w[asset] <= hi + 1e-6

    apy = portfolio.expected_apy(returns, w)
    manual_apy = ((1 + returns.mean()) ** 52 - 1).mul(w).sum()
    assert apy == pytest.approx(float(manual_apy))

    risk = portfolio.tvl_weighted_risk(returns, w, rm="MV")
    manual_risk = rp.Risk_Contribution(w, returns, returns.cov(), rm="MV").sum()
    assert risk == pytest.approx(float(manual_risk))


def test_tracking_error_matches_manual_calculation():
    idx = pd.date_range("2024-01-01", periods=6, freq="W", tz="UTC")
    returns = pd.DataFrame(
        {
            "A": [0.012, -0.008, 0.01, 0.004, -0.002, 0.006],
            "B": [0.009, 0.011, -0.005, 0.007, 0.003, 0.008],
        },
        index=idx,
    )
    weights = pd.Series({"A": 0.6, "B": 0.4})

    periodic_te, annual_te = portfolio.tracking_error(returns, weights, freq=52)

    norm_weights = weights / weights.sum()
    portfolio_returns = returns.fillna(0.0).mul(norm_weights, axis=1).sum(axis=1)
    benchmark = float(portfolio_returns.mean())
    active = portfolio_returns - benchmark
    expected_periodic = float(active.std(ddof=1))
    expected_annual = expected_periodic * np.sqrt(52)

    assert periodic_te == pytest.approx(expected_periodic)
    assert annual_te == pytest.approx(expected_annual)


def test_apy_performance_summary_with_explicit_nav() -> None:
    idx = pd.date_range("2024-03-01", periods=5, freq="W", tz="UTC")
    returns = pd.DataFrame(
        {
            "A": [0.01, -0.004, 0.012, 0.006, -0.003],
            "B": [0.008, 0.009, -0.002, 0.007, 0.005],
        },
        index=idx,
    )
    weights = pd.Series({"A": 0.55, "B": 0.45})

    nav = portfolio.performance.nav_series(returns, weights, initial=100.0)
    metrics, nav_path = portfolio.apy_performance_summary(
        returns,
        weights,
        freq=52,
        initial_nav=100.0,
        nav=nav,
    )

    pd.testing.assert_series_equal(nav, nav_path)

    norm_weights = weights / weights.sum()
    portfolio_returns = returns.fillna(0.0).mul(norm_weights, axis=1).sum(axis=1)
    total_growth = float((1.0 + portfolio_returns).prod())
    realised_total = total_growth - 1.0
    realised_apy = total_growth ** (52 / len(portfolio_returns)) - 1.0
    expected = portfolio.expected_apy(returns, norm_weights, freq=52)
    expected_periodic = (1.0 + expected) ** (1.0 / 52) - 1.0
    active = portfolio_returns - expected_periodic
    tracking_periodic = float(active.std(ddof=1))
    tracking_annual = tracking_periodic * np.sqrt(52)

    assert metrics["expected_apy"] == pytest.approx(expected)
    assert metrics["realized_total_return"] == pytest.approx(realised_total)
    assert metrics["realized_apy"] == pytest.approx(realised_apy)
    assert metrics["active_apy"] == pytest.approx(realised_apy - expected)
    assert metrics["tracking_error_periodic"] == pytest.approx(tracking_periodic)
    assert metrics["tracking_error_annualized"] == pytest.approx(tracking_annual)
    assert metrics["final_nav"] == pytest.approx(nav.iloc[-1])
