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


def test_rebalance_engine_handles_asset_dropout_and_addition():
    idx = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    returns = pd.DataFrame(
        {
            "A": [0.01, -0.02, 0.03, 0.0],
            "B": [0.02, 0.01, -0.01, 0.02],
            "C": [0.015, 0.0, np.nan, np.nan],
            "D": [np.nan, np.nan, 0.05, 0.01],
        },
        index=idx,
    )

    targets = {
        idx[0]: {"A": 0.5, "B": 0.3, "C": 0.2},
        idx[3]: {"A": 0.4, "B": 0.3, "D": 0.3},
    }

    result = portfolio.rebalance_portfolio(
        returns,
        rebalance_schedule=[idx[0], idx[3]],
        target_weights=targets,
    )

    weights = result.weights
    assert pytest.approx(1.0) == weights.loc[idx[0]].sum()
    assert weights.loc[idx[2], "C"] == pytest.approx(0.0)
    assert weights.loc[idx[2], ["A", "B"]].sum() == pytest.approx(1.0)
    assert weights.loc[idx[3], "D"] == pytest.approx(0.3)
    assert pytest.approx(1.0) == weights.loc[idx[3]].sum()
    assert result.nav.iloc[-1] > 0


def test_schedule_helpers_enforce_alignment():
    idx = pd.date_range("2024-02-01", periods=5, freq="D", tz="UTC")
    returns = pd.DataFrame(
        {
            "A": np.linspace(0.0, 0.02, num=5),
            "B": np.linspace(0.01, -0.01, num=5),
        },
        index=idx,
    )

    manual_schedule = portfolio.schedule_from_user_weights(
        dates=[idx[0], idx[2], idx[4]],
        weights={"A": 0.6, "B": 0.4},
        returns_index=returns.index,
    )
    assert manual_schedule.index.equals(idx[[0, 2, 4]])
    assert all(manual_schedule.sum(axis=1).round(8) == 1.0)

    optimised = {
        idx[0]: pd.Series({"A": 0.7, "B": 0.3}),
        idx[3]: pd.Series({"A": 0.2, "B": 0.8}),
    }
    opt_schedule = portfolio.schedule_from_optimizations(
        optimised,
        returns_index=returns.index,
    )
    assert opt_schedule.index.equals(idx[[0, 3]])
    assert all(opt_schedule.sum(axis=1).round(8) == 1.0)

    with pytest.raises(ValueError):
        portfolio.schedule_from_user_weights(
            dates=[idx[0], idx[-1] + pd.Timedelta(days=1)],
            weights={"A": 0.5, "B": 0.5},
            returns_index=returns.index,
        )
