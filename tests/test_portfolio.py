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
