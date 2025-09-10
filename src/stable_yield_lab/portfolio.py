from __future__ import annotations

import pandas as pd

from .risk_metrics import _require_riskfolio


def allocate_mean_variance(
    returns: pd.DataFrame,
    *,
    bounds: dict[str, tuple[float, float]] | None = None,
    risk_measure: str = "MV",
    rf: float = 0.0,
    l: float = 0.0,
) -> pd.Series:
    """Optimize portfolio weights using mean-variance framework.

    Parameters
    ----------
    returns: pd.DataFrame
        Wide DataFrame of historical returns (rows=observations, cols=pools).
    bounds: dict[str, tuple[float, float]] | None, optional
        Mapping of asset -> (min_weight, max_weight). Bounds are applied
        post-optimization and weights re-normalized. Defaults to None.
    risk_measure: str, optional
        Risk measure understood by riskfolio-lib (e.g. "MV", "MAD").
    rf: float, optional
        Risk-free rate for the optimizer. Defaults to 0.0.
    l: float, optional
        Risk aversion factor used by riskfolio-lib. Defaults to 0.0.
    """
    _require_riskfolio()
    import riskfolio as rp

    port = rp.Portfolio(returns=returns)
    port.assets_stats(method_mu="hist", method_cov="hist")
    weights_df = port.optimization(model="Classic", rm=risk_measure, obj="Sharpe", rf=rf, l=l)
    weights = weights_df.squeeze()

    if bounds:
        lo = pd.Series({k: v[0] for k, v in bounds.items()})
        hi = pd.Series({k: v[1] for k, v in bounds.items()})
        weights = weights.clip(lower=lo, upper=hi)

    weights = weights / weights.sum()
    return weights.reindex(returns.columns)


def expected_apy(returns: pd.DataFrame, weights: pd.Series, *, freq: int = 52) -> float:
    """Compute portfolio expected APY given periodic returns and weights."""
    weights = weights.reindex(returns.columns).fillna(0)
    mean_returns = returns.mean()
    apy_assets = (1 + mean_returns) ** freq - 1
    return float((weights * apy_assets).sum())


def tvl_weighted_risk(returns: pd.DataFrame, weights: pd.Series, *, rm: str = "MV") -> float:
    """Compute TVL-weighted risk of a portfolio using riskfolio risk measures."""
    _require_riskfolio()
    import riskfolio as rp

    weights = weights.reindex(returns.columns).fillna(0)
    cov = returns.cov()
    rc = rp.Risk_Contribution(weights, returns, cov, rm=rm)
    return float(rc.sum())
