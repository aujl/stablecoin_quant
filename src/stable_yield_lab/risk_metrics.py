from __future__ import annotations


import pandas as pd


def _require_riskfolio() -> None:
    try:
        import riskfolio as _  # noqa: F401
    except Exception as exc:  # pragma: no cover - only raised when missing
        raise RuntimeError(
            "riskfolio-lib is required for these metrics. Install with: \n"
            "  pip install 'riskfolio-lib'\n"
            "or enable the optional extra if using Poetry."
        ) from exc


def summary_statistics(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute basic portfolio statistics per asset using riskfolio-lib formulas.

    Parameters
    - returns: wide DataFrame of periodic returns (index=date, columns=assets)
    """
    _require_riskfolio()
    import riskfolio as rp

    stats = rp.Sharpe_Risk(returns)
    # rp.Sharpe_Risk returns a DataFrame with common risk stats
    return stats


def efficient_frontier(
    returns: pd.DataFrame,
    freq: int = 52,
    model: str = "Classic",
    risk_measure: str = "MV",
    obj: str = "Sharpe",
    rf: float = 0.0,
    l: float = 0.0,
) -> pd.DataFrame:
    """Compute an efficient frontier set of portfolios using riskfolio-lib.

    Returns a DataFrame with weights per asset for frontier points.
    """
    _require_riskfolio()
    import riskfolio as rp

    Y = returns.copy()
    port = rp.Portfolio(returns=Y)
    method_mu = "hist"
    method_cov = "hist"
    port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)
    frontier = port.efficient_frontier(model=model, rm=risk_measure, obj=obj, rf=rf, l=l, points=20)
    return frontier


def risk_contributions(weights: pd.Series, returns: pd.DataFrame) -> pd.Series:
    """Compute marginal risk contributions for a given weights vector."""
    _require_riskfolio()
    import riskfolio as rp

    port = rp.Portfolio(returns=returns)
    port.assets_stats(method_mu="hist", method_cov="hist", d=0.94)
    rc = port.risk_contribution(weights=weights, rm="MV")
    return rc
