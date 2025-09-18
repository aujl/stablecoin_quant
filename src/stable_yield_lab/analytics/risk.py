from __future__ import annotations

from typing import Any

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


def _call_assets_stats(portfolio: Any, *, method_mu: str, method_cov: str) -> None:
    """Invoke ``assets_stats`` with graceful fallback across riskfolio versions."""

    try:
        portfolio.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)
    except TypeError:
        portfolio.assets_stats(method_mu=method_mu, method_cov=method_cov)


def summary_statistics(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute basic portfolio statistics per asset using riskfolio-lib formulas."""

    _require_riskfolio()
    import riskfolio as rp

    stats_obj = rp.Sharpe_Risk(returns)
    if isinstance(stats_obj, pd.DataFrame):
        stats = stats_obj
    elif isinstance(stats_obj, pd.Series):
        stats = stats_obj.to_frame().T
    else:
        stats = pd.DataFrame([stats_obj])

    if isinstance(returns, pd.DataFrame) and not stats.empty:
        if stats.shape[1] == returns.shape[1]:
            stats.columns = list(returns.columns)
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
    _call_assets_stats(port, method_mu=method_mu, method_cov=method_cov)
    frontier = port.efficient_frontier(model=model, rm=risk_measure, obj=obj, rf=rf, l=l, points=20)
    if not isinstance(frontier, pd.DataFrame):
        frontier = pd.DataFrame(frontier)
    return frontier


def risk_contributions(weights: pd.Series, returns: pd.DataFrame) -> pd.Series:
    """Compute marginal risk contributions for a given weights vector."""
    _require_riskfolio()
    import riskfolio as rp

    port = rp.Portfolio(returns=returns)
    _call_assets_stats(port, method_mu="hist", method_cov="hist")
    rc = port.risk_contribution(weights=weights, rm="MV")
    return rc
