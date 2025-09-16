"""Portfolio performance utilities.

This module offers simple analytics for cumulative returns and net asset value
(NAV) paths. Both functions assume **discrete compounding** and periodic
rebalancing to fixed weights.
"""

from __future__ import annotations

import pandas as pd


def cumulative_return(series: pd.Series) -> pd.Series:
    r"""Compute the cumulative return of a return series.

    Parameters
    ----------
    series:
        Periodic simple returns expressed as decimal fractions (e.g. ``0.01``
        for 1 %).

    Returns
    -------
    pandas.Series
        Cumulative return where ``R_t = \prod_{i=1}^t (1 + r_i) - 1``.

    Notes
    -----
    The formula assumes reinvestment of gains each period and no leverage.
    """
    if series.empty:
        return series.copy()
    return (1 + series).cumprod() - 1


def nav_series(
    returns: pd.DataFrame,
    weights: pd.Series | None = None,
    initial: float = 1.0,
) -> pd.Series:
    r"""Generate a net asset value series from asset returns.

    Given asset returns :math:`r_{i,t}` and target weights :math:`w_i`, the
    portfolio return per period is

    .. math::
       r_{p,t} = \sum_i w_i r_{i,t}

    The NAV then evolves as

    .. math::
       NAV_t = NAV_{t-1} (1 + r_{p,t}), \quad NAV_0 = \text{initial}

    Parameters
    ----------
    returns:
        Wide DataFrame of periodic returns (index is time, columns are assets).
    weights:
        Target portfolio weights. If ``None`` an equally-weighted portfolio is
        assumed. The weights are normalised to sum to one at each call.
    initial:
        Starting NAV value. Units of the result follow the units of ``initial``.

    Returns
    -------
    pandas.Series
        NAV values for each period.

    Notes
    -----
    Assumes constant-mix rebalancing and simple (non-log) returns.
    """
    if returns.empty:
        return pd.Series(dtype=float)

    if weights is None:
        weights = pd.Series(1 / returns.shape[1], index=returns.columns)
    else:
        weights = weights.reindex(returns.columns).fillna(0)

    total = weights.sum()
    if total == 0:
        raise ValueError("weights sum to zero")
    weights = weights / total

    portfolio_ret = returns.mul(weights, axis=1).sum(axis=1)
    cum = cumulative_return(portfolio_ret)
    return initial * (1 + cum)
