r"""Performance analytics helpers for discrete-compounding return data.

The functions in this module operate on periodic **simple returns** and assume
that gains are reinvested every period without external cash flows. Under these
assumptions the cumulative growth of each series follows the discrete
compounding identity :math:`G_t = \prod_{i=1}^t (1 + r_i)`.
"""

from __future__ import annotations

import pandas as pd


def cumulative_return(series: pd.Series) -> pd.Series:
    r"""Compute the cumulative return of a single return series.

    The cumulative return through period :math:`t` is defined as

    .. math::
       R_t = \prod_{i=1}^t (1 + r_i) - 1,

    where :math:`r_i` are periodic simple returns expressed as decimal
    fractions (``0.01`` corresponds to +1%). The calculation assumes discrete
    compounding with full reinvestment of gains and losses.

    Parameters
    ----------
    series:
        Periodic simple returns indexed by time.

    Returns
    -------
    pandas.Series
        Cumulative returns aligned with ``series``.
    """
    if series.empty:
        return series.copy()
    return (1.0 + series).cumprod() - 1.0


def nav_series(
    returns: pd.DataFrame,
    weights: pd.Series | None = None,
    initial: float = 1.0,
    *,
    rebalance_cost_bps: float = 0.0,
    rebalance_fixed_fee: float = 0.0,
) -> pd.Series:
    r"""Generate a portfolio net asset value (NAV) path from asset returns.

    Given asset returns :math:`r_{i,t}` and target portfolio weights
    :math:`w_i`, the portfolio's simple return per period is the weighted sum

    .. math::
       r_{p,t} = \sum_i w_i r_{i,t}.

    With discrete compounding and rebalancing back to ``weights`` each period,
    the NAV recursion is

    .. math::
       \text{NAV}_t = \text{NAV}_{t-1} (1 + r_{p,t}), \qquad \text{NAV}_0 = \text{initial}.

    Parameters
    ----------
    returns:
        Wide DataFrame of periodic returns (rows are timestamps, columns are
        assets) expressed as decimal fractions.
    weights:
        Target portfolio weights. If ``None`` an equally weighted portfolio is
        assumed. Missing assets receive a weight of zero. The weights must sum
        to a non-zero value and are normalised to 1.
    initial:
        Starting NAV value. Units are preserved in the output.
    rebalance_cost_bps:
        Variable cost applied per unit of turnover at each rebalance, expressed
        in basis points. A value of 10 represents a 0.10% fee on the traded
        notional ``0.5 * sum(|w_{t}^{+} - w_i|)``.
    rebalance_fixed_fee:
        Fixed currency fee deducted whenever the rebalance requires trading
        (i.e. the turnover is positive).

    Returns
    -------
    pandas.Series
        NAV values for each period.

    Notes
    -----
    The returned series stores the realised periodic returns and turnover in
    ``Series.attrs["net_returns"]`` and ``Series.attrs["turnover"]`` for
    downstream analysis.

    Raises
    ------
    ValueError
        If the provided weights sum to zero after alignment with ``returns``.
    """
    if returns.empty:
        return pd.Series(dtype=float)

    if weights is None:
        weights = pd.Series(1.0 / returns.shape[1], index=returns.columns)
    else:
        weights = weights.reindex(returns.columns).fillna(0.0)

    total = float(weights.sum())
    if total == 0.0:
        raise ValueError("weights sum to zero")
    norm_weights = weights / total

    clean_returns = returns.fillna(0.0)

    nav_prev = float(initial)
    nav_path: list[float] = []
    net_returns: list[float] = []
    turnovers: list[float] = []

    cost_rate = rebalance_cost_bps / 10_000.0
    turnover_tol = 1e-12

    for _, row in clean_returns.iterrows():
        holdings_after = nav_prev * norm_weights * (1.0 + row)
        nav_before_rebalance = float(holdings_after.sum())

        if nav_before_rebalance <= 0.0:
            net_return = -1.0 if nav_prev > 0.0 else 0.0
            nav_prev = 0.0
            nav_path.append(nav_prev)
            net_returns.append(net_return)
            turnovers.append(0.0)
            continue

        weights_after = holdings_after / nav_before_rebalance
        turnover = 0.5 * float((weights_after - norm_weights).abs().sum())
        if turnover < turnover_tol:
            turnover = 0.0

        turnover_cost = nav_before_rebalance * turnover * cost_rate if turnover else 0.0
        fixed_cost = rebalance_fixed_fee if turnover else 0.0
        total_cost = turnover_cost + fixed_cost

        nav_after_cost = nav_before_rebalance - total_cost
        if nav_after_cost < 0.0:
            nav_after_cost = 0.0

        net_return = (nav_after_cost / nav_prev - 1.0) if nav_prev > 0.0 else 0.0
        nav_prev = nav_after_cost

        nav_path.append(nav_prev)
        net_returns.append(net_return)
        turnovers.append(turnover)

    nav_series = pd.Series(nav_path, index=clean_returns.index, dtype=float)
    if nav_path:
        nav_series.attrs["net_returns"] = pd.Series(
            net_returns, index=clean_returns.index, dtype=float
        )
        nav_series.attrs["turnover"] = pd.Series(
            turnovers, index=clean_returns.index, dtype=float
        )
    return nav_series


def nav_trajectories(returns: pd.DataFrame, *, initial_investment: float) -> pd.DataFrame:
    r"""Compute individual asset NAV trajectories from periodic returns.

    Each asset is assumed to start with the same capital ``initial_investment``
    and evolves according to

    .. math::
       \text{NAV}_{i,t} = \text{NAV}_{i,t-1} (1 + r_{i,t}),

    where :math:`r_{i,t}` are the asset's periodic simple returns. Missing
    returns are treated as zero performance for the corresponding period.


    Parameters
    ----------
    returns:
        Wide DataFrame of periodic returns (rows are timestamps, columns are
        assets) expressed as decimal fractions.
    initial_investment:
        Starting capital per asset. Units are preserved in the output.

    Returns
    -------
    pandas.DataFrame
        NAV values for each asset over time.
    """
    if returns.empty:
        return returns.copy()


    growth = (1.0 + returns.fillna(0.0)).cumprod()
    return growth * float(initial_investment)


def yield_trajectories(returns: pd.DataFrame) -> pd.DataFrame:
    r"""Compute cumulative yield trajectories for each asset.

    The cumulative yield through period :math:`t` for each asset is the
    discrete compounding of periodic returns:

    .. math::
       Y_{i,t} = \prod_{k=1}^t (1 + r_{i,k}) - 1.

    Missing returns are interpreted as zero for the corresponding period.

    Parameters
    ----------
    returns:
        Wide DataFrame of periodic returns (rows are timestamps, columns are
        assets) expressed as decimal fractions.

    Returns
    -------
    pandas.DataFrame
        Cumulative return for each asset as decimal fractions.

    """
    if returns.empty:
        return returns.copy()
    return (1.0 + returns.fillna(0.0)).cumprod() - 1.0
