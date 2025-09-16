r"""Performance analytics helpers for discrete-compounding return data.

The functions in this module operate on periodic **simple returns** and assume
that gains are reinvested every period without external cash flows. Under these
assumptions the cumulative growth of each series follows the discrete
compounding identity :math:`G_t = \prod_{i=1}^t (1 + r_i)`.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import tzinfo

import pandas as pd
from pandas.tseries.frequencies import to_offset


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

    Returns
    -------
    pandas.Series
        NAV values for each period.

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
    portfolio_ret = clean_returns.mul(norm_weights, axis=1).sum(axis=1)
    compounded = cumulative_return(portfolio_ret)
    return float(initial) * (1.0 + compounded)


def _align_timezone(index: pd.DatetimeIndex, tz: tzinfo | None) -> pd.DatetimeIndex:
    if tz is None:
        if index.tz is not None:
            return index.tz_localize(None)
        return index

    if index.tz is None:
        return index.tz_localize(tz)
    return index.tz_convert(tz)


def _rebalance_schedule(
    index: pd.DatetimeIndex,
    calendar: str | pd.DatetimeIndex | Iterable[pd.Timestamp] | pd.Timestamp | None,
) -> pd.DatetimeIndex:
    if calendar is None:
        schedule = pd.DatetimeIndex([], tz=index.tz)
    elif isinstance(calendar, str):
        try:
            offset = to_offset(calendar)
        except ValueError as exc:
            raise ValueError(f"invalid calendar frequency: {calendar}") from exc
        schedule = pd.date_range(start=index[0], end=index[-1], freq=offset, tz=index.tz)
    elif isinstance(calendar, pd.Timestamp):
        schedule = pd.DatetimeIndex([calendar])
    elif isinstance(calendar, pd.DatetimeIndex):
        schedule = calendar.copy()
    elif isinstance(calendar, Iterable):
        schedule = pd.DatetimeIndex(list(calendar))
    else:
        raise TypeError("unsupported calendar specification")

    schedule = _align_timezone(schedule, index.tz)
    return schedule.intersection(index)


def nav_with_rebalance(
    returns: pd.DataFrame,
    calendar: str | pd.DatetimeIndex | Iterable[pd.Timestamp] | pd.Timestamp | None,
    *,
    weights: pd.Series | None = None,
    initial: float = 1.0,
) -> pd.Series:
    r"""Simulate a portfolio NAV path with periodic rebalancing.

    The function models a buy-and-hold portfolio that is *only* rebalanced to
    the target weights on the provided ``calendar``. Between those events the
    asset weights drift according to realised returns. All returns are treated
    as discrete simple returns and missing observations are interpreted as zero
    performance. Assets that are absent from ``weights`` receive a zero
    allocation whenever the portfolio is rebalanced.

    Under these assumptions the portfolio NAV evolves as

    .. math::
       \text{NAV}_t = \text{NAV}_{t-1} (1 + r_{p,t}), \qquad
       r_{p,t} = \sum_i w_{i,t^-} r_{i,t},

    where :math:`w_{i,t^-}` denotes the weights immediately before period
    :math:`t`. After each period the drifted weights satisfy

    .. math::
       w_{i,t^+} = \frac{w_{i,t^-} (1 + r_{i,t})}{1 + r_{p,t}},

    and these weights are reset to the target allocation only on calendar
    dates.

    Parameters
    ----------
    returns:
        Wide DataFrame of periodic simple returns (rows are timestamps,
        columns are assets). The index must be a ``DatetimeIndex`` and will be
        processed in chronological order.
    calendar:
        Rebalancing schedule. Accepted inputs include a pandas frequency alias
        (e.g. ``"W-MON"`` or ``"M"``), a ``DatetimeIndex``/sequence of
        timestamps, or a single timestamp. Only dates that are present in
        ``returns.index`` trigger a rebalance. If the calendar does not contain
        any in-sample dates the portfolio is rebalanced only on the first
        observation and then allowed to drift. Passing ``None`` has the same
        effect as providing an empty calendar.
    weights:
        Target weights applied on rebalance dates. If ``None`` the portfolio is
        equally weighted across all assets. Missing assets receive a weight of
        zero. The weights are normalised to sum to one.
    initial:
        Starting NAV value. Units are preserved in the output.

    Returns
    -------
    pandas.Series
        Simulated NAV values aligned with ``returns``.

    Raises
    ------
    TypeError
        If ``returns.index`` is not a ``DatetimeIndex``.
    ValueError
        If the normalised weights sum to zero.
    """
    if returns.empty:
        return pd.Series(dtype=float)

    ordered_returns = returns.sort_index()
    if not isinstance(ordered_returns.index, pd.DatetimeIndex):
        raise TypeError("returns index must be a DatetimeIndex")

    clean_returns = ordered_returns.fillna(0.0)
    if clean_returns.shape[1] == 0:
        return pd.Series(dtype=float)

    if weights is None:
        target = pd.Series(1.0 / clean_returns.shape[1], index=clean_returns.columns)
    else:
        target = weights.reindex(clean_returns.columns).fillna(0.0)

    total_weight = float(target.sum())
    if total_weight == 0.0:
        raise ValueError("weights sum to zero")

    target_weights = target / total_weight

    index = clean_returns.index
    schedule_index = _rebalance_schedule(index, calendar)
    if schedule_index.empty or schedule_index[0] != index[0]:
        schedule_index = schedule_index.union(pd.DatetimeIndex([index[0]]))
    schedule = set(schedule_index)

    nav_path: list[float] = []
    nav_value = float(initial)
    current_weights = target_weights.copy()

    for timestamp, period_returns in clean_returns.iterrows():
        if timestamp in schedule:
            current_weights = target_weights.copy()

        portfolio_return = float(period_returns.mul(current_weights).sum())
        nav_value *= 1.0 + portfolio_return
        nav_path.append(nav_value)

        gross = current_weights * (1.0 + period_returns)
        gross_total = float(gross.sum())
        if gross_total == 0.0:
            current_weights = pd.Series(0.0, index=current_weights.index)
        else:
            current_weights = gross / gross_total

    return pd.Series(nav_path, index=index)


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
