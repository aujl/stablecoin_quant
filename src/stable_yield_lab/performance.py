r"""Performance analytics helpers for discrete-compounding return data.

The functions in this module operate on periodic **simple returns** and assume
that gains are reinvested every period without external cash flows. Under these
assumptions the cumulative growth of each series follows the discrete
compounding identity :math:`G_t = \prod_{i=1}^t (1 + r_i)`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
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


@dataclass
class RealisedAPYResult:
    """Container for realised APY analytics per asset."""

    realised_apy: float
    observations: int
    window_start: pd.Timestamp | None
    window_end: pd.Timestamp | None
    coverage_days: float
    warning: str


def _coverage_in_days(index: Iterable[pd.Timestamp]) -> float:
    timestamps = pd.Index(index).sort_values()
    if timestamps.empty:
        return float("nan")
    if len(timestamps) == 1:
        return float("nan")
    span = (timestamps.max() - timestamps.min()).total_seconds() / 86400.0
    avg_period = span / (len(timestamps) - 1)
    return float(span + avg_period) if avg_period > 0 else float("nan")


def estimate_realised_apy(
    returns: pd.DataFrame,
    *,
    lookback_days: int | None = None,
    min_observations: int = 4,
) -> pd.DataFrame:
    """Estimate realised APY per asset from periodic simple returns.

    The function compounds observed returns within an optional lookback window
    and annualises the cumulative growth using the realised time coverage.

    Parameters
    ----------
    returns:
        Wide DataFrame of periodic simple returns indexed by timestamp. Values
        should be expressed as decimal fractions (``0.01`` for +1%).
    lookback_days:
        Optional trailing window size in days. When provided only observations
        newer than ``now - lookback_days`` are considered.
    min_observations:
        Minimum non-missing observations required to produce an APY estimate.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by asset name with columns:
        ``realised_apy``, ``realised_apy_observations``,
        ``realised_apy_window_start``, ``realised_apy_window_end``,
        ``realised_apy_coverage_days`` and ``realised_apy_warning``.
    """

    if returns.empty:
        return pd.DataFrame(
            columns=
            [
                "realised_apy",
                "realised_apy_observations",
                "realised_apy_window_start",
                "realised_apy_window_end",
                "realised_apy_coverage_days",
                "realised_apy_warning",
            ]
        )

    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("returns index must be a DatetimeIndex")

    window_end = returns.index.max()
    if lookback_days is not None:
        cutoff = window_end - pd.Timedelta(days=int(lookback_days))
        windowed = returns.loc[returns.index >= cutoff]
    else:
        windowed = returns

    results: list[RealisedAPYResult] = []
    names: list[str] = []

    for name, series in windowed.items():
        clean = series.dropna()
        observations = int(clean.shape[0])
        warning = ""
        realised = float("nan")
        coverage_days = float("nan")
        window_start: pd.Timestamp | None = None
        window_end_col: pd.Timestamp | None = None

        if observations == 0:
            warning = "No observations within lookback window."
        else:
            window_start = pd.Timestamp(clean.index.min())
            window_end_col = pd.Timestamp(clean.index.max())
            if observations < int(min_observations):
                warning = (
                    f"Only {observations} observations (< {int(min_observations)})"
                    " within lookback window."
                )
            elif observations == 1:
                warning = "Only a single observation; cannot annualise returns."
            else:
                coverage_days = _coverage_in_days(clean.index)
                if not math.isfinite(coverage_days) or coverage_days <= 0:
                    warning = "Insufficient time coverage to annualise returns."
                else:
                    log_growth = float(np.log1p(clean).sum())
                    annual_factor = 365.25 / coverage_days
                    realised = math.expm1(log_growth * annual_factor)

        results.append(
            RealisedAPYResult(
                realised_apy=realised,
                observations=observations,
                window_start=window_start,
                window_end=window_end_col,
                coverage_days=coverage_days,
                warning=warning,
            )
        )
        names.append(str(name))

    df = pd.DataFrame(
        {
            "realised_apy": [r.realised_apy for r in results],
            "realised_apy_observations": [r.observations for r in results],
            "realised_apy_window_start": [r.window_start for r in results],
            "realised_apy_window_end": [r.window_end for r in results],
            "realised_apy_coverage_days": [r.coverage_days for r in results],
            "realised_apy_warning": [r.warning for r in results],
        },
        index=names,
    )
    return df
