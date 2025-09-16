r"""Performance analytics helpers for discrete-compounding return data.

The functions in this module operate on periodic **simple returns** and assume
that gains are reinvested every period without external cash flows. Under these
assumptions the cumulative growth of each series follows the discrete
compounding identity :math:`G_t = \prod_{i=1}^t (1 + r_i)`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class HorizonAPYDiagnostics:
    r"""Annualised APY diagnostics for a single pool over a discrete horizon.

    Attributes
    ----------
    pool:
        Pool identifier (column name from the returns DataFrame).
    apy:
        Annualised percentage yield over the evaluated horizon expressed as a
        decimal fraction.
    periods:
        Number of observed (non-missing) return periods within the horizon.
    missing_pct:
        Fraction of missing observations relative to the horizon length
        (``0`` means fully observed, ``1`` means all periods missing).
    volatility:
        Standard deviation of observed period returns (decimal fraction).
    """

    pool: str
    apy: float
    periods: int
    missing_pct: float
    volatility: float


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


def horizon_apy_diagnostics(
    returns: pd.DataFrame,
    *,
    periods_per_year: int,
    horizon: int | None = None,
) -> pd.DataFrame:
    r"""Summarise annualised APY and data coverage diagnostics per pool.

    Parameters
    ----------
    returns:
        Wide DataFrame of periodic simple returns (rows are timestamps, columns
        are pools) expressed as decimal fractions.
    periods_per_year:
        Number of compounding periods in a year (e.g. ``52`` for weekly
        returns). Must be positive.
    horizon:
        Optional look-back window expressed as a number of periods. When
        provided, only the most recent ``horizon`` rows are considered.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by pool name with columns ``["apy", "periods",
        "missing_pct", "volatility"]``. ``apy`` is the annualised yield over
        the selected horizon, ``periods`` counts observed (non-missing)
        returns, ``missing_pct`` measures the share of missing observations,
        and ``volatility`` records the standard deviation of observed returns.
        Missing returns are treated as ``0`` during compounding to preserve the
        horizon length; the ``missing_pct`` column surfaces the resulting data
        gaps.

    Raises
    ------
    ValueError
        If ``periods_per_year`` or ``horizon`` (when provided) are not
        positive.
    """

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    if returns.empty:
        return pd.DataFrame(columns=["apy", "periods", "missing_pct", "volatility"], dtype=float)

    if horizon is not None:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        window = returns.tail(horizon)
    else:
        window = returns

    diagnostics: list[HorizonAPYDiagnostics] = []
    for pool, series in window.items():
        total_periods = len(series)
        if total_periods == 0:
            continue

        observed = series.dropna()
        periods = int(observed.size)
        missing_pct = float((total_periods - periods) / total_periods)
        volatility = float(observed.std(ddof=0)) if periods > 0 else float("nan")

        if periods == 0:
            apy = float("nan")
        else:
            filled = series.fillna(0.0)
            growth = float((1.0 + filled).prod())
            if growth < 0.0:
                apy = float("nan")
            else:
                exponent = periods_per_year / total_periods
                apy = float(growth ** exponent - 1.0)

        diagnostics.append(
            HorizonAPYDiagnostics(
                pool=pool,
                apy=apy,
                periods=periods,
                missing_pct=missing_pct,
                volatility=volatility,
            )
        )

    if not diagnostics:
        return pd.DataFrame(columns=["apy", "periods", "missing_pct", "volatility"], dtype=float)

    df = pd.DataFrame([asdict(d) for d in diagnostics]).set_index("pool")
    return df[["apy", "periods", "missing_pct", "volatility"]]
