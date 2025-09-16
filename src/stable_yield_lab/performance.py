r"""Performance analytics helpers for discrete-compounding return data.

The functions in this module operate on periodic **simple returns** and assume
that gains are reinvested every period without external cash flows. Under these
assumptions the cumulative growth of each series follows the discrete
compounding identity :math:`G_t = \prod_{i=1}^t (1 + r_i)`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Literal

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


@dataclass(frozen=True)
class APYEstimate:
    r"""Structured APY estimate with metadata for downstream reporting."""

    summary: pd.DataFrame
    frequency: str
    periods_per_year: float
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    performance_fee_bps: float
    management_fee_bps: float

    def to_frame(self) -> pd.DataFrame:
        """Return a defensive copy of the summary table."""

        return self.summary.copy()


def estimate_pool_apy(
    returns: pd.Series | pd.DataFrame,
    *,
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
    frequency: str = "D",
    perf_fee_bps: float = 0.0,
    mgmt_fee_bps: float = 0.0,
) -> APYEstimate:
    r"""Estimate annual percentage yield (APY) from periodic simple returns.

    The function assumes discrete compounding with full reinvestment. Given
    periodic simple returns :math:`r_t`, the cumulative growth over ``T``
    observations is

    .. math::
       G_T = \prod_{t=1}^T (1 + r_t).

    The corresponding sample return is ``G_T - 1``. To annualise the result we
    exponentiate by the ratio of periods per year :math:`n_f` (inferred from the
    provided ``frequency``) to the number of observed periods ``T``:

    .. math::
       \text{APY} = G_T^{n_f / T} - 1.

    Performance and management fees supplied in basis points reduce the annual
    growth factor by a multiplicative factor :math:`1 - (\text{perf} + \text{mgmt}) / 10^4`
    with a floor at ``-100%`` on the resulting APY.

    Parameters
    ----------
    returns:
        Series or DataFrame of periodic simple returns indexed by
        ``DatetimeIndex``.
    start, end:
        Optional inclusive temporal bounds. If the filtered window contains no
        observations a ``ValueError`` is raised.
    frequency:
        Pandas offset alias used to infer the number of periods per year for
        annualisation (e.g. ``"D"`` for daily, ``"W"`` for weekly).
    perf_fee_bps, mgmt_fee_bps:
        Optional fee adjustments expressed in basis points.

    Returns
    -------
    APYEstimate
        Structured summary with gross and net APY plus metadata for each input
        series.
    """

    frame = _coerce_to_dataframe(returns)
    if frame.empty:
        raise ValueError("returns data is empty")

    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("returns index must be a DatetimeIndex")

    ordered = frame.sort_index()

    start_ts = pd.Timestamp(start) if start is not None else None
    end_ts = pd.Timestamp(end) if end is not None else None

    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        raise ValueError("start must be on or before end")

    if start_ts is not None:
        ordered = ordered.loc[ordered.index >= start_ts]
    if end_ts is not None:
        ordered = ordered.loc[ordered.index <= end_ts]

    ordered = ordered.dropna(how="all")
    if ordered.empty:
        raise ValueError("no returns available in the requested window")

    window_start = ordered.index.min()
    window_end = ordered.index.max()

    periods_per_year = _periods_per_year(frequency)
    fee_multiplier = 1.0 - (float(perf_fee_bps) + float(mgmt_fee_bps)) / 10_000.0

    periods: dict[Any, float] = {}
    totals: dict[Any, float] = {}
    annualisation: dict[Any, float] = {}
    gross: dict[Any, float] = {}
    net: dict[Any, float] = {}

    for column in ordered.columns:
        series = ordered[column].dropna()
        if series.empty:
            periods[column] = float("nan")
            totals[column] = float("nan")
            annualisation[column] = float("nan")
            gross[column] = float("nan")
            net[column] = float("nan")
            continue

        compounded = cumulative_return(series)
        total_return = float(compounded.iloc[-1])
        observations = float(len(series))

        periods[column] = observations
        totals[column] = total_return

        annual_factor = periods_per_year / observations
        annualisation[column] = annual_factor

        gross_apy = (1.0 + total_return) ** annual_factor - 1.0
        gross[column] = gross_apy

        net_growth = (1.0 + gross_apy) * fee_multiplier
        net_apy = net_growth - 1.0
        if net_apy < -1.0:
            net_apy = -1.0
        net[column] = net_apy

    summary = pd.DataFrame(
        {
            "periods": pd.Series(periods, dtype=float),
            "total_return": pd.Series(totals, dtype=float),
            "annualization_factor": pd.Series(annualisation, dtype=float),
            "gross_apy": pd.Series(gross, dtype=float),
            "net_apy": pd.Series(net, dtype=float),
        }
    )
    summary = summary.reindex(ordered.columns)

    return APYEstimate(
        summary=summary,
        frequency=frequency,
        periods_per_year=periods_per_year,
        window_start=window_start,
        window_end=window_end,
        performance_fee_bps=float(perf_fee_bps),
        management_fee_bps=float(mgmt_fee_bps),
    )


def _coerce_to_dataframe(returns: pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(returns, pd.Series):
        name = returns.name or "value"
        return returns.to_frame(name=name)
    return returns.copy()


def _periods_per_year(freq: str) -> float:
    try:
        offset = to_offset(freq)
    except ValueError as exc:
        raise ValueError(f"invalid frequency alias: {freq!r}") from exc

    rule = getattr(offset, "rule_code", None)
    base_code = (rule or offset.freqstr).upper()
    if "-" in base_code:
        base_code = base_code.split("-")[0]
    base_code = base_code.rstrip("S")

    mapping: dict[str, float] = {
        "B": 252.0,
        "D": 365.0,
        "W": 52.0,
        "M": 12.0,
        "Q": 4.0,
        "A": 1.0,
        "Y": 1.0,
    }

    multiplier = float(getattr(offset, "n", 1))
    if base_code in mapping:
        return mapping[base_code] / multiplier

    nanos = getattr(offset, "nanos", None)
    if nanos:
        year = pd.Timedelta(days=365)
        return float(year / pd.Timedelta(nanos, unit="ns"))

    delta = getattr(offset, "delta", None)
    if delta:
        year = pd.Timedelta(days=365)
        return float(year / delta)

    raise ValueError(f"unsupported frequency for annualisation: {freq!r}")


def horizon_apys(
    trajectory: pd.DataFrame | pd.Series,
    *,
    lookbacks: Mapping[str, int | str | pd.Timedelta],
    value_type: Literal["nav", "yield"] = "nav",
    periods_per_year: float | None = None,
) -> pd.DataFrame:
    r"""Convert NAV or cumulative yield paths into annualised realised APYs.

    For a lookback window of length :math:`L` (in years) the realised annual
    percentage yield (APY) is computed from the growth factor :math:`G`
    observed over the window:

    .. math::
       \text{APY} = G^{1/L} - 1, \qquad G = \frac{\text{NAV}_{t}}{\text{NAV}_{t-L}}.

    The function accepts NAV trajectories (level data) or cumulative yields. In
    the latter case the input is first converted into NAV space via
    ``1 + cumulative_yield`` so that growth factors remain comparable across
    pools regardless of the initial capital. Lookbacks may be provided either
    as positive integers (number of periods) or as pandas-compatible
    ``Timedelta`` specifications such as ``"52W"`` or ``"90D"``.

    Parameters
    ----------
    trajectory:
        NAV levels or cumulative yields indexed by time with one column per
        asset.
    lookbacks:
        Mapping of human-readable labels to lookback definitions. Integer
        values are interpreted as a number of observation periods; strings and
        :class:`~pandas.Timedelta` objects are converted via
        :func:`pandas.to_timedelta`.
    value_type:
        ``"nav"`` when ``trajectory`` already represents NAV levels, or
        ``"yield"`` when the input is cumulative yield (``G - 1``).
    periods_per_year:
        Optional frequency override used when lookbacks are specified in
        integer periods and cannot be inferred from the index. If omitted the
        function attempts to infer the observation frequency from a
        ``DatetimeIndex``.

    Returns
    -------
    pandas.DataFrame
        Realised APYs expressed as decimal fractions. The rows correspond to
        asset names (trajectory columns) and columns align with the provided
        ``lookbacks`` labels.

    Raises
    ------
    ValueError
        If an integer lookback is provided without a resolvable
        ``periods_per_year`` and the index frequency cannot be inferred, or if
        any lookback definition is non-positive.
    TypeError
        When a time-based lookback is requested but the index is not a
        ``DatetimeIndex``.
    """

    if isinstance(trajectory, pd.Series):
        name = trajectory.name or "value"
        data = trajectory.astype(float).to_frame(name=name)
    else:
        data = trajectory.astype(float)

    if data.empty:
        return pd.DataFrame(index=data.columns, columns=list(lookbacks.keys()), dtype=float)

    if not lookbacks:
        return pd.DataFrame(index=data.columns, dtype=float)

    nav = data.sort_index()
    nav = nav.astype(float)

    value_type = value_type.lower()
    if value_type == "yield":
        nav = 1.0 + nav
    elif value_type != "nav":
        raise ValueError(f"Unsupported value_type '{value_type}'. Use 'nav' or 'yield'.")

    idx = nav.index
    period_length: pd.Timedelta | None = None
    if isinstance(idx, pd.DatetimeIndex) and len(idx) >= 2:
        diffs = idx.to_series().diff().dropna()
        if not diffs.empty:
            median = diffs.median()
            if pd.notna(median) and median > pd.Timedelta(0):
                period_length = pd.Timedelta(median)

    per_year = float(periods_per_year) if periods_per_year is not None else None
    if per_year is None and period_length is not None:
        per_year = float(pd.Timedelta(days=365) / period_length)

    assets = list(nav.columns)
    result = pd.DataFrame(index=assets, columns=list(lookbacks.keys()), dtype=float)
    latest = nav.iloc[-1]

    year_td = pd.Timedelta(days=365)

    for label, raw_lookback in lookbacks.items():
        if isinstance(raw_lookback, Integral):
            periods = int(raw_lookback)
            if periods <= 0:
                raise ValueError(f"Lookback '{label}' must be positive; received {periods}.")
            if len(nav) <= periods:
                continue
            base = nav.iloc[-(periods + 1)]
            if per_year is None:
                raise ValueError(
                    "periods_per_year must be provided or inferrable when using integer lookbacks"
                )
            years = periods / per_year
        else:
            try:
                delta = pd.to_timedelta(raw_lookback)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid lookback '{label}': {raw_lookback!r}") from exc
            if delta <= pd.Timedelta(0):
                raise ValueError(f"Lookback '{label}' must be positive; received {raw_lookback!r}.")
            if not isinstance(idx, pd.DatetimeIndex):
                raise TypeError("Timedelta lookbacks require a DatetimeIndex.")
            target = idx[-1] - delta
            window = nav.loc[:target]
            if window.empty:
                continue
            base = window.iloc[-1]
            years = float(delta / year_td)

        if years <= 0:
            continue

        growth = latest / base
        valid = (base > 0) & (latest > 0) & growth.notna() & (growth > 0)
        apy_series = pd.Series(float("nan"), index=assets)
        if valid.any():
            exponent = 1.0 / years
            apy_series.loc[valid] = growth.loc[valid].pow(exponent) - 1.0
        result[label] = apy_series

    return result

