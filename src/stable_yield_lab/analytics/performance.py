r"""Performance analytics helpers for discrete-compounding return data.

The functions in this module operate on periodic **simple returns** and assume
that gains are reinvested every period without external cash flows. Under these
assumptions the cumulative growth of each series follows the discrete
compounding identity :math:`G_t = \prod_{i=1}^t (1 + r_i)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable, Mapping

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


@dataclass(frozen=True)
class RebalanceScenario:
    """Parameters describing a portfolio rebalance experiment."""

    calendar: Iterable[pd.Timestamp] | pd.DatetimeIndex
    cost_bps: float = 0.0


@dataclass(frozen=True)
class ScenarioRunResult:
    """Container for aggregated scenario metrics and paths."""

    metrics: pd.DataFrame
    navs: pd.DataFrame
    returns: pd.DataFrame


@dataclass(frozen=True)
class _ScenarioPath:
    nav: pd.Series
    returns: pd.Series
    total_cost: float


def _normalise_weights(weights: pd.Series, columns: pd.Index) -> pd.Series:
    aligned = weights.reindex(columns).fillna(0.0)
    total = float(aligned.sum())
    if total == 0.0:
        raise ValueError("weights sum to zero after alignment with returns")
    return aligned / total


def _prepare_calendar(
    calendar: Iterable[pd.Timestamp] | pd.DatetimeIndex | None,
    index: pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    if calendar is None:
        return pd.DatetimeIndex([], tz=index.tz)
    if isinstance(calendar, pd.DatetimeIndex):
        cal = calendar
    else:
        cal = pd.DatetimeIndex(pd.to_datetime(list(calendar)))
    if index.tz is not None:
        if cal.tz is None:
            cal = cal.tz_localize(index.tz)
        else:
            cal = cal.tz_convert(index.tz)
    else:
        if cal.tz is not None:
            cal = cal.tz_convert(None)
    return cal.intersection(index)


def _infer_periods_per_year(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 1.0
    diffs = index.to_series().diff().dropna()
    if diffs.empty:
        return 1.0
    avg_days = diffs.dt.total_seconds().mean() / 86_400.0
    if avg_days <= 0:
        return float(len(index))
    return 365.25 / avg_days


def _simulate_rebalanced_portfolio(
    returns: pd.DataFrame,
    weights: pd.Series,
    scenario: RebalanceScenario,
    *,
    initial_nav: float,
) -> _ScenarioPath:
    if returns.empty:
        empty = pd.Series(dtype=float, index=returns.index)
        return _ScenarioPath(nav=empty, returns=empty, total_cost=0.0)

    clean_returns = returns.fillna(0.0)
    weights = _normalise_weights(weights, clean_returns.columns)
    calendar = _prepare_calendar(scenario.calendar, clean_returns.index)
    rebalance_mask = pd.Series(clean_returns.index.isin(calendar), index=clean_returns.index)

    nav = float(initial_nav)
    holdings = weights * nav
    nav_path: list[float] = []
    period_returns: list[float] = []
    total_cost = 0.0
    cost_rate = float(scenario.cost_bps) / 10_000.0

    for timestamp, row in clean_returns.iterrows():
        nav_before = nav
        holdings = holdings * (1.0 + row)
        nav = float(holdings.sum())

        if rebalance_mask.loc[timestamp]:
            if nav > 0.0:
                current_weights = holdings / nav
                diff = weights - current_weights
                traded_value = float(diff.abs().sum()) * nav
            else:
                traded_value = 0.0
            cost = traded_value * cost_rate
            if cost:
                nav -= cost
                total_cost += cost
            holdings = weights * nav

        period_return = (nav - nav_before) / nav_before if nav_before != 0 else 0.0
        nav_path.append(nav)
        period_returns.append(period_return)

    nav_series = pd.Series(nav_path, index=clean_returns.index, name="nav")
    returns_series = pd.Series(period_returns, index=clean_returns.index, name="return")
    return _ScenarioPath(nav=nav_series, returns=returns_series, total_cost=total_cost)


def run_rebalance_scenarios(
    returns: pd.DataFrame,
    weights: pd.Series,
    scenarios: Mapping[str, RebalanceScenario],
    *,
    benchmark: str | None = None,
    initial_nav: float = 1.0,
) -> ScenarioRunResult:
    """Evaluate portfolio performance under alternative rebalance calendars.

    Parameters
    ----------
    returns:
        Wide DataFrame of periodic simple returns with datetime index.
    weights:
        Target portfolio weights aligned with ``returns`` columns.
    scenarios:
        Mapping of scenario name to calendar/cost assumptions.
    benchmark:
        Scenario name used as reference for tracking error. When ``None`` the
        benchmark is a frictionless strategy that rebalances every period.
    initial_nav:
        Starting portfolio value.

    Returns
    -------
    ScenarioRunResult
        Object containing per-scenario metrics plus NAV/return trajectories.
    """

    if returns.empty:
        empty = pd.DataFrame(index=returns.index)
        return ScenarioRunResult(
            metrics=pd.DataFrame(columns=["realized_apy", "total_cost", "tracking_error", "terminal_nav"]),
            navs=empty,
            returns=empty,
        )

    if not scenarios:
        raise ValueError("at least one scenario must be provided")

    paths: dict[str, _ScenarioPath] = {}
    for name, scenario in scenarios.items():
        paths[name] = _simulate_rebalanced_portfolio(
            returns,
            weights,
            scenario,
            initial_nav=initial_nav,
        )

    if benchmark is not None:
        if benchmark not in paths:
            raise KeyError(f"benchmark '{benchmark}' not found in scenarios")
        benchmark_returns = paths[benchmark].returns
    else:
        benchmark_path = _simulate_rebalanced_portfolio(
            returns,
            weights,
            RebalanceScenario(calendar=returns.index, cost_bps=0.0),
            initial_nav=initial_nav,
        )
        benchmark_returns = benchmark_path.returns

    benchmark_returns = benchmark_returns.reindex(returns.index, fill_value=0.0)
    periods_per_year = _infer_periods_per_year(returns.index)

    metrics_rows: list[dict[str, float | str]] = []
    nav_data: dict[str, pd.Series] = {}
    return_data: dict[str, pd.Series] = {}

    for name, path in paths.items():
        nav_series = path.nav.reindex(returns.index, fill_value=float("nan"))
        return_series = path.returns.reindex(returns.index, fill_value=0.0)
        nav_data[name] = nav_series
        return_data[name] = return_series

        total_periods = len(return_series)
        total_growth = nav_series.iloc[-1] / initial_nav if total_periods else float("nan")
        if total_periods and total_growth > 0.0:
            realized_apy = total_growth ** (periods_per_year / total_periods) - 1.0
        else:
            realized_apy = float("nan")

        diff = (return_series - benchmark_returns).fillna(0.0)
        if len(diff) > 1:
            tracking_error = float(diff.std(ddof=0) * sqrt(periods_per_year))
        else:
            tracking_error = 0.0

        metrics_rows.append(
            {
                "scenario": name,
                "realized_apy": float(realized_apy),
                "total_cost": float(path.total_cost),
                "tracking_error": tracking_error,
                "terminal_nav": float(nav_series.iloc[-1]),
            }
        )

    metrics = pd.DataFrame(metrics_rows).set_index("scenario")
    navs = pd.DataFrame(nav_data)
    returns_df = pd.DataFrame(return_data)
    return ScenarioRunResult(metrics=metrics, navs=navs, returns=returns_df)
