from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping

import pandas as pd

from . import performance
from .risk_metrics import _require_riskfolio


@dataclass(frozen=True)
class RebalancePortfolioResult:
    """Outputs from :func:`rebalance_portfolio`."""

    weights: pd.DataFrame
    nav: pd.Series
    returns: pd.Series


def _normalise_weights(returns: pd.DataFrame, weights: pd.Series | None) -> pd.Series:
    """Align and normalise weight vector against ``returns`` columns."""

    if returns.empty:
        return pd.Series(dtype=float)

    if weights is None:
        return pd.Series(1.0 / returns.shape[1], index=returns.columns, dtype=float)

    aligned = weights.reindex(returns.columns).fillna(0.0)
    total = float(aligned.sum())
    if total == 0.0:
        raise ValueError("weights sum to zero")
    return aligned / total


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


def _normalise_schedule_index(
    dates: Iterable[pd.Timestamp],
    reference: pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    """Align a list of timestamps to the timezone of ``reference``."""

    idx = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    if reference.tz is not None:
        if idx.tz is None:
            idx = idx.tz_localize(reference.tz)
        else:
            idx = idx.tz_convert(reference.tz)
    elif idx.tz is not None:
        idx = idx.tz_convert(None)
    return idx


def schedule_from_user_weights(
    *,
    dates: Iterable[pd.Timestamp],
    weights: Mapping[str, float] | pd.Series,
    returns_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Construct a rebalance schedule from user supplied weights."""

    schedule_index = _normalise_schedule_index(dates, returns_index)
    if schedule_index.empty:
        raise ValueError("at least one rebalance date must be provided")
    if not pd.Index(schedule_index).isin(returns_index).all():
        missing = schedule_index[~schedule_index.isin(returns_index)]
        raise ValueError(f"rebalance dates not present in returns index: {missing.tolist()}")

    weight_series = pd.Series(weights, dtype=float)
    if weight_series.empty:
        raise ValueError("weights must contain at least one asset")
    total = float(weight_series.sum())
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value")
    norm = weight_series / total

    columns = list(norm.index)
    schedule = pd.DataFrame([norm] * len(schedule_index), index=schedule_index, columns=columns)
    return schedule


def schedule_from_optimizations(
    optimizations: Mapping[pd.Timestamp, pd.Series],
    *,
    returns_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Normalise optimisation outputs into a weight schedule."""

    if not optimizations:
        raise ValueError("optimizations cannot be empty")

    aligned_index = _normalise_schedule_index(optimizations.keys(), returns_index)

    columns: list[str] = []
    series_rows: list[pd.Series] = []
    for series in optimizations.values():
        weights = pd.Series(series, dtype=float)
        for asset in weights.index:
            if asset not in columns:
                columns.append(asset)
        series_rows.append(weights)

    schedule_rows: list[pd.Series] = []
    for ts, weights in zip(aligned_index, series_rows):
        aligned = weights.reindex(columns).fillna(0.0)
        total = float(aligned.sum())
        if total <= 0.0:
            raise ValueError("optimization weights must sum to a positive value")
        schedule_rows.append(aligned / total)

    schedule = pd.DataFrame(schedule_rows, index=aligned_index, columns=columns)
    return schedule.sort_index()


def rebalance_portfolio(
    returns: pd.DataFrame,
    *,
    rebalance_schedule: Iterable[pd.Timestamp],
    target_weights: Mapping[pd.Timestamp, Mapping[str, float] | pd.Series],
    initial_nav: float = 1.0,
) -> RebalancePortfolioResult:
    """Simulate a simple rebalance strategy with piecewise-constant weights."""

    if returns.empty:
        index = pd.DatetimeIndex(returns.index)
        empty_df = pd.DataFrame(index=index)
        empty_series = pd.Series(index=index, dtype=float)
        return RebalancePortfolioResult(weights=empty_df, nav=empty_series, returns=empty_series)

    returns = returns.copy()
    returns.index = pd.DatetimeIndex(returns.index)
    returns = returns.sort_index()
    returns_index = pd.DatetimeIndex(returns.index)

    schedule_index = _normalise_schedule_index(rebalance_schedule, returns_index)
    if schedule_index.empty:
        raise ValueError("rebalance_schedule must contain at least one timestamp")
    if not schedule_index.isin(returns_index).all():
        missing = schedule_index[~schedule_index.isin(returns_index)]
        raise ValueError(f"rebalance dates not found in returns index: {missing.tolist()}")

    normalised_targets: dict[pd.Timestamp, pd.Series] = {}
    for ts_raw, weights in target_weights.items():
        ts = _normalise_schedule_index([ts_raw], returns_index)[0]
        normalised_targets[ts] = pd.Series(weights, dtype=float)

    columns: list[str] = list(returns.columns)
    for series in normalised_targets.values():
        for asset in series.index:
            if asset not in columns:
                columns.append(asset)

    schedule_rows: list[pd.Series] = []
    for ts in schedule_index:
        weights = normalised_targets.get(ts)
        if weights is None:
            raise KeyError(f"missing target weights for {ts}")
        aligned = weights.reindex(columns).fillna(0.0)
        total = float(aligned.sum())
        if total <= 0.0:
            raise ValueError("target weights must sum to a positive value")
        schedule_rows.append(aligned / total)

    schedule_df = pd.DataFrame(schedule_rows, index=schedule_index, columns=columns).sort_index()
    weights_schedule = schedule_df.reindex(returns_index).ffill().bfill().fillna(0.0)

    adjusted_weights = pd.DataFrame(0.0, index=returns_index, columns=columns)
    aligned_returns = returns.reindex(columns=columns)
    for ts in returns_index:
        ret_row = aligned_returns.loc[ts]
        weights_row = weights_schedule.loc[ts].copy()
        valid_mask = ret_row.notna()
        if valid_mask.any():
            weights_row = weights_row.where(valid_mask, other=0.0)
            total = float(weights_row.sum())
            if total <= 0.0:
                valid_assets = ret_row.index[valid_mask]
                equal = pd.Series(1.0 / len(valid_assets), index=valid_assets, dtype=float)
                weights_row.loc[:] = 0.0
                weights_row.loc[equal.index] = equal
            else:
                weights_row = weights_row / total
        else:
            weights_row.loc[:] = 0.0
        adjusted_weights.loc[ts] = weights_row

    clean_returns = aligned_returns.fillna(0.0)
    portfolio_returns = clean_returns.mul(adjusted_weights, axis=1).sum(axis=1)

    nav_values: list[float] = []
    nav = float(initial_nav)
    for value in portfolio_returns:
        nav *= 1.0 + float(value)
        nav_values.append(nav)

    nav_series = pd.Series(nav_values, index=returns_index, dtype=float)
    return RebalancePortfolioResult(weights=adjusted_weights, nav=nav_series, returns=portfolio_returns)


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


def tracking_error(
    returns: pd.DataFrame | pd.Series,
    weights: pd.Series | None = None,
    *,
    freq: int = 52,
    target_periodic_return: float | None = None,
) -> tuple[float, float]:
    """Compute periodic and annualised tracking error for a rebalanced portfolio.

    Parameters
    ----------
    returns:
        Either a wide DataFrame of asset returns or a Series of portfolio
        returns. The returns must be periodic simple returns expressed as
        decimal fractions.
    weights:
        Target weights per asset when ``returns`` is a DataFrame. Missing
        weights default to zero and the vector is re-normalised to sum to one.
    freq:
        Number of compounding periods per year. Must be positive.
    target_periodic_return:
        Optional benchmark periodic return expressed as a decimal fraction. If
        omitted, the realised mean of the portfolio returns is used.

    Returns
    -------
    tuple[float, float]
        Periodic tracking error followed by its annualised counterpart. When
        insufficient observations are available the function returns
        ``(nan, nan)``.
    """

    if freq <= 0:
        raise ValueError("freq must be positive")

    if isinstance(returns, pd.DataFrame):
        if returns.empty:
            return float("nan"), float("nan")
        if weights is None:
            raise ValueError("weights are required when returns is a DataFrame")
        norm_weights = _normalise_weights(returns, weights)
        portfolio_returns = returns.fillna(0.0).mul(norm_weights, axis=1).sum(axis=1)
    else:
        portfolio_returns = returns.dropna()

    if portfolio_returns.empty:
        return float("nan"), float("nan")

    benchmark = float(target_periodic_return) if target_periodic_return is not None else float(
        portfolio_returns.mean()
    )
    active = portfolio_returns - benchmark
    active = active.dropna()
    if active.size < 2:
        return float("nan"), float("nan")

    periodic_te = float(active.std(ddof=1))
    annualised_te = periodic_te * math.sqrt(freq)
    return periodic_te, annualised_te


def apy_performance_summary(
    returns: pd.DataFrame,
    weights: pd.Series | None = None,
    *,
    freq: int = 52,
    initial_nav: float = 1.0,
    nav: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Summarise expected versus realised APY for a rebalanced portfolio.

    The function compares the static expectation from :func:`expected_apy`
    against the realised performance implied by a rebalance-aware NAV path.

    Parameters
    ----------
    returns:
        Wide DataFrame of periodic simple returns, indexed by timestamp.
    weights:
        Target weights per asset. When ``None`` an equal-weight portfolio is
        assumed.
    freq:
        Compounding periods per year used to annualise results.
    initial_nav:
        Starting NAV used for the aggregated portfolio path.
    nav:
        Optional externally computed NAV path. When provided it must align with
        ``returns.index``.

    Returns
    -------
    tuple[pandas.Series, pandas.Series]
        Tuple containing (metrics, nav_path). ``metrics`` is a Series with
        expected APY, realised APY, realised total return, active APY and
        tracking error figures. ``nav_path`` is the NAV trajectory used for the
        realised calculations.
    """

    if freq <= 0:
        raise ValueError("freq must be positive")

    if returns.empty:
        empty_nav = pd.Series(dtype=float)
        metrics = pd.Series(
            {
                "expected_apy": float("nan"),
                "realized_apy": float("nan"),
                "realized_total_return": float("nan"),
                "active_apy": float("nan"),
                "tracking_error_periodic": float("nan"),
                "tracking_error_annualized": float("nan"),
                "horizon_periods": 0.0,
                "horizon_years": float("nan"),
                "final_nav": float(initial_nav),
            },
            dtype=float,
        )
        return metrics, empty_nav

    norm_weights = _normalise_weights(returns, weights)
    clean_returns = returns.fillna(0.0)
    portfolio_returns = clean_returns.mul(norm_weights, axis=1).sum(axis=1)

    if nav is None:
        nav_path = performance.nav_series(clean_returns, norm_weights, initial=initial_nav)
    else:
        nav_path = nav.reindex(clean_returns.index)
        if nav_path.isna().any():
            raise ValueError("nav path contains NaN after aligning with returns index")

    if nav_path.empty:
        final_nav = float(initial_nav)
        realised_total = float("nan")
        realised_apy = float("nan")
        periods = 0
        horizon_years = float("nan")
    else:
        final_nav = float(nav_path.iloc[-1])
        periods = nav_path.shape[0]
        realised_total = final_nav / float(initial_nav) - 1.0
        realised_apy = (1.0 + realised_total) ** (freq / periods) - 1.0
        horizon_years = periods / freq

    expected = expected_apy(returns, norm_weights, freq=freq)
    expected_periodic = (1.0 + expected) ** (1.0 / freq) - 1.0
    te_periodic, te_annualised = tracking_error(
        portfolio_returns,
        freq=freq,
        target_periodic_return=expected_periodic,
    )

    metrics = pd.Series(
        {
            "expected_apy": expected,
            "realized_apy": realised_apy,
            "realized_total_return": realised_total,
            "active_apy": realised_apy - expected,
            "tracking_error_periodic": te_periodic,
            "tracking_error_annualized": te_annualised,
            "horizon_periods": float(periods),
            "horizon_years": horizon_years if periods else float("nan"),
            "final_nav": final_nav,
        },
        dtype=float,
    )

    return metrics, nav_path
