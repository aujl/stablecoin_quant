"""Portfolio return attribution using realised returns and weight schedules."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class AttributionResult:
    """Container holding attribution outputs for downstream reporting."""

    portfolio: dict[str, Any]
    by_pool: pd.DataFrame
    by_window: pd.DataFrame
    period_returns: pd.Series


def _ensure_datetime_index(index: pd.Index, *, label: str) -> pd.DatetimeIndex:
    """Return a datetime index, coercing the input when possible."""

    if isinstance(index, pd.DatetimeIndex):
        dt_index = index
    else:
        dt_index = pd.to_datetime(index, utc=True, errors="coerce")
    if dt_index.isna().any():  # type: ignore[truthy-function]
        raise TypeError(f"{label} index must be datetime-like")
    return pd.DatetimeIndex(dt_index).sort_values()


def _infer_periods_per_year(index: pd.DatetimeIndex) -> float:
    """Infer the periodicity of the return series expressed as periods per year."""

    if len(index) < 2:
        return 1.0
    diffs = index.to_series().diff().dropna().dt.total_seconds()
    mean_seconds = float(diffs.mean()) if not diffs.empty else 0.0
    if mean_seconds <= 0:
        return 1.0
    seconds_per_year = 365.25 * 24 * 3600
    return seconds_per_year / mean_seconds


def _prepare_weight_schedule(
    weight_schedule: pd.DataFrame | pd.Series,
    returns_index: pd.DatetimeIndex,
    columns: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Align the weight schedule to the return index and compute window labels."""

    if isinstance(weight_schedule, pd.Series):
        schedule = pd.DataFrame([weight_schedule], index=[returns_index[0]])
    else:
        schedule = weight_schedule.copy()
        if "timestamp" in schedule.columns and not isinstance(schedule.index, pd.DatetimeIndex):
            schedule = schedule.set_index("timestamp")

    if schedule.empty:
        raise ValueError("weight_schedule must contain at least one row")

    schedule.index = _ensure_datetime_index(schedule.index, label="weight_schedule")
    schedule = schedule.loc[~schedule.index.duplicated(keep="last")]
    schedule = schedule.sort_index()
    schedule = schedule.reindex(columns=columns).fillna(0.0)

    aligned = schedule.reindex(returns_index, method="ffill")
    if aligned.isna().any().any():
        raise ValueError("weight_schedule does not cover the full return history")

    window_labels = pd.Series(schedule.index, index=schedule.index)
    window_labels = window_labels.reindex(returns_index, method="ffill")
    if window_labels.isna().any():
        raise ValueError("weight_schedule does not cover the full return history")

    return aligned, window_labels


def compute_attribution(
    returns: pd.DataFrame,
    weight_schedule: pd.DataFrame | pd.Series | None,
    *,
    periods_per_year: float | None = None,
    initial_nav: float = 1.0,
) -> AttributionResult:
    """Compute realised performance attribution by pool and rebalance window.

    Parameters
    ----------
    returns:
        DataFrame of periodic simple returns expressed as decimal fractions. Rows
        represent timestamps and columns represent pools.
    weight_schedule:
        Target weights per pool. A wide DataFrame or Series indexed by
        rebalance timestamps. ``None`` falls back to equal weights across the
        available pools.
    periods_per_year:
        Annualisation factor used to convert realised total return into APY. If
        omitted the value is inferred from the timestamp spacing.
    initial_nav:
        Starting capital used to scale contributions.

    Notes
    -----
    For each period :math:`t` the capital change attributed to pool :math:`i`
    is ``ΔNAV_{i,t} = NAV_{t-1} · w_{i,t} · r_{i,t}``, where ``w`` are the
    schedule weights and ``r`` are realised simple returns. Pool level
    contributions normalise the sum of these capital changes by the initial
    capital ``NAV_0`` yielding an additive decomposition of the total realised
    simple return. Rebalance window contributions aggregate ``ΔNAV`` over each
    interval defined by the weight schedule. The realised APY is computed from
    the geometric growth factor ``G = NAV_T / NAV_0`` via ``APY = G^{P/T} - 1``
    where ``P`` denotes ``periods_per_year`` and ``T`` the number of observed
    periods. Contribution shares scale this APY by each component's share of the
    total simple return.
    """

    if returns.empty:
        empty = pd.DataFrame(
            columns=[
                "pool",
                "avg_weight",
                "nav_contribution",
                "return_contribution",
                "return_share",
                "apy_contribution",
            ]
        )
        windows = pd.DataFrame(
            columns=[
                "window_start",
                "window_end",
                "periods",
                "start_nav",
                "end_nav",
                "nav_change",
                "window_return",
                "return_contribution",
                "return_share",
                "apy_contribution",
                "window_apy",
            ]
        )
        portfolio = {
            "initial_nav": float(initial_nav),
            "final_nav": float(initial_nav),
            "total_return": 0.0,
            "realized_apy": 0.0,
            "periods": 0,
            "periods_per_year": periods_per_year or float("nan"),
        }
        return AttributionResult(
            portfolio=portfolio, by_pool=empty, by_window=windows, period_returns=pd.Series(dtype=float)
        )

    initial_value = float(initial_nav)
    returns = returns.copy()
    returns_index = _ensure_datetime_index(returns.index, label="returns")
    returns.index = returns_index
    returns = returns.sort_index()
    columns = list(returns.columns)
    if weight_schedule is None:
        if not columns:
            raise ValueError("returns must contain columns when weight_schedule is None")
        weight_schedule = pd.Series(1.0 / len(columns), index=columns)
    aligned_weights, window_labels = _prepare_weight_schedule(weight_schedule, returns_index, columns)

    weight_sums = aligned_weights.sum(axis=1)
    if (weight_sums <= 0).any():
        raise ValueError("weight_schedule rows must sum to a positive value")
    norm_weights = aligned_weights.div(weight_sums, axis=0).fillna(0.0)

    clean_returns = returns.fillna(0.0).astype(float)
    norm_weights = norm_weights.astype(float)

    nav = initial_value
    pool_nav_contrib = pd.Series(0.0, index=columns, dtype=float)
    weight_accum = pd.Series(0.0, index=columns, dtype=float)
    window_stats: dict[pd.Timestamp, dict[str, Any]] = {}
    period_returns = []

    for timestamp in clean_returns.index:
        nav_prev = nav
        weights_row = norm_weights.loc[timestamp].reindex(columns).fillna(0.0)
        returns_row = clean_returns.loc[timestamp].reindex(columns).fillna(0.0)

        period_ret = float((weights_row * returns_row).sum())
        period_returns.append(period_ret)
        delta_nav_by_pool = nav_prev * weights_row * returns_row
        delta_nav = float(delta_nav_by_pool.sum())
        nav = nav_prev + delta_nav

        pool_nav_contrib += delta_nav_by_pool
        weight_accum += weights_row

        window_key = pd.Timestamp(window_labels.loc[timestamp])
        stats = window_stats.get(window_key)
        if stats is None:
            stats = {
                "window_start": window_key,
                "window_end": timestamp,
                "start_nav": nav_prev,
                "end_nav": nav,
                "nav_change": 0.0,
                "periods": 0,
            }
            window_stats[window_key] = stats
        stats["window_end"] = timestamp
        stats["end_nav"] = nav
        stats["nav_change"] = float(stats["nav_change"]) + delta_nav
        stats["periods"] = int(stats["periods"]) + 1

    total_return = (nav / initial_value) - 1.0
    periods = len(clean_returns)
    if periods_per_year is None:
        periods_per_year = _infer_periods_per_year(returns_index)
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    horizon_years = periods / periods_per_year if periods_per_year else float("nan")
    growth = nav / float(initial_nav)
    realized_apy = (growth ** (periods_per_year / periods)) - 1.0 if periods > 0 else 0.0

    contrib_returns = pool_nav_contrib / initial_value
    close_to_zero = math.isclose(total_return, 0.0, abs_tol=1e-12)
    if close_to_zero:
        return_share = contrib_returns * 0.0
    else:
        return_share = contrib_returns / total_return
    apy_contrib = return_share * realized_apy
    avg_weight = weight_accum / periods

    by_pool = pd.DataFrame(
        {
            "pool": columns,
            "avg_weight": avg_weight.values,
            "nav_contribution": pool_nav_contrib.values,
            "return_contribution": contrib_returns.values,
            "return_share": return_share.values,
            "apy_contribution": apy_contrib.values,
        }
    )

    window_records: list[dict[str, Any]] = []
    for key in sorted(window_stats.keys()):
        stats = window_stats[key]
        start_nav = float(stats["start_nav"])
        nav_change = float(stats["nav_change"])
        end_nav = float(stats["end_nav"])
        periods_in_window = int(stats["periods"])
        window_return = nav_change / start_nav if start_nav else 0.0
        return_contribution = nav_change / initial_value
        if close_to_zero:
            window_share = 0.0
        else:
            window_share = return_contribution / total_return
        if periods_in_window > 0:
            window_apy = ((1.0 + window_return) ** (periods_per_year / periods_in_window)) - 1.0
        else:
            window_apy = float("nan")
        window_records.append(
            {
                "window_start": stats["window_start"],
                "window_end": stats["window_end"],
                "periods": periods_in_window,
                "start_nav": start_nav,
                "end_nav": end_nav,
                "nav_change": nav_change,
                "window_return": window_return,
                "return_contribution": return_contribution,
                "return_share": window_share,
                "apy_contribution": window_share * realized_apy,
                "window_apy": window_apy,
            }
        )

    by_window = pd.DataFrame(window_records)

    portfolio_summary = {
        "initial_nav": initial_value,
        "final_nav": nav,
        "total_return": total_return,
        "realized_apy": realized_apy,
        "periods": periods,
        "periods_per_year": periods_per_year,
        "horizon_years": horizon_years,
    }

    return AttributionResult(
        portfolio=portfolio_summary,
        by_pool=by_pool,
        by_window=by_window,
        period_returns=pd.Series(period_returns, index=returns_index, name="portfolio_return"),
    )


def load_weight_schedule(path: str | bytes | Any) -> pd.DataFrame:
    """Load a weight schedule from CSV in either wide or long format."""

    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError("weight schedule CSV must contain a 'timestamp' column")
    if {"name", "weight"}.issubset(df.columns):
        schedule = df.pivot(index="timestamp", columns="name", values="weight")
    else:
        schedule = df.set_index("timestamp")
    schedule.index = pd.to_datetime(schedule.index, utc=True)
    return schedule.sort_index()
