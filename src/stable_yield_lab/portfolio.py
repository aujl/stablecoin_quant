from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .risk_metrics import _require_riskfolio


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


@dataclass(frozen=True)
class RebalanceResult:
    """Container for the outputs of :func:`rebalance_portfolio`.

    Attributes
    ----------
    weights:
        DataFrame with the same index as the input returns and columns for every
        asset encountered across the simulation. Each row represents the
        portfolio weights applied at the beginning of that period after
        rebalancing and normalising for asset availability.
    portfolio_returns:
        Period-by-period portfolio returns obtained by compounding the asset
        level returns with the simulated weights.
    nav:
        Net Asset Value path assuming an initial NAV of 1.0 by default.
    """

    weights: pd.DataFrame
    portfolio_returns: pd.Series
    nav: pd.Series


def _ensure_datetime_index(index: Sequence[pd.Timestamp | str] | pd.Index) -> pd.DatetimeIndex:
    if isinstance(index, pd.DatetimeIndex):
        return index.tz_convert("UTC") if index.tz is not None else index.tz_localize("UTC")
    converted = pd.to_datetime(list(index), utc=True)
    return pd.DatetimeIndex(converted)


def _to_utc_timestamp(value: pd.Timestamp | str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _as_weight_series(weights: Mapping[str, float] | pd.Series) -> pd.Series:
    if isinstance(weights, pd.Series):
        series = weights.astype(float).copy()
    elif isinstance(weights, Mapping):
        series = pd.Series(dict(weights), dtype=float)
    else:
        raise TypeError("Weights must be provided as a mapping or pandas Series.")
    series = series.reindex(sorted(series.index)).fillna(0.0)
    return series


def _normalise_row(row: pd.Series) -> pd.Series:
    row = row.fillna(0.0)
    total = row.sum()
    if pd.isna(total) or total <= 0:
        return row * 0.0
    return row / total


def _normalise_frame_rows(frame: pd.DataFrame) -> pd.DataFrame:
    normalised = frame.copy()
    for idx, row in normalised.iterrows():
        normalised.loc[idx] = _normalise_row(row)
    return normalised.fillna(0.0)


def _resolve_target_frame(
    schedule: pd.DatetimeIndex,
    target_weights: (
        pd.DataFrame
        | Mapping[Any, Mapping[str, float] | pd.Series]
        | Callable[[pd.Timestamp], Mapping[str, float] | pd.Series]
    ),
) -> pd.DataFrame:
    if isinstance(target_weights, pd.DataFrame):
        frame = target_weights.copy()
        frame.index = _ensure_datetime_index(frame.index)
        missing = schedule.difference(frame.index)
        if not missing.empty:
            raise KeyError(f"Missing target weights for rebalance dates: {missing.tolist()}")
        frame = frame.reindex(schedule)
        frame = frame.fillna(0.0)
        return _normalise_frame_rows(frame)

    rows: list[pd.Series] = []
    columns: set[str] = set()

    if isinstance(target_weights, Mapping):
        keyed: dict[pd.Timestamp, pd.Series] = {}
        for key, value in target_weights.items():
            ts = _to_utc_timestamp(key)
            keyed[ts] = _as_weight_series(value)
            columns.update(keyed[ts].index.tolist())
        missing = schedule.difference(pd.DatetimeIndex(keyed.keys()))
        if not missing.empty:
            raise KeyError(f"Missing target weights for rebalance dates: {missing.tolist()}")
        for ts in schedule:
            row = keyed[ts].reindex(sorted(columns)).fillna(0.0)
            rows.append(row)
    elif callable(target_weights):
        for ts in schedule:
            row = _as_weight_series(target_weights(ts))
            columns.update(row.index.tolist())
            rows.append(row)
        rows = [row.reindex(sorted(columns)).fillna(0.0) for row in rows]
    else:
        raise TypeError(
            "target_weights must be a DataFrame, mapping keyed by rebalance date, or callable returning weights."
        )

    if not rows:
        raise ValueError("No target weights produced for the provided schedule.")

    frame = pd.DataFrame(rows, index=schedule)
    frame = frame.fillna(0.0)
    return _normalise_frame_rows(frame)


def rebalance_portfolio(
    returns: pd.DataFrame,
    *,
    rebalance_schedule: Sequence[pd.Timestamp | str] | pd.Index,
    target_weights: (
        pd.DataFrame
        | Mapping[Any, Mapping[str, float] | pd.Series]
        | Callable[[pd.Timestamp], Mapping[str, float] | pd.Series]
    ),
    initial_weights: Mapping[str, float] | pd.Series | None = None,
    initial_nav: float = 1.0,
) -> RebalanceResult:
    """Simulate a rebalancing strategy over a returns panel.

    Parameters
    ----------
    returns:
        Asset return panel (index = timestamps, columns = assets). Values are
        simple periodic returns (e.g. weekly growth rates). Missing values denote
        an asset being unavailable that period.
    rebalance_schedule:
        Iterable of timestamps where the portfolio is rebalanced to target
        weights. The timestamps must exist in ``returns.index`` once normalised
        to UTC.
    target_weights:
        Target weights provided either as a DataFrame keyed by rebalance date, a
        mapping from rebalance date to weights, or a callable returning weights
        for each rebalance date. The callable is invoked with a
        :class:`pandas.Timestamp` and must return a mapping or Series of weights.
    initial_weights:
        Optional starting weights used prior to the first rebalance event. If
        omitted the first rebalance defines the initial allocation. When
        provided they are normalised to sum to one across available assets.
    initial_nav:
        Starting net asset value used to compound portfolio returns.
    """

    if returns.empty:
        raise ValueError("returns DataFrame must not be empty.")

    returns = returns.copy()
    returns.index = _ensure_datetime_index(returns.index)
    returns = returns.sort_index()

    schedule = _ensure_datetime_index(rebalance_schedule)
    schedule = schedule.sort_values().unique()
    if schedule.size == 0:
        raise ValueError("rebalance_schedule must contain at least one timestamp.")
    missing_dates = schedule.difference(returns.index)
    if not missing_dates.empty:
        raise ValueError(
            "Rebalance dates not present in returns index: "
            f"{[ts.isoformat() for ts in missing_dates]}"
        )

    target_frame = _resolve_target_frame(schedule, target_weights)

    all_assets = returns.columns.union(target_frame.columns)
    returns = returns.reindex(columns=all_assets)
    target_frame = target_frame.reindex(columns=all_assets, fill_value=0.0)

    if initial_weights is not None:
        current_weights = _normalise_row(
            _as_weight_series(initial_weights).reindex(all_assets, fill_value=0.0)
        )
    else:
        current_weights = pd.Series(0.0, index=all_assets)

    nav_values: list[float] = []
    portfolio_returns: list[float] = []
    weight_records: list[pd.Series] = []
    current_nav = float(initial_nav)

    for ts, period_returns in returns.sort_index().iterrows():
        if ts in target_frame.index:
            current_weights = target_frame.loc[ts].reindex(all_assets, fill_value=0.0)

        period_returns = period_returns.fillna(pd.NA)
        available_mask = period_returns.notna()
        adjusted_weights = current_weights.copy()
        adjusted_weights.loc[~available_mask] = 0.0

        if available_mask.any():
            active_assets = all_assets[available_mask]
            total_active = adjusted_weights.loc[active_assets].sum()
            if total_active <= 0:
                adjusted_weights.loc[active_assets] = 1.0 / len(active_assets)
            else:
                adjusted_weights.loc[active_assets] /= total_active
        else:
            adjusted_weights[:] = 0.0

        weight_records.append(adjusted_weights.copy())

        if available_mask.any():
            clean_returns = period_returns.fillna(0.0)
            prev_nav = current_nav
            holdings_before = adjusted_weights * prev_nav
            holdings_after = holdings_before * (1.0 + clean_returns)
            current_nav = float(holdings_after.sum())
            portfolio_return = 0.0 if prev_nav == 0 else current_nav / prev_nav - 1.0
            if current_nav > 0:
                current_weights = holdings_after.reindex(all_assets, fill_value=0.0) / current_nav
            else:
                current_weights = pd.Series(0.0, index=all_assets)
        else:
            portfolio_return = 0.0

        portfolio_returns.append(float(portfolio_return))
        nav_values.append(float(current_nav))

    weights = pd.DataFrame(weight_records, index=returns.index, columns=all_assets).fillna(0.0)
    portfolio_return_series = pd.Series(portfolio_returns, index=returns.index)
    nav_series = pd.Series(nav_values, index=returns.index)

    return RebalanceResult(weights=weights, portfolio_returns=portfolio_return_series, nav=nav_series)


def schedule_from_optimizations(
    optimisation_results: Mapping[Any, Mapping[str, float] | pd.Series],
    *,
    returns_index: pd.Index | None = None,
) -> pd.DataFrame:
    """Convert optimisation outputs into a rebalance schedule DataFrame."""

    entries = [(_to_utc_timestamp(key), value) for key, value in optimisation_results.items()]
    if not entries:
        raise ValueError("optimisation_results must not be empty.")

    entries.sort(key=lambda item: item[0])

    rows = []
    columns: set[str] = set()
    timestamps = []
    for ts, value in entries:
        weights = _as_weight_series(value)
        columns.update(weights.index.tolist())
        rows.append(weights)
        timestamps.append(ts)

    timestamp_index = pd.DatetimeIndex(timestamps)
    frame = pd.DataFrame(rows, index=timestamp_index).reindex(columns=sorted(columns), fill_value=0.0)
    frame = _normalise_frame_rows(frame)

    if returns_index is not None:
        returns_idx = _ensure_datetime_index(returns_index)
        missing = frame.index.difference(returns_idx)
        if not missing.empty:
            raise ValueError(
                "Optimisation schedule contains dates outside the returns index: "
                f"{[ts.isoformat() for ts in missing]}"
            )

    return frame


def schedule_from_user_weights(
    *,
    dates: Sequence[pd.Timestamp | str],
    weights: (
        Mapping[str, float] | pd.Series | Sequence[Mapping[str, float] | pd.Series]
    ),
    returns_index: pd.Index | None = None,
) -> pd.DataFrame:
    """Construct a rebalance schedule from user-provided weights."""

    if not dates:
        raise ValueError("dates must not be empty.")

    timestamp_index = _ensure_datetime_index(dates)
    timestamp_index = timestamp_index.sort_values()

    if isinstance(weights, (Mapping, pd.Series)):
        weight_series = [_as_weight_series(weights) for _ in timestamp_index]
    else:
        weight_list = list(weights)
        if len(weight_list) != len(timestamp_index):
            raise ValueError("Length of weights sequence must match number of dates.")
        weight_series = [_as_weight_series(w) for w in weight_list]

    columns: set[str] = set()
    for series in weight_series:
        columns.update(series.index.tolist())

    frame = pd.DataFrame(weight_series, index=timestamp_index).reindex(columns=sorted(columns), fill_value=0.0)
    frame = _normalise_frame_rows(frame)

    if returns_index is not None:
        returns_idx = _ensure_datetime_index(returns_index)
        missing = frame.index.difference(returns_idx)
        if not missing.empty:
            raise ValueError(
                "Manual schedule contains dates outside the returns index: "
                f"{[ts.isoformat() for ts in missing]}"
            )

    return frame
