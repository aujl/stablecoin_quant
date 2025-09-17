"""Rebalancing utilities for portfolio weight schedules and turnover metrics.

The functions in this module generate deterministic outputs that can feed into
visualisation components.  The default engine applies a simple momentum style
overlay to derive time-varying target weights, computes the resulting trading
turnover and estimates trading fees given a basis-point cost assumption.  The
logic is intentionally light-weight to keep the demo fast while providing
realistic shaped data for charts.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RebalanceResult:
    """Container for the outputs of a rebalancing run.

    Attributes
    ----------
    target_weights:
        Post-trade portfolio weights (rows are timestamps, columns are assets).
    pre_trade_weights:
        Portfolio weights immediately before rebalancing, after market drift.
    turnover:
        Fraction of portfolio traded on each rebalance date (decimal form).
    fees:
        Trading fees paid each period expressed as fraction of portfolio NAV.
    """

    target_weights: pd.DataFrame
    pre_trade_weights: pd.DataFrame
    turnover: pd.Series
    fees: pd.Series


def _normalise_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure each row of ``df`` sums to one while handling zero rows.

    Rows that sum to zero (or contain only NaNs) are replaced with equal
    weights across all available columns.  Empty frames are returned unchanged.
    """

    if df.empty:
        return df

    cols = df.columns
    if len(cols) == 0:
        return df

    normalised = df.copy()
    equal = pd.Series(1.0 / len(cols), index=cols)
    for idx, row in normalised.iterrows():
        if not row.notna().any():
            normalised.loc[idx] = equal
            continue
        total = float(row.sum(skipna=True))
        if total == 0.0:
            normalised.loc[idx] = equal
        else:
            normalised.loc[idx] = row / total
    return normalised.fillna(0.0)


def _prepare_targets(
    returns: pd.DataFrame,
    target_weights: pd.DataFrame | pd.Series | None,
) -> pd.DataFrame:
    """Derive a target weight schedule aligned with ``returns``.

    When ``target_weights`` is ``None`` a rolling momentum proxy is used to
    produce time-varying allocations.  A provided Series is broadcast to all
    periods, while a DataFrame is forward-filled to cover every timestamp.
    """

    if returns.empty:
        return returns.copy()

    cols = returns.columns
    index = returns.index

    if target_weights is None:
        window = min(4, len(returns))
        momentum = returns.rolling(window=window, min_periods=1).mean()
        positive = momentum.clip(lower=0.0)
        weights = positive.div(positive.sum(axis=1).replace(0.0, pd.NA), axis=0)
        weights = weights.reindex(index=index, columns=cols)
    elif isinstance(target_weights, pd.Series):
        base = target_weights.reindex(cols).fillna(0.0)
        weights = pd.DataFrame([base] * len(index), index=index, columns=cols)
    else:
        weights = target_weights.reindex(columns=cols)
        if not weights.index.equals(index):
            weights = weights.reindex(index=index, method="ffill")
        weights = weights.fillna(method="ffill")

    weights = weights.reindex(index=index, columns=cols).fillna(0.0)
    return _normalise_rows(weights)


def run_rebalance(
    returns: pd.DataFrame,
    *,
    target_weights: pd.DataFrame | pd.Series | None = None,
    trading_cost_bps: float = 5.0,
) -> RebalanceResult:
    """Simulate periodic rebalancing and compute turnover/fees.

    Parameters
    ----------
    returns:
        Wide DataFrame of periodic simple returns. Rows are timestamps and
        columns are asset identifiers.
    target_weights:
        Optional schedule of desired weights. When ``None`` a simple momentum
        overlay is used to derive targets endogenously.
    trading_cost_bps:
        Assumed round-trip trading cost in basis points applied to the traded
        notionals each period.
    """

    if returns.empty or returns.shape[1] == 0:
        empty_df = returns.copy()
        empty_series = pd.Series(index=returns.index, dtype=float)
        return RebalanceResult(empty_df, empty_df, empty_series, empty_series)

    clean_returns = returns.fillna(0.0)
    targets = _prepare_targets(clean_returns, target_weights)

    index = clean_returns.index
    cols = clean_returns.columns

    pre_trade = pd.DataFrame(0.0, index=index, columns=cols)
    post_trade = pd.DataFrame(0.0, index=index, columns=cols)
    turnover = pd.Series(0.0, index=index, dtype=float)
    fees = pd.Series(0.0, index=index, dtype=float)

    post = targets.iloc[0].copy()
    if float(post.sum()) == 0.0:
        post[:] = 1.0 / len(cols)
    post_trade.iloc[0] = post
    pre_trade.iloc[0] = post

    turnover.iloc[0] = 0.0
    fees.iloc[0] = 0.0

    for i in range(1, len(clean_returns)):
        ret = clean_returns.iloc[i]
        drifted = post * (1.0 + ret)
        total = float(drifted.sum())
        if total > 0.0:
            drifted = drifted / total
        else:
            drifted = post.copy()
        pre_trade.iloc[i] = drifted

        target_row = targets.iloc[i].copy()
        target_total = float(target_row.sum())
        if target_total == 0.0:
            target_row = post.copy()
        else:
            target_row = target_row / target_total
        post_trade.iloc[i] = target_row

        trade_amount = float((target_row - drifted).abs().sum()) * 0.5
        turnover.iloc[i] = trade_amount
        fees.iloc[i] = trade_amount * (trading_cost_bps / 10_000.0)

        post = target_row

    return RebalanceResult(post_trade, pre_trade, turnover, fees)


__all__ = ["RebalanceResult", "run_rebalance"]
