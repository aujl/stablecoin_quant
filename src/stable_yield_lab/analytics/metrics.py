from __future__ import annotations

from collections.abc import Iterable, Sequence
import math
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ..core import Pool, PoolRepository


def _coerce_float(value: object) -> float:
    """Best-effort conversion to ``float`` returning ``nan`` on failure."""

    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def weighted_mean(values: Sequence[object], weights: Sequence[object]) -> float:
    """Compute a weighted mean while skipping ``NaN`` pairs and zero weight sums."""

    vals = list(values)
    wts = list(weights)
    if not vals or not wts or len(vals) != len(wts):
        return float("nan")

    contributions: list[float] = []
    cleaned_weights: list[float] = []
    for raw_value, raw_weight in zip(vals, wts):
        value = _coerce_float(raw_value)
        weight = _coerce_float(raw_weight)
        if math.isnan(value) or math.isnan(weight):
            continue
        contributions.append(value * weight)
        cleaned_weights.append(weight)

    if not contributions:
        return float("nan")

    weight_sum = math.fsum(cleaned_weights)
    if not math.isfinite(weight_sum) or weight_sum == 0.0:
        return float("nan")

    numerator = math.fsum(contributions)
    if not math.isfinite(numerator):
        return float("nan")

    return numerator / weight_sum


def net_apy(
    base_apy: float,
    reward_apy: float = 0.0,
    *,
    perf_fee_bps: float = 0.0,
    mgmt_fee_bps: float = 0.0,
) -> float:
    """Compute net APY after applying performance and management fees."""

    base = _coerce_float(base_apy)
    reward = _coerce_float(reward_apy)
    perf = _coerce_float(perf_fee_bps)
    mgmt = _coerce_float(mgmt_fee_bps)

    if not all(math.isfinite(component) for component in (base, reward, perf, mgmt)):
        return float("nan")

    gross = base + reward
    fee_fraction = (perf + mgmt) / 10_000.0
    if not math.isfinite(fee_fraction):
        return float("nan")

    net_growth = (1.0 + gross) * (1.0 - fee_fraction)
    if not math.isfinite(net_growth):
        return float("nan")

    return max(net_growth - 1.0, -1.0)


def add_net_apy_column(
    df: pd.DataFrame,
    *,
    perf_fee_bps: float = 0.0,
    mgmt_fee_bps: float = 0.0,
    out_col: str = "net_apy",
) -> pd.DataFrame:
    """Append a net APY column computed via :func:`net_apy` for each row."""

    if df.empty:
        return df.copy()

    out = df.copy()
    out[out_col] = [
        net_apy(
            row.get("base_apy", 0.0),
            row.get("reward_apy", 0.0),
            perf_fee_bps=perf_fee_bps,
            mgmt_fee_bps=mgmt_fee_bps,
        )
        for _, row in out.iterrows()
    ]
    return out


def hhi(df: pd.DataFrame, value_col: str, group_col: str | None = None) -> pd.DataFrame:
    """Compute the Herfindahl–Hirschman Index of concentration."""

    if df.empty:
        if group_col is None:
            return pd.DataFrame({"hhi": pd.Series(dtype=float)})
        return pd.DataFrame(columns=[group_col, "hhi"], dtype=float)

    values = pd.to_numeric(df[value_col], errors="coerce")

    if group_col is None:
        valid = values.dropna()
        total = float(valid.sum())
        if total <= 0.0:
            return pd.DataFrame({"hhi": [float("nan")]})
        shares = (valid / total) ** 2
        return pd.DataFrame({"hhi": [float(shares.sum())]})

    data = pd.DataFrame({group_col: df[group_col], value_col: values})

    def _group_hhi(series: pd.Series) -> float:
        valid = pd.to_numeric(series, errors="coerce").dropna()
        total = float(valid.sum())
        if total <= 0.0:
            return float("nan")
        shares = (valid / total) ** 2
        return float(shares.sum())

    result = data.groupby(group_col)[value_col].apply(_group_hhi).reset_index(name="hhi")
    return result


class Metrics:
    """Namespace exposing common analytics helpers for backwards compatibility."""

    @staticmethod
    def weighted_mean(values: Sequence[object], weights: Sequence[object]) -> float:
        return weighted_mean(values, weights)

    @staticmethod
    def portfolio_apr(pools: Iterable[Pool], weights: Sequence[object] | None = None) -> float:
        arr = list(pools)
        if not arr:
            return float("nan")
        vals = [p.base_apy for p in arr]
        if weights is None:
            weights = [p.tvl_usd for p in arr]
        return weighted_mean(vals, list(weights))

    @staticmethod
    def groupby_chain(repo: PoolRepository) -> pd.DataFrame:
        df = repo.to_dataframe()
        if df.empty:
            return df
        g = (
            df.groupby("chain")
            .agg(
                pools=("name", "count"),
                tvl=("tvl_usd", "sum"),
                apr_avg=("base_apy", "mean"),
                apr_wavg=(
                    "base_apy",
                    lambda x: weighted_mean(
                        x.tolist(),
                        df.loc[x.index, "tvl_usd"].tolist(),
                    ),
                ),
            )
            .reset_index()
        )
        return g

    @staticmethod
    def top_n(repo: PoolRepository, n: int = 10, key: str = "base_apy") -> pd.DataFrame:
        df = repo.to_dataframe()
        if df.empty:
            return df
        return df.sort_values(key, ascending=False).head(n)

    @staticmethod
    def net_apy(
        base_apy: float,
        reward_apy: float = 0.0,
        *,
        perf_fee_bps: float = 0.0,
        mgmt_fee_bps: float = 0.0,
    ) -> float:
        return net_apy(
            base_apy,
            reward_apy,
            perf_fee_bps=perf_fee_bps,
            mgmt_fee_bps=mgmt_fee_bps,
        )

    @staticmethod
    def add_net_apy_column(
        df: pd.DataFrame,
        *,
        perf_fee_bps: float = 0.0,
        mgmt_fee_bps: float = 0.0,
        out_col: str = "net_apy",
    ) -> pd.DataFrame:
        return add_net_apy_column(
            df,
            perf_fee_bps=perf_fee_bps,
            mgmt_fee_bps=mgmt_fee_bps,
            out_col=out_col,
        )

    @staticmethod
    def hhi(df: pd.DataFrame, value_col: str, group_col: str | None = None) -> pd.DataFrame:
        return hhi(df, value_col=value_col, group_col=group_col)


__all__ = [
    "Metrics",
    "add_net_apy_column",
    "hhi",
    "net_apy",
    "weighted_mean",
]
