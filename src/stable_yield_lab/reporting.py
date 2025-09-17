from __future__ import annotations

from pathlib import Path
import math
from typing import Any

import pandas as pd

from . import Metrics, PoolRepository, attribution, performance

_RISK_COLUMNS = [
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "negative_period_share",
]


def _ensure_outdir(outdir: str | Path) -> Path:
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _risk_metrics(series: pd.Series) -> dict[str, float]:
    metrics = {col: float("nan") for col in _RISK_COLUMNS}
    if series is None or series.empty:
        return metrics

    clean = series.dropna()
    if clean.empty:
        return metrics

    mean_return = float(clean.mean())
    std_return = float(clean.std(ddof=1))
    if std_return > 0.0:
        metrics["sharpe_ratio"] = mean_return / std_return

    downside = clean[clean < 0.0]
    if not downside.empty:
        downside_std = float((downside.pow(2).mean()) ** 0.5)
        if downside_std > 0.0:
            metrics["sortino_ratio"] = mean_return / downside_std

    nav = (1.0 + clean).cumprod()
    if not nav.empty:
        drawdown = nav / nav.cummax() - 1.0
        metrics["max_drawdown"] = float(drawdown.min())

    metrics["negative_period_share"] = float(clean.lt(0.0).mean())
    return metrics


def _weighted_portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    if returns.empty or weights.empty:
        return pd.Series(dtype=float)

    aligned_weights = weights.reindex(returns.columns).fillna(0.0)
    aligned_weights = aligned_weights[aligned_weights > 0.0]
    if aligned_weights.empty:
        return pd.Series(dtype=float)

    subset = returns.loc[:, aligned_weights.index]
    if subset.empty:
        return pd.Series(dtype=float)

    weight_sum = (~subset.isna()).mul(aligned_weights, axis=1).sum(axis=1)
    weighted_sum = subset.mul(aligned_weights, axis=1).sum(axis=1, min_count=1)
    portfolio = weighted_sum / weight_sum
    return portfolio.dropna()


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna()
    if not mask.any():
        return float("nan")
    aligned_weights = weights.reindex(values.index).astype(float)
    masked = aligned_weights[mask]
    if masked.empty:
        return float("nan")
    total = float(masked.sum())
    if total <= 0.0:
        return float("nan")
    return float((values[mask] * masked).sum() / total)


def _realised_apy_statistics(
    returns: pd.DataFrame,
    *,
    lookback_days: int | None,
    min_observations: int,
) -> tuple[pd.Series, pd.Series, dict[str, str]]:
    if returns.empty:
        empty_index = returns.columns
        return (
            pd.Series(float("nan"), index=empty_index, dtype=float),
            pd.Series(0, index=empty_index, dtype=int),
            {},
        )

    window = returns.sort_index()
    if lookback_days is not None and not window.empty:
        cutoff = window.index.max() - pd.Timedelta(days=lookback_days)
        window = window.loc[window.index >= cutoff]

    realised: dict[str, float] = {}
    observations: dict[str, int] = {}
    warnings_map: dict[str, str] = {}

    for column in window.columns:
        series = window[column].dropna()
        count = int(series.shape[0])
        observations[column] = count
        if count < min_observations:
            warnings_map[column] = f"Only {count} observations available in lookback window"
            realised[column] = float("nan")
            continue

        growth = float((1.0 + series).prod())
        if growth <= 0.0:
            warnings_map[column] = "Non-positive cumulative growth in lookback window"
            realised[column] = float("nan")
            continue

        freq = performance._infer_periods_per_year(series.index) if count > 1 else performance._infer_periods_per_year(window.index)
        if not math.isfinite(freq) or freq <= 0.0:
            freq = 1.0
        realised[column] = growth ** (freq / count) - 1.0

    realised_series = pd.Series(realised, dtype=float).reindex(window.columns, fill_value=float("nan"))
    obs_series = pd.Series(observations, dtype=int).reindex(window.columns, fill_value=0)
    return realised_series, obs_series, warnings_map


def cross_section_report(
    repo: PoolRepository,
    outdir: str | Path,
    *,
    perf_fee_bps: float = 0.0,
    mgmt_fee_bps: float = 0.0,
    top_n: int = 20,
    returns: pd.DataFrame | None = None,
    weight_schedule: pd.DataFrame | pd.Series | None = None,
    attribution_periods_per_year: float | None = None,
    attribution_initial_nav: float = 1.0,
    attribution_console: bool = True,
    return_attribution: bool = False,
    realised_apy_lookback_days: int | None = 90,
    realised_apy_min_observations: int = 3,
) -> dict[str, Path] | tuple[dict[str, Path], attribution.AttributionResult | None]:
    """Generate file-first CSV outputs for the given snapshot repository.

    Parameters
    ----------
    repo:
        Snapshot repository describing pools at a point in time.
    outdir:
        Directory where CSV reports are written.
    perf_fee_bps, mgmt_fee_bps:
        Fee assumptions applied before writing ``net_apy`` values.
    top_n:
        Number of pools to include in ``topN.csv``.
    returns:
        Optional wide DataFrame of realised periodic returns (index timestamp,
        columns pool name). When provided, the ``concentration.csv`` output
        includes Sharpe ratio, Sortino ratio, maximum drawdown, and the share of
        negative periods for each pool and aggregate grouping.
    weight_schedule:
        Optional weight schedule passed to the attribution engine when
        ``return_attribution`` is ``True``.
    attribution_periods_per_year:
        Annualisation factor used for attribution reporting. When omitted the
        cadence is inferred from ``returns``.
    attribution_initial_nav:
        Starting NAV used for attribution capital accounting.
    attribution_console:
        Print attribution headline metrics to stdout when ``True``.
    return_attribution:
        Return the attribution object alongside ``paths`` when enabled.
    realised_apy_lookback_days:
        Rolling window (in days) used to compute realised APY statistics.
    realised_apy_min_observations:
        Minimum number of observations required before reporting realised APY.

    Returns
    -------
    dict[str, Path] | tuple[dict[str, Path], attribution.AttributionResult | None]
        Mapping of report label to the written CSV path. When
        ``return_attribution`` is ``True`` the attribution payload is returned as
        the second tuple element.

    Writes the following CSVs:
      - pools.csv: all pools with net_apy column
      - by_chain.csv: aggregated by chain with TVL-weighted APY
      - by_source.csv: aggregated by source (protocol)
      - by_stablecoin.csv: aggregated by stablecoin symbol
      - topN.csv: top-N pools by base_apy
      - concentration.csv: HHI metrics across chain and stablecoin augmented
        with realised risk statistics when ``returns`` are supplied
    """
    out = _ensure_outdir(outdir)
    paths: dict[str, Path] = {}

    df = repo.to_dataframe()
    df = Metrics.add_net_apy_column(df, perf_fee_bps=perf_fee_bps, mgmt_fee_bps=mgmt_fee_bps)
    df["realised_apy"] = float("nan")
    df["realised_apy_observations"] = 0
    df["realised_apy_warning"] = pd.Series(pd.NA, index=df.index, dtype="object")

    warnings_records: list[dict[str, Any]] = []
    returns_df: pd.DataFrame | None = None

    # Concentration metrics enriched with realised risk metrics
    hhi_total = Metrics.hhi(df, value_col="tvl_usd")
    hhi_chain = Metrics.hhi(df, value_col="tvl_usd", group_col="chain")
    hhi_stable = Metrics.hhi(df, value_col="tvl_usd", group_col="stablecoin")

    metrics_map: dict[str, dict[str, float]] = {}
    if returns is not None and not returns.empty and not df.empty:
        returns_df = returns.copy()
        returns_df = returns_df.sort_index()
        returns_df = returns_df.apply(pd.to_numeric, errors="coerce")

        metadata = df.set_index("name")
        returns_df = returns_df.reindex(columns=metadata.index)

        realised_series, obs_series, warn_map = _realised_apy_statistics(
            returns_df,
            lookback_days=realised_apy_lookback_days,
            min_observations=realised_apy_min_observations,
        )
        df.loc[:, "realised_apy"] = df["name"].map(realised_series)
        df.loc[:, "realised_apy_observations"] = (
            df["name"].map(obs_series).fillna(0).astype(int)
        )
        df.loc[:, "realised_apy_warning"] = df["name"].map(warn_map).astype("object")
        warnings_records = [
            {"pool": pool, "message": message}
            for pool, message in warn_map.items()
            if message
        ]

        weights = metadata.get("tvl_usd", pd.Series(dtype=float)).astype(float)

        total_series = _weighted_portfolio_returns(returns_df, weights)
        metrics_map["total"] = _risk_metrics(total_series)

        for chain, names in metadata.groupby("chain").groups.items():
            chain_weights = weights.loc[list(names)]
            series = _weighted_portfolio_returns(returns_df, chain_weights)
            metrics_map[f"chain:{chain}"] = _risk_metrics(series)

        for stable, names in metadata.groupby("stablecoin").groups.items():
            stable_weights = weights.loc[list(names)]
            series = _weighted_portfolio_returns(returns_df, stable_weights)
            metrics_map[f"stablecoin:{stable}"] = _risk_metrics(series)

        for pool_name in returns_df.columns:
            metrics_map[f"pool:{pool_name}"] = _risk_metrics(returns_df[pool_name])

    def _metrics_for(scope: str) -> dict[str, float]:
        values = metrics_map.get(scope)
        if not values:
            return {col: float("nan") for col in _RISK_COLUMNS}
        return {col: values.get(col, float("nan")) for col in _RISK_COLUMNS}

    conc_records: list[dict[str, float]] = []

    total_hhi = float(hhi_total["hhi"].iloc[0]) if not hhi_total.empty else float("nan")
    record = {"scope": "total", "hhi": total_hhi}
    record.update(_metrics_for("total"))
    conc_records.append(record)

    if not hhi_chain.empty:
        for _, row in hhi_chain.iterrows():
            scope = f"chain:{row['chain']}"
            record = {"scope": scope, "hhi": float(row["hhi"])}
            record.update(_metrics_for(scope))
            conc_records.append(record)

    if not hhi_stable.empty:
        for _, row in hhi_stable.iterrows():
            scope = f"stablecoin:{row['stablecoin']}"
            record = {"scope": scope, "hhi": float(row["hhi"])}
            record.update(_metrics_for(scope))
            conc_records.append(record)

    existing_scopes = {rec["scope"] for rec in conc_records}
    for scope, values in metrics_map.items():
        if scope in existing_scopes:
            continue
        record = {"scope": scope, "hhi": float("nan")}
        record.update({col: values.get(col, float("nan")) for col in _RISK_COLUMNS})
        conc_records.append(record)

    conc_all = pd.DataFrame(conc_records)
    paths["concentration"] = out / "concentration.csv"
    conc_all.to_csv(paths["concentration"], index=False)

    # Aggregations and realised APY summaries
    by_chain = Metrics.groupby_chain(repo)
    if not by_chain.empty and not df.empty and "chain" in df.columns:
        chain_realised = (
            df.groupby("chain")[["realised_apy", "tvl_usd"]]
            .apply(lambda g: _weighted_average(g["realised_apy"], g["tvl_usd"]))
            .rename("realised_apy_wavg")
        )
        by_chain = by_chain.merge(chain_realised, on="chain", how="left")
    else:
        by_chain["realised_apy_wavg"] = float("nan")
    paths["by_chain"] = out / "by_chain.csv"
    by_chain.to_csv(paths["by_chain"], index=False)

    def _agg(source_df: pd.DataFrame, key: str) -> pd.DataFrame:
        if source_df.empty:
            return source_df
        g = source_df.groupby(key).agg(
            pools=("name", "count"),
            tvl=("tvl_usd", "sum"),
            apr_avg=("base_apy", "mean"),
            apr_wavg=(
                "base_apy",
                lambda x: (x * source_df.loc[x.index, "tvl_usd"]).sum() / source_df.loc[x.index, "tvl_usd"].sum(),
            ),
        )
        return g.reset_index()

    by_source = _agg(df, "source")
    by_stable = _agg(df, "stablecoin")
    paths["by_source"] = out / "by_source.csv"
    paths["by_stablecoin"] = out / "by_stablecoin.csv"
    by_source.to_csv(paths["by_source"], index=False)
    by_stable.to_csv(paths["by_stablecoin"], index=False)

    top = df.sort_values("base_apy", ascending=False).head(top_n)
    paths["topN"] = out / "topN.csv"
    top.to_csv(paths["topN"], index=False)

    warnings_df = pd.DataFrame(warnings_records, columns=["pool", "message"])
    paths["warnings"] = out / "warnings.csv"
    warnings_df.to_csv(paths["warnings"], index=False)

    paths["pools"] = out / "pools.csv"
    df.to_csv(paths["pools"], index=False)

    attribution_result: attribution.AttributionResult | None = None
    if return_attribution:
        if returns_df is not None and not returns_df.empty:
            schedule = weight_schedule
            if schedule is not None:
                if isinstance(schedule, pd.Series):
                    schedule = schedule.reindex(returns_df.columns)
                else:
                    schedule = schedule.reindex(columns=returns_df.columns)
            periods_per_year = (
                attribution_periods_per_year
                if attribution_periods_per_year is not None
                else performance._infer_periods_per_year(returns_df.index)
            )
            attribution_result = attribution.compute_attribution(
                returns_df,
                schedule,
                periods_per_year=periods_per_year,
                initial_nav=attribution_initial_nav,
            )
            paths["attribution_by_pool"] = out / "attribution_by_pool.csv"
            paths["attribution_by_window"] = out / "attribution_by_window.csv"
            attribution_result.by_pool.to_csv(paths["attribution_by_pool"], index=False)
            attribution_result.by_window.to_csv(paths["attribution_by_window"], index=False)
            if attribution_console:
                print(
                    "Realized APY:",
                    attribution_result.portfolio.get("realized_apy"),
                )
        else:
            attribution_result = None

    return (paths, attribution_result) if return_attribution else paths
