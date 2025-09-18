from __future__ import annotations

from pathlib import Path

import pandas as pd

from stable_yield_lab.core import PoolRepository

from ..analytics.metrics import Metrics

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


def _infer_periods_per_year(index: pd.DatetimeIndex) -> float | None:
    if index.size < 2:
        return None

    ordered = index.sort_values()
    diffs = ordered.to_series().diff().dropna()
    if diffs.empty:
        return None

    avg_days = diffs.dt.total_seconds().mean() / 86_400.0
    if avg_days and avg_days > 0.0:
        return 365.25 / avg_days
    return None


def _annualized_return(series: pd.Series) -> float | None:
    if series.empty:
        return None
    index = series.index
    if not isinstance(index, pd.DatetimeIndex):
        return None
    periods_per_year = _infer_periods_per_year(index)
    if periods_per_year is None:
        return None

    growth = float((1.0 + series).prod())
    periods = series.shape[0]
    if growth <= 0.0 or periods <= 0:
        return None

    return float(growth ** (periods_per_year / periods) - 1.0)


def _tvl_weighted_average(values: pd.Series, weights: pd.Series) -> float:
    mask = (~values.isna()) & (~weights.isna()) & (weights > 0.0)
    if not mask.any():
        return float("nan")

    vals = values.loc[mask].astype(float).tolist()
    wts = weights.loc[mask].astype(float).tolist()
    return Metrics.weighted_mean(vals, wts)


def cross_section_report(
    repo: PoolRepository,
    outdir: str | Path,
    *,
    perf_fee_bps: float = 0.0,
    mgmt_fee_bps: float = 0.0,
    top_n: int = 20,
    returns: pd.DataFrame | None = None,
    realised_apy_lookback_days: int | None = 365,
    realised_apy_min_observations: int = 5,
) -> dict[str, Path]:
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
    realised_apy_lookback_days:
        Number of trailing days considered when computing realised APY. Set to
        ``None`` to use the full history available.
    realised_apy_min_observations:
        Minimum number of non-null observations required before a realised APY
        is calculated. Pools with fewer observations emit a warning.

    Returns
    -------
    dict[str, Path]
        Mapping of report label to the written CSV path.

    Writes the following CSVs:
      - pools.csv: all pools with net_apy column
      - by_chain.csv: aggregated by chain with TVL-weighted APY
      - by_source.csv: aggregated by source (protocol)
      - by_stablecoin.csv: aggregated by stablecoin symbol
      - topN.csv: top-N pools by base_apy
      - concentration.csv: HHI metrics across chain and stablecoin augmented
        with realised risk statistics when ``returns`` are supplied
      - warnings.csv: per-pool realised APY warnings and observation counts
    """
    out = _ensure_outdir(outdir)
    paths: dict[str, Path] = {}

    df = repo.to_dataframe()
    df = Metrics.add_net_apy_column(df, perf_fee_bps=perf_fee_bps, mgmt_fee_bps=mgmt_fee_bps)

    metrics_index = pd.Index(df["name"], name="name") if not df.empty else pd.Index([], name="name")
    realised_metrics = pd.DataFrame(index=metrics_index)
    realised_metrics["realised_apy"] = pd.Series(float("nan"), index=metrics_index, dtype=float)
    realised_metrics["realised_apy_observations"] = pd.Series(
        pd.NA, index=metrics_index, dtype="Int64"
    )
    realised_metrics["realised_apy_warning"] = pd.Series(pd.NA, index=metrics_index, dtype="string")
    warnings_records: list[dict[str, object]] = []

    metrics_map: dict[str, dict[str, float]] = {}
    if returns is not None and not returns.empty and not df.empty:
        returns_df = returns.copy()
        if isinstance(returns_df, pd.Series):
            returns_df = returns_df.to_frame()
        returns_df.index = pd.to_datetime(returns_df.index, utc=True, errors="coerce")
        returns_df = returns_df.loc[~returns_df.index.isna()]
        returns_df = returns_df.sort_index()
        returns_df = returns_df.apply(pd.to_numeric, errors="coerce")
        returns_df = returns_df.reindex(columns=metrics_index)
        returns_df = returns_df.dropna(how="all")

        if (
            realised_apy_lookback_days is not None
            and realised_apy_lookback_days > 0
            and not returns_df.empty
        ):
            cutoff = returns_df.index.max() - pd.Timedelta(days=int(realised_apy_lookback_days))
            returns_df = returns_df.loc[returns_df.index >= cutoff]
            returns_df = returns_df.dropna(how="all")

        metadata = df.set_index("name")
        min_required = max(1, realised_apy_min_observations)

        for pool_name in metrics_index:
            series = returns_df.get(pool_name, pd.Series(dtype=float))
            series = series.dropna()
            observations = int(series.shape[0])
            warning: str | None = None
            realised_value = float("nan")

            if observations < min_required:
                shortfall_text = (
                    f"Only {min_required - 1} observations or fewer in history lookback "
                    f"(available {observations}; minimum {min_required})"
                )
                if realised_apy_lookback_days is None or realised_apy_lookback_days <= 0:
                    warning = shortfall_text
                else:
                    warning = (
                        f"{shortfall_text} within the past {int(realised_apy_lookback_days)} days"
                    )
            else:
                realised = _annualized_return(series)
                if realised is None:
                    warning = "Unable to annualise returns with available history"
                else:
                    realised_value = realised

            realised_metrics.loc[pool_name, "realised_apy"] = realised_value
            realised_metrics.loc[pool_name, "realised_apy_observations"] = observations
            if warning is not None:
                realised_metrics.loc[pool_name, "realised_apy_warning"] = warning
                warnings_records.append(
                    {
                        "pool": pool_name,
                        "observations": observations,
                        "message": warning,
                    }
                )

        if not returns_df.empty:
            weights = metadata.get("tvl_usd", pd.Series(dtype=float)).astype(float)
            returns_aligned = returns_df.reindex(columns=metadata.index)

            total_series = _weighted_portfolio_returns(returns_aligned, weights)
            metrics_map["total"] = _risk_metrics(total_series)

            for chain, names in metadata.groupby("chain").groups.items():
                chain_weights = weights.loc[list(names)]
                series = _weighted_portfolio_returns(returns_aligned, chain_weights)
                metrics_map[f"chain:{chain}"] = _risk_metrics(series)

            for stable, names in metadata.groupby("stablecoin").groups.items():
                stable_weights = weights.loc[list(names)]
                series = _weighted_portfolio_returns(returns_aligned, stable_weights)
                metrics_map[f"stablecoin:{stable}"] = _risk_metrics(series)

            for pool_name in returns_aligned.columns:
                metrics_map[f"pool:{pool_name}"] = _risk_metrics(returns_aligned[pool_name])

    df = df.join(realised_metrics, on="name")
    if "realised_apy" not in df.columns:
        df["realised_apy"] = pd.Series(dtype=float)
    if "realised_apy_observations" not in df.columns:
        df["realised_apy_observations"] = pd.Series(dtype="Int64")
    if "realised_apy_warning" not in df.columns:
        df["realised_apy_warning"] = pd.Series(dtype="string")

    paths["pools"] = out / "pools.csv"
    df.to_csv(paths["pools"], index=False)

    warnings_df = pd.DataFrame(warnings_records, columns=["pool", "observations", "message"])
    if warnings_df.empty:
        warnings_df = pd.DataFrame(columns=["pool", "observations", "message"])
    else:
        warnings_df["observations"] = warnings_df["observations"].astype("Int64")
        warnings_df["message"] = warnings_df["message"].astype("string")
    paths["warnings"] = out / "warnings.csv"
    warnings_df.to_csv(paths["warnings"], index=False)

    # Aggregations
    by_chain = Metrics.groupby_chain(repo)
    chain_realised = pd.DataFrame(
        columns=["chain", "realised_apy_avg", "realised_apy_wavg", "realised_apy_observations"]
    )
    if not df.empty:
        chain_realised = (
            df.groupby("chain")
            .agg(
                realised_apy_avg=("realised_apy", "mean"),
                realised_apy_wavg=(
                    "realised_apy",
                    lambda x: _tvl_weighted_average(x, df.loc[x.index, "tvl_usd"]),
                ),
                realised_apy_observations=(
                    "realised_apy_observations",
                    lambda x: int(x.fillna(0).sum()),
                ),
            )
            .reset_index()
        )
    chain_realised = chain_realised.astype({"realised_apy_observations": "Int64"}, errors="ignore")
    if by_chain.empty:
        by_chain = chain_realised
    else:
        by_chain = by_chain.merge(chain_realised, on="chain", how="left")
    paths["by_chain"] = out / "by_chain.csv"
    by_chain.to_csv(paths["by_chain"], index=False)

    def _agg(df: pd.DataFrame, key: str) -> pd.DataFrame:
        if df.empty:
            return df
        g = df.groupby(key).agg(
            pools=("name", "count"),
            tvl=("tvl_usd", "sum"),
            apr_avg=("base_apy", "mean"),
            apr_wavg=(
                "base_apy",
                lambda x: (x * df.loc[x.index, "tvl_usd"]).sum() / df.loc[x.index, "tvl_usd"].sum(),
            ),
        )
        return g.reset_index()

    by_source = _agg(df, "source")
    by_stable = _agg(df, "stablecoin")
    paths["by_source"] = out / "by_source.csv"
    paths["by_stablecoin"] = out / "by_stablecoin.csv"
    by_source.to_csv(paths["by_source"], index=False)
    by_stable.to_csv(paths["by_stablecoin"], index=False)

    # Top N
    top = df.sort_values("base_apy", ascending=False).head(top_n)
    paths["topN"] = out / "topN.csv"
    top.to_csv(paths["topN"], index=False)

    # Concentration metrics enriched with realised risk metrics
    hhi_total = Metrics.hhi(df, value_col="tvl_usd")
    hhi_chain = Metrics.hhi(df, value_col="tvl_usd", group_col="chain")
    hhi_stable = Metrics.hhi(df, value_col="tvl_usd", group_col="stablecoin")

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

    return paths
