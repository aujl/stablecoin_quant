from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import Metrics, PoolRepository

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


def cross_section_report(
    repo: PoolRepository,
    outdir: str | Path,
    *,
    perf_fee_bps: float = 0.0,
    mgmt_fee_bps: float = 0.0,
    top_n: int = 20,
    returns: pd.DataFrame | None = None,
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
    """
    out = _ensure_outdir(outdir)
    paths: dict[str, Path] = {}

    df = repo.to_dataframe()
    df = Metrics.add_net_apy_column(df, perf_fee_bps=perf_fee_bps, mgmt_fee_bps=mgmt_fee_bps)
    paths["pools"] = out / "pools.csv"
    df.to_csv(paths["pools"], index=False)

    # Aggregations
    by_chain = Metrics.groupby_chain(repo)
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

    metrics_map: dict[str, dict[str, float]] = {}
    if returns is not None and not returns.empty and not df.empty:
        returns_df = returns.copy()
        returns_df = returns_df.sort_index()
        returns_df = returns_df.apply(pd.to_numeric, errors="coerce")

        metadata = df.set_index("name")
        returns_df = returns_df.reindex(columns=metadata.index)

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

    return paths
