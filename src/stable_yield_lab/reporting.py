from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import Metrics, PoolRepository, attribution


def _ensure_outdir(outdir: str | Path) -> Path:
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    return p


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
    attribution_console: bool = False,
    return_attribution: bool = False,
) -> dict[str, Path] | tuple[dict[str, Path], attribution.AttributionResult | None]:
    """Generate file-first CSV outputs for the given snapshot repository.

    Writes the following CSVs:
      - pools.csv: all pools with net_apy column
      - by_chain.csv: aggregated by chain with TVL-weighted APY
      - by_source.csv: aggregated by source (protocol)
      - by_stablecoin.csv: aggregated by stablecoin symbol
      - topN.csv: top-N pools by base_apy
      - concentration.csv: HHI metrics across chain and stablecoin
      - attribution_by_pool.csv: realised APY contributions by pool (optional)
      - attribution_by_window.csv: realised APY contributions per rebalance window (optional)

    When ``return_attribution`` is ``True`` a tuple ``(paths, AttributionResult)``
    is returned, otherwise only the ``paths`` dictionary is produced. The
    attribution CSVs and optional console summary are emitted only when
    ``returns`` contains historical performance data.
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

    # Concentration metrics
    hhi_total = Metrics.hhi(df, value_col="tvl_usd")
    hhi_chain = Metrics.hhi(df, value_col="tvl_usd", group_col="chain")
    hhi_stable = Metrics.hhi(df, value_col="tvl_usd", group_col="stablecoin")
    conc = {
        "scope": ["total"],
        "hhi": list(hhi_total["hhi"]) if not hhi_total.empty else [float("nan")],
    }
    conc_df = pd.DataFrame(conc)
    hhi_chain["scope"] = "chain:" + hhi_chain["chain"].astype(str)
    hhi_stable["scope"] = "stablecoin:" + hhi_stable["stablecoin"].astype(str)
    conc_all = pd.concat([conc_df, hhi_chain[["scope", "hhi"]], hhi_stable[["scope", "hhi"]]], ignore_index=True)
    paths["concentration"] = out / "concentration.csv"
    conc_all.to_csv(paths["concentration"], index=False)

    attr_result: attribution.AttributionResult | None = None
    if returns is not None and not returns.empty:
        attr_result = attribution.compute_attribution(
            returns,
            weight_schedule,
            periods_per_year=attribution_periods_per_year,
        )
        pool_path = out / "attribution_by_pool.csv"
        window_path = out / "attribution_by_window.csv"
        attr_result.by_pool.to_csv(pool_path, index=False)
        attr_result.by_window.to_csv(window_path, index=False)
        paths["attribution_by_pool"] = pool_path
        paths["attribution_by_window"] = window_path

        if attribution_console:
            summarize_attribution(attr_result)

    if return_attribution:
        return paths, attr_result
    return paths


def summarize_attribution(result: attribution.AttributionResult, *, top_n: int = 5) -> None:
    """Print a concise attribution summary to stdout."""

    portfolio = result.portfolio
    total_apy = portfolio.get("realized_apy", float("nan"))
    total_return = portfolio.get("total_return", float("nan"))
    print(f"Realised APY: {total_apy:.2%} ({total_return:.2%} total return)")

    pool_df = result.by_pool.sort_values("apy_contribution", ascending=False).head(top_n)
    if not pool_df.empty:
        print("Top pool contributions:")
        for _, row in pool_df.iterrows():
            print(
                f"  {row['pool']}: {row['apy_contribution']:.2%} APY share"
                f" ({row['return_share']:.2%} of total return)"
            )

    window_df = result.by_window.sort_values("window_start").head(top_n)
    if not window_df.empty:
        print("Rebalance window contributions:")
        for _, row in window_df.iterrows():
            start = row["window_start"]
            end = row["window_end"]
            print(
                f"  {pd.to_datetime(start).date()} → {pd.to_datetime(end).date()}:"
                f" {row['apy_contribution']:.2%} APY share"
            )
