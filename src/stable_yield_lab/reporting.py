from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import Metrics, PoolRepository, Visualizer


def _ensure_outdir(outdir: str | Path) -> Path:
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def cross_section_report(
    repo: PoolRepository,
    outdir: str | Path,
    period_returns: pd.DataFrame | None = None,
    *,
    perf_fee_bps: float = 0.0,
    mgmt_fee_bps: float = 0.0,
    top_n: int = 20,
    include_period_return_hist: bool = False,
    include_rolling_apy_box: bool = False,
    rolling_window: int = 30,
    rolling_periods_per_year: int = 52,
) -> dict[str, Path]:
    """Generate file-first CSV outputs for the given snapshot repository.

    Writes the following CSVs:
      - pools.csv: all pools with net_apy column
      - by_chain.csv: aggregated by chain with TVL-weighted APY
      - by_source.csv: aggregated by source (protocol)
      - by_stablecoin.csv: aggregated by stablecoin symbol
      - topN.csv: top-N pools by base_apy
      - concentration.csv: HHI metrics across chain and stablecoin
    Optional charts (PNG) are exported when ``period_returns`` is provided and
    the corresponding toggle is enabled:
      - period_return_histogram.png: histogram of periodic return samples
      - rolling_apy_boxplot.png: box plot of annualised rolling APYs

    Returns a dict of file label -> path for convenience.
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
    conc_all = pd.concat(
        [conc_df, hhi_chain[["scope", "hhi"]], hhi_stable[["scope", "hhi"]]], ignore_index=True
    )
    paths["concentration"] = out / "concentration.csv"
    conc_all.to_csv(paths["concentration"], index=False)

    returns_df = None
    if period_returns is not None:
        returns_df = period_returns.to_frame() if isinstance(period_returns, pd.Series) else period_returns
    if returns_df is not None and not returns_df.empty:
        if include_period_return_hist:
            hist_path = out / "period_return_histogram.png"
            Visualizer.hist_period_returns(
                returns_df,
                title="Distribution of Period Returns",
                save_path=str(hist_path),
                show=False,
            )
            paths["period_return_histogram"] = hist_path
        if include_rolling_apy_box:
            box_path = out / "rolling_apy_boxplot.png"
            Visualizer.boxplot_rolling_apy(
                returns_df,
                window=rolling_window,
                periods_per_year=rolling_periods_per_year,
                title=f"{rolling_window}-period Rolling APY Distribution",
                save_path=str(box_path),
                show=False,
            )
            paths["rolling_apy_boxplot"] = box_path

    return paths
