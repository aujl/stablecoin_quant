from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from . import Metrics, PoolRepository


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
    rolling_windows: Sequence[int] = (4, 12),
    periods_per_year: int = 52,
    target_field: str = "net_apy",
) -> dict[str, Path]:
    """Generate file-first CSV outputs for the given snapshot repository.

    Writes the following CSVs:
      - pools.csv: all pools with net_apy column
      - by_chain.csv: aggregated by chain with TVL-weighted APY
      - by_source.csv: aggregated by source (protocol)
      - by_stablecoin.csv: aggregated by stablecoin symbol
      - topN.csv: top-N pools by base_apy
      - concentration.csv: HHI metrics across chain and stablecoin
      - rolling_apy.csv: rolling annualised yields derived from historical returns
      - drawdowns.csv / drawdown_summary.csv: pathwise and summary drawdowns
      - realised_vs_target.csv: realised APY vs configured target metric
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

    if returns is not None and not returns.empty:
        aligned_returns = returns.sort_index()
        if not isinstance(aligned_returns.index, pd.DatetimeIndex):
            raise ValueError("returns must be indexed by timestamps")

        names_series = df.get("name")
        if names_series is None:
            pool_names: list[str] = []
        else:
            pool_names = list(dict.fromkeys(names_series.dropna().tolist()))
        available_names = [name for name in pool_names if name in aligned_returns.columns]

        if available_names:
            returns_subset = aligned_returns[available_names]

            # Rolling APY calculations (annualised from discrete compounding)
            rolling_frames: list[pd.DataFrame] = []
            returns_for_comp = returns_subset.fillna(0.0)
            for window in rolling_windows:
                if window <= 0:
                    continue
                if len(returns_for_comp) < window:
                    continue
                growth = (1.0 + returns_for_comp).rolling(window=window, min_periods=window).apply(
                    lambda arr: float(pd.Series(arr).prod()),
                    raw=True,
                )
                apy = growth.pow(periods_per_year / window) - 1.0
                tidy = (
                    apy.reset_index()
                    .rename(columns={apy.index.name or "index": "timestamp"})
                    .melt(id_vars="timestamp", var_name="name", value_name="rolling_apy")
                )
                tidy["window"] = window
                tidy = tidy.dropna(subset=["rolling_apy"])
                if not tidy.empty:
                    rolling_frames.append(tidy)

            if rolling_frames:
                rolling_df = pd.concat(rolling_frames, ignore_index=True)
                rolling_path = out / "rolling_apy.csv"
                rolling_df.to_csv(rolling_path, index=False)
                paths["rolling_apy"] = rolling_path

            # Drawdown trajectories
            nav = (1.0 + returns_for_comp).cumprod()
            running_max = nav.cummax()
            drawdowns = nav.divide(running_max).subtract(1.0)
            drawdown_long = (
                drawdowns.reset_index()
                .rename(columns={drawdowns.index.name or "index": "timestamp"})
                .melt(id_vars="timestamp", var_name="name", value_name="drawdown")
            )
            drawdown_path = out / "drawdowns.csv"
            drawdown_long.to_csv(drawdown_path, index=False)
            paths["drawdowns"] = drawdown_path

            drawdown_summary = pd.DataFrame(
                {
                    "name": drawdowns.columns,
                    "max_drawdown": drawdowns.min().values,
                    "current_drawdown": drawdowns.iloc[-1].values,
                }
            )
            summary_path = out / "drawdown_summary.csv"
            drawdown_summary.to_csv(summary_path, index=False)
            paths["drawdown_summary"] = summary_path

            # Realised vs target APY comparison
            realised_returns = returns_subset.mean()
            realised_apy = (1.0 + realised_returns) ** periods_per_year - 1.0
            df_targets = df.set_index("name")
            if target_field in df_targets.columns:
                targets = df_targets[target_field]
            else:
                targets = pd.Series(index=df_targets.index, dtype=float)
            targets = targets.reindex(realised_apy.index)
            comparison = pd.DataFrame(
                {
                    "name": realised_apy.index,
                    "realised_apy": realised_apy.values,
                    "target_apy": targets.values,
                }
            )
            comparison["realised_minus_target"] = comparison["realised_apy"] - comparison["target_apy"]
            comparison_path = out / "realised_vs_target.csv"
            comparison.to_csv(comparison_path, index=False)
            paths["realised_vs_target"] = comparison_path

    return paths
