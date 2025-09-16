from __future__ import annotations

from collections.abc import Mapping, Sequence
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
    perf_fee_schedule: Sequence[tuple[float, float] | Mapping[str, float]] | None = None,
    mgmt_fee_schedule: Sequence[tuple[float, float] | Mapping[str, float]] | None = None,
    realized_returns: Mapping[str, Sequence[float] | pd.Series] | pd.DataFrame | None = None,
    periods_per_year: int | None = None,
    top_n: int = 20,
) -> dict[str, Path]:
    """Generate file-first CSV outputs for the given snapshot repository.

    Writes the following CSVs:
      - pools.csv: all pools with net_apy column
      - by_chain.csv: aggregated by chain with TVL-weighted APY
      - by_source.csv: aggregated by source (protocol)
      - by_stablecoin.csv: aggregated by stablecoin symbol
      - topN.csv: top-N pools by base_apy
      - concentration.csv: HHI metrics across chain and stablecoin
    Returns a dict of file label -> path for convenience.
    """
    out = _ensure_outdir(outdir)
    paths: dict[str, Path] = {}

    df = repo.to_dataframe()
    df = Metrics.add_net_apy_column(
        df,
        perf_fee_bps=perf_fee_bps,
        mgmt_fee_bps=mgmt_fee_bps,
        perf_fee_schedule=perf_fee_schedule,
        mgmt_fee_schedule=mgmt_fee_schedule,
        realized_returns=realized_returns,
        periods_per_year=periods_per_year,
    )
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

    return paths
